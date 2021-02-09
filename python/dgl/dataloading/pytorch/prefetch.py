from queue import Queue

from torch import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor

import torch
from torch.cuda import nvtx

import numpy as np
import pytorch_extension


class PreDataLoaderIter:
    def __init__(self, dataloader_iter, device, nfeat, labels, HtoD_stream, batch_inputs_conn, block_transform=None):
        self.block_transform = block_transform
        self.batch_inputs_conn = batch_inputs_conn
        self.HtoD_stream: torch.cuda.Stream = HtoD_stream
        self.labels = labels
        self.device = device
        self.nfeat = nfeat
        self.dataloader_iter = dataloader_iter

        self.is_first = True
        self.data_future = None  # Future[blocks, batch_inputs, batch_labels, seeds_length]
        self.prefetch_next_executor = ThreadPoolExecutor(max_workers=2)

        self.queue = Queue(maxsize=3)
        self.sample()

    def sample(self):
        def sample_f():
            try:
                while True:
                    v = self.raw__next__()
                    self.queue.put(v)
            except StopIteration:
                self.queue.put(None)

        self.prefetch_next_executor.submit(sample_f)

    def __next__(self):
        if self.is_first:
            self.is_first = False
            v = self.next_data()
        else:
            v = self.data_future.result()

        if v is None:
            raise StopIteration
        else:
            self.data_future = self.prefetch_next_executor.submit(self.next_data)

        return v

    def next_data(self):
        nvtx.range_push("w")
        v = self.queue.get()
        nvtx.range_pop()
        if v is None:
            return None

        input_nodes, seeds, blocks = v

        nvtx.range_push("send")
        self.batch_inputs_conn.send(input_nodes.to(self.device))
        nvtx.range_pop()

        if self.block_transform:
            blocks = list(map(self.block_transform, blocks))

        with torch.cuda.stream(self.HtoD_stream):
            nvtx.range_push("dg")
            blocks = [blk.to(self.device) for blk in blocks]
            nvtx.range_pop()

        nvtx.range_push("dl")
        batch_labels = self.labels[seeds]
        nvtx.range_pop()

        batch_inputs = self.batch_inputs_conn.recv()

        return blocks, batch_inputs, batch_labels, len(seeds)

    def raw__next__(self):
        return self.dataloader_iter.__next__()


def f(device, nfeat: torch.Tensor, conn):
    executor = ThreadPoolExecutor(max_workers=1)

    nfeat_size = nfeat.size()
    assert len(nfeat_size) == 2
    nfeat_dim = nfeat_size[1]

    prev_nodes_argsort_index = torch.empty(0, dtype=torch.int64, device=device)
    sorted_prev_nodes = torch.empty(0, dtype=torch.int64, device=device)
    prev = torch.empty(0, nfeat_dim, device=device)

    try:
        while True:
            curr_nodes_device = conn.recv()

            nvtx.range_push("pre")
            curr_nodes_argsort_index = torch.argsort(curr_nodes_device)
            sorted_curr_nodes = curr_nodes_device[curr_nodes_argsort_index]
            nvtx.range_pop()

            nvtx.range_push("c1")
            rest_nodes = pytorch_extension.intersect1d(sorted_curr_nodes, sorted_prev_nodes)
            nvtx.range_pop()

            nvtx.range_push("c2")
            rest_sorted_curr_index = torch.searchsorted(sorted_curr_nodes, rest_nodes)
            rest_curr_index = curr_nodes_argsort_index[rest_sorted_curr_index]

            rest_sorted_prev_index = torch.searchsorted(sorted_prev_nodes, rest_nodes)
            rest_prev_index = prev_nodes_argsort_index[rest_sorted_prev_index]
            nvtx.range_pop()

            nvtx.range_push("c3")
            supplement_nodes = pytorch_extension.setdiff1d(sorted_curr_nodes, sorted_prev_nodes)
            nvtx.range_pop()

            nvtx.range_push("c4")
            supplement_sorted_curr_index = torch.searchsorted(sorted_curr_nodes, supplement_nodes)
            supplement_curr_index = torch.take(curr_nodes_argsort_index, supplement_sorted_curr_index)
            nvtx.range_pop()

            nvtx.range_push("c5")
            curr = torch.empty(curr_nodes_device.size()[0], nfeat_dim, device=device)
            curr.index_copy_(0, rest_curr_index, prev[rest_prev_index])
            nvtx.range_pop()

            nvtx.range_push("dfs")
            nfeat_slice = nfeat[supplement_nodes]
            nvtx.range_pop()

            nvtx.range_push("dft")
            nfeat_slice_device = nfeat_slice.to(device, non_blocking=True)
            nvtx.range_pop()

            curr.index_copy_(0, supplement_curr_index, nfeat_slice_device)

            conn.send(curr)

            nvtx.range_push("del")
            del prev_nodes_argsort_index
            del sorted_prev_nodes
            del prev
            nvtx.range_pop()

            prev_nodes_argsort_index = curr_nodes_argsort_index
            sorted_prev_nodes = sorted_curr_nodes
            prev = curr

    except EOFError:
        pass


def init_feat_slice_process(device, nfeat):
    ctx = mp.get_context('spawn')

    parent_conn, child_conn = ctx.Pipe()

    feat_slice_process = ctx.Process(target=f, args=(device, nfeat, child_conn,))
    feat_slice_process.start()

    return feat_slice_process, parent_conn


class PreDataLoader:
    def __init__(self, dataloader, num_epochs: int, device, nfeat, labels, feat_slice_process_parent_conn,
                 block_transform=None):
        self.block_transform = block_transform
        self.HtoD_stream = torch.cuda.Stream(device=device)
        self.labels = labels
        self.nfeat = nfeat
        self.device = device
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.feat_slice_process_parent_conn = feat_slice_process_parent_conn

        self.iter_count = 0
        self.next_iter = None

    def __iter__(self):
        assert self.iter_count < self.num_epochs
        self.iter_count += 1

        if self.dataloader.is_distributed:
            return self.raw__iter__()
        else:
            pre_data_loader_iter = PreDataLoaderIter(self.raw__iter__(), self.device, self.nfeat, self.labels,
                                                     self.HtoD_stream, self.feat_slice_process_parent_conn,
                                                     self.block_transform)

            return pre_data_loader_iter

    def raw__iter__(self):
        return self.dataloader.__iter__()
