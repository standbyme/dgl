from torch import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor

import torch
from torch.cuda import nvtx


class PreDataLoaderIter:
    def __init__(self, dataloader_iter, device, nfeat, labels, HtoD_stream, batch_inputs_conn):
        self.batch_inputs_conn = batch_inputs_conn
        self.HtoD_stream: torch.cuda.Stream = HtoD_stream
        self.labels = labels
        self.device = device
        self.nfeat = nfeat
        self.dataloader_iter = dataloader_iter

        self.is_first = True
        self.data_future = None  # Future[blocks, batch_inputs, batch_labels, seeds_length]
        self.prefetch_next_executor = ThreadPoolExecutor(max_workers=1)

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
        try:
            v = self.raw__next__()
        except StopIteration:
            return None

        input_nodes, seeds, blocks = v

        nvtx.range_push("send")
        self.batch_inputs_conn.send(input_nodes)
        nvtx.range_pop()

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


def f(device, nfeat, conn):
    try:
        while True:
            input_nodes = conn.recv()

            nvtx.range_push("dfs")
            nfeat_slice = nfeat[input_nodes]
            nvtx.range_pop()

            nvtx.range_push("dft")
            batch_inputs = nfeat_slice.to(device, non_blocking=True)
            nvtx.range_pop()

            conn.send(batch_inputs)

            nvtx.range_push("del")
            del input_nodes
            del nfeat_slice
            del batch_inputs
            nvtx.range_pop()

    except EOFError:
        pass


def init_feat_slice_process(device, nfeat):
    ctx = mp.get_context('spawn')

    parent_conn, child_conn = ctx.Pipe()

    feat_slice_process = ctx.Process(target=f, args=(device, nfeat, child_conn,))
    feat_slice_process.start()

    return feat_slice_process, parent_conn


class PreDataLoader:
    def __init__(self, dataloader, num_epochs: int, device, nfeat, labels, feat_slice_process_parent_conn):
        self.HtoD_stream = torch.cuda.Stream(device=device)
        self.labels = labels
        self.nfeat = nfeat
        self.device = device
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.feat_slice_process_parent_conn = feat_slice_process_parent_conn

        self.iter_count = 0

    def __iter__(self):
        assert self.iter_count < self.num_epochs
        self.iter_count += 1

        if self.dataloader.is_distributed:
            return self.raw__iter__()
        else:
            pre_data_loader_iter = PreDataLoaderIter(self.raw__iter__(), self.device, self.nfeat, self.labels,
                                                     self.HtoD_stream, self.feat_slice_process_parent_conn)
            return pre_data_loader_iter

    def raw__iter__(self):
        return self.dataloader.__iter__()
