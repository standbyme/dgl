from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Callable

import torch
from torch.cuda import nvtx

from torch import multiprocessing as mp


@dataclass
class CommonArg:
    device: torch.device
    nfeat: torch.Tensor
    labels: torch.Tensor
    block_transform: Callable = None

    slice_queue: Queue = Queue()
    compute_queue: Queue = Queue()

    slice_stream = None


class PrefetchDataLoaderIter:
    def __init__(self, dataloader_iter, common_arg: CommonArg):
        self.common_arg = common_arg
        self.dataloader_iter = dataloader_iter

        self.is_finished = False

        self.fill_slice_queue()

    def fill_slice_queue(self):
        if not self.is_finished:
            try:
                v = self.raw__next__()
            except StopIteration:
                v = None
                self.is_finished = True

            self.common_arg.slice_queue.put_nowait(v)

    def __next__(self):
        self.fill_slice_queue()

        v = self.common_arg.compute_queue.get()
        if v is None:
            raise StopIteration

        blocks_device, batch_inputs_device, seeds = v

        with torch.cuda.stream(self.common_arg.slice_stream):
            nvtx.range_push("dl")
            batch_labels = self.common_arg.labels[seeds]
            nvtx.range_pop()

        return blocks_device, batch_inputs_device, batch_labels, len(seeds)

    def raw__next__(self):
        return self.dataloader_iter.__next__()


def transfer_f(queue: Queue, conn, buffers: Queue, device):
    HtoD_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(HtoD_stream):
        while True:
            buffer = queue.get()

            nvtx.range_push("dft")
            batch_inputs_device = buffer.to(device, non_blocking=False)
            nvtx.range_pop()

            conn.send(batch_inputs_device)

            del batch_inputs_device

            buffers.put_nowait(buffer)


def f(device, in_feats, conn):
    feature_dim = in_feats.shape[1]

    # init buffer: start
    buffers = Queue()
    total_node_amount_per_batch = 4096 * 5 * 10 * 15
    for _ in range(3):
        # noinspection PyArgumentList
        buffer = torch.empty(total_node_amount_per_batch, feature_dim, pin_memory=True)
        assert buffer.is_pinned()
        buffers.put_nowait(buffer)
    # init buffer: end

    queue: Queue = Queue()
    transfer_thread = Thread(target=transfer_f, daemon=True, args=(queue, conn, buffers, device,))
    transfer_thread.start()

    try:
        while True:
            input_nodes = conn.recv()
            input_nodes: torch.Tensor

            nvtx.range_push("dfs")
            buffer: torch.Tensor = buffers.get()
            buffer = buffer.resize_(input_nodes.shape[0], feature_dim)
            torch.index_select(in_feats, 0, input_nodes, out=buffer)
            nvtx.range_pop()

            queue.put_nowait(buffer)

            del input_nodes

    except EOFError:
        pass


class PrefetchDataLoader:
    def __init__(self, dataloader, num_epochs: int, common_arg: CommonArg):
        self.common_arg = common_arg
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.iter_count = 0

        self.HtoD_stream = torch.cuda.Stream(device=common_arg.device)
        self.common_arg.slice_stream = torch.cuda.Stream(device=common_arg.device)
        self.feature_dim = self.common_arg.nfeat.shape[1]

        self.buffers = Queue()
        self.init_buffers()

        self.transfer_queue: Queue = Queue()

        self.slice_thread = Thread(target=self.slice, daemon=True)
        self.transfer_thread = Thread(target=self.transfer, daemon=True)

        self.slice_thread.start()
        self.transfer_thread.start()

        self.ctx = mp.get_context('spawn')

        parent_conn, child_conn = self.ctx.Pipe()
        self.parent_conn = parent_conn

        feat_slice_process = self.ctx.Process(target=f,
                                              args=(self.common_arg.device, self.common_arg.nfeat, child_conn,),
                                              daemon=True)
        feat_slice_process.start()

    def init_buffers(self):
        total_node_amount_per_batch = 4096 * 5 * 10 * 15
        for _ in range(3):
            buffer = torch.empty(total_node_amount_per_batch, self.feature_dim).pin_memory()
            self.buffers.put_nowait(buffer)

    def __iter__(self):
        assert self.iter_count < self.num_epochs
        self.iter_count += 1

        return PrefetchDataLoaderIter(self.raw__iter__(), self.common_arg)

    def raw__iter__(self):
        return self.dataloader.__iter__()

    def slice(self):
        while True:
            v = self.common_arg.slice_queue.get()
            if v is None:
                self.transfer_queue.put_nowait(None)
                continue

            input_nodes, seeds, blocks = v

            self.parent_conn.send(input_nodes)

            if self.common_arg.block_transform:
                blocks = list(map(self.common_arg.block_transform, blocks))

            self.transfer_queue.put_nowait((blocks, seeds))

    def transfer(self):
        while True:
            v = self.transfer_queue.get()
            if v is None:
                self.common_arg.compute_queue.put_nowait(None)
                continue

            blocks_cpu, seeds = v

            with torch.cuda.stream(self.HtoD_stream):
                nvtx.range_push("dg")
                blocks_device = [blk.to(self.common_arg.device, non_blocking=True) for blk in blocks_cpu]
                nvtx.range_pop()

            batch_inputs_device = self.parent_conn.recv()
            self.common_arg.compute_queue.put_nowait((blocks_device, batch_inputs_device, seeds))
