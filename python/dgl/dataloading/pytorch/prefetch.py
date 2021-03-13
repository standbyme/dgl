from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Callable

import torch
from dgl.dataloading.pytorch.presample import PreSampleDataLoader
from dgl.utils import memory_pool_add_track_stream
from torch.cuda import nvtx


@dataclass
class CommonArg:
    device: torch.device
    nfeat: torch.Tensor
    labels: torch.Tensor
    block_transform: Callable = None

    slice_queue: Queue = Queue()
    compute_queue: Queue = Queue()

    slice_stream = None


class PreDataLoaderIter:
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


class PreDataLoader:
    def __init__(self, dataloader, num_epochs: int, common_arg: CommonArg):
        self.common_arg = common_arg
        self.dataloader = PreSampleDataLoader(dataloader, num_epochs)
        self.num_epochs = num_epochs
        self.iter_count = 0

        memory_pool_add_track_stream(torch.cuda.current_stream())

        self.HtoD_stream = torch.cuda.Stream(device=common_arg.device)
        self.common_arg.slice_stream = torch.cuda.Stream(device=common_arg.device)

        self.buffers = Queue()
        self.init_buffers()

        self.transfer_queue: Queue = Queue()

        self.slice_thread = Thread(target=self.slice, daemon=True)
        self.transfer_thread = Thread(target=self.transfer, daemon=True)

        self.slice_thread.start()
        self.transfer_thread.start()

    def init_buffers(self):
        buffer_1 = torch.empty(2500000, 100).pin_memory()
        buffer_2 = torch.empty(2500000, 100).pin_memory()
        buffer_3 = torch.empty(2500000, 100).pin_memory()

        self.buffers.put_nowait(buffer_1)
        self.buffers.put_nowait(buffer_2)
        self.buffers.put_nowait(buffer_3)

    def __iter__(self):
        assert self.iter_count < self.num_epochs
        self.iter_count += 1

        return PreDataLoaderIter(self.raw__iter__(), self.common_arg)

    def raw__iter__(self):
        return self.dataloader.__iter__()

    def slice(self):
        while True:
            v = self.common_arg.slice_queue.get()
            if v is None:
                self.transfer_queue.put_nowait(None)
                continue

            input_nodes, seeds, blocks = v

            nvtx.range_push("dfs")
            buffer = self.buffers.get()
            torch.index_select(self.common_arg.nfeat, 0, input_nodes, out=buffer)
            nvtx.range_pop()

            if self.common_arg.block_transform:
                blocks = list(map(self.common_arg.block_transform, blocks))

            self.transfer_queue.put_nowait((blocks, buffer, seeds))

    def transfer(self):
        while True:
            v = self.transfer_queue.get()
            if v is None:
                self.common_arg.compute_queue.put_nowait(None)
                continue

            blocks_cpu, batch_inputs_cpu, seeds = v

            with torch.cuda.stream(self.HtoD_stream):
                nvtx.range_push("dg")
                blocks_device = [blk.to(self.common_arg.device, non_blocking=True) for blk in blocks_cpu]
                nvtx.range_pop()

                nvtx.range_push("dft")
                batch_inputs_device = batch_inputs_cpu.to(self.common_arg.device, non_blocking=True)
                nvtx.range_pop()

            self.buffers.put_nowait(batch_inputs_cpu)
            self.common_arg.compute_queue.put_nowait((blocks_device, batch_inputs_device, seeds))
