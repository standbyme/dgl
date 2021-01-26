from torch import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue

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
        self.iter_queue = Queue()
        self.is_first = True
        self.data_future = None  # Future[blocks, batch_inputs, batch_labels, seeds_length]
        self.prefetch_next_executor = ThreadPoolExecutor(max_workers=2)

        self.prefetch_next_executor.submit(self.pre__next__)

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
        v = self.iter_queue.get()
        if v is None:
            return None

        input_nodes, seeds, blocks = v

        nvtx.range_push("send")
        self.batch_inputs_conn.send(input_nodes)
        nvtx.range_pop()

        with torch.cuda.stream(self.HtoD_stream):
            nvtx.range_push("dg")
            blocks = [blk.to(self.device, non_blocking=True) for blk in blocks]
            nvtx.range_pop()

        nvtx.range_push("dl")
        batch_labels = self.labels[seeds]
        nvtx.range_pop()

        batch_inputs = self.batch_inputs_conn.recv()

        return blocks, batch_inputs, batch_labels, len(seeds)

    def pre__next__(self):
        while True:
            try:
                result = self.raw__next__()
                self.iter_queue.put_nowait(result)
            except StopIteration:
                self.iter_queue.put_nowait(None)
                break

    def raw__next__(self):
        return self.dataloader_iter.__next__()


def f(device, nfeat, conn):
    try:
        while True:
            input_nodes = conn.recv()

            nvtx.range_push("dfs")
            nfeat_slice = nfeat[input_nodes]
            nvtx.range_pop()

            nvtx.range_push("pin")
            nfeat_slice_pin = nfeat_slice.pin_memory()
            nvtx.range_pop()

            nvtx.range_push("dft")
            batch_inputs = nfeat_slice_pin.to(device, non_blocking=True)
            nvtx.range_pop()

            conn.send(batch_inputs)

            nvtx.range_push("del")
            del input_nodes
            del nfeat_slice
            del nfeat_slice_pin
            del batch_inputs
            nvtx.range_pop()

    except EOFError:
        pass


class PreDataLoader:
    def __init__(self, dataloader, num_epochs: int, device, nfeat, labels):
        self.HtoD_stream = torch.cuda.Stream(device=device)
        self.labels = labels
        self.nfeat = nfeat
        self.device = device
        self.dataloader = dataloader
        self.num_epochs = num_epochs

        parent_conn, child_conn = mp.Pipe()
        self.parent_conn = parent_conn

        feat_slice_process = mp.Process(target=f, args=(device, nfeat, child_conn,))
        feat_slice_process.start()

        self.prefetch_iter_executor = ThreadPoolExecutor(max_workers=num_epochs)
        self.queue = Queue()
        self.iter_count = 0

        self.pre__iter__()

    def __iter__(self):
        assert self.iter_count < self.num_epochs
        self.iter_count += 1

        if self.dataloader.is_distributed:
            return self.dataloader.__iter__()
        else:
            return self.queue.get()

    def pre__iter__(self):
        for _ in range(self.num_epochs):
            self.prefetch_iter_executor.submit(
                lambda: self.queue.put_nowait(
                    PreDataLoaderIter(self.raw__iter__(), self.device, self.nfeat, self.labels, self.HtoD_stream,
                                      self.parent_conn)))

    def raw__iter__(self):
        return self.dataloader.__iter__()
