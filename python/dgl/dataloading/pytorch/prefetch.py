from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue

import torch
from torch.cuda import nvtx


class PreDataLoaderIter:
    def __init__(self, dataloader_iter, device, nfeat, labels, HtoD_stream):
        self.HtoD_stream = HtoD_stream
        self.labels = labels
        self.device = device
        self.nfeat = nfeat
        self.dataloader_iter = dataloader_iter
        self.iter_queue = Queue()
        self.is_first = True
        self.data_future = None  # Future[blocks, batch_inputs, batch_labels, seeds]
        self.prefetch_next_executor = ThreadPoolExecutor(max_workers=3)

        self.prefetch_next_executor.submit(self.prefetch__next__)

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

        def f():
            with torch.cuda.stream(self.HtoD_stream):
                nvtx.range_push("dg")
                blocks_v = [blk.to(self.device, non_blocking=True) for blk in blocks]
                nvtx.range_pop()
                return blocks_v

        blocks_future = self.prefetch_next_executor.submit(f)

        nvtx.range_push("dfs")
        nfeat_slice = self.nfeat[input_nodes].pin_memory()
        nvtx.range_pop()

        nvtx.range_push("dl")
        batch_labels = self.labels[seeds]
        nvtx.range_pop()

        blocks = blocks_future.result()
        with torch.cuda.stream(self.HtoD_stream):
            nvtx.range_push("dft")
            batch_inputs = nfeat_slice.to(self.device, non_blocking=True)
            nvtx.range_pop()

        return blocks, batch_inputs, batch_labels, seeds

    def prefetch__next__(self):
        while True:
            try:
                result = self.raw__next__()
                self.iter_queue.put_nowait(result)
            except StopIteration:
                self.iter_queue.put_nowait(None)
                break

    def raw__next__(self):
        return self.dataloader_iter.__next__()


class PreDataLoader:
    def __init__(self, dataloader, num_epochs: int, device, nfeat, labels):
        self.HtoD_stream = torch.cuda.Stream(device=device)

        self.labels = labels
        self.nfeat = nfeat
        self.device = device
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.iter_count = 0
        self.prefetch_iter_executor = ThreadPoolExecutor(max_workers=num_epochs)
        self.queue = Queue()
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
                    PreDataLoaderIter(self.raw__iter__(), self.device, self.nfeat, self.labels, self.HtoD_stream)))

    def raw__iter__(self):
        return self.dataloader.__iter__()
