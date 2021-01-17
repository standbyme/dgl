from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue


class PreDataLoaderIter:
    def __init__(self, dataloader_iter):
        self.dataloader_iter = dataloader_iter
        self.prefetch_next_executor = ThreadPoolExecutor(max_workers=1)
        self.queue = Queue()
        self.prefetch_next_executor.submit(self.prefetch__next__)

    def __next__(self):
        v = self.queue.get()

        if v is None:
            raise StopIteration
        return v

    def prefetch__next__(self):
        while True:
            try:
                result = self.raw__next__()
                self.queue.put(result)
            except StopIteration:
                self.queue.put(None)
                break

    def raw__next__(self):
        return self.dataloader_iter.__next__()


class PreDataLoader:
    def __init__(self, dataloader, num_epochs: int):
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.prefetch_iter_executor = ThreadPoolExecutor(max_workers=num_epochs)
        self.queue = Queue()
        self.pre__iter__()
        self.iter_count = 0

    def __iter__(self):
        assert self.iter_count < self.num_epochs
        self.iter_count += 1

        if self.dataloader.is_distributed:
            return self.dataloader.__iter__()
        else:
            return self.queue.get()

    def pre__iter__(self):
        for _ in range(self.num_epochs):
            self.prefetch_iter_executor.submit(lambda: self.queue.put(PreDataLoaderIter(self.raw__iter__())))

    def raw__iter__(self):
        return self.dataloader.__iter__()
