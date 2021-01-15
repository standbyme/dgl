from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue


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
            self.prefetch_iter_executor.submit(lambda: self.queue.put(self.raw__iter__()))

    def raw__iter__(self):
        return self.dataloader.__iter__()
