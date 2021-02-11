from queue import Queue, Empty


class PreSampleDataLoaderIter:
    def __init__(self, dataloader_iter):
        self.dataloader_iter = dataloader_iter

        self.queue = Queue()

        self.init_queue()

    def __next__(self):
        try:
            v = self.queue.get_nowait()
        except Empty:
            raise StopIteration
        return v

    def raw__next__(self):
        return self.dataloader_iter.__next__()

    def init_queue(self):
        try:
            while True:
                v = self.raw__next__()
                self.queue.put_nowait(v)
        except StopIteration:
            pass


class PreSampleDataLoader:
    def __init__(self, dataloader, num_epochs: int):
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.iter_count = 0

        self.iter_queue = Queue()

        self.init_queue()

    def __iter__(self):
        assert self.iter_count < self.num_epochs
        self.iter_count += 1

        if self.dataloader.is_distributed:
            return self.dataloader.__iter__()
        else:
            return self.iter_queue.get_nowait()

    def raw__iter__(self):
        return self.dataloader.__iter__()

    def init_queue(self):
        for _ in range(self.num_epochs):
            pre_data_loader_iter = PreSampleDataLoaderIter(self.raw__iter__())
            self.iter_queue.put_nowait(pre_data_loader_iter)
