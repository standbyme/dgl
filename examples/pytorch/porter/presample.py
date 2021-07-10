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
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        v = PreSampleDataLoaderIter(self.raw__iter__())
        return v

    def raw__iter__(self):
        return self.dataloader.__iter__()
