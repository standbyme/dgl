from dataclasses import dataclass

import torch
from dgl.utils.internal import memory_pool_sync_free
from dgl.dataloading.pytorch.presample import PreSampleDataLoader


@dataclass
class CommonArg:
    memory_pool_free_every: int
    nfeat: torch.Tensor


class PreDataLoaderIter:
    def __init__(self, dataloader_iter, common_arg: CommonArg):
        self.common_arg = common_arg
        self.dataloader_iter = dataloader_iter
        self.count = 0

    def __next__(self):
        self.count += 1
        if self.count % self.common_arg.memory_pool_free_every == 0:
            memory_pool_sync_free()

        return self.raw__next__()

    def raw__next__(self):
        return self.dataloader_iter.__next__()


class PreDataLoader:
    def __init__(self, dataloader, num_epochs: int, common_arg: CommonArg):
        self.common_arg = common_arg
        self.dataloader = PreSampleDataLoader(dataloader, num_epochs)
        self.num_epochs = num_epochs
        self.iter_count = 0

    def __iter__(self):
        assert self.iter_count < self.num_epochs
        self.iter_count += 1

        return PreDataLoaderIter(self.raw__iter__(), self.common_arg)

    def raw__iter__(self):
        return self.dataloader.__iter__()
