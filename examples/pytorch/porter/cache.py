from dataclasses import dataclass

import torch
from torch.cuda import nvtx

import numpy as np


@dataclass
class CompressResult:
    supplement_nodes: torch.Tensor
    supplement_curr_index: torch.Tensor

    rest_prev_index: torch.Tensor
    rest_curr_index: torch.Tensor


class Cache:
    def __init__(self, device, nfeat_dim: int):
        self.device = device
        self.nfeat_dim = nfeat_dim

    def compress(self, curr_nodes_device: torch.Tensor) -> CompressResult:
        pass

    def decompress(self, compress_result: CompressResult, supplement_nfeat_slice: torch.Tensor) -> torch.Tensor:
        pass


class RecycleCache(Cache):
    def __init__(self, device, nfeat_dim: int):
        super().__init__(device, nfeat_dim)

        self.prev_nodes_argsort_index = torch.empty(0, dtype=torch.int64, device=device)
        self.sorted_prev_nodes = torch.empty(0, dtype=torch.int64, device=device)
        self.prev = torch.empty(0, self.nfeat_dim, device=device)

    def compress(self, curr_nodes_device: torch.Tensor) -> CompressResult:
        assert curr_nodes_device.is_cuda

        nvtx.range_push("pre")
        curr_nodes_argsort_index = torch.argsort(curr_nodes_device)
        sorted_curr_nodes = curr_nodes_device[curr_nodes_argsort_index]
        nvtx.range_pop()

        nvtx.range_push("c1")
        rest_nodes = np.intersect1d(sorted_curr_nodes.cpu(), self.sorted_prev_nodes.cpu())
        nvtx.range_pop()

        nvtx.range_push("c2")
        rest_sorted_curr_index = torch.searchsorted(sorted_curr_nodes, torch.from_numpy(rest_nodes).cuda())
        rest_curr_index = curr_nodes_argsort_index[rest_sorted_curr_index]

        rest_sorted_prev_index = torch.searchsorted(self.sorted_prev_nodes, torch.from_numpy(rest_nodes).cuda())
        rest_prev_index = self.prev_nodes_argsort_index[rest_sorted_prev_index]
        nvtx.range_pop()

        nvtx.range_push("c3")
        supplement_nodes = np.setdiff1d(sorted_curr_nodes.cpu(), self.sorted_prev_nodes.cpu())
        nvtx.range_pop()

        nvtx.range_push("c4")
        supplement_sorted_curr_index = torch.searchsorted(sorted_curr_nodes, torch.from_numpy(supplement_nodes).cuda())
        supplement_curr_index = torch.take(curr_nodes_argsort_index, supplement_sorted_curr_index)
        nvtx.range_pop()

        self.prev_nodes_argsort_index = curr_nodes_argsort_index
        self.sorted_prev_nodes = sorted_curr_nodes

        return CompressResult(torch.from_numpy(supplement_nodes), supplement_curr_index
                              , rest_prev_index, rest_curr_index)

    def decompress(self, compress_result: CompressResult, supplement_nfeat_slice: torch.Tensor) -> torch.Tensor:
        assert supplement_nfeat_slice.is_cuda
        assert supplement_nfeat_slice.shape[0] == compress_result.supplement_nodes.shape[0]
        assert supplement_nfeat_slice.shape[0] == compress_result.supplement_curr_index.shape[0]

        nvtx.range_push("decompress")
        size = len(compress_result.supplement_curr_index) + len(compress_result.rest_curr_index)
        curr = torch.empty(size, self.nfeat_dim, device=self.device)
        curr.index_copy_(0, compress_result.rest_curr_index, self.prev[compress_result.rest_prev_index])
        curr.index_copy_(0, compress_result.supplement_curr_index, supplement_nfeat_slice)
        nvtx.range_pop()

        self.prev = curr

        return curr
