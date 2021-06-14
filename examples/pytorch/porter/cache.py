from dataclasses import dataclass

import torch
from torch.cuda import nvtx

import pytorch_extension


@dataclass
class CompressArg:
    prev_nodes_argsort_index: torch.Tensor
    sorted_prev_nodes: torch.Tensor


@dataclass
class DecompressArg:
    supplement_nodes: torch.Tensor
    supplement_curr_index: torch.Tensor

    rest_prev_index: torch.Tensor
    rest_curr_index: torch.Tensor


@dataclass
class CompressResult:
    compress_arg: CompressArg
    decompress_arg: DecompressArg


class RecycleCache:
    def __init__(self, device, nfeat_dim: int):
        self.device = device
        self.nfeat_dim = nfeat_dim

        self.stream = torch.cuda.Stream(device=device)
        self.stream_ptr = self.stream.cuda_stream

    def compress(self, curr_nodes_device: torch.Tensor, compress_arg: CompressArg) -> CompressResult:
        assert curr_nodes_device.is_cuda

        with torch.cuda.stream(self.stream):
            nvtx.range_push("pre")
            curr_nodes_argsort_index = torch.argsort(curr_nodes_device)
            sorted_curr_nodes = curr_nodes_device[curr_nodes_argsort_index]
            nvtx.range_pop()

            nvtx.range_push("c1")
            rest_nodes = pytorch_extension.intersect1d(sorted_curr_nodes, compress_arg.sorted_prev_nodes,
                                                       self.stream_ptr)
            nvtx.range_pop()

            nvtx.range_push("c2")
            rest_sorted_curr_index = torch.searchsorted(sorted_curr_nodes, rest_nodes)
            rest_curr_index = curr_nodes_argsort_index[rest_sorted_curr_index]

            rest_sorted_prev_index = torch.searchsorted(compress_arg.sorted_prev_nodes, rest_nodes)
            rest_prev_index = compress_arg.prev_nodes_argsort_index[rest_sorted_prev_index]
            nvtx.range_pop()

            nvtx.range_push("c3")
            supplement_nodes = pytorch_extension.setdiff1d(sorted_curr_nodes, compress_arg.sorted_prev_nodes,
                                                           self.stream_ptr)
            nvtx.range_pop()

            nvtx.range_push("c4")
            supplement_sorted_curr_index = torch.searchsorted(sorted_curr_nodes,
                                                              supplement_nodes.to(self.device))
            supplement_curr_index = torch.take(curr_nodes_argsort_index, supplement_sorted_curr_index)
            nvtx.range_pop()

            compress_arg = CompressArg(curr_nodes_argsort_index, sorted_curr_nodes)
            decompress_arg = DecompressArg(supplement_nodes, supplement_curr_index, rest_prev_index, rest_curr_index)

        self.stream.synchronize()
        return CompressResult(compress_arg, decompress_arg)

    def decompress(self, decompress_arg: DecompressArg,
                   supplement_nfeat_slice: torch.Tensor,
                   prev: torch.Tensor) -> torch.Tensor:
        assert supplement_nfeat_slice.is_cuda
        assert supplement_nfeat_slice.shape[0] == decompress_arg.supplement_nodes.shape[0]
        assert supplement_nfeat_slice.shape[0] == decompress_arg.supplement_curr_index.shape[0]

        nvtx.range_push("decompress")
        size = len(decompress_arg.supplement_curr_index) + len(decompress_arg.rest_curr_index)
        curr = torch.empty(size, self.nfeat_dim, device=self.device)
        curr.index_copy_(0, decompress_arg.rest_curr_index, prev[decompress_arg.rest_prev_index])
        curr.index_copy_(0, decompress_arg.supplement_curr_index, supplement_nfeat_slice)
        nvtx.range_pop()

        return curr
