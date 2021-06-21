from dataclasses import dataclass

import torch
from torch.cuda import nvtx


@dataclass
class CompressArg:
    prev_index_map: torch.Tensor
    prev_nodes: torch.Tensor


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

        prev_index_map = compress_arg.prev_index_map
        prev_nodes = compress_arg.prev_nodes

        with torch.cuda.stream(self.stream):
            prev_index_map.fill_(-1)
            prev_index_map.index_copy_(0, prev_nodes, torch.arange(prev_nodes.shape[0], device=self.device))

            mix = prev_index_map[curr_nodes_device]
            rest_flag: torch.Tensor = (mix >= 0)
            supplement_flag: torch.Tensor = ~rest_flag

            rest_curr_index: torch.Tensor = rest_flag.nonzero(as_tuple=True)[0]
            rest_prev_index: torch.Tensor = mix[rest_flag]

            supplement_nodes: torch.Tensor = curr_nodes_device[supplement_flag]
            supplement_curr_index: torch.Tensor = supplement_flag.nonzero(as_tuple=True)[0]

            compress_arg = CompressArg(prev_index_map, curr_nodes_device)
            decompress_arg = DecompressArg(supplement_nodes, supplement_curr_index, rest_prev_index, rest_curr_index)

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
