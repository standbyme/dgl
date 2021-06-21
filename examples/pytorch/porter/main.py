import dgl
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import nvtx

import time
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.reddit import RedditDataset

from GCN import GCN
from GraphSAGE import SAGE
from GAT import GAT
from cache import RecycleCache, CompressArg

from dgl.utils import memory_pool_add_track_stream


def compute_acc(pred, _labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == _labels).float().sum() / len(pred)


def evaluate(model, g, _nfeat, _labels, val_nid, test_nid, _device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        if args.model == "gcn":
            raise NotImplemented()
        elif args.model == "graphsage":
            pred = model.inference(g, _nfeat, _device)
        elif args.model == "gat":
            pred = model.inference(g, _nfeat, args.head, _device)
        else:
            raise NotImplemented()
    model.train()
    return compute_acc(pred[val_nid], _labels[val_nid]), compute_acc(pred[test_nid], _labels[test_nid]), pred


def load_subtensor(_nfeat, _labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = _nfeat[input_nodes].to(device)
    batch_labels = _labels[seeds]
    return batch_inputs, batch_labels


def run(_args, _device, _data):
    # Unpack data
    train_nid, val_nid, test_nid, _in_feats, _labels, _n_classes, _nfeat, g = _data

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in _args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=_args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=_args.num_workers)

    # Define model and optimizer
    if _args.model == "gcn":
        model = GCN(_in_feats, _args.num_hidden, _n_classes, _args.num_layers, F.relu, _args.dropout)
    elif _args.model == "graphsage":
        model = SAGE(_args, _in_feats, _args.num_hidden, _n_classes, _args.num_layers, F.relu, _args.dropout)
    elif _args.model == "gat":
        model = GAT(_args, _in_feats, _args.num_hidden, _n_classes, _args.num_layers, _args.head, F.relu)
    else:
        raise NotImplemented()

    model = model.to(_device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=_args.lr, weight_decay=_args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(_args.num_epochs):
        nvtx.range_push("e")
        tic = time.time()

        prev_index_map: th.Tensor = th.empty(graph.num_nodes(), dtype=th.int64, device=device)
        prev_nodes = th.empty(0, dtype=th.int64, device=device)
        compress_arg = CompressArg(prev_index_map, prev_nodes)
        prev = th.empty(0, in_feats, device=device)

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            compress_result = cache.compress(input_nodes.cuda(), compress_arg)

            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.int().to(_device) for blk in blocks]

            # Load the input features as well as output labels
            supplement_nfeat_slice, batch_labels = load_subtensor(_nfeat, _labels, seeds
                                                                  , compress_result.decompress_arg.supplement_nodes)
            batch_inputs = cache.decompress(compress_result.decompress_arg, supplement_nfeat_slice, prev)

            nvtx.range_push("c")
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nvtx.range_pop()  # c

            compress_arg = compress_result.compress_arg
            prev = batch_inputs

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if _args.log and step % _args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print(
                    f'Epoch {epoch:05d} | Step {step:05d} | Loss {loss.item():.4f} | Train Acc {acc.item():.4f} | '
                    f'Speed (samples/sec) {np.mean(iter_tput[3:]):.4f} | GPU {gpu_mem_alloc:.1f} MB')

        toc = time.time()
        nvtx.range_pop()  # e
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 2:
            avg += toc - tic

        if _args.eval and epoch % _args.eval_every == 0 and epoch != 0:
            eval_acc, test_acc, pred = evaluate(model, g, _nfeat, _labels, val_nid, test_nid, _device)
            if _args.save_pred:
                np.savetxt(_args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
            print('Eval Acc {:.4f}'.format(eval_acc))
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_test_acc = test_acc
            print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 1)))
    return best_test_acc


def split_dataset_idx(_graph: dgl.DGLHeteroGraph):
    block_size = 200_000

    _train_idx_size = block_size
    _val_idx_size = block_size * 10
    _test_idx_size = block_size * 25

    _train_idx = th.linspace(0, _train_idx_size - 1, _train_idx_size, dtype=th.int64)
    _val_idx = th.linspace(_train_idx_size, _train_idx_size + _val_idx_size - 1, _val_idx_size, dtype=th.int64)
    _test_idx = th.linspace(_train_idx_size + _val_idx_size, _train_idx_size + _val_idx_size + _test_idx_size - 1,
                            _test_idx_size, dtype=th.int64)

    return _train_idx, _val_idx, _test_idx


def load_data(_args, _device):
    _dataset = _args.dataset
    if _args.env == "tiny":
        feat_dim = 100
    elif _args.env == "large":
        feat_dim = 500
    else:
        raise NotImplemented()

    if _dataset == 'reddit':
        reddit_dataset = RedditDataset()
        _graph = reddit_dataset[0]
        _nfeat = _graph.ndata['feat']
        _labels = _graph.ndata['label'].to(_device)
        _train_idx, _val_idx, _test_idx = split_dataset_idx(_graph)
    elif _dataset == 'livejournal':
        edge = th.load('./dataset/livejournal/edge.pt').long()
        _u, _v = edge[0], edge[1]
        _graph = dgl.graph((_u, _v))
        _nfeat = th.rand((_graph.num_nodes(), feat_dim), dtype=th.float32)
        _labels = th.randint(10, (_graph.num_nodes(),), device=_device, dtype=th.int64)
        _train_idx, _val_idx, _test_idx = split_dataset_idx(_graph)
    elif _dataset == 'enwiki':
        edge = th.load('./dataset/wikipedia_link_en/edge.pt').long()
        _u, _v = edge[0], edge[1]
        _graph = dgl.graph((_u, _v))
        _nfeat = th.rand((_graph.num_nodes(), feat_dim), dtype=th.float32)
        _labels = th.randint(10, (_graph.num_nodes(),), device=_device, dtype=th.int64)
        _train_idx, _val_idx, _test_idx = split_dataset_idx(_graph)
    else:
        _data = DglNodePropPredDataset(name=f'ogbn-{_dataset}')
        _splitted_idx = _data.get_idx_split()
        _train_idx, _val_idx, _test_idx = _splitted_idx['train'], _splitted_idx['valid'], _splitted_idx['test']
        if _dataset == "mag":
            _train_idx, _val_idx, _test_idx = _train_idx['paper'], _val_idx['paper'], _test_idx['paper']

        _graph, _labels = _data[0]
        if _dataset == "mag":
            _graph = dgl.edge_type_subgraph(_graph, [('paper', 'cites', 'paper')])
            _labels = _labels["paper"]

        _nfeat = _graph.ndata.pop('feat')
        _labels = _labels[:, 0].to(_device)

    return _train_idx, _val_idx, _test_idx, _labels, _nfeat, _graph


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-times', type=int, default=10)
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument("--log", action='store_true', default=False)
    argparser.add_argument("--eval", action='store_true', default=False)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--head', type=int, default=4)
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--model', type=str, required=True, choices=['gcn', 'graphsage', 'gat'])
    argparser.add_argument('--dataset', type=str, required=True,
                           choices=['reddit', 'arxiv', 'products', 'mag', 'enwiki', 'livejournal'])
    argparser.add_argument('--env', type=str, required=True,
                           choices=['tiny', 'large'])
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    train_idx, val_idx, test_idx, labels, nfeat, graph = load_data(args, device)

    if args.model == "gcn" or args.model == "gat":
        print('Total edges before adding self-loop {}'.format(graph.num_edges()))
        graph = graph.remove_self_loop().add_self_loop()
        print('Total edges after adding self-loop {}'.format(graph.num_edges()))

    in_feats = nfeat.shape[1]
    n_classes = (labels.max() + 1).item()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph
    cache = RecycleCache(device, in_feats)

    memory_pool_add_track_stream(th.cuda.current_stream())

    # Run 10 times
    test_accs = []
    for i in range(args.num_times):
        v = run(args, device, data)
        if v != 0:
            v = v.cpu().numpy()
        test_accs.append(v)
        print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
