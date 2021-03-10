import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
from dgl.dataloading.pytorch.prefetch import PreDataLoader, CommonArg
from ogb.nodeproppred import DglNodePropPredDataset

from torch.cuda import nvtx

class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 num_heads,
                 activation):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv((in_feats, in_feats), n_hidden, num_heads=num_heads, activation=activation))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_hidden,
                                             num_heads=num_heads, activation=activation))
        self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_classes,
                                         num_heads=num_heads, activation=None))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h.log_softmax(dim=-1)

    def inference(self, g, x, num_heads, device):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                y = th.zeros(g.num_nodes(), self.n_hidden * num_heads if l != len(self.layers) - 1 else self.n_classes)
            else:
                y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.num_dst_nodes()]
                if l < self.n_layers - 1:
                    h = layer(block, (h, h_dst)).flatten(1) 
                else:
                    h = layer(block, (h, h_dst))
                    h = h.mean(1)
                    h = h.log_softmax(dim=-1)

                y[output_nodes] = h.cpu()

            x = y
        return y.to(device)

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, test_nid, num_heads, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, num_heads, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid]), pred

def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    th.index_select(nfeat, 0, input_nodes, out=buffer)
    batch_inputs = buffer.to(device)
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels

#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g, num_heads = data

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    pre_dataloader = PreDataLoader(dataloader, args.num_epochs, CommonArg(device, nfeat, labels, lambda x: x.int()))
    # Define model and optimizer
    model = GAT(in_feats, args.num_hidden, n_classes, args.num_layers, num_heads, F.relu)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        nvtx.range_push("e")
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (blocks, batch_inputs, batch_labels, seeds_length) in enumerate(pre_dataloader):
            tic_step = time.time()

            nvtx.range_push("c")
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = F.nll_loss(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nvtx.range_pop()  # c

            iter_tput.append(seeds_length / (time.time() - tic_step))
            if args.log and step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        nvtx.range_pop()  # e
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 2:
            avg += toc - tic
        if args.eval and epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, test_acc, pred = evaluate(model, g, nfeat, labels, val_nid, test_nid, num_heads, device)
            if args.save_pred:
                np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
            print('Eval Acc {:.4f}'.format(eval_acc))
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_test_acc = test_acc
            print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

    print('Avg Throughput: {}'.format(np.mean(iter_tput[3:])))
    return best_test_acc


def split_dataset_idx(_graph: dgl.DGLHeteroGraph):
    num_nodes = _graph.num_nodes()
    block_size = num_nodes // 100

    _train_idx_size = block_size
    _val_idx_size = block_size * 10
    _test_idx_size = block_size * 25

    _train_idx = th.linspace(0, _train_idx_size - 1, _train_idx_size, dtype=th.int64)
    _val_idx = th.linspace(_train_idx_size, _train_idx_size + _val_idx_size - 1, _val_idx_size, dtype=th.int64)
    _test_idx = th.linspace(_train_idx_size + _val_idx_size, _train_idx_size + _val_idx_size + _test_idx_size - 1,
                            _test_idx_size, dtype=th.int64)

    return _train_idx, _val_idx, _test_idx


def load_data(_dataset, _device):
    if _dataset == 'enwiki':
        edge = th.load('./dataset/wikipedia_link_en/edge.pt').long()
        _u, _v = edge[0], edge[1]
        _graph = dgl.graph((_u, _v))
        _nfeat = th.rand((_graph.num_nodes(), 100), dtype=th.float32)
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
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=512)
    argparser.add_argument('--val-batch-size', type=int, default=512)
    argparser.add_argument("--log", action='store_true', default=False)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument("--eval", action='store_true', default=False)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--num-workers', type=int, default=8,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--head', type=int, default=4)
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--dataset', type=str, required=True, choices=['arxiv', 'products', 'mag', 'enwiki'])
    args = argparser.parse_args()
    
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    train_idx, val_idx, test_idx, labels, nfeat, graph = load_data(args.dataset, device)

    in_feats = nfeat.shape[1]
    n_classes = (labels.max() + 1).item()

    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph, args.head

    # Run 10 times
    test_accs = []
    for i in range(args.num_times):
        v = run(args, device, data)
        if v != 0:
            v = v.cpu().numpy()
        test_accs.append(v)
        print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
