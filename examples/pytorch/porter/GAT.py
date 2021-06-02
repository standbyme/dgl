import torch as th
import torch.nn as nn

import dgl
import dgl.nn.pytorch as dglnn

import tqdm


class GAT(nn.Module):
    def __init__(self,
                 args,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 num_heads,
                 activation):
        super().__init__()
        self.args = args
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
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.args.num_workers)

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
