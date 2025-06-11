import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGE(torch.nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.dropout = args.dropout
        self.hidden = args.hidden_dim
        self.num_layers = args.stu_layers
        if 'ogb' in args.dataset:
            self.embedding = AtomEncoder(self.hidden)
        else:
            self.embedding = Linear(args.num_features, self.hidden)
        self.conv1 = SAGEConv(self.hidden, self.hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.hidden, self.hidden))
        self.BN = BatchNorm1d(self.hidden)
        self.BNs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.BNs.append(BatchNorm1d(self.hidden))

        self.lin1 = Linear(self.hidden, self.hidden)
        self.lin2 = Linear(self.hidden, args.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        x = self.conv1(x, edge_index)
        x = self.BN(x)
        middle_feats = []
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        middle_feats.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.BNs[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            middle_feats.append(x)
        x = global_mean_pool(x, batch)
        out = F.relu(self.lin1(x))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(out)
        return out, middle_feats, batch

    def __repr__(self):
        return self.__class__.__name__
