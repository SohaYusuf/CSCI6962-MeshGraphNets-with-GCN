import torch.nn as nn
from .blocks import EdgeBlock, NodeBlock
from utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, out_size))
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module


class GCNModel(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, lay_norm=True):
        super(GCNModel, self).__init__()

        layers = []
        
        # Input layer
        layers.append(GCNConv(in_size, hidden_size))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(GCNConv(hidden_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
        
        # Output layer
        layers.append(GCNConv(hidden_size, out_size))
        
        # Optional layer normalization
        if lay_norm:
            layers.append(nn.LayerNorm(normalized_shape=out_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

# def build_gcn(in_size, hidden_size, out_size, num_layers=1, lay_norm=True):
#     layers = []
    
#     # Input layer
#     layers.append(GCNConv(in_size, hidden_size))
#     layers.append(nn.SiLU(inplace=True))
    
#     # Hidden layers
#     for _ in range(num_layers - 2):
#         layers.append(GCNConv(hidden_size, hidden_size))
#         layers.append(nn.SiLU(inplace=True))
    
#     # Output layer
#     layers.append(GCNConv(hidden_size, out_size))
    
#     # Optional layer normalization
#     if lay_norm:
#         layers.append(nn.LayerNorm(normalized_shape=out_size))

#     return nn.Sequential(*layers)


def build_gcn(in_size, hidden_size, out_size, num_layers=1, lay_norm=True):
    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn = GCNConv(in_size, hidden_size)
            self.out = nn.Linear(hidden_size, out_size)
        def forward(self, x, edge_index):
            h = self.gcn(x, edge_index).relu()
            z = self.out(h)
            return z
    model = GCN()
    return model

class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        node_attr, _, edge_attr, _ = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)
        
        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)



class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()


        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        # nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        # eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)

        nb_custom_func = build_gcn(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_gcn(eb_input_dim, hidden_size, hidden_size)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)



class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size)
        
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)
        
        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph):

        graph= self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded







