import torch
from torch_geometric.utils import add_self_loops, to_undirected, coalesce


@torch.no_grad()
def prep_edges(edge_index, N):
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    edge_index     = to_undirected(edge_index, num_nodes=N)
    edge_index, _  = coalesce(edge_index, None, N, N)
    return edge_index


@torch.no_grad()
def node_homophily(edge_index, labels):
    src, dst = edge_index
    return (labels[src] == labels[dst]).float().mean().item()
