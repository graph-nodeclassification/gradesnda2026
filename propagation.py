import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax

from configs import TAU, BETA
from graph_utils import prep_edges


@torch.no_grad()
def compute_dgat(x, edge_index, K, tau=TAU, beta=BETA):
    N          = x.size(0)
    edge_index = prep_edges(edge_index, N)
    row, col   = edge_index[0], edge_index[1]

    x_norm = F.normalize(x.float(), p=2, dim=1)
    sim    = (x_norm[row] * x_norm[col]).sum(dim=1)

    pos_mask = sim >= tau
    neg_mask = ~pos_mask

    def channel_prop(mask, H_in):
        if not mask.any():
            return torch.zeros_like(H_in)
        r, c = row[mask], col[mask]
        s    = sim[mask]
        a    = pyg_softmax(s, r, num_nodes=N)
        out  = torch.zeros_like(H_in)
        out.scatter_add_(0,
            r.unsqueeze(1).expand(-1, H_in.size(1)),
            a.unsqueeze(1) * H_in[c])
        return out

    H = x.float().clone()
    for _ in range(K):
        H_pos = channel_prop(pos_mask, H)
        H_neg = channel_prop(neg_mask, H)
        H     = H_pos - beta * H_neg

    return torch.cat([H, x.float()], dim=1)
