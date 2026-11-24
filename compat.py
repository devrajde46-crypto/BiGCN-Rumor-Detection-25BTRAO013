import torch
from torch import nn

class GCNConv(nn.Module):
    """Symmetrically normalized directed GCN with self loops (native PyTorch)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.lin.weight)
    def forward(self, x, edge_index):
        n = x.size(0)
        s,t = edge_index
        keep = s != t
        loops = torch.arange(n, device=x.device)
        s,t = torch.cat((s[keep],loops)),torch.cat((t[keep],loops))
        degree = x.new_zeros(n).index_add_(0,t,x.new_ones(t.numel()))
        inv = degree.rsqrt()
        h = self.lin(x)
        return h.new_zeros(h.shape).index_add_(0,t,h[s]*(inv[s]*inv[t])[:,None])+self.bias

def scatter_mean(x, batch, dim=0):
    n = int(batch.max())+1
    return x.new_zeros((n,x.size(1))).index_add_(0,batch,x)/torch.bincount(batch,minlength=n).to(x.dtype)[:,None]
