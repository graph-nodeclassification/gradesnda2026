import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, dropout=0.5):
        super().__init__()
        self.fc      = nn.Linear(in_dim, n_classes, bias=True)
        self.dropout = dropout

    def forward(self, H):
        H = F.dropout(H, p=self.dropout, training=self.training)
        return F.log_softmax(self.fc(H), dim=1)
