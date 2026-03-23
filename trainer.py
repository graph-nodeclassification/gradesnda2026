import torch
import torch.nn.functional as F

from configs import DROPOUT, LR, WEIGHT_DECAY
from model import LinearClassifier


def train_epoch(model, opt, H, labels, mask):
    model.train()
    opt.zero_grad()
    F.nll_loss(model(H)[mask], labels[mask]).backward()
    opt.step()


@torch.no_grad()
def evaluate(model, H, labels, mask):
    model.eval()
    return (model(H).argmax(1)[mask] == labels[mask]).float().mean().item()


def run_split(H, labels, train_mask, val_mask, test_mask,
              n_classes, device, epochs, patience):
    H      = H.to(device)
    labels = labels.to(device)

    model = LinearClassifier(H.size(1), n_classes, DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state, wait = 0.0, None, 0

    for _ in range(1, epochs + 1):
        train_epoch(model, opt, H, labels, train_mask)
        val_acc = evaluate(model, H, labels, val_mask)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return evaluate(model, H, labels, test_mask)
