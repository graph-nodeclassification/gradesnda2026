import time
import torch

from configs import K_VALUES, RUNS, EPOCHS, PATIENCE, TAU, BETA, DROPOUT, LR, WEIGHT_DECAY, DATA_ROOT, DATASETS
from data import load_dataset
from graph_utils import node_homophily
from propagation import compute_dgat
from trainer import run_split


def benchmark_dataset(name, device):
    print(f"\n{'='*65}")
    print(f"  Dataset : {name.upper()}   Device: {device}")
    print(f"{'='*65}")

    ds, data, split_type = load_dataset(name, DATA_ROOT)
    N  = data.num_nodes
    C  = ds.num_classes
    h  = node_homophily(data.edge_index, data.y)

    n_splits_avail = data.train_mask.size(1) if split_type == "multi" else 1
    n_splits = min(RUNS, n_splits_avail)

    print(f"  Nodes      : {N}")
    print(f"  Edges      : {data.num_edges}")
    print(f"  Features   : {data.num_node_features}")
    print(f"  Classes    : {C}")
    print(f"  Homophily  : {h:.3f}")
    print(f"  Split mode : {split_type}  ({n_splits_avail} available, using {n_splits})")

    results = {}

    for K in K_VALUES:
        t0   = time.time()
        H    = compute_dgat(data.x, data.edge_index, K)
        tpre = time.time() - t0

        accs = []
        for split_idx in range(n_splits):
            torch.manual_seed(split_idx)

            if split_type == "multi":
                tr = data.train_mask[:, split_idx].to(device)
                va = data.val_mask[:, split_idx].to(device)
                te = data.test_mask[:, split_idx].to(device)
            else:
                tr = data.train_mask.to(device)
                va = data.val_mask.to(device)
                te = data.test_mask.to(device)

            acc = run_split(H, data.y.to(device), tr, va, te, C, device, EPOCHS, PATIENCE)
            accs.append(acc)

        accs_t = torch.tensor(accs)
        mean   = accs_t.mean().item() * 100
        std    = accs_t.std().item()  * 100
        results[f"K={K}"] = (mean, std)

        n_eff = 1 if split_type == "single" else n_splits
        print(f"    K={K}  |  {mean:.2f}% ± {std:.2f}%  (precompute {tpre:.3f}s, {n_eff} splits)")

    return results


def print_table(all_results):
    col_w = 18
    print(f"\n\n{'='*80}")
    print(f"  DGAT — Test Accuracy mean ± std %")
    print(f"  (tau={TAU}, beta={BETA}, dropout={DROPOUT}, lr={LR}, wd={WEIGHT_DECAY})")
    print(f"{'='*80}")

    header = f"  {'K':<6}" + "".join(f"  {d:<{col_w}}" for d in all_results)
    print(header)
    print(f"  {'-'*75}")

    for K in K_VALUES:
        key = f"K={K}"
        row = f"  {key:<6}"
        for ds in all_results:
            if key in all_results[ds]:
                mean, std = all_results[ds][key]
                cell = f"{mean:.2f}±{std:.2f}"
            else:
                cell = "N/A"
            row += f"  {cell:<{col_w}}"
        print(row)

    print(f"{'='*80}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}
    t_total     = time.time()

    for ds_name in DATASETS:
        try:
            all_results[ds_name] = benchmark_dataset(ds_name, device)
        except Exception as e:
            print(f"\n  [WARN] {ds_name} failed: {e}")
            all_results[ds_name] = {}

    print_table(all_results)
    print(f"  Total runtime : {time.time() - t_total:.1f}s")
