from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor


def load_dataset(name, root):
    name_l = name.lower()

    if name_l in ("cornell", "texas", "wisconsin"):
        ds   = WebKB(root=root, name=name)
        data = ds[0]
        return ds, data, "multi"

    if name_l in ("chameleon", "squirrel"):
        try:
            ds = WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=True)
        except TypeError:
            ds = WikipediaNetwork(root=root, name=name)
        data       = ds[0]
        split_type = "multi" if data.train_mask.dim() == 2 else "single"
        return ds, data, split_type

    if name_l == "actor":
        ds         = Actor(root=f"{root}/Actor")
        data       = ds[0]
        split_type = "multi" if data.train_mask.dim() == 2 else "single"
        return ds, data, split_type

    raise ValueError(f"Unknown dataset: {name}")
