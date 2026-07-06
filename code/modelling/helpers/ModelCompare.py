import glob
from torch.utils.data import WeightedRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np

class _AugmentedSubset(Dataset):
    """Wraps a subset of dataset with its own augment flag."""

    def __init__(self, dataset, indices: list, augment: bool):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original = self.dataset.augment
        self.dataset.augment = self.augment
        try:
            return self.dataset[self.indices[idx]]
        finally:
            self.dataset.augment = original
          

def get_loaders(
    paths,
    dataset_type,
    val_split: float = 0.2,
    batch_size: int = 16,
    seed: int = 42,
):
    full_dataset = dataset_type(paths, augment=False)

    n = len(full_dataset)
    n_val  = max(int(n * val_split), 1)
    n_train = n - n_val

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()

    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    train_set = _AugmentedSubset(full_dataset, train_indices, augment=True)
    val_set = _AugmentedSubset(full_dataset, val_indices,   augment=False)

    weights = []
    for idx in train_indices:
        d  = np.load(full_dataset.paths[idx], allow_pickle=True)
        frac = float(d["scd_pixel_fraction"])
        weights.append(1.0 + frac)

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set,   batch_size=batch_size, shuffle=False,   num_workers=0)

    print(f"Train: {n_train} | Val: {n_val}")
    return train_loader, val_loader, val_set

REGION_GROUPS = {
    'Brazil': ['Brazil'],
    'USA': ['USA'],
    'Karoo': ['Karoo'],
    'Russia': ['Russia', 'Russia2', 'Russia3'],
    'Australia': ['Australia']
}

def _region_paths(region_name):
    paths = []
    for d in REGION_GROUPS[region_name]:
        paths.extend(glob.glob(f"/content/drive/MyDrive/IRP/SNIPCluster/{d}/*.npz"))
    return sorted(paths)


def train_fold(train_paths, val_paths, epochs, model_info, batch_size=16):
    """ 
    Train one cross-validation fold, model/architecture flexible!
    """

    train_set = model_info['dataset'](train_paths, augment=True)
    val_set = model_info['dataset'](val_paths, augment=False)

    w = []
    for p in train_set.paths:
        d = np.load(p, allow_pickle=True)
        w.append(1.0 + float(d["scd_pixel_fraction"]))
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model_info['model'](in_channels=4, base_filters=64)
    criterion = model_info['loss']
    fn = model_info['train_fn']

    vls, tls, precisions, recalls, f1s, model, val_loader =  fn(model, criterion, train_loader, val_loader, epochs)
    metrics = {'vls': vls,
              'tls': tls,
              'precisions': precisions,
              'recalls': recalls,
              'f1s': f1s,
              }

    return metrics, model, val_loader


def train_model(paths, epochs, model_info):
    train_loader, val_loader, _ = get_loaders(paths, model_info['dataset'])

    model = model_info['model'](in_channels=4, base_filters=64)
    criterion = model_info['loss']
    fn = model_info['train_fn']

    vls, tls, precisions, recalls, f1s, model, val_loader =  fn(model, criterion, train_loader, val_loader, epochs)
    metrics = {'vls': vls,
              'tls': tls,
              'precisions': precisions,
              'recalls': recalls,
              'f1s': f1s,
              }

    return metrics, model, val_loader


def _run_cv_comparison(epochs, models):

    cv_results = {}
    best_states = {}
    loaders = {}
    for held_out in REGION_GROUPS:
        for model, model_info in models.items():
            print(f'Model: {model} ')
            print(f"\n=== fold: holding out {held_out} ===")
            val_paths = _region_paths(held_out)
            train_paths = [p for r in REGION_GROUPS if r != held_out for p in _region_paths(r)]

            metrics, best_model, val_loader = train_fold(train_paths, val_paths, epochs, model_info)
            if model not in cv_results:
                cv_results[model] = {} # Initialize dictionary for each model
            cv_results[model][held_out] = metrics # Store metrics per held_out region
            best_states[model] = best_model
            loaders[model] = val_loader


    return cv_results, best_states, loaders

def _run_grouped_comparison(paths, epochs, models, cv=False):

    cv_results = {}
    best_states = {}
    loaders = {}
    for model, model_info in models.items():
        print(f'Model: {model} ')

        metrics, best_model, val_loader = train_model(paths, epochs, model_info)
        if model not in cv_results:
            cv_results[model] = {} # Initialize dictionary for each model
        cv_results[model] = metrics # Store metrics per held_out region
        best_states[model] = best_model
        loaders[model] = val_loader

    return cv_results, best_states, loaders

def compare(models, epochs, paths=None, cv=True):

    if cv:
        return _run_cv_comparison(epochs, models)
    else:
        if paths is not None:
            return _run_grouped_comparison(paths, epochs, models)
        else:
            return('please provide paths (as a list)')
            

