import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from code.modelling.blocks import _AugmentedSubset
from sklearn.model_selection import train_test_split
from code.modelling.MaskRCNN import MaskRCNN, MaskRCNNDataset, rcnn_train
from sklearn.metrics import roc_auc_score
from code.MaskRCNNFunctions import collate_fn

DATASET_ARGS = {
    "default": {
        "batch_size": 16,
        "collate_fn": None,
    },
    "MaskRCNN": {
        "batch_size": 2,
        "collate_fn": collate_fn,
    },
    "FPNCentreNet": {
        "batch_size": 4,
        "collate_fn": None}}

REGION_GROUPS = {
    'Russia': ['Russia', 'Russia2', 'Russia3'],
    'Brazil': ['Brazil'],
    'USA': ['USA', 'Texas'],
    'Karoo': ['Karoo'],
    'Australia': ['Australia'],
    'France': ['France'],
    'UK': ['UKQuantock', 'UKTraining', 'EastQuantock']
}

def _region_paths(region_name, root):
    paths = []
    for d in REGION_GROUPS[region_name]:
        paths.extend(glob.glob(f"{root}/ScalesCombined/{d}/*.npz"))
        negs = glob.glob(f"{root}/Negatives/{d}/*.npz")
        paths.extend(negs)
    return sorted(paths)

def train_fold(model_type, train_paths, val_paths, epochs, model_info, batch_size=8):
    """ 
    Train one cross-validation fold, model/architecture flexible!
    """

    train_set = model_info['dataset'](train_paths, augment=True)
    val_set = model_info['dataset'](val_paths, augment=False)

    loader_args = DATASET_ARGS.get(model_type, DATASET_ARGS["default"]).copy()

    if batch_size is not None:
        loader_args["batch_size"] = batch_size

    # Remove collate_fn if None, otherwise DataLoader may complain in some cases
    if loader_args.get("collate_fn") is None:
        loader_args.pop("collate_fn")


    train_loader = DataLoader(train_set,
                            shuffle=True,
                            num_workers=0,
                            **loader_args)

    val_loader = DataLoader(val_set,
                            shuffle=False,
                            num_workers=0,
                            **loader_args)
    
    model = model_info['model']()
    criterion = model_info['loss']
    fn = model_info['train_fn']

    vls, tls, precisions, recalls, f1s, best_model, val_loader =  fn(model, criterion, train_loader, val_loader, epochs)
    metrics = {'vls': vls,
              'tls': tls,
              'precisions': precisions,
              'recalls': recalls,
              'f1s': f1s,
              }

    return metrics, best_model, val_loader


def train_model(model_type, paths, epochs, model_info, batch_size=8):

    # get train-val split
    train_paths, val_paths = train_test_split(paths, test_size=0.2)

    # init datasets (custom per model)
    train_set = model_info['dataset'](train_paths, augment=True)
    val_set = model_info['dataset'](val_paths, augment=False)

    w = []
    for p in train_set.paths:
        d = np.load(p, allow_pickle=True)
        w.append(1.0 + float(d["scd_pixel_fraction"]))

    # decide loader args and init loaders
    loader_args = DATASET_ARGS.get(model_type, DATASET_ARGS["default"]).copy()

    if batch_size is not None:
        loader_args["batch_size"] = batch_size

    # Remove collate_fn if None
    if loader_args.get("collate_fn") is None:
        loader_args.pop("collate_fn")

    train_loader = DataLoader(train_set,
                            shuffle=True,
                            num_workers=0,
                            **loader_args)

    val_loader = DataLoader(val_set,
                            shuffle=False,
                            num_workers=0,
                            **loader_args)

    # build models, losses and train fns
    model = model_info['model']()
    if model_type != 'MaskRCNN':
        criterion = model_info['loss']()
    else:
        criterion = None
        
    fn = model_info['train_fn']

    # train and return metric lists
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
            train_paths.extend(glob.glob(f"/content/drive/MyDrive/IRP/Tiles/UKNegatives/*.npz"))

            metrics, best_model, val_loader = train_fold(model, train_paths, val_paths, epochs, model_info)
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

        metrics, best_model, val_loader = train_model(model, paths, epochs, model_info)
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
            
# only considering maskRCNN now
def train_for_deployment(model, held_out, root):

    val_paths = _region_paths(held_out)
    train_paths = []
    for region_name in REGION_GROUPS.keys():
        if region_name == held_out:
            continue
        else:
            train_paths.extend(_region_paths(region_name, root))

    train_paths.extend(glob.glob(f'{root}/UknegApproved/*.npz'))
    if held_out == 'France':
        dirs = ['EastQuantock', 'UKTraining', 'UKQuantock']
        for d in dirs:
            train_paths.extend(glob.glob(f"{root}/{d}/*.npz"))

    val_set = MaskRCNNDataset(val_paths)
    train_set = MaskRCNNDataset(train_paths)

    val_loader = DataLoader(val_set, batch_size=4, collate_fn=collate_fn)
    train_loader = DataLoader(train_set, batch_size=4, collate_fn=collate_fn)
    
    _, train_losses, val_precisions, val_recalls, val_f1s, model, val_loader = rcnn_train(model, None, train_loader, val_loader, epochs=50)

    return model.state_dict()

    
