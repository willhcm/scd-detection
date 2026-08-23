# copy and pasted from code/modelling/TrainingFunctions.
# functionality was refactored in that file, so old code has been transferred for completeness. 

from glob import glob 
import numpy as np
from torch.utils.data import DataLoader
from somewhere import _region_paths, REGION_GROUPS, DATASET_ARGS, train_fold # just because the lines were annoying me 
from sklearn.model_selection import train_test_split

# ------------------------------- Depreciated Code ---------------------------------------#


def _run_grouped_comparison(paths, epochs, models, cv=False):
    """Was used to compare several model types in a 80/20 random split"""

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

def _run_cv_comparison(epochs, models, root):
    """Was used to compare several model types in a region-level LOOCV"""
    cv_results = {}
    best_states = {}
    loaders = {}
    for held_out in REGION_GROUPS:
        for model, model_info in models.items():
            print(f'Model: {model} ')
            print(f"\n=== fold: holding out {held_out} ===")
            val_paths = _region_paths(held_out, root)
            train_paths = [p for r in REGION_GROUPS if r != held_out for p in _region_paths(r, root)]
            train_paths.extend(glob.glob(f"{root}/UknegApproved/*.npz"))

            metrics, best_model, val_loader = train_fold(model, train_paths, val_paths, epochs, model_info)
            if model not in cv_results:
                cv_results[model] = {} # Initialize dictionary for each model
            cv_results[model][held_out] = metrics # Store metrics per held_out region
            best_states[model] = best_model
            loaders[model] = val_loader

    return cv_results, best_states, loaders


def compare(models, epochs, paths=None, cv=True, root=None):
    """Public function for model comparison functions"""
    if cv:
        return _run_cv_comparison(epochs, models, root)
    else:
        if paths is not None:
            return _run_grouped_comparison(paths, epochs, models)
        else:
            return('please provide paths (as a list)')

def train_model(model_type, paths, epochs, model_info, batch_size=8):
    """ trains one model (model agnostic when model_info dict is used correctly) 
    using 80/20 test train split 
    
    Returns results and best_states """

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
