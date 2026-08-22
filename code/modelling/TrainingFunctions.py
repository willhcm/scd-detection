import glob
from pathlib import Path
import sys

CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from modelling.MaskRCNN import MaskRCNN, MaskRCNNDataset, rcnn_train
from helpers.MaskRCNNFunctions import collate_fn
from modelling.Veto import VetoDataset
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

MASKRCNN_INFO = {'model': MaskRCNN, 'loss': None, 'train_fn': rcnn_train, 'dataset': MaskRCNNDataset}

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
    Train one cross-validation fold, MaskRCNN specific now that this is the only model being used.
    previous versions used to be model agnostic.

    Returns results and best_states
    """

    train_set = model_info['dataset'](train_paths, augment=True)
    val_set = model_info['dataset'](val_paths, augment=False)

    train_loader = DataLoader(train_set,
                            shuffle=True,
                            num_workers=0,
                            batch_size=batch_size,
                            collate_fn=collate_fn)

    val_loader = DataLoader(val_set,
                            shuffle=False,
                            num_workers=0,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    
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

# written up from colab for reproducibility 
def train_veto_classifier(model, epochs, paths):
    """
    Trains the Veto classifier model using the provided paths and number of epochs.
    Returns the learning metrics and the best model state."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    t, v = train_test_split(paths, test_size = 0.2)
    
    train_set = VetoDataset(t)
    val_set = VetoDataset(v)

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True)

    val_loader = DataLoader(
        val_set,
        batch_size=8)

    # get training set wide scalar stats
    training_scalars = np.stack([np.load(path)["scalar_features"].astype(np.float32) for path in train_set.paths], axis=0)
    scalar_mean = training_scalars.mean(axis=0)
    scalar_std = np.maximum(training_scalars.std(axis=0),1e-6)
    model.set_scalar_statistics(scalar_mean,scalar_std)

    decision_threshold = 0.85
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5, weight_decay=1e-4)
    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # label 1 = RETAIN
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=device))

    train_losses, val_losses, f1s, precisions, recalls = [], [], [], [], []

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):

        # Training
        model.train()

        running_loss = 0.0
        train_samples = 0

        for rgb_local, dem_context, scalar_features, labels in train_loader:

            rgb_local = rgb_local.to(device,
                                    dtype=torch.float32,
                                    non_blocking=True)

            dem_context = dem_context.to(device,
                                        dtype=torch.float32,
                                        non_blocking=True)

            scalar_features = scalar_features.to(device,
                                                dtype=torch.float32,
                                                non_blocking=True)

            # view(-1), rather than squeeze(), remains safe
            # when the final batch contains one sample.
            labels = labels.to(device, dtype=torch.float32, non_blocking=True).view(-1)

            optimizer.zero_grad(set_to_none=True)

            logits = model(rgb_local, dem_context, scalar_features).view(-1)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]

            running_loss += loss.item() * batch_size
            train_samples += batch_size

        epoch_train_loss = running_loss / max(train_samples, 1)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()

        running_val_loss = 0.0
        val_samples = 0

        all_probabilities = []
        all_labels = []

        with torch.inference_mode():

            for rgb_local, dem_context, scalar_features, labels in val_loader:

                rgb_local = rgb_local.to(device,
                                        dtype=torch.float32,
                                        non_blocking=True)

                dem_context = dem_context.to(device,
                                            dtype=torch.float32,
                                            non_blocking=True)

                scalar_features = scalar_features.to(device,
                                                    dtype=torch.float32,
                                                    non_blocking=True)

                labels = labels.to(device,
                                dtype=torch.float32,
                                non_blocking=True
                                ).view(-1)

                logits = model(rgb_local, dem_context, scalar_features).view(-1)

                loss = criterion(logits, labels)

                probabilities = torch.sigmoid(logits)

                batch_size = labels.shape[0]

                running_val_loss += loss.item() * batch_size
                val_samples += batch_size
                all_probabilities.append(probabilities.cpu())

                all_labels.append(labels.cpu())

        epoch_val_loss = (running_val_loss / max(val_samples, 1))

        val_losses.append(epoch_val_loss)

        probabilities = torch.cat(all_probabilities).numpy()

        labels_numpy = torch.cat(all_labels).numpy().astype(np.int64)

        predictions = (probabilities >= decision_threshold).astype(np.int64)

        # calculate metrics
        epoch_precision = precision_score(labels_numpy, predictions,
                                        pos_label=1, average="binary",
                                        zero_division=0)
        epoch_recall = recall_score(labels_numpy, predictions, pos_label=1,
                                    average="binary", zero_division=0)

        epoch_f1 = f1_score(labels_numpy, predictions, pos_label=1,
                            average="binary", zero_division=0)

        precisions.append(epoch_precision)
        recalls.append(epoch_recall)
        f1s.append(epoch_f1)

        tn, fp, fn, tp = confusion_matrix(labels_numpy, predictions, labels=[0, 1]).ravel()

        # Step the scheduler
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train {epoch_train_loss:.4f} | "
            f"val {epoch_val_loss:.4f} | "
            f"P {epoch_precision:.3f} | "
            f"R {epoch_recall:.3f} | "
            f"F1 {epoch_f1:.3f} | "
            f"TP {tp} FP {fp} FN {fn} TN {tn}"
        )

    learning_metrics = {'f1s': f1s,
                        'precisions': precisions,
                        'recalls': recalls,
                        'val_losses': val_losses,
                        'train_losses': train_losses}
    
    return learning_metrics, best_state


#Mask R-CNN specific
def cross_validate(epochs, root):
    """
    Performs cross-validation for the Mask R-CNN model across different regions.
    Returns the cross-validation results and the best model states for each held-out region.
    """
    cv_results = {}
    best_states = {}
    for held_out in REGION_GROUPS:
        model = MaskRCNN()
        val_paths = _region_paths(held_out, root)
        train_paths = [p for r in REGION_GROUPS if r != held_out for p in _region_paths(r, root)]
        train_paths.extend(glob.glob(f"{root}/UknegApproved/*.npz"))
        metrics, best_model, _ = train_fold(model, train_paths, val_paths, epochs, MASKRCNN_INFO)
        cv_results[held_out] = metrics # Store metrics per held_out region
        best_states[held_out] = best_model

    return cv_results, best_states

            
# only considering maskRCNN now
def train_for_deployment(model, held_out, root):
    """
    Trains the Mask R-CNN model for deployment on a specific held-out region.
    Returns the trained model state dictionary."""

    val_paths = _region_paths(held_out, root)
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
