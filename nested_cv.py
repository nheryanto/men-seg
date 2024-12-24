import os
import numpy as np
from sklearn.model_selection import KFold
import glob
from os.path import join, basename
import json

SEED = 12345
def generate_crossval_split(train_identifiers, seed=SEED, n_splits=5):
    splits = {}
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_indices, test_idx) in enumerate(kfold.split(train_identifiers)):
        outer_train_keys = np.array(train_identifiers)[train_indices]
        outer_val_keys = np.array(train_identifiers)[test_idx]
        
        splits[i] = {}
        splits[i]["outer_train"] = [key.replace(".nii.gz", "") for key in outer_train_keys]
        splits[i]["outer_val"] = [key.replace(".nii.gz", "") for key in outer_val_keys]
        
        for j, (train_idx, val_idx) in enumerate(kfold.split(outer_train_keys)):
            inner_train_keys = np.array(outer_train_keys)[train_idx]
            inner_val_keys = np.array(outer_train_keys)[val_idx]
            splits[i][j] = {}
            splits[i][j]["inner_train"] = [key.replace(".nii.gz", "") for key in inner_train_keys]
            splits[i][j]["inner_val"] = [key.replace(".nii.gz", "") for key in inner_val_keys]
        
    return splits

all_keys_sorted = sorted(os.listdir("data/Dataset999_BraTS23/labelsTr"))
splits = generate_crossval_split(all_keys_sorted, seed=SEED, n_splits=5)

cross_val_split = {}
for outer_fold in [0,1,2,3,4]:
    print(f"splitting outer fold {outer_fold}")
    outer_train_cases = splits[outer_fold]["outer_train"]
    outer_val_cases = splits[outer_fold]["outer_val"]
    
    outer_train_slices, outer_val_slices = [], []
    for case in outer_train_cases:
        slices = [basename(slice) for slice in glob.glob(join("data/Dataset999_BraTS23/preprocessed/npy/gts", f"{case}*"))]
        outer_train_slices.extend(slices)
    for case in outer_val_cases:
        slices = [basename(slice) for slice in glob.glob(join("data/Dataset999_BraTS23/preprocessed/npy/gts", f"{case}*"))]
        outer_val_slices.extend(slices)
        
    cross_val_split[outer_fold] = {}
    cross_val_split[outer_fold]["outer_train"] = outer_train_slices
    cross_val_split[outer_fold]["outer_val"] = outer_val_slices
    
    for inner_fold in [0,1,2,3,4]:
        print(f"splitting inner fold {inner_fold}")
        inner_train_cases = splits[outer_fold][inner_fold]["inner_train"]
        inner_val_cases = splits[outer_fold][inner_fold]["inner_val"]

        inner_train_slices, inner_val_slices = [], []
    
        for case in inner_train_cases:
            slices = [basename(slice) for slice in glob.glob(join("data/Dataset999_BraTS23/preprocessed/npy/gts", f"{case}*"))]
            inner_train_slices.extend(slices)
        for case in inner_val_cases:
            slices = [basename(slice) for slice in glob.glob(join("data/Dataset999_BraTS23/preprocessed/npy/gts", f"{case}*"))]
            inner_val_slices.extend(slices)
        
        cross_val_split[outer_fold][inner_fold] = {}
        cross_val_split[outer_fold][inner_fold]["inner_train"] = inner_train_slices
        cross_val_split[outer_fold][inner_fold]["inner_val"] = inner_val_slices

    print(f"outer fold: {outer_fold}, train: {len(outer_train_slices)}, val: {len(outer_val_slices)}")
    
with open("nested_cv.json", "w") as f:
    json.dump(cross_val_split, f)