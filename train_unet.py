import warnings
warnings.filterwarnings("ignore")

# SEEDING
import random, os
import numpy as np
import torch

# SEED = 3407
SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED) # single GPU, use manual_seed_all for multi GPUs
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(SEED)

from time import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import monai
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchinfo import summary

from unet import UNet, UNetEncoder, UNetDecoder
from metrics import get_tensor_dice
from dataloading import MyDataset
from utils import print_to_log_file, plot_epochs

from os import makedirs
from os.path import join
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_epochs", type=int, default=50,
    help="Number of epochs to train, default 50."
)
parser.add_argument(
    "--fold", type=str, default="0",
    help="Fold to train (0, 1, 2, 3, 4), default 0."
)

args = parser.parse_args()
kfold = [args.fold]

img_dir = "../storage/data/Dataset999_BraTS23/preprocessed/npy/imgs"
gt_dir = "../storage/data/Dataset999_BraTS23/preprocessed/npy/gts"

deep_supervision = False
ds_scale = None
if deep_supervision:
    ds_scale = np.array([1.0, 0.5, 0.25])
    ds_scale_norm = ds_scale / ds_scale.sum()
bbox_mask_as_channel = False
    
device = "cuda"
num_epochs = args.num_epochs
batch_size = 16
num_workers = 8

use_val_best = True
initial_lr = 5e-5
weight_decay = 1e-2
reduce_lr_patience = 5
early_stop = False
early_stop_patience = None
if early_stop:
    early_stop_patience = 7

modality = ["t1c"]
# modality = ["t1c", "t2f", "t1n", "t2w"]
bbox_shift = 25

n_stages = 4
in_channels = 3
out_channels = [64,128,256,512]
kernel_sizes = [3,3,3,3]
strides = [1,2,2,2]
# drop_rate = 0.4
drop_rate = None
    
timestamp = datetime.now()
timestamp = "%d_%d_%d_%02.0d_%02.0d_%02.0d" % (timestamp.year, timestamp.month, timestamp.day,
                                               timestamp.hour, timestamp.minute, timestamp.second)
model_dir = join("model", "unet", timestamp)
makedirs(model_dir, exist_ok=True)
config = {}
config["modality"] = modality
config["img_dir"] = img_dir
config["gt_dir"] = gt_dir

config["deep_supervision"] = deep_supervision
if deep_supervision:
    config["ds_scale"] = ds_scale.tolist()
    config["ds_scale_norm"] = ds_scale_norm.tolist()
config["bbox_mask_as_channel"] = bbox_mask_as_channel
config["bbox_shift"] = bbox_shift

config["device"] = device
config["num_epochs"] = num_epochs
config["batch_size"] = batch_size
config["num_workers"] = num_workers

config["use_val_best"] = use_val_best
config["learning_rate"] = initial_lr
config["weight_decay"] = weight_decay
config["reduce_lr_patience"] = reduce_lr_patience
config["early_stop"] = early_stop
if early_stop:
    config["early_stop_patience"] = early_stop_patience

config["n_stages"] = n_stages
config["in_channels"] = in_channels
config["out_channels"] = out_channels
config["kernel_sizes"] = kernel_sizes
config["strides"] = strides
config["drop_rate"] = drop_rate

with open(join(model_dir, "config.json"), "w") as f:
    json.dump(config, f)

with open("nested_cv.json", "r") as f:
    cross_val_split = json.load(f)
data_set = {}
data_loader = {}
for outer_fold in kfold:
    data_set[outer_fold] = {}
    data_loader[outer_fold] = {}    
    for inner_fold in [0]:
        data_set[outer_fold][inner_fold] = {}
        data_loader[outer_fold][inner_fold] = {}
        for phase in ["inner_train", "inner_val"]:   
            case_ids = cross_val_split[str(outer_fold)][str(inner_fold)][phase]
            data_set[outer_fold][inner_fold][phase] = MyDataset(
                img_dir=img_dir,
                gt_dir=gt_dir,
                modality=modality,
                case_ids=case_ids,
                shift=bbox_shift,
                ds_scale=ds_scale,
                bbox_mask_as_channel=bbox_mask_as_channel
            )
            if "train" in phase:
                shuffle = True
            else:
                shuffle = False
            data_loader[outer_fold][inner_fold][phase] = DataLoader(
                dataset=data_set[outer_fold][inner_fold][phase],
                batch_size=batch_size,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=g,
                shuffle=shuffle,
                pin_memory=True
            )        

dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

unet_encoder = UNetEncoder(
    n_stages=n_stages,
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_sizes=kernel_sizes,
    strides=strides,
    drop_rate=drop_rate
)
unet_decoder = UNetDecoder(
    unet_encoder,
    num_classes=1,
    deep_supervision=deep_supervision
)
model = UNet(
    unet_encoder,
    unet_decoder  
)
summary(model, depth=7)

load_checkpoint = False
checkpoint_dir = ""

for outer_fold in kfold:
    fold_dir = join(model_dir, f"fold{outer_fold}")
    makedirs(fold_dir, exist_ok=True)
    
    timestamp = datetime.now()
    log_file = join(fold_dir, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                    (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second))
    
    for inner_fold in [0]:
        model = UNet(
            unet_encoder,
            unet_decoder  
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay,
        )    

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9,
            patience=reduce_lr_patience,
            cooldown=0
        )

        best_val_loss, best_val_dice = 1e10, 0.0
        early_stop_ctr = 0
        epochs = range(num_epochs)

        epoch_time = [0. for _ in epochs]
        mean_train_loss = [1e10 for _ in epochs]
        mean_val_loss = [1e10 for _ in epochs]
        mean_val_dice = [0. for _ in epochs]

        start = time()
        for epoch in epochs:
            print_to_log_file(log_file, "")
            print_to_log_file(log_file, f"outer fold {outer_fold}, inner fold {inner_fold}, epoch {epoch}")

            current_lr = optimizer.param_groups[0]["lr"]
            print_to_log_file(log_file, f"learning rate: {current_lr:.6f}")

            epoch_start = time()
            # TRAIN
            model.train()
            pbar = tqdm(data_loader[outer_fold][inner_fold]["inner_train"])
            train_loss = [1e10 for _ in range(len(pbar))]

            for step, batch in enumerate(pbar):
                image = batch["data"]
                target = batch["target"]

                image = image.to(device)
                if isinstance(target, list):
                    target = [i.to(device) for i in target] # deep supervision needs downsampled target
                else:
                    target = target.to(device)

                optimizer.zero_grad()
                preds = model(image)

                if deep_supervision:
                    l_dice = sum([ds_scale_norm[i] * dice_loss(seg[0], seg[1]) for i, seg in enumerate(zip(preds, target))])
                    l_bce = sum([ds_scale_norm[i] * bce_loss(seg[0], seg[1]) for i, seg in enumerate(zip(preds, target))])
                else:
                    l_dice = dice_loss(preds, target)
                    l_bce = bce_loss(preds, target)

                loss = 1.0 * l_dice + 1.0 * l_bce

                train_loss[step] = loss.item()
                loss.backward()
                optimizer.step()

                pbar.set_description(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, train loss: {loss.item():.4f}")

            mean_train_loss[epoch] = sum(train_loss) / len(train_loss)  

            # VALIDATION    
            model.eval()
            pbar = tqdm(data_loader[outer_fold][inner_fold]["inner_val"])
            val_loss = [1e10 for _ in range(len(pbar))]
            val_dice = [0. for _ in range(len(pbar))]
            with torch.no_grad():            
                for step, batch in enumerate(pbar):
                    image = batch["data"]
                    target = batch["target"]

                    image = image.to(device)
                    if isinstance(target, list):
                        target = [i.to(device) for i in target] # deep supervision needs downsampled target
                    else:
                        target = target.to(device)

                    preds = model(image)

                    if deep_supervision:
                        l_dice = sum([ds_scale_norm[i] * dice_loss(seg[0], seg[1]) for i, seg in enumerate(zip(preds, target))])
                        l_bce = sum([ds_scale_norm[i] * bce_loss(seg[0], seg[1]) for i, seg in enumerate(zip(preds, target))])
                        preds, target = preds[0], target[0]
                    else:
                        l_dice = dice_loss(preds, target)
                        l_bce = bce_loss(preds, target)
                    loss = 1.0 * l_dice + 1.0 * l_bce

                    val_loss[step] = loss.item()
                    preds = (torch.sigmoid(preds) > 0.5).long()
                    val_dice[step] = get_tensor_dice(preds, target)

                    pbar.set_description(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, val loss: {loss.item():.4f}")

                mean_val_loss[epoch] = sum(val_loss) / len(val_loss)   
                mean_val_dice[epoch] = sum(val_dice) / len(val_dice)

            lr_scheduler.step(mean_val_loss[epoch]) 

            epoch_time[epoch] = time() - epoch_start

            print_to_log_file(log_file, f"train mean loss: {mean_train_loss[epoch]:.4f}")
            print_to_log_file(log_file, f"val mean loss: {mean_val_loss[epoch]:.4f}") 
            print_to_log_file(log_file, f"val mean dice: {mean_val_dice[epoch]:.4f}")

            checkpoint = {
                "epoch": epoch,
                "early_stop_ctr": early_stop_ctr,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_dice": best_val_dice,
                "dice": mean_val_dice[epoch]
            }
            torch.save(checkpoint, join(fold_dir, "checkpoint_latest.pth"))
            
            if mean_val_loss[epoch] <= best_val_loss and mean_val_dice[epoch] >= best_val_dice:
                early_stop_ctr = 0
                best_epoch = epoch

                print_to_log_file(log_file, f"new best val dice: {best_val_dice:.4f} -> {mean_val_dice[epoch]:.4f}")
                print_to_log_file(log_file, f"new best val loss: {best_val_loss:.4f} -> {mean_val_loss[epoch]:.4f}")

                best_val_loss = mean_val_loss[epoch]
                best_val_dice = mean_val_dice[epoch]

                checkpoint["early_stop_ctr"] = early_stop_ctr
                checkpoint["best_epoch"] = best_epoch
                checkpoint["best_val_loss"] = best_val_loss
                checkpoint["best_val_dice"] = best_val_dice
                torch.save(checkpoint, join(fold_dir, "checkpoint_best.pth"))
            else:
                if early_stop:
                    early_stop_ctr += 1
                    print_to_log_file(log_file, f"early stopping counter: {early_stop_ctr} out of {early_stop_patience}")

                    if early_stop_ctr >= early_stop_patience:
                        print_to_log_file(log_file, f"early stopping at epoch {epoch}")
                        break

        plot_epochs(train_loss=mean_train_loss[:epoch+1], val_loss=mean_val_loss[:epoch+1], val_dice=mean_val_dice[:epoch+1], 
                    epoch_time=epoch_time[:epoch+1], output_dir=fold_dir)

        total_time = time() - start
        print_to_log_file(log_file, "")
        print_to_log_file(log_file, f"running {epoch + 1} epochs took a total of {total_time:.2f} seconds")
    
    # ACTUAL VALIDATION
    preds_dir = join(fold_dir, "prediction")
    makedirs(preds_dir, exist_ok=True)
    
    if use_val_best:
        checkpoint = torch.load(join(fold_dir, "checkpoint_best.pth"), map_location="cpu")
    else:
        checkpoint = torch.load(join(fold_dir, "checkpoint_latest.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
        
    if deep_supervision:
        model.set_deep_supervision_enabled(False)
    model.eval()        
    with torch.no_grad():
        val_dice = 0.
        case_ids = cross_val_split[str(outer_fold)]["outer_val"]
        val_set = MyDataset(
            img_dir=img_dir,
            gt_dir=gt_dir,
            modality=modality,
            case_ids=case_ids,
            shift=bbox_shift,
            ds_scale=None,
            bbox_mask_as_channel=bbox_mask_as_channel
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=True
        )
        pbar = tqdm(val_loader)
        for step, batch in enumerate(pbar):            
            image = batch["data"]        
            target = batch["target"]
            image, target = image.to(device), target.to(device)
            
            preds = model(image)
            preds = (torch.sigmoid(preds) > 0.5).long()
            current_dice = get_tensor_dice(preds, target)
            val_dice += current_dice
            
            index = batch["index"]
            case_id = cross_val_split[str(outer_fold)]["outer_val"][index]
            # post-processing
            # preds = preds & bbox_mask # mask prediction with bbox mask
            preds = preds.detach().cpu().numpy().squeeze() # move back to cpu & numpy
            preds = preds[..., 8:-8, 8:-8] # crop back to (240,240)
            np.save(join(preds_dir, case_id), preds)
            
            pbar.set_description(f"running outer fold {outer_fold} validation, dice: {current_dice:.4f}")
        print_to_log_file(log_file, f"outer fold {outer_fold} validation mean dice: {val_dice / step:.4f}")              
    if deep_supervision:
        model.set_deep_supervision_enabled(True)