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

from unet import UNet, UNetEncoder, UNetDecoder, BBUNet, BBUNetDecoder
from litemedsam import MedSAM_Lite
from MedSAM.tiny_vit_sam import TinyViT
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

from metrics import get_tensor_dice
from dataloading import MyDataset
from utils import print_to_log_file, plot_epochs

from os import makedirs, listdir
from os.path import join, isfile, isdir
import json, csv
import argparse

import skimage
from medpy.metric import binary
import hausdorff

def compute_tp_fp_fn_tn(mask_ref, mask_pred):
    tp = np.sum(mask_ref & mask_pred)
    fp = np.sum((~mask_ref) & mask_pred)
    fn = np.sum(mask_ref & (~mask_pred))
    tn = np.sum((~mask_ref) & (~mask_pred))
    return tp, fp, fn, tn

def compute_metrics(ground_truth, prediction):
    tp, fp, fn, tn = compute_tp_fp_fn_tn(ground_truth, prediction)
    dice = (2 * tp) / (2 * tp + fn + fp) if (2* tp + fp + fn) != 0 else 0.
    prec = tp / (tp + fp) if (tp + fp) != 0 else 0.
    rec = tp / (tp + fn) if (tp + fn) != 0 else 0.
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) != 0 else 0.
    return dice, prec, rec, acc

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, 
    help="Trained model: bb_unet or litemedsam."
)
parser.add_argument(
    "--modality", type=str,
    help="MRI modality: single or multi."
)
parser.add_argument(
    "--experiment", type=str,
    help="Experiment setup (only provide ONE), e.g. nested_single_relu."
)

args = parser.parse_args()
if args.modality == "single":
    modality = ["t1c"]
elif args.modality == "multi":
    modality = ["t1c", "t2w", "t2f"]
elif args.modality == "multi_all":
    modality = ["t1c", "t2f", "t1n", "t2w"]
else:
    print("Modality not recognized.")
    
model_dir = None
if not isdir(join("model", args.model)):
    print("Model not recognized.")
else:
    model_dir = join("model", args.model)

exp_dir = None
if not isdir(join(model_dir, args.experiment)):
    print(join(model_dir, args.experiment))
    print("Experiment not recognized.")
else:
    img_dir = "../storage/data/Dataset999_BraTS23/preprocessed/npy/imgs"
    gt_dir = "../storage/data/Dataset999_BraTS23/preprocessed/npy/gts"

    device = "cuda"
    use_val_best = True
    bbox_shift = 25

    with open("nested_cv.json", "r") as f:
        cross_val_split = json.load(f)

    output_dir = join("model", "inference", args.model)
    preds_dir = join(output_dir, args.experiment)
    makedirs(preds_dir, exist_ok=True)
    
    metrics = []
    fold = 0
    
    exp_dir = join(model_dir, args.experiment)
    print(exp_dir)
    for exp_datetime in sorted(listdir(exp_dir)):
        fold_dir = join(exp_dir, exp_datetime, f"fold{fold}")
        if args.model == "bb_unet":
            unet_encoder = UNetEncoder(
                n_stages=4,
                in_channels=3,
                out_channels=[64,128,256,512],
                kernel_sizes=[3,3,3,3],
                strides=[1,2,2,2],
                drop_rate=0.4
            )
            bbunet_decoder = BBUNetDecoder(
                unet_encoder,
                bb_pool_ratio=[1,2,4],
                bb_stride=[1,2,4],
                num_classes=1,
                deep_supervision=False
            )
            model = BBUNet(
                unet_encoder,
                bbunet_decoder  
            )
        elif args.model == "unet":
            unet_encoder = UNetEncoder(
                n_stages=4,
                in_channels=3,
                out_channels=[64,128,256,512],
                kernel_sizes=[3,3,3,3],
                strides=[1,2,2,2],
                drop_rate=0.4
            )
            unet_decoder = UNetDecoder(
                unet_encoder,
                num_classes=1,
                deep_supervision=False
            )
            model = UNet(
                unet_encoder,
                unet_decoder  
            )
        elif args.model == "litemedsam":
            medsam_lite_image_encoder = TinyViT(
                img_size=256,
                in_chans=3,
                embed_dims=[
                    64, ## (64, 256, 256)
                    128, ## (128, 128, 128)
                    160, ## (160, 64, 64)
                    320 ## (320, 64, 64) 
                ],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            )
            medsam_lite_prompt_encoder = PromptEncoder(
                embed_dim=256,
                image_embedding_size=(64, 64),
                input_image_size=(256, 256),
                mask_in_chans=16
            )
            medsam_lite_mask_decoder = MaskDecoder(
                num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                        depth=2,
                        embedding_dim=256,
                        mlp_dim=2048,
                        num_heads=8,
                    ),
                    transformer_dim=256,
                    iou_head_depth=3,
                    iou_head_hidden_dim=256,
            )
            model = MedSAM_Lite(
                    image_encoder = medsam_lite_image_encoder,
                    mask_decoder = medsam_lite_mask_decoder,
                    prompt_encoder = medsam_lite_prompt_encoder
                )

        if use_val_best:
            checkpoint = torch.load(join(fold_dir, "checkpoint_best.pth"), map_location="cpu")
        else:
            checkpoint = torch.load(join(fold_dir, "checkpoint_latest.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False

        model.eval()
        with torch.no_grad():
            summary(model, col_names=["num_params", "trainable"])
                    
            val_dice = 0.
            case_ids = cross_val_split[str(fold)]["outer_val"]
            val_set = MyDataset(
                img_dir=img_dir,
                gt_dir=gt_dir,
                modality=modality,
                case_ids=case_ids,
                shift=bbox_shift
            )
            val_loader = DataLoader(
                dataset=val_set,
                batch_size=1,
                num_workers=0,
                worker_init_fn=seed_worker,
                generator=g,
                shuffle=False,
                pin_memory=True
            )
            pbar = tqdm(val_loader)
            for step, batch in enumerate(pbar):
                image = batch["data"]
                target = batch["target"]
                image, target = image.to(device), target.to(device)
                
                if args.model == "unet":
                    preds = model(image)
                else:
                    bbox_mask = batch["bbox_mask"]
                    bbox_mask = bbox_mask.to(device)
                    if args.model == "bb_unet":
                        preds = model(image, bbox_mask)
                    elif args.model == "litemedsam":
                        bbox = batch["bbox"]
                        bbox = bbox.to(device)
                        preds, _ = model(image, bbox)
                    
                preds = (torch.sigmoid(preds) > 0.5).long()
                current_dice = get_tensor_dice(preds, target)
                val_dice += current_dice

                index = batch["index"]
                case_id = cross_val_split[str(fold)]["outer_val"][index]

                preds = preds.detach().cpu().numpy().squeeze() # move back to cpu & numpy                
                preds = preds[..., 8:-8, 8:-8] # crop back to (240,240)
                # post-processing
                if args.model in ["bb_unet", "litemedsam"]:
                    bbox_mask = bbox_mask.detach().cpu().numpy().squeeze()
                    bbox_mask = bbox_mask[..., 8:-8, 8:-8] 
                    preds = np.uint8(preds) & np.uint8(bbox_mask) # mask prediction with bbox mask
                    
                np.save(join(preds_dir, case_id), preds)

                target = target.detach().cpu().numpy().squeeze()
                target = target[..., 8:-8, 8:-8]
                target = np.uint8(target)
                
                dice, prec, rec, acc = compute_metrics(target, preds)
                gt_area, pred_area = target.sum(), preds.sum()

                skimage_hd, fast_hd, medpy_hd, medpy_hd95 = 0., 0., 0., 0.
                if pred_area > 0:
                    skimage_hd = skimage.metrics.hausdorff_distance(preds, target)
                    fast_hd = hausdorff.hausdorff_distance(preds, target)
                    medpy_hd = binary.hd(preds, target)
                    medpy_hd95 = binary.hd95(preds, target)

                metrics.append([fold, case_id, dice, prec, rec, acc, gt_area, pred_area, 
                                skimage_hd, fast_hd, medpy_hd, medpy_hd95])

                pbar.set_description(f"postprocessing & evaluating fold {fold} validation, current dice: {dice:.4f}")
        fold += 1

        with open(join(output_dir, f"postprocess_{args.experiment}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "case_id", "dice_score", "precision", "recall", "accuracy", 
                             "gt_area", "pred_area", "skimage_hd", "fast_hd", "medpy_hd", "medpy_hd95"])
            writer.writerows(metrics)