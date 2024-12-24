from os.path import join
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize
    
class MyDataset(Dataset): 
    def __init__(self, img_dir, gt_dir, case_ids, modality=["t1c"], shift=25, ds_scale=None, bbox_mask_as_channel=False):
        self.gt_dir = gt_dir
        self.img_dir = img_dir
        self.case_ids = case_ids
        self.modality_dict = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}
        self.modality = [self.modality_dict[mod] for mod in modality]
        self.shift = shift
        self.ds_scale = ds_scale
        self.bbox_mask_as_channel = bbox_mask_as_channel
        
    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]

        # load modalities & ground truth segmentation
        img_npy = np.load(join(self.img_dir, case_id), 'r', allow_pickle=True)
        gt_npy = np.load(join(self.gt_dir, case_id), 'r', allow_pickle=True)

        # pad to (256, 256)
        img = np.pad(img_npy, ((0, 0), (8, 8), (8, 8)), mode='edge')
        gt_crop = np.pad(gt_npy, ((8, 8), (8, 8)), mode='edge')

        brain_mask = img[-1]
        img = img[self.modality]
        if len(self.modality) == 1:
            img = np.repeat(img, 3, axis=0)

        gt = np.uint8(gt_crop > 0)       
        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)        
        if self.shift is not None:
            H, W = gt.shape
            x_min = max(0, x_min - np.random.randint(5, self.shift))
            x_max = min(W, x_max + np.random.randint(5, self.shift))
            y_min = max(0, y_min - np.random.randint(5, self.shift))
            y_max = min(H, y_max + np.random.randint(5, self.shift))
        
        bbox = np.array([x_min, y_min, x_max, y_max])
        bbox_mask = np.zeros_like(gt)
        bbox_mask[y_min:y_max, x_min:x_max] = 1
        if self.bbox_mask_as_channel:
            # append bbox_mask as last channel
            if len(self.modality) == 1:
                img[-1] = bbox_mask
            elif len(self.modality) > 1:
                img = np.concatenate((img, np.expand_dims(bbox_mask, axis=0)), axis=0)
        
        if self.ds_scale is not None:
            old_size = np.array(gt.shape)
            new_size = [scale * old_size for scale in self.ds_scale]
            gt = [resize(image=gt.astype(float), output_shape=size,
                         order=0, mode='edge', clip=True,
                         anti_aliasing=False) for size in new_size]
            gt = [np.expand_dims(_gt, axis=0).astype(np.float32) for _gt in gt]
        else:
            gt = np.expand_dims(gt, axis=0).astype(np.float32)
            
        # expand dims for correct batch generation in dataloaders
        bbox_mask = np.expand_dims(bbox_mask, axis=0)
        brain_mask = np.expand_dims(brain_mask, axis=0)

        return {
            "data": img.astype(np.float32),
            "bbox_mask": bbox_mask.astype(np.float32),
            "brain_mask": brain_mask.astype(np.float32),
            "bbox": bbox.astype(np.float32),
            "target": gt, # keep as is for deep supervision
            "index": index
        }