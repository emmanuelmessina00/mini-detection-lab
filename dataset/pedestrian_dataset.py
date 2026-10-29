import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.download_data import download_data


class PennFudanDataset(Dataset):
    def __init__(self,num_samples=200,image_size=(559,536),transform=None, root_dir="data/PennFudanPed"):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir,"PNGImages")
        self.mask_dir = os.path.join(root_dir,"PedMasks")
        if not os.path.exists(root_dir) or len(os.listdir(root_dir)) == 0:
            download_data("data")

        self.imgs = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".png")])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask_np=np.array(mask)
        obj_ids=np.unique(mask_np)
        obj_ids = obj_ids[obj_ids !=0]

        boxes=[]

        for obj_id in obj_ids:
            pos=np.argwhere(mask_np==obj_id)
            ymin,xmin=pos.min(axis=0)
            ymax,xmax=pos.max(axis=0)

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels=torch.ones((len(boxes),),dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        img_tensor=self.transform(img)
        return img_tensor, target

if __name__ == "__main__":
    dataset = PennFudanDataset()
    print(dataset.imgs)