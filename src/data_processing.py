import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class SingleClassSegmentationDataset(Dataset):
    def __init__(self, dataset, class_labels, image_size=352, transform=None):
        
        self.items = dataset
        self.class_labels = class_labels
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        image = Image.open(item["img_path"]).convert("RGB")
        mask = Image.open(item["mask_path"]).convert("L")
        class_name = item["label"]

        class_index = self.class_labels.index(class_name)
        background_index = 0

        mask_np = np.array(mask) > 0
        final_mask = np.full(mask_np.shape, background_index, dtype=np.uint8)
        final_mask[mask_np] = class_index

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        final_mask = Image.fromarray(final_mask).resize((self.image_size, self.image_size), Image.NEAREST)

        if self.transform:
            image, final_mask = self.transform(image, final_mask)

        return {
            "image": image,
            "labels": torch.from_numpy(np.array(final_mask)).long()
        }
    

class SegmentationCollator:
    def __init__(self, processor, class_labels):
        self.processor = processor
        self.class_labels = class_labels

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        labels = [item["labels"] for item in batch]

        prompts = self.class_labels * len(images)
        expanded_images = [img for img in images for _ in self.class_labels]

        inputs = self.processor(
            images=expanded_images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        return {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
            "labels": torch.stack(labels)
        }
