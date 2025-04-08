from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from PIL import Image
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    CLIPSegProcessor,
    CLIPSegForImageSegmentation,
)
from transformers.modeling_outputs import ModelOutput

from .config import ClipSegMultiClassConfig
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

def flatten_outputs(preds, targets, num_classes):
    """Flatten predictions and targets to 1D arrays, filter ignored labels."""
    preds = preds.cpu().numpy().reshape(-1)
    targets = targets.cpu().numpy().reshape(-1)

    mask = (targets >= 0) & (targets < num_classes)
    return preds[mask], targets[mask]

def compute_metrics(all_preds, all_targets, num_classes, average="macro"):
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    return metrics


@dataclass
class ClipSegMultiClassOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.LongTensor] = None


class ClipSegMultiClassModel(PreTrainedModel):
    config_class = ClipSegMultiClassConfig
    base_model_prefix = "clipseg_multiclass"

    def __init__(self, config: ClipSegMultiClassConfig):
        super().__init__(config)

        self.config = config
        self.class_labels = config.class_labels
        self.num_classes = config.num_classes
        self.processor = CLIPSegProcessor.from_pretrained(config.model)
        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(config.model)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ClipSegMultiClassOutput:

        if pixel_values is None or input_ids is None:
            raise ValueError("Both `pixel_values` and `input_ids` must be provided.")

        pixel_values = pixel_values.to(self.device)
        input_ids = input_ids.to(self.device)

        outputs = self.clipseg(pixel_values=pixel_values, input_ids=input_ids)
        raw_logits = outputs.logits  # shape: [B * C, H, W]

        B = raw_logits.shape[0] // self.num_classes
        C = self.num_classes
        H, W = raw_logits.shape[-2:]

        logits = raw_logits.view(B, C, H, W)  # [B, C, H, W]
        pred = torch.argmax(logits, dim=1)   # [B, H, W]

        loss = self.loss_fct(logits, labels.long()) if labels is not None else None

        return ClipSegMultiClassOutput(
            loss=loss,
            logits=logits,
            predictions=pred
        )

    @torch.no_grad()
    def predict(self, images: Union[List, "PIL.Image.Image"]) -> torch.Tensor:
        self.eval()
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(
            images=[img for img in images for _ in self.class_labels],
            text=self.class_labels * len(images),
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        output = self.forward(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"]
        )
        return output.predictions

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> dict:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import numpy as np

        self.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device)     # [B * C, 3, H, W]
                input_ids = batch["input_ids"].to(self.device)           # [B * C, T]
                labels = batch["labels"].to(self.device)                 # [B, H, W]

                outputs = self.forward(pixel_values=pixel_values, input_ids=input_ids)
                preds = outputs.predictions  # [B, H, W]

                for pred, label in zip(preds, labels):
                    pred = pred.cpu().flatten()
                    label = label.cpu().flatten()

                    mask = label != 0
                    pred = pred[mask]
                    label = label[mask]

                    all_preds.append(pred)
                    all_targets.append(label)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

