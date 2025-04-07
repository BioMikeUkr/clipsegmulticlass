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
