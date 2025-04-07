from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

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

    def _prepare_inputs(self, images: List[Image.Image]):
        prompts = self.class_labels * len(images)
        expanded_images = [img for img in images for _ in self.class_labels]

        inputs = self.processor(
            images=expanded_images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        return inputs

    def forward(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
) ->     ClipSegMultiClassOutput:

        if isinstance(images, Image.Image):
            images = [images]

        device = self.device

        if images is not None:
            inputs = self._prepare_inputs(images)
            pixel_values = inputs["pixel_values"].to(device)
            input_ids = inputs["input_ids"].to(device)
        else:
            raise ValueError("`images` must be provided for CLIPSeg.")

        outputs = self.clipseg(pixel_values=pixel_values, input_ids=input_ids)
        raw_logits = outputs.logits

        B = len(images)
        C = len(self.class_labels)
        H, W = raw_logits.shape[-2:]
        logits = raw_logits.view(B, C, H, W)

        probs = logits

        pred = torch.argmax(probs, dim=1)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.long())

        return ClipSegMultiClassOutput(
            loss=loss,
            logits=logits,
            predictions=pred
        )

    @torch.no_grad()
    def predict(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        self.eval()
        output = self.forward(images=images)
        return output.predictions
