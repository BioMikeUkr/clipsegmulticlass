from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class ClipSegMultiClassConfig(PretrainedConfig):
    model_type = "clipseg-multiclass"
    is_composition = False

    def __init__(
        self,
        class_labels=None,
        label2color=None,
        model="CIDAS/clipseg-rd64-refined",
        image_size=352,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.class_labels = class_labels or []
        self.label2color = label2color or {
            i: [int(255 * (i / max(1, len(class_labels) - 1))), 0, 255 - int(255 * (i / max(1, len(class_labels) - 1)))]
            for i in range(len(class_labels))
        }

        self.num_classes = len(self.class_labels) 
        self.model = model
        self.image_size = image_size
