# 🧠 ClipSegMultiClass

Multi-class semantic segmentation with CLIP-based zero-shot guidance.  
Built on top of [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined), this version adds support for multiple class prompts and training on a custom dataset with pixel-wise masks.

## 🤖 Model

**Model Name:** `BioMike/clipsegmulticlass_v1`  
**Base:** `CIDAS/clipseg-rd64-refined`  
**Input:** RGB image + list of class prompts  
**Output:** Segmentation map with one class label per pixel

### 🏷️ Supported classes
- Pig
- Horse
- Sheep

(background class is automatically handled as `0`)

## 📈 Evaluation

| Model                       | Precision | Recall | F1 Score | Accuracy |
|----------------------------|-----------|--------|----------|----------|
| CIDAS/clipseg-rd64-refined | 0.5239    | 0.2114 | 0.2882   | 0.2665   |
| BioMike/clipsegmulticlass_v1 | 0.7460    | 0.5035 | 0.6009   | 0.6763   |

Evaluation is done excluding background pixels. Metrics are macro-averaged over all classes.

## 🚀 Demo

Try it out here 👉 [Gradio Space](https://huggingface.co/spaces/BioMike/clipsegmulticlass)

<p align="center">
  <img src="https://huggingface.co/spaces/BioMike/clipsegmulticlass/resolve/main/demo.gif" width="600"/>
</p>

## 📦 Usage

```python
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ClipSegMultiClassModel
from config import ClipSegMultiClassConfig

# Load model and config
model = ClipSegMultiClassModel.from_pretrained("trained_clipseg_multiclass").to("cuda").eval()
config = model.config  # contains label2color

# Load image
image = Image.open("pigs.jpg").convert("RGB")

# Run inference
mask = model.predict(image)  # shape: [1, H, W]

# Visualize
def visualize_mask(mask_tensor: torch.Tensor, label2color: dict):
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.squeeze(0)

    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)  # [H, W]
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in label2color.items():
        color_mask[mask_np == class_idx] = color

    return color_mask

color_mask = visualize_mask(mask, config.label2color)

plt.imshow(color_mask)
plt.axis("off")
plt.title("Predicted Segmentation Mask")
plt.show()
