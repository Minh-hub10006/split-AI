import torch
import io
import requests
import cv2
import numpy as np
import torch.nn.functional as F
from src.core.yaml_config import YAMLConfig

# ===== LOAD MODEL (chỉ dùng decoder) =====
cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model
checkpoint = torch.load("weight/dfine_l_coco.pth", map_location="cpu")
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)
model.eval()

image_path = "test.jpg"

# ===== GỬI ẢNH LÊN ENCODER SERVER =====
with open(image_path, "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/encode",
        files={"file": f}
    )

feat = torch.load(io.BytesIO(response.content))

# ===== CHẠY DECODER =====
with torch.no_grad():
    outputs = model.decoder(feat,None)

logits = outputs["pred_logits"][0]  
boxes = outputs["pred_boxes"][0]     

# ===== TÍNH SCORE =====
probs = F.softmax(logits, dim=-1)
scores, labels = probs.max(-1)

# ===== LỌC THEO THRESHOLD =====
threshold = 0.5
keep = scores > threshold

boxes = boxes[keep]
scores = scores[keep]
labels = labels[keep]
print("Max score:", scores.max().item())
print("Num boxes:", len(boxes))
# ===== VẼ BOX =====
image = cv2.imread(image_path)
h, w, _ = image.shape

for box, score in zip(boxes, scores):
    cx, cy, bw, bh = box

    
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(image, f"{score:.2f}",
                (x1, max(0, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2)

cv2.imwrite("result.jpg", image)


print("Done. Saved result.jpg")
