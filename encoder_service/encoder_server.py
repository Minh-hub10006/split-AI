import torch
import io
import torchvision.transforms as T  
from PIL import Image               
from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import Response

from src.core.yaml_config import YAMLConfig

app = FastAPI()

cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model
# Tạo model rỗng từ file .yml
checkpoint = torch.load("weight/dfine_l_coco.pth", map_location="cpu")
# Load trọng số được train
if "model" in checkpoint: # Checkpoint là file lưu trữ trạng thái model tại 1 thời điểm
    state_dict = checkpoint["model"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.eval()

# Định nghĩa Transform
transform = T.Compose([
    T.Resize((640, 640)), # D-FINE mặc định thường dùng 640x640
    T.ToTensor(),
])

@app.post("/encode") # Khi có HTTP POST tới /encode thì chạy hàm này
async def encode(file: UploadFile = File(...)):
    # Đọc ảnh từ request
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Tiền xử lý
    x = transform(image).unsqueeze(0) # Tạo batch dimension [1, 3, 640, 640]

    with torch.no_grad():
        feat = model.encoder(model.backbone(x))
    
    # Lưu tensor vào bộ nhớ đệm để gửi đi
    buffer = io.BytesIO()
    torch.save(feat, buffer)
    buffer.seek(0)

    return Response(
        content=buffer.getvalue(),
        media_type="application/octet-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("encoder_server:app", host="127.0.0.1", port=8000, reload=True)
# uvicorn encoder_server:app --reload