from ultralytics import YOLO
import torch

# 确保有 GPU 环境
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 加载模型并移动到 GPU
model = YOLO(r'F:\work\Porject\SLS\model\detection\SLS_BJ_HandPhone_Detect_v8n_p2_20260409\weights\best.pt').to(device)

# 执行导出
results = model.export(
    format='onnx',
    opset=11,      # 建议提高 opset 到 11 或更高，opset 9 对 FP16 支持较差
    half=False,     # 现在 GPU 环境下该参数将生效
    imgsz=640,
    device=device,
    name="TCL_Hand_Detect_20250915"
)