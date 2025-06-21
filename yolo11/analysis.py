from thop.profile import profile
import torch 
from model import YOLOv11
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dummy_input = torch.randn(1, 3, 640, 640).to(device)

print('device',device)
model = YOLOv11().build_model(version='n', num_classes=80).to(device)

print("dummy_input device: ", dummy_input.device)

# with torch.autocast(device_type=device.type, dtype=torch.float16):
#     flops, params = profile(model, (dummy_input,), verbose=False)

# print(f"Params: {params / 1e6:.3f}M")
# print(f"GFLOPs: {flops / 1e9:.3f}")

summary(model,input_data = dummy_input)