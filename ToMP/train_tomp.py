import torch
from models.tompnet import ToMPNet
from got10k_dataset import GOT10kDataset
import os
from tqdm import tqdm
from utils.box_ops import box_cxcywh_to_xyxy


VAL_PATH = "/home/artemis/DNN/datasets/GOT-10k/val_data/val"
WEIGHTS_PATH = "/home/artemis/DNN/ToMP/models/tomp_got10k.pth"

torch.set_printoptions(linewidth=500)
torch.set_printoptions(precision=3)
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigma = 1.0/2.0

# Dataset and DataLoader
dataset = GOT10kDataset(VAL_PATH, sigma=sigma)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = ToMPNet(sigma=sigma).to(device)
if os.path.exists(WEIGHTS_PATH):
    print(f"Loading weights from {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion_cls = torch.nn.BCEWithLogitsLoss()
criterion_bbox = torch.nn.MSELoss()

model.train()

# sample = dataset[0] 
# dataset.show_sample(sample)
torch.cuda.empty_cache()

for epoch in range(10):
    total_cls_loss, total_box_loss = 0.0, 0.0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        # Each batch entry is a list of length batch_size, each containing a list of length num_frames
        # We use the first frame as template, second as search (adjust if num_frames > 2)
        # Collate lists into tensors
   

        sample = dataset[0]
        print(sample['seq_name'])
        print(sample['frame_ids'])
        training_images = []
        training_images.append(sample['noncentered_images'][0].to(device).unsqueeze(0))  # [C, H, W]
        training_images.append(sample['noncentered_images'][1].to(device).unsqueeze(0))  # [C, H, W]
        test_image = sample['noncentered_images'][2].to(device).unsqueeze(0)
        heatmap = sample['noncentered_heatmaps'][2].to(device).unsqueeze(0)  # [1, H, W]
        training_boxes = []
        training_boxes.append(sample['noncentered_bbox'][0].to(device).unsqueeze(0))
        training_boxes.append(sample['noncentered_bbox'][1].to(device).unsqueeze(0))
        test_box = sample['noncentered_bbox'][2].to(device).unsqueeze(0)

        print("test_box before",test_box)

        # Convert test_box from [B, x, y, w, h] to [B, l, t, r, b] in the
        # search window frame
        test_box_ltrb = torch.zeros_like(test_box)
        test_box_ltrb[:, 0] = test_box[:, 0] 
        test_box_ltrb[:, 1] = test_box[:, 1]
        test_box_ltrb[:, 2] = test_box[:, 0] + test_box[:, 2]   # r = x + w/2
        test_box_ltrb[:, 3] = test_box[:, 1] + test_box[:, 3]   # b = y + h/2
        test_box = test_box_ltrb
        print("test box", test_box)

        y_cls, ltrb, max_locations = model(training_images,test_image,training_boxes)
        # print("y_cls", y_cls)
        print("max_locations", max_locations)
        print("ltrb", ltrb)

        # print("y_cls shape:", y_cls.shape)
        # print("heatmap:", torch.sigmoid(y_cls))

        cls_loss = criterion_cls(y_cls, heatmap) 
        box_loss = criterion_bbox(ltrb, test_box)

        loss = cls_loss + box_loss
        # loss = cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_cls_loss += cls_loss.item()
        total_box_loss += box_loss.item()
print(f"Epoch {epoch+1} - Cls: {total_cls_loss:.4f}, IoU: {total_box_loss:.4f}")
torch.save(model.state_dict(), WEIGHTS_PATH)
