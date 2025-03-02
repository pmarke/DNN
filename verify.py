import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. PyTorch will use GPU.")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("CUDA is not available. PyTorch will use CPU.")

import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())


# Create a 500x500 green image (BGR format, green is (0, 255, 0))
green_image = np.zeros((500, 500, 3), dtype=np.uint8)
green_image[:] = (0, 255, 0)

# Display the image
cv2.imshow('Green Image', green_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()