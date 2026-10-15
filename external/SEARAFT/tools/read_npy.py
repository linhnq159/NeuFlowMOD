import numpy as np
import torch
from core.utils.flow_viz import flow_to_image
import cv2

file_npy = "/data/linhnq11/MOD/CPP/mod/build/npy_dump/35_9/frame_2.npy" # [H, W, 2] float 32

img = cv2.imread("output.jpg")
print("img.shape : ", img.shape)

# Try to peek at raw header (first 100 bytes)
with open(file_npy, 'rb') as f:
    header = f.read(128)
    print("Raw header (first 128 bytes):")
    print(header[:128])
    print("\nHex dump (first 64 bytes):")
    print(header[:64].hex())

flow = np.load(file_npy) # [H, W, 2]

# flow = flow.permute(2, 0, 1) # [2, H, W]

print("flow shape: ", flow.shape)

print(flow.shape)

flow_save = flow

flow_vis = flow_to_image(flow_save, convert_to_bgr=True)

cv2.imwrite(f"output.jpg", flow_vis)

flow = flow.unsqueeze(0) # [1, 2, H, W]

flow = flow.float()