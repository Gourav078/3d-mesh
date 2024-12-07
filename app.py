import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d

# Load pre-trained model and feature extractor
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# Load and preprocess the image
image = Image.open("pic.jpg")
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32

new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

# Extract features for depth prediction
inputs = feature_extractor(images=image, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Postprocess predicted depth
pad = 16
output = predicted_depth.squeeze(0).cpu().numpy() * 1000.0  # Convert to millimeters
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

# Display image and depth map
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.pause(5)

# Convert images to Open3D format
width, height = image.size
depth_image = (output * 255 / np.max(output)).astype(np.uint8)  # Normalize and convert to 8-bit
depth_o3d = o3d.geometry.Image(depth_image)

image_np = np.array(image)  # Convert PIL image to numpy array
image_o3d = o3d.geometry.Image(image_np)

# Create RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    image_o3d,
    depth_o3d,
    # depth_scale=1000.0,  # Depth values are in millimeters
    # depth_trunc=3.0,  # Truncate depth values greater than 3 meters
    convert_rgb_to_intensity=False  # Keep original RGB colors
)

# Define camera intrinsics
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsics.set_intrinsics(
    width=width, height=height, fx=500, fy=500, cx=width / 2, cy=height / 2
)

# Generate the point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
