import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import matplotlib.pyplot as plt

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Sample image array (128, 128, 3)
image_array = np.random.randint(0, 256, size=(128, 128, 3)).astype(np.uint8)

# Add batch dimension to the image array
image_array = np.transpose(image_array, (2, 0, 1))
image_array = np.expand_dims(image_array, axis=0)

# Encode image using CLIP
inputs = clip_processor(images=image_array, return_tensors="pt", padding=True)
with torch.no_grad():
    image_features = clip_model.get_image_features(**inputs)

print(image_features.shape)
# Resize semantic map to (128, 128, 1)
semantic_map = torch.nn.functional.interpolate(image_features, size=(1, 128, 128), mode="bicubic", align_corners=False)

# Convert semantic map to numpy array
semantic_map_np = semantic_map.squeeze(0).permute(1, 2, 0).numpy()

# Concatenate original image and semantic map
concatenated_array = np.concatenate((image_array.squeeze(0), semantic_map_np), axis=0)

print(concatenated_array.shape)  # Output: (128, 128, 4)

# Visualize the original image and its semantic map
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_array.squeeze(0))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(semantic_map_np.squeeze(), cmap='viridis')
plt.title('Semantic Map')
plt.axis('off')

plt.show()