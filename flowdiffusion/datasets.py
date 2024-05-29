from torch.utils.data import Dataset
import os
from glob import glob
import torch
from utils import get_paths, get_paths_from_dir
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T
import random
from torchvideotransforms import video_transforms, volume_transforms
from einops import rearrange
# from vidaug import augmentors as va

import h5py
from transformers import CLIPProcessor, CLIPModel
random.seed(0)
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '/home/yilong/Documents/videopredictor/flowdiffusion/gmflow')
import get_flow

sys.path.insert(0, '/home/yilong/Documents/videopredictor/flowdiffusion/clip_processor_f3rm/f3rm/scripts')
import get_clip_features

### Sequential Datasets: given first frame, predict all the future frames


def visualize_semantic_map(image_tensor, semantic_map_np):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')

    # Visualize the semantic map
    plt.subplot(1, 2, 2)
    plt.imshow(semantic_map_np, cmap='viridis')
    plt.title('Semantic Map')
    plt.axis('off')

    plt.show()

def get_image_with_semantic_map(clip_model, clip_processor, image_array):
    inputs = clip_processor(images=image_array, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    print(image_features.shape)
    # Resize semantic map to (128, 128, 1)
    semantic_map = torch.nn.functional.interpolate(image_features, size=(128, 128), mode="bicubic", align_corners=False)

    # Convert semantic map to numpy array
    semantic_map_np = semantic_map.squeeze(0).numpy()

    # Concatenate original image and semantic map
    concatenated_array = np.concatenate((image_array, semantic_map_np), axis=2)

    return concatenated_array.shape  # Output: (4, 128, 128)

def visualize_RGB(image1, image2):
    """
    Visualize two (128, 128, 3) RGB images side by side.

    Parameters:
    image1 (numpy.ndarray): The first RGB image array with shape (128, 128, 3).
    image2 (numpy.ndarray): The second RGB image array with shape (128, 128, 3).
    """
    fig, axes = plt.subplots(1, 2)  # Create a figure with two subplots
    
    axes[0].imshow(image1)
    axes[0].axis('off')  # Turn off axis for the first subplot
    axes[0].set_title('Image 1')  # Set a title for the first subplot
    
    axes[1].imshow(image2)
    axes[1].axis('off')  # Turn off axis for the second subplot
    axes[1].set_title('Image 2')  # Set a title for the second subplot
    
    plt.show()

class Datasethdf5RGB(Dataset):
    def __init__(self, path='../datasets/', semantic_map=False, frame_skip=0, random_crop=False):
        if semantic_map:
            print("Preparing RGB data from hdf5 dataset with semantic channel (RGB + semantic) ...")
        else:
            print("Preparing RGB data from hdf5 dataset ...")
        
        self.frame_skip = frame_skip

        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)
        self.tasks = []
        self.obs = []
        self.next_obs = []


        for seq_dir in sequence_dirs:
            print(f'Loading from {seq_dir}')
            task = seq_dir.split("/")[-2].replace('_', ' ')
            with h5py.File(seq_dir, 'r') as f:
                data = f['data']
                for demo in tqdm(data):
                    next_obs = f['data'][demo]['next_obs']['sideview_image'][self.frame_skip:][::self.frame_skip+1]/255.0
                    obs = f['data'][demo]['obs']['sideview_image'][::self.frame_skip+1][:len(next_obs)]/255.0
                    next_obs_semantic = get_clip_features.get_clip_features(next_obs, task)
                    obs_semantic = get_clip_features.get_clip_features(obs, task)
                    for i in range(len(obs)):
                        if semantic_map:
                            get_clip_features.get_clip_features(torch.tensor(obs[i]).permute(2, 0, 1), task)
                            self.tasks.append(task)
                        else:
                            self.obs.append(obs[i])
                            self.next_obs.append(next_obs[i])
                            self.tasks.append(task)
        
        self.transform = video_transforms.Compose([
                volume_transforms.ClipToTensor()
        ])
        print('Done')

    def get_samples(self, idx):
        return [self.obs[idx], self.next_obs[idx]]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        samples = self.get_samples(idx)
        x_cond = torch.from_numpy(rearrange(samples[0], "h w c -> c h w")).float()
        # x = rearrange(images[:, 1:], "c f h w -> (f c) h w")
        x = torch.from_numpy(rearrange(samples[1], 'h w c -> c h w')).float()
        task = self.tasks[idx]
        return x, x_cond, task

def visualize_depth_image(rgb_image, depth_channel):
    import cv2
    from matplotlib import pyplot as plt
    # Display RGB image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('RGB Image')
    plt.axis('off')

    # Display Depth image
    plt.subplot(1, 2, 2)
    plt.imshow(depth_channel[:,:,0], cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Depth Channel')
    plt.axis('off')

    plt.show()

class Datasethdf5RGBD(Dataset):
    def __init__(self, path='../datasets/', semantic_map=False, frame_skip=0, random_crop=False):
        if semantic_map:
            print("Preparing RGBD data from hdf5 dataset with semantic channel (RGBD + semantic) ...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            print("Preparing RGBD data from hdf5 dataset ...")
        
        self.frame_skip = frame_skip

        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)
        self.tasks = []
        self.obs = []
        self.next_obs = []
        self.depth_max = 1.099
        self.depth_min = 0.507


        for seq_dir in sequence_dirs:
            print(f'Loading from {seq_dir}')
            task = seq_dir.split("/")[-2].replace('_', ' ')
            with h5py.File(seq_dir, 'r') as f:
                data = f['data']
                for demo in tqdm(data):
                    next_obs = f['data'][demo]['next_obs']['sideview_image'][self.frame_skip:][::self.frame_skip+1]/255.0
                    obs = f['data'][demo]['obs']['sideview_image'][::self.frame_skip+1][:len(next_obs)]/255.0
                    
                    next_obs_depth = f['data'][demo]['next_obs']['sideview_depth'][self.frame_skip:][::self.frame_skip+1]
                    obs_depth = f['data'][demo]['obs']['sideview_depth'][::self.frame_skip+1][:len(next_obs)]
                    #obs_depth = np.clip(obs_depth, self.depth_min, self.depth_max) # Clipping is problematic
                    obs_depth = (obs_depth - np.min(obs_depth)) / (np.max(obs_depth) - np.min(obs_depth)) # Normalize
                    #next_obs_depth = np.clip(next_obs_depth, self.depth_min, self.depth_max)
                    next_obs_depth = (next_obs_depth - np.min(next_obs_depth)) / (np.max(next_obs_depth) - np.min(next_obs_depth)) # Normalize

                    obs = np.concatenate((obs, obs_depth), axis=3)
                    next_obs = np.concatenate((next_obs, next_obs_depth), axis=3)
                    for i in range(len(obs)):
                        # clip depth
                        if semantic_map:
                            self.obs.append(get_image_with_semantic_map(clip_model, clip_processor, obs[i]))
                            self.next_obs.append(get_image_with_semantic_map(clip_model, clip_processor, next_obs[i]))
                            self.tasks.append(task)
                        else:
                            self.obs.append(obs[i])
                            self.next_obs.append(next_obs[i])
                            self.tasks.append(task)
        
        self.transform = video_transforms.Compose([
                volume_transforms.ClipToTensor()
        ])
        print('Done')

    def get_samples(self, idx):
        return [self.obs[idx], self.next_obs[idx]]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        samples = self.get_samples(idx)
        x_cond = torch.from_numpy(rearrange(samples[0], "h w c -> c h w")).float()
        # x = rearrange(images[:, 1:], "c f h w -> (f c) h w")
        x = torch.from_numpy(rearrange(samples[1], 'h w c -> c h w')).float()
        task = self.tasks[idx]
        return x, x_cond, task
    
class Datasethdf5Flow(Dataset):
    def __init__(self, path='../datasets/', semantic_map=False, frame_skip=0, random_crop=False):
        if semantic_map:
            print("Preparing RGBD data from hdf5 dataset with semantic channel (RGBD + semantic) ...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            print("Preparing RGBD data from hdf5 dataset ...")
        
        self.frame_skip = frame_skip

        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)
        self.tasks = []
        self.obs = []
        self.next_obs = []
        self.depth_max = 1.099
        self.depth_min = 0.507

        resume = '/home/yilong/Documents/videopredictor/flowdiffusion/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth'

        flow_model = get_flow.get_gmflow_model(resume)


        for seq_dir in sequence_dirs:
            print(f'Loading from {seq_dir}')
            task = seq_dir.split("/")[-2].replace('_', ' ')
            with h5py.File(seq_dir, 'r') as f:
                data = f['data']
                for demo in tqdm(data):
                    next_obs = f['data'][demo]['next_obs']['sideview_image'][self.frame_skip:][::self.frame_skip+1]/255.0
                    obs = f['data'][demo]['obs']['sideview_image'][::self.frame_skip+1][:len(next_obs)]/255.0

                    for i in range(len(obs)):
                        flow = get_flow.get_gmflow_flow(flow_model, obs[i], next_obs[i])
                        if semantic_map:
                            self.obs.append(get_image_with_semantic_map(clip_model, clip_processor, obs[i]))
                            self.next_obs.append(get_image_with_semantic_map(clip_model, clip_processor, next_obs[i]))
                            self.tasks.append(task)
                        else:
                            self.obs.append(obs[i])
                            self.next_obs.append(flow)
                            self.tasks.append(task)
        
        self.transform = video_transforms.Compose([
                volume_transforms.ClipToTensor()
        ])
        print('Done')

    def get_samples(self, idx):
        return [self.obs[idx], self.next_obs[idx]]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        samples = self.get_samples(idx)
        x_cond = torch.from_numpy(rearrange(samples[0], "h w c -> c h w")).float()
        # x = rearrange(images[:, 1:], "c f h w -> (f c) h w")
        x = torch.from_numpy(rearrange(samples[1], 'h w c -> c h w')).float()
        task = self.tasks[idx]
        return x, x_cond, task


if __name__ == "__main__":
    dataset = SequentialNavDataset("../datasets/thor")
    x, x_cond, task = dataset[2]
    print(x.shape)
    print(x_cond.shape)
    print(task)

