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
import scipy.ndimage

import sys

# sys.path.insert(0, '/users/ysong135/Desktop/videopredictor/flowdiffusion/gmflow') # Oscar
sys.path.insert(0, '/home/yilong/Documents/videopredictor/flowdiffusion/gmflow') # Local
import get_flow

# sys.path.insert(0, '/users/ysong135/Desktop/videopredictor/flowdiffusion/clip_processor_f3rm/f3rm') # Oscar
sys.path.insert(0, '/home/yilong/Documents/videopredictor/flowdiffusion/clip_processor_f3rm/f3rm') # Local
import scripts.get_clip_features
from features.clip import clip

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

class CLIPArgs:
    model_name: str = "ViT-L/14@336px"
    skip_center_crop: bool = True
    batch_size: int = 64

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the CLIP model parameters."""
        return {
            "model_name": cls.model_name,
            "skip_center_crop": cls.skip_center_crop,
        }

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

        if semantic_map:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model, preprocess = clip.load(CLIPArgs.model_name, device=device)


        for seq_dir in sequence_dirs:
            print(f'Loading from {seq_dir}')
            task = seq_dir.split("/")[-2].replace('_', ' ')
            with h5py.File(seq_dir, 'r') as f:
                data = f['data']
                for demo in tqdm(data):
                    next_obs = f['data'][demo]['next_obs']['sideview_image'][self.frame_skip:][::self.frame_skip+1]/255.0
                    obs = f['data'][demo]['obs']['sideview_image'][::self.frame_skip+1][:len(next_obs)]/255.0
                    if semantic_map:
                        next_obs_semantic = scripts.get_clip_features.get_clip_features(next_obs, task, device, model, preprocess)
                        obs_semantic = scripts.get_clip_features.get_clip_features(obs, task, device, model, preprocess)
                        next_obs_heatmap = np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], 1))
                        obs_heatmap = np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], 1))
                        for i in range(obs.shape[0]):
                            next_obs_heatmap[i] = np.expand_dims(scipy.ndimage.zoom(next_obs_semantic[i], zoom=(128/9, 128/9), order=1), axis=-1)
                            obs_heatmap[i] = np.expand_dims(scipy.ndimage.zoom(obs_semantic[i], zoom=(128/9, 128/9), order=1), axis=-1)

                        obs = np.concatenate((obs, obs_heatmap), axis=3)
                        next_obs = np.concatenate((next_obs, next_obs_heatmap), axis=3)

                    for i in range(len(obs)):
                        self.tasks.append(task)
                        self.obs.append(obs[i])
                        self.next_obs.append(next_obs[i])
        
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

        if semantic_map:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model, preprocess = clip.load(CLIPArgs.model_name, device=device)


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

                    if semantic_map:
                        next_obs_semantic = scripts.get_clip_features.get_clip_features(next_obs, task, device, model, preprocess)
                        obs_semantic = scripts.get_clip_features.get_clip_features(obs, task, device, model, preprocess)
                        next_obs_heatmap = np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], 1))
                        obs_heatmap = np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], 1))
                        for i in range(obs.shape[0]):
                            next_obs_heatmap[i] = np.expand_dims(scipy.ndimage.zoom(next_obs_semantic[i], zoom=(128/9, 128/9), order=1), axis=-1)
                            obs_heatmap[i] = np.expand_dims(scipy.ndimage.zoom(obs_semantic[i], zoom=(128/9, 128/9), order=1), axis=-1)

                        obs = np.concatenate((obs, obs_heatmap), axis=3)
                        next_obs = np.concatenate((next_obs, next_obs_heatmap), axis=3)

                    obs = np.concatenate((obs, obs_depth), axis=3)
                    next_obs = np.concatenate((next_obs, next_obs_depth), axis=3)
                    
                    for i in range(len(obs)):
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
        
        if semantic_map:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model, preprocess = clip.load(CLIPArgs.model_name, device=device)

        for seq_dir in sequence_dirs:
            print(f'Loading from {seq_dir}')
            task = seq_dir.split("/")[-2].replace('_', ' ')
            with h5py.File(seq_dir, 'r') as f:
                data = f['data']
                for demo in tqdm(data):
                    next_obs = f['data'][demo]['next_obs']['sideview_image'][self.frame_skip:][::self.frame_skip+1]/255.0
                    obs = f['data'][demo]['obs']['sideview_image'][::self.frame_skip+1][:len(next_obs)]/255.0
                    if semantic_map:
                        obs_semantic = scripts.get_clip_features.get_clip_features(obs, task, device, model, preprocess)
                        obs_heatmap = np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], 1))
                        for i in range(obs.shape[0]):
                            obs_heatmap[i] = np.expand_dims(scipy.ndimage.zoom(obs_semantic[i], zoom=(128/9, 128/9), order=1), axis=-1)

                        obs = np.concatenate((obs, obs_heatmap), axis=3)

                    for i in range(len(obs)):
                        flow = get_flow.get_gmflow_flow(flow_model, obs[i][:,:,:3], next_obs[i])
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
