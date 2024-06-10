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

#from raft import RAFT
from torchvision.models.optical_flow import raft_large

import sys

# sys.path.insert(0, '/users/ysong135/Desktop/videopredictor/flowdiffusion/gmflow') # Oscar
sys.path.insert(0, '/home/yilong/Documents/videopredictor/flowdiffusion/gmflow') # Local
import get_flow

# sys.path.insert(0, '/users/ysong135/Desktop/videopredictor/flowdiffusion/clip_processor_f3rm/f3rm') # Oscar
sys.path.insert(0, '/home/yilong/Documents/videopredictor/flowdiffusion/clip_processor_f3rm/f3rm') # Local
import scripts.get_clip_features
from features.clip import clip

cameras_RGB = ['agentview_image']
cameras_RGBD = ['agentview_depth']

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
                    for camera in cameras_RGB:
                        next_obs = f['data'][demo]['next_obs'][camera][self.frame_skip:][::self.frame_skip+1]/255.0
                        obs = f['data'][demo]['obs'][camera][::self.frame_skip+1][:len(next_obs)]/255.0
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
                    for camera_RGB, camera_RGBD in zip(cameras_RGB, cameras_RGBD):
                        next_obs = f['data'][demo]['next_obs'][camera_RGB][self.frame_skip:][::self.frame_skip+1]/255.0
                        obs = f['data'][demo]['obs'][camera_RGB][::self.frame_skip+1][:len(next_obs)]/255.0
                        
                        next_obs_depth = f['data'][demo]['next_obs'][camera_RGBD][self.frame_skip:][::self.frame_skip+1]
                        obs_depth = f['data'][demo]['obs'][camera_RGBD][::self.frame_skip+1][:len(next_obs)]
                        obs_depth = np.clip(obs_depth, self.depth_min, self.depth_max) # Clipping is problematic
                        obs_depth = (obs_depth - np.min(obs_depth)) / (np.max(obs_depth) - np.min(obs_depth)) # Normalize
                        next_obs_depth = np.clip(next_obs_depth, self.depth_min, self.depth_max)
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
    

def get_raft_flow(image1, image2):
    model = RAFT(pretrained=True)

    # Convert images to tensors
    image1_tensor = torch.from_numpy(np.array(image1)).permute(2, 0, 1).float()
    image2_tensor = torch.from_numpy(np.array(image2)).permute(2, 0, 1).float()

    # Normalize images
    image1_tensor = image1_tensor / 255.0
    image2_tensor = image2_tensor / 255.0

    # Add batch dimension
    image1_tensor = image1_tensor.unsqueeze(0)
    image2_tensor = image2_tensor.unsqueeze(0)

    # Compute optical flow
    flow = model(image1_tensor, image2_tensor)

    # Extract the flow tensor
    flow_tensor = flow[0].permute(1, 2, 0).detach().cpu().numpy()

    print(flow_tensor.shape)

    return flow_tensor


def visualize_flow(rgb_image1, rgb_image2, optical_flow):
    x = np.arange(0, 128, 1)
    y = np.arange(0, 128, 1)
    x, y = np.meshgrid(x, y)

    # Extracting flow components
    u = optical_flow[:,:,0]
    v = optical_flow[:,:,1]

    # Plotting the images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plotting the RGB image
    axes[0].imshow(rgb_image1)
    axes[0].set_title('Current obs')

    axes[1].imshow(rgb_image2)
    axes[1].set_title('Next obs')

    # Plotting the optical flow vectors
    axes[2].imshow(np.zeros((128, 128)), cmap='gray')  # Displaying a blank image
    axes[2].quiver(x, y, u, v, color='r', angles='xy', scale_units='xy', scale=1)
    axes[2].set_title('Optical Flow Visualization')

    plt.show()
    
class Datasethdf5Flow(Dataset):
    def __init__(self, path='../datasets/', semantic_map=False, frame_skip=0, random_crop=False):
        if semantic_map:
            print("Preparing flow data from hdf5 dataset with semantic channel (flow + semantic) ...")
        else:
            print("Preparing flow data from hdf5 dataset ...")
        
        self.frame_skip = frame_skip

        sequence_dirs = glob(f"{path}/**/*.hdf5", recursive=True)
        self.tasks = []
        self.obs = []
        self.next_obs = []
        self.depth_max = 1.099
        self.depth_min = 0.507

        resume = '/home/yilong/Documents/videopredictor/flowdiffusion/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth'

        flow_model = get_flow.get_gmflow_model(resume)
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if semantic_map:
            model, preprocess = clip.load(CLIPArgs.model_name, device=device)

        for seq_dir in sequence_dirs:
            print(f'Loading from {seq_dir}')
            task = seq_dir.split("/")[-2].replace('_', ' ')
            with h5py.File(seq_dir, 'r') as f:
                data = f['data']
                for demo in tqdm(data):
                    for camera in cameras_RGB:
                        next_obs = f['data'][demo]['next_obs'][camera][self.frame_skip:][::self.frame_skip+1]/255.0
                        obs = f['data'][demo]['obs'][camera][::self.frame_skip+1][:len(next_obs)]/255.0
                        if semantic_map:
                            obs_semantic = scripts.get_clip_features.get_clip_features(obs, task, device, model, preprocess)
                            obs_heatmap = np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], 1))
                            for i in range(obs.shape[0]):
                                obs_heatmap[i] = np.expand_dims(scipy.ndimage.zoom(obs_semantic[i], zoom=(128/9, 128/9), order=1), axis=-1)

                            obs = np.concatenate((obs, obs_heatmap), axis=3)

                        for i in range(len(obs)):
                            # flow = get_flow.get_gmflow_flow(flow_model, obs[i][:,:,:3], next_obs[i]) # GMFlow doesn't seem to work
                            # flow = get_raft_flow(obs[i][:,:,:3], next_obs[i])

                            model = raft_large(pretrained=True, progress=False).to(device)
                            model = model.eval()
                            flow = model(torch.tensor(obs[i][:,:,:3]).to(device).unsqueeze(0).permute(0, 3, 1, 2).float(), 
                                         torch.tensor(next_obs[i]).to(device).unsqueeze(0).permute(0, 3, 1, 2).float())[-1]
                            flow = flow.squeeze(0).permute(1,2,0).cpu().detach().numpy()
                            visualize_flow(obs[i][:,:,:3], next_obs[i], flow)

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
