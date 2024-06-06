import sys
sys.path.insert(0, '/home/yilong/Documents/videopredictor/flowdiffusion/') # Local
# sys.path.insert(0, '/users/ysong135/Desktop/videopredictor/flowdiffusion/clip_processor_f3rm/f3rm') # Oscar

from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetRGB, UnetRGBD, UnetFlow
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import Datasethdf5RGB, Datasethdf5RGBD, Datasethdf5Flow
from torch.utils.data import Subset
import argparse
import numpy as np
from matplotlib import pyplot as plt
import csv

from torchvision import transforms
import imageio
import torch
from os.path import splitext


checkpoint_num = 24
sample_steps = 100
target_size = (128, 128)


def visualize_prediction(img1, img2, img3):
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))
    img3 = np.transpose(img3, (1, 2, 0))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Current Frame')
    
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Ground Truth')
    
    axes[2].imshow(img3)
    axes[2].axis('off')
    axes[2].set_title('Prediction')
    
    plt.show()

    

def next_frame_predictor(modality, semantic=False):

    if modality == 'RGB':
        unet = UnetRGB(semantic)
        channels = 3
    elif modality == 'RGBD':
        unet = UnetRGBD(semantic)
        channels = 4
    elif modality == 'flow':
        unet = UnetFlow(semantic)
        channels = 2

    if semantic and modality != 'flow':
        channels += 1

    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusion(
        channels=channels,
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[None],
        valid_set=[None],
        train_lr=1e-4,
        train_num_steps=100,
        save_and_sample_every=100,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =8,
        results_folder ='results/pretrain',
        fp16 =True,
        amp=True,
        channels=channels
        )

    trainer.load(checkpoint_num)

    return trainer


def predict_next_frame(img, text, model):
    channels = img.shape[0]
    guidance_weight = 0
    batch_size = 1
    next_frame = model.sample(img.unsqueeze(0), [text], batch_size, guidance_weight).cpu()
    next_frame = next_frame[0].reshape(-1, channels, *target_size)
    return next_frame.squeeze(0)


def main():
    model = next_frame_predictor('RGB', semantic=False)

    datasets_path = "/home/yilong/Documents/videopredictor/datasets/"

    train_set = Datasethdf5RGB(
                path=datasets_path,
                semantic_map=False,
                frame_skip=3,
                random_crop=True
            )
    valid_inds = [i for i in range(0, len(train_set), len(train_set)//20)][:20]
    valid_set = Subset(train_set, valid_inds)

    obs_next_text = valid_set[np.random.randint(len(valid_set))]
    obs = obs_next_text[0]
    next_obs = obs_next_text[1]
    instruction = obs_next_text[2]

    pred = predict_next_frame(obs, instruction, model)

    visualize_prediction(obs, pred, next_obs)



if __name__ == "__main__":
    main()