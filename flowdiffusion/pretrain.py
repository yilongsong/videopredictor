from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetRGB, UnetRGBD, UnetRGBDFlow
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import SequentialDatasetv2
from datasets import Datasethdf5RGB, Datasethdf5RGBD, Datasethdf5RGBDFlow
from torch.utils.data import Subset
import argparse

def main(args):
    valid_n = 3
    sample_per_seq = 2
    target_size = (128, 128)

    datasets_path = "/users/ysong135/scratch/datasets/" # Oscar
    # datasets_path = "/home/yilong/Documents/videopredictor/datasets/" # Local

    if args.mode == 'inference':
        if args.modality == 'RGB':
            train_set = Datasethdf5RGB(
                path=datasets_path,
                semantic_map=args.semantic,
                frame_skip=3,
                random_crop=True
            )
        elif args.modality == 'RGBD':
            train_set = Datasethdf5RGBD(
                path=datasets_path,
                semantic_map=args.semantic,
                frame_skip=3,
                random_crop=True
            )
        elif args.modality == 'OF':
            train_set = Datasethdf5RGBDFlow(
                path=datasets_path,
                semantic_map=args.semantic,
                frame_skip=3,
                random_crop=True
            )
        valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
        valid_set = Subset(train_set, valid_inds)
    else:
        if args.modality == 'RGB':
            train_set = Datasethdf5RGB(
                path=datasets_path,
                semantic_map=args.semantic,
                frame_skip=3,
                random_crop=True
            )
        elif args.modality == 'RGBD':
            train_set = Datasethdf5RGBD(
                path=datasets_path,
                semantic_map=args.semantic,
                frame_skip=3,
                random_crop=True
            )
        elif args.modality == 'OF':
            train_set = Datasethdf5RGBDFlow(
                path=datasets_path,
                semantic_map=args.semantic,
                frame_skip=3,
                random_crop=True
            )
        valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
        valid_set = Subset(train_set, valid_inds)
        #train_set.obs, train_set.next_obs = train_set.obs[:-valid_n], train_set.next_obs[:-valid_n]

    if args.modality == 'RGB':
        unet = UnetRGB()
    elif args.modality == 'RGBD':
        unet = UnetRGBD()
    elif args.modality == 'OF':
        unet = UnetRGBDFlow()

    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusion(
        channels=3*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps =60000,
        save_and_sample_every =2,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =4,
        valid_batch_size =4,
        gradient_accumulate_every = 1,
        num_samples=valid_n, 
        results_folder ='../results/pretrain',
        fp16 =True,
        amp=True,
    )

    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)

    if args.mode == 'train':
        trainer.train()
    else:
        import numpy as np
        from torchvision import transforms
        import imageio
        import torch
        from os.path import splitext

        c = 3
        if args.modality == 'RGBD':
            c = 4
        elif args.modality == 'OF':
            c = 1
        text = args.text
        guidance_weight = args.guidance_weight
        obs_next_text = valid_set[np.random.randint(len(valid_set))]
        obs = obs_next_text[0]
        next_obs = obs_next_text[1]
        instruction = obs_next_text[2]
        batch_size = 1
        output = trainer.sample(obs.unsqueeze(0), [text], batch_size, guidance_weight).cpu()
        output = output[0].reshape(-1, c, *target_size)
        output = torch.cat([obs.unsqueeze(0), output], dim=0)
        output_gt = torch.cat([obs.unsqueeze(0), next_obs.unsqueeze(0)], dim=0)

        output_gif = '../examples/' + text.replace(' ', '_') + '_out.gif'
        output_gt_gif = '../examples/' + text.replace(' ', '_') + '_out_gt.gif'
        if args.modality == 'RGB':
            output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
            output_gt = (output_gt.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
        elif args.modality == 'RGBD':
            output = (output.cpu().numpy().transpose(0, 2, 3, 1)[:,:,:,-1].clip(0, 1) * 255).astype('uint8')
            output_gt = (output_gt.cpu().numpy().transpose(0, 2, 3, 1)[:,:,:,-1].clip(0, 1) * 255).astype('uint8')
        imageio.mimsave(output_gif, output, duration=200, loop=1000)
        imageio.mimsave(output_gt_gif, output_gt, duration=200, loop=1000)
        print(f'Generated {output_gif}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'inference']) # set to 'inference' to generate samples
    parser.add_argument('-o', '--modality', type=str, default='RGB', choices=['RGB', 'RGBD', 'OF'])
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None) # set to checkpoint number to resume training or generate samples
    parser.add_argument('-t', '--text', type=str, default=None) # set to text to generate samples
    parser.add_argument('-n', '--sample_steps', type=int, default=100) # set to number of steps to sample
    parser.add_argument('-g', '--guidance_weight', type=int, default=0) # set to positive to use guidance
    parser.add_argument('-s', '--semantic', action='store_true', help='Add semantic channel')
    args = parser.parse_args()
    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    main(args)