import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from utils import dist_util, logger
from data.WHUloader import WHU
from data.Masloader import Mas
from data.SV2loader import SV2
from data.Inrialoader import Inria
import torchvision.utils as vutils
from utils.utils import staple
from utils.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
from tqdm import tqdm

seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    if args.data_name == 'SpaceNet':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = SpaceNet(args, args.data_dir, transform_test)
        args.in_ch = 4

    elif args.data_name == 'WHU':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = WHU(args, args.data_dir, transform_test)
        args.in_ch = 4

    elif args.data_name == 'Mas':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = Mas(args, args.data_dir, transform_test)
        args.in_ch = 4

    elif args.data_name == 'Inria':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = Inria(args, args.data_dir, transform_test)
        args.in_ch = 4

    elif args.data_name == 'SV2':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = SV2(args, args.data_dir, transform_test)
        args.in_ch = 4

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion.")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    progress_bar = tqdm(total=len(data), desc="test", unit="image")
    for _ in range(len(data)):
        print("The number of total data is ", len(data))
        b, m, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$

        if args.data_name == 'SpaceNet':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'WHU':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'Mas':
            slice_ID=path[0].split('.')[0]
        elif args.data_name == 'SV2':
            slice_ID=path[0].split('.')[0]
        elif args.data_name == 'Inria':
            slice_ID=path[0].split('.')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()

            co = th.tensor(cal_out)
            enslist.append(sample[:,-1,:,:])

            if args.debug:
                if args.data_name == 'SpaceNet':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                elif args.data_name == 'Inria':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                elif args.data_name == 'WHU':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                elif args.data_name == 'Mas':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                compose = th.cat(tup,0)
                vutils.save_image(compose, fp = os.path.join(args.out_dir, str(slice_ID)+'_output'+str(i)+".jpg"), nrow = 1, padding = 10)
                progress_bar.update(1)
        ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_'+".jpg"), nrow = 1, padding = 10)

def create_argparser():
    defaults = dict(
        data_name = 'WHU',
        data_dir="../data/WHU/test/",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,
        gpu_dev = "0",
        out_dir='',
        multi_gpu = None, #"0,1,2"
        debug = False,
        rescale_timesteps = False,
        rescale_learned_sigmas = False,
        noise_schedule = "linear",
        diffusion_steps = 100,
        attention_resolutions = 16,
        use_scale_shift_norm = False,
        learn_sigma = True,
        num_heads = 1,
        num_res_blocks = 2,
        class_cond = False,
        num_channels = 128,
        image_size = 512
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
