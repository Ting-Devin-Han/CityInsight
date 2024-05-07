
import sys
import argparse
sys.path.append("../")
sys.path.append("./")
from utils import dist_util, logger
from diffusion.resample import create_named_schedule_sampler
from data.WHUloader import WHU
from data.Masloader import Mas
from data.Inrialoader import Inria
from data.SpaceNetloader import SpaceNet
from data.SV2loader import SV2
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from utils.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn

def count_parameters(model):
    para_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    para_count_kb = para_count / 1024
    para_count_mb = para_count_kb / 1024
    return para_count_mb

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader from", args.data_name)

    if args.data_name == 'SpaceNet':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = SpaceNet(args, args.data_dir, transform_train)
        args.in_ch = 4

    elif args.data_name == 'WHU':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = WHU(args, args.data_dir, transform_train)
        args.in_ch = 4

    elif args.data_name == 'Inria':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = Inria(args, args.data_dir, transform_train)
        args.in_ch = 4

    elif args.data_name == 'Mas':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = Mas(args, args.data_dir, transform_train)
        args.in_ch = 4

    elif args.data_name == 'SV2':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = SV2(args, args.data_dir, transform_train)
        args.in_ch = 4
        
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion..")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    total_params_model = count_parameters(model)
    print(f"Total number of model trainable parameters: {total_params_model}")


    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("Start training.")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_name = 'WHU',
        data_dir="../data/WHU/train/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None, 
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, 
        out_dir ='../WHU_checkpoint/',
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
