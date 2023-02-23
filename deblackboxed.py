import os.path as osp
import os
import datetime
import random
import re
import pandas as pd
import torch
import numpy as np
from utils.visualization import motion2fig, motion2bvh
import matplotlib.pyplot as plt
import sys as _sys
from utils.data import motion_from_raw, to_cpu
from utils.pre_run import GenerateOptions, load_all_form_checkpoint
from utils.data import Joint, Edge  # to be used in 'eval'


def sample(args, g_ema, device, mean_latent):
    print('Sampling...')

    seed_rnd_mult = args.motions * 10000
    seeds = np.array([])
    n_motions = 1#args.motions * 10
    while np.unique(seeds).shape[0] != n_motions:  # refrain from duplicates in seeds
        seeds = (np.random.random(n_motions) * seed_rnd_mult).astype(int)
    generated_motion = pd.DataFrame(index=seeds, columns=['motion', 'W'], dtype=object)

    for i, seed in enumerate(seeds):
        print(i)
        rnd_generator = torch.Generator(device=device).manual_seed(int(seed))
        sample_z = torch.randn(1, args.latent, device=device, generator=rnd_generator)
        motion, W, _ = g_ema(
            [sample_z], truncation=args.truncation, truncation_latent=mean_latent,
            return_sub_motions=args.return_sub_motions, return_latents=True)

        if (i + 1) % 1000 == 0:
            print(f'Done sampling {i + 1} motions.')

        # to_cpu is used becuase advanced python versions cannot as
        # sign a cuda object to a dataframe
        generated_motion.loc[seed, 'motion'] = to_cpu(motion)
        generated_motion.loc[seed, 'W'] = to_cpu(W)

        filter = np.ones(generated_motion.shape[0], dtype=bool)
    generated_motion = generated_motion[filter]
    return generated_motion


def get_motion_std(args, motion):
    if args.entity == 'Edge' and args.glob_pos:
        assert args.foot
        std = motion[:, :3, -3, :].norm(p=2, dim=1).std()
    else:
        raise 'this case is not supported yet'
    return std


def get_gen_mot_np(args, generated_motion, mean_joints, std_joints):
    # part 1: align data type
    if isinstance(generated_motion, pd.Series):
        index = generated_motion.index
        if not isinstance(generated_motion.iloc[0], list) and \
                generated_motion.iloc[0].ndim == 4 and generated_motion.iloc[0].shape[0] > 1:
            generated_motion = generated_motion.apply(
                lambda motions: torch.unsqueeze(motions, 1))  # add a batch dimension
            generated_motion = generated_motion.apply(list)  # part2 expects lists
        generated_motion = generated_motion.tolist()
    else:
        assert isinstance(generated_motion, list)
        index = range(len(generated_motion))

    # part 2: torch to np
    for i in np.arange(len(generated_motion)):
        if not isinstance(generated_motion[i], list):
            generated_motion[i] = generated_motion[i].transpose(1, 2).detach().cpu().numpy()
            assert generated_motion[i].shape[:3] == std_joints.shape[:3] or args.return_sub_motions
        else:
            generated_motion[i], _ = get_gen_mot_np(args, generated_motion[i], mean_joints, std_joints)

    return generated_motion, index


def cluster(motions, mean_joints, std_joints, entity, edge_rot_dict_general):
    from Clustering.PyPltModiViewer import ModiViewer
    from Clustering.MotionClassificationModel import MotionClassification

    Viewer = ModiViewer(motions, mean_joints, std_joints, entity=entity, edge_rot_dict_general=edge_rot_dict_general)
   # MotionClassification(motions[0]['motion'])




def generate(args, g_ema, device, mean_joints, std_joints, entity):
    type2func = {'sample': sample}
    with torch.no_grad():
        g_ema.eval()
        mean_latent = g_ema.mean_latent(args.truncation_mean)
        generated_motions = type2func[args.type](args, g_ema, device, mean_latent)

    if entity.str() == 'Joint':
        edge_rot_dict_general = None
    else:
        _, _, _, edge_rot_dict_general = motion_from_raw(args, np.load(args.path, allow_pickle=True))
        edge_rot_dict_general['std_tensor'] = edge_rot_dict_general['std_tensor'].cpu()
        edge_rot_dict_general['mean_tensor'] = edge_rot_dict_general['mean_tensor'].cpu()

    if args.out_path is not None:
        out_path = args.out_path
        os.makedirs(out_path, exist_ok=True)
    else:
        time_str = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
        out_path = osp.join(osp.splitext(args.ckpt)[0] + '_files', f'{time_str}_{args.type}')
        if args.type == 'sample':
            out_path = f'{out_path}_{args.motions}'
        os.makedirs(out_path, exist_ok=True)
    root_out_path = out_path
    if not isinstance(generated_motions, tuple):
        generated_motions = (generated_motions,)
    for generated_motion in generated_motions:
        out_path = root_out_path
        if isinstance(generated_motion, tuple):
            out_path = osp.join(out_path, generated_motion[0])
            os.makedirs(out_path, exist_ok=True)
            generated_motion = generated_motion[1]

        if not isinstance(generated_motion, pd.DataFrame):
            generated_motion = pd.DataFrame(columns=['motion'], data=to_cpu(generated_motion))

        cluster(generated_motions, mean_joints, std_joints, entity, edge_rot_dict_general)

    return root_out_path


def main(args_not_parsed):
    parser = GenerateOptions()
    args = parser.parse_args(args_not_parsed)
    device = args.device
    g_ema, discriminator, checkpoint, entity, mean_joints, std_joints = load_all_form_checkpoint(args.ckpt, args)
    out_path = generate(args, g_ema, device, mean_joints, std_joints, entity=entity)
    return out_path


if __name__ == "__main__":
    main(_sys.argv[1:])
