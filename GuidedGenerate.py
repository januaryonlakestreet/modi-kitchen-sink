import math
import os.path as osp
import os
import datetime
import random
import re
import sys

import pandas as pd
import torch
import numpy as np
from utils.visualization import motion2fig, motion2bvh
import matplotlib.pyplot as plt
import sys as _sys
from utils.data import motion_from_raw, to_cpu
from utils.pre_run import GenerateOptions, load_all_form_checkpoint
from utils.data import Joint, Edge  # to be used in 'eval'

CurrentSeeds = {}


def sample(args, g_ema, device, mean_latent, SeedsToUse):
    global CurrentSeeds
    # print('Sampling...')
    seeds = np.array([])
    n_motions = 10  # args.motions * 10
    if len(SeedsToUse) == 0:
        seed_rnd_mult = 18000000000
        n_motions = 5  # args.motions * 10
        while np.unique(seeds).shape[0] != n_motions:  # refrain from duplicates in seeds
            for x in range(n_motions):
                seeds = np.append(seeds, int(random.randint(-seed_rnd_mult, seed_rnd_mult)))
            # seeds = (np.random.random(n_motions) * seed_rnd_mult).astype(int)
            CurrentSeeds = seeds
    else:
        seeds = SeedsToUse
    generated_motion = pd.DataFrame(index=seeds, columns=['motion', 'W'], dtype=object)

    for i, seed in enumerate(seeds):
        # print(i)
        rnd_generator = torch.Generator(device=device).manual_seed(int(seed))
        sample_z = torch.randn(1, args.latent, device=device, generator=rnd_generator)
        motion, W, _ = g_ema(
            [sample_z], truncation=args.truncation, truncation_latent=mean_latent,
            return_sub_motions=args.return_sub_motions, return_latents=True)

        if (i + 1) % 1000 == 0:
            print(f'Done sampling {i + 1} motions.')

        # to_cpu is used becuase advanced python versions cannot as
        # sign a cuda object to a dataframe

        # print(motion.shape, "motion shape")
        # print(W.shape,"w shape")
        # print(seed,"seed")
        try:
            generated_motion.loc[seed, 'motion'] = to_cpu(motion)
            generated_motion.loc[seed, 'W'] = to_cpu(W)
        except ValueError as err:
            print("t")

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


def generate_seeds(motion, width):
    values = []
    generated = np.rint((np.random.normal(motion, width, size=100)))
    LargestValue = max(generated)
    for x in range(100):
        if values.count(generated[x]) != 0:
            generated[x] = generated[x] + LargestValue


        values.append(int(generated[x]))
    return np.unique(values)


def generate(args, g_ema, device, mean_joints, std_joints, entity):
    type2func = {'sample': sample}
    with torch.no_grad():
        g_ema.eval()
        mean_latent = g_ema.mean_latent(args.truncation_mean)
        generated_motions = type2func[args.type](args, g_ema, device, mean_latent, SeedsToUse=[])
        _, _, _, edge_rot_dict_general = motion_from_raw(args, np.load(args.path, allow_pickle=True))
        edge_rot_dict_general['std_tensor'] = edge_rot_dict_general['std_tensor'].cpu()
        edge_rot_dict_general['mean_tensor'] = edge_rot_dict_general['mean_tensor'].cpu()

        ClassToFind = 0
        MaxGuesses = 100
        CurrentBestScore = -math.inf
        CurrentBestSeed = 0
        StuckInRutCheck = 3
        CurrentRut = 0
        totalevaluated = 0
        global CurrentSeeds
        for y in range(876):

            for x in range(MaxGuesses):
                Value = EvaluateMotion(generated_motions, edge_rot_dict_general)
                Valuesnp = np.asarray(Value.detach().cpu())
                _CurrentScore = CurrentBestScore
                for _x in range(Valuesnp.shape[0]):
                    totalevaluated += 1
                    if Valuesnp[_x][11] > CurrentBestScore:
                        CurrentBestScore = Valuesnp[_x][11]
                        CurrentBestSeed = CurrentSeeds[_x]
                if CurrentBestScore == _CurrentScore:
                    CurrentRut += 1

                print("after", totalevaluated, "guess the best seed for class", y, "is ", CurrentBestSeed,
                      "with a score of ", CurrentBestScore)
                if CurrentRut >= StuckInRutCheck:

                    CurrentRut = 0
                    print("Was stuck in local area wiping seeds and starting again")
                    generated_motions = type2func[args.type](args, g_ema, device, mean_latent, SeedsToUse=[])
                else:
                    NewSeeds = generate_seeds(CurrentBestSeed, 100)
                    CurrentSeeds = np.asarray(NewSeeds, dtype=int)
                    generated_motions = type2func[args.type](args, g_ema, device, mean_latent, SeedsToUse=CurrentSeeds)

            print("after",totalevaluated," guesses current best seed for class ", y, "is ", CurrentSeeds)

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

    return root_out_path


def main(args_not_parsed):
    parser = GenerateOptions()
    args = parser.parse_args(args_not_parsed)
    device = args.device
    g_ema, discriminator, checkpoint, entity, mean_joints, std_joints = load_all_form_checkpoint(args.ckpt, args)
    out_path = generate(args, g_ema, device, mean_joints, std_joints, entity=entity)
    return out_path


# classifier functions

def classifierModel():
    from evaluation.models.stgcn import STGCN
    modelpath = "evaluation/2023-02-21model.pt"
    model = STGCN(in_channels=3,
                  num_class=876,
                  graph_args={"layout": 'openpose', "strategy": "spatial"},
                  edge_importance_weighting=True,
                  device='cuda')
    model = model.to('cuda')
    state_dict = torch.load(modelpath, map_location='cuda')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def ProcessGeneratedMotion(generated_motion_np, edge_rot_dict_general):
    from utils.data import anim_from_edge_rot_dict, un_normalize, edge_rot_dict_from_edge_motion_data, motion_from_raw
    from Motion import Animation

    edge_rot_dict_general['std_tensor'] = edge_rot_dict_general['std_tensor'].cpu()
    edge_rot_dict_general['mean_tensor'] = edge_rot_dict_general['mean_tensor'].cpu()
    # edge_rot_dict_general['offsets_no_root'] /= 100  ## not needed in humanact

    generated_motions = []

    # get anim for xyz positions
    motion_data = un_normalize(generated_motion_np, mean=edge_rot_dict_general['mean'].transpose(0, 2, 1, 3),
                               std=edge_rot_dict_general['std'].transpose(0, 2, 1, 3))
    anim_dicts, frame_mults, is_sub_motion = edge_rot_dict_from_edge_motion_data(motion_data, type='sample',
                                                                                 edge_rot_dict_general=edge_rot_dict_general)

    for j, (anim_dict, frame_mult) in enumerate(zip(anim_dicts, frame_mults)):
        anim, names = anim_from_edge_rot_dict(anim_dict, root_name='Hips')
        # compute global positions using anim
        positions = Animation.positions_global(anim)

        # sample joints relevant to 15 joints skeleton
        positions_15joints = positions[:,
                             [7, 6, 15, 16, 17, 10, 11, 12, 0, 23, 24, 25, 19, 20, 21]]  # openpose order R then L
        positions_15joints = positions_15joints.transpose(1, 2, 0)
        positions_15joints_oriented = positions_15joints.copy()
        positions_15joints_oriented = positions_15joints_oriented[:, [0, 2, 1]]
        positions_15joints_oriented[:, 1, :] = -1 * positions_15joints_oriented[:, 1, :]

        generated_motions.append(positions_15joints_oriented)

    generated_motions = np.asarray(generated_motions)
    return generated_motions


def EvaluateMotion(motion, EdgeRotDictGeneral):
    from torch.utils.data import DataLoader
    def calculate_activation_statistics(activations):
        activations = activations.cpu().detach().numpy()
        # activations = activations.cpu().numpy()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def compute_features(model, iterator):
        device = 'cuda'
        activations = []
        predictions = []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                # for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
                batch_for_model = {}
                batch_for_model['x'] = batch.to(device).float()
                model(batch_for_model)
                activations.append(batch_for_model['features'])
                predictions.append(batch_for_model['yhat'])
                # labels.append(batch_for_model['y'])
            activations = torch.cat(activations, dim=0)
            predictions = torch.cat(predictions, dim=0)
            # labels = torch.cat(labels, dim=0)
            # shape torch.Size([16, 15, 3, 64]) (batch, joints, xyz, frames)
        return activations, predictions

    def ConvertGeneratedMotionToNumpy(generated_motion):
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

        return generated_motion, index

    MotionsNumpy = ConvertGeneratedMotionToNumpy(pd.Series(motion['motion']))[0]
    MotionLocation = []
    EdgeRotDictCopy = EdgeRotDictGeneral
    MotionLocation.append(ProcessGeneratedMotion(list(MotionsNumpy), EdgeRotDictCopy))
    iterator_generated = DataLoader(MotionLocation[0], batch_size=64, shuffle=False, num_workers=6)
    generated_features, generated_predictions = compute_features(classifierModel(), iterator_generated)
    generated_stats = calculate_activation_statistics(generated_features)
    results = generated_predictions.max(dim=1).indices
    return generated_predictions


if __name__ == "__main__":
    main(_sys.argv[1:])
