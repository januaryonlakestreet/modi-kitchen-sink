import torch
from evaluation.models.stgcn import STGCN
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


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
            generated_motion[i] = np.reshape(generated_motion[i], (4, 23, 64))
    return generated_motion, index

class Exp:
    def __init__(self, motions, mean_joints, std_joint, entity, edge_rot_dict_general):
        self.Model = self.LoadModel()
        self.motions = pd.Series(motions[0]['motion'])
        self.MeanJoints = mean_joints
        self.StdJoints = std_joint
        self.entity = entity
        self.EdgeRotDictGeneral = edge_rot_dict_general
        self.IteratorGenerated = DataLoader(ConvertGeneratedMotionToNumpy(self.motions)[0], batch_size=64, shuffle=False, num_workers=8)
        self.generated_features, self.generated_predictions = self.compute_features(self.Model, self.IteratorGenerated)
        print("g")

    def LoadModel(self):
        num_classes = 15
        device = 'cpu'
        modelpath = "D:\phdmethods\MoDi-main\evaluation\checkpoint_0300_mixamo_acc_0.74_train_test_split_smaller_arch.tar"

        model = STGCN(in_channels=3,
                      num_class=num_classes,
                      graph_args={"layout": 'openpose', "strategy": "spatial"},
                      edge_importance_weighting=True,
                      device=device)
        model = model.to(device)
        state_dict = torch.load(modelpath, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def GetModel(self):
        return self.Model


    def compute_features(self,model, iterator):
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