from evaluation.models.stgcn import STGCN
import torch
from torch.utils.data import DataLoader

class MotionClassification:
    def __init__(self, GeneratedMotions):
        self.MixamoCheckPointModelPath = "evaluation/checkpoint_0300_mixamo_acc_0.74_train_test_split_smaller_arch.tar"
        self.HumanAct12CheckPointModelPath = "evaluation/checkpoint_0300_mixamo_acc_0.74_train_test_split_smaller_arch.tar"
        self.Model = self.InitalizeModel(device='cuda',ModelPath=self.MixamoCheckPointModelPath)
        self.GeneratedMotions = GeneratedMotions
        self.GeneratePredictions()
    def InitalizeModel(self, device, ModelPath, dataset='mixamo'):
        num_classes = 0
        if dataset == 'mixamo':
            num_classes = 15
        elif dataset == 'humanact12':
            num_classes = 12
        model = STGCN(in_channels=3,
                      num_class=num_classes,
                      graph_args={"layout": 'openpose', "strategy": "spatial"},
                      edge_importance_weighting=True,
                      device=device)
        model = model.to(device)
        state_dict = torch.load(ModelPath, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

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

    def GeneratePredictions(self):
     self.GeneratedMotions = self.GeneratedMotions[:, :15]
     self.GeneratedMotions -= self.GeneratedMotions[:, 8:9, :, :]  # locate root joint of all frames at origin
     iterator_generated = DataLoader(self.GeneratedMotions, batch_size=64, shuffle=False, num_workers=6)

     generated_features, generated_predictions = self.compute_features(self.Model, iterator_generated)
     #generated_stats = calculate_activation_statistics(generated_features)
     print("h")