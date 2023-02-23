import os
import os.path as osp
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from utils.visualization import motion2bvh
from flask import Flask, render_template
import threading
import webbrowser
from utils.data import anim_from_edge_rot_dict, un_normalize, edge_rot_dict_from_edge_motion_data, motion_from_raw
from Motion import Animation
from evaluation.models.stgcn import STGCN
from torch.utils.data import DataLoader
from matplotlib.widgets import Button


def ShowViewer():
    plt.show()


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


def convert_motions_to_location(generated_motion_np, edge_rot_dict_general):
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


def initialize_model(device, modelpath, dataset='mixamo'):
    if dataset == 'mixamo':
        num_classes = 876
    elif dataset == 'humanact12':
        num_classes = 12
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


class ModiViewer:
    def __init__(self, motions, mean_joints, std_joint, entity, edge_rot_dict_general, Predictions=None):
        self.GeneratedMotions = pd.Series(motions[0]['motion'])
        self.MeanJoints = mean_joints
        self.Predictions = Predictions
        self.StdJoints = std_joint
        self.Entity = entity
        self.EdgeRotDictGeneral = edge_rot_dict_general
        self.SaveSelectedMotionLocation = './clustering/static/animations/'
        self.ClusterFig, self.ClusterAxes = plt.subplots()
        self.ClusterAxes.set_title("Modi clustering")
        self.ClusterFig.canvas.mpl_connect('pick_event', self.OnClick)
        self.ClusterFig.canvas.mpl_connect('close_event', self.onClose)
        # self.ElbowMethod()
        # self.SilhouetteScore()
        self.GenerateClusters()
       # self.GenerateClusters3D()
        self.port = 23336
        self.NewBrowser = True
        self.host_name = "0.0.0.0"
        self.app = Flask(__name__)
        self.app.add_url_rule('/', 'index', self.ViewAnimation)
        threading.Thread(
            target=lambda: self.app.run(host=self.host_name, port=self.port, debug=True, use_reloader=False)).start()
        ShowViewer()

    def SilhouetteScore(self):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        silhouette_avg = []

        MotionsNumpy = ConvertGeneratedMotionToNumpy(self.GeneratedMotions)[0]
        MotionsNumpy = np.reshape(MotionsNumpy, [len(MotionsNumpy), 4 * 23 * 64])

        K = range(2, 10)

        for k in K:
            # initialise kmeans
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(MotionsNumpy)
            cluster_labels = kmeans.labels_
            silhouette_avg.append(silhouette_score(MotionsNumpy, cluster_labels))
        plt.plot(K, silhouette_avg, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('silhouette score')
        plt.title('silhouette analysis for optimal K')
        plt.show()

    def ElbowMethod(self):
        from sklearn.cluster import KMeans
        from sklearn import metrics
        from scipy.spatial.distance import cdist

        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        MotionsNumpy = ConvertGeneratedMotionToNumpy(self.GeneratedMotions)[0]
        MotionsNumpy = np.reshape(MotionsNumpy, [len(MotionsNumpy), 4 * 23 * 64])
        K = range(1, 1000)

        for k in K:
            print(str(k))
            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k, n_init='auto').fit(MotionsNumpy)
            kmeanModel.fit(MotionsNumpy)

            distortions.append(sum(np.min(cdist(MotionsNumpy, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / MotionsNumpy.shape[0])
            inertias.append(kmeanModel.inertia_)

            mapping1[k] = sum(np.min(cdist(MotionsNumpy, kmeanModel.cluster_centers_,
                                           'euclidean'), axis=1)) / MotionsNumpy.shape[0]
            mapping2[k] = kmeanModel.inertia_

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()

    def PredictLabels(self, MotionsNumpy):
        MotionLocation = []
        EdgeRotDictCopy = self.EdgeRotDictGeneral
        MotionLocation.append(convert_motions_to_location(list(MotionsNumpy), EdgeRotDictCopy))
        iterator_generated = DataLoader(MotionLocation[0], batch_size=64, shuffle=False, num_workers=6)
        modelpath = "evaluation/checkpoint_0300_mixamo_acc_0.74_train_test_split_smaller_arch.tar"

        # initialize model
        model = initialize_model('cuda', modelpath, 'mixamo')
        # compute features of generated motions
        generated_features, generated_predictions = compute_features(model, iterator_generated)
        generated_stats = calculate_activation_statistics(generated_features)
        results = generated_predictions.max(dim=1).indices
        return results, generated_stats

    def GenerateClusters3D(self):
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        self.ClusterAxes.remove()
        dax = self.ClusterFig.add_subplot(projection='3d')
        dax.grid(False)
        dax.axis('off')
        MotionsNumpy = ConvertGeneratedMotionToNumpy(self.GeneratedMotions)[0]
        ClassPredictions, _ = self.PredictLabels(MotionsNumpy)
        UniquePredictions = np.unique(np.asarray(ClassPredictions.cpu()))
        ClassPredictionsAsArray = np.asarray(ClassPredictions.cpu())
        results = list(zip(MotionsNumpy,ClassPredictionsAsArray))
        MotionsNumpy.clear()
        for x in range(len(results)):
            if results[x][1] == 11: #or results[x][1] == 12
                MotionsNumpy.append(results[x][0])
        MotionsNumpy = np.reshape(MotionsNumpy, [len(MotionsNumpy), 4 * 23 * 64])

        K_ = KMeans(n_clusters=8)
        reduced_data_tsne = TSNE(n_components=3, perplexity=50).fit_transform(np.array(MotionsNumpy))
        Labels = K_.fit_predict(reduced_data_tsne)
        Colours = ['red', 'green', 'blue', 'olive', 'purple', 'cyan', 'magenta', 'Coral','firebrick','sienna','teal','plum','crimson']

        for x in range(8):
            dax.scatter(reduced_data_tsne[Labels == x, 0], reduced_data_tsne[Labels == x, 1],reduced_data_tsne[Labels == x, 2],
                                     c=Colours[x], picker=True)
            cluster_center = K_.cluster_centers_[x]
            dax.scatter(cluster_center[0], cluster_center[1],cluster_center[2], c="black")



    def GenerateClusters(self):
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans

        MotionsNumpy = ConvertGeneratedMotionToNumpy(self.GeneratedMotions)[0]

        MotionLocation = []
        EdgeRotDictCopy = self.EdgeRotDictGeneral
        MotionLocation.append(convert_motions_to_location(list(MotionsNumpy), EdgeRotDictCopy))
        iterator_generated = DataLoader(MotionLocation[0], batch_size=64, shuffle=False, num_workers=6)
       # modelpath = "evaluation/checkpoint_0300_mixamo_acc_0.74_train_test_split_smaller_arch.tar"
        modelpath = "evaluation/2023-02-21model.pt"
        # initialize model
        model = initialize_model('cuda', modelpath, 'mixamo')
        # compute features of generated motions
        generated_features, generated_predictions = compute_features(model, iterator_generated)
        generated_stats = calculate_activation_statistics(generated_features)
        results = generated_predictions.max(dim=1).indices

        MotionsNumpy = np.reshape(MotionsNumpy, [len(MotionsNumpy), 4 * 23 * 64])

        K_ = KMeans(n_clusters=8)
        reduced_data_tsne = TSNE(n_components=3, perplexity=50).fit_transform(np.array(MotionsNumpy))
        Labels = K_.fit_predict(reduced_data_tsne)

        AllData = list(zip(np.asarray(results.cpu()), Labels, reduced_data_tsne))
        Colours = ['red', 'green', 'blue', 'olive', 'purple', 'cyan', 'magenta', 'Coral']
        for x in range(8):
            self.ClusterAxes.scatter(x=reduced_data_tsne[Labels == x, 0], y=reduced_data_tsne[Labels == x, 1], s=1.0,
                                     c=Colours[x], picker=True)
            cluster_center = K_.cluster_centers_[x]
            self.ClusterAxes.scatter(x=cluster_center[0], y=cluster_center[1], s=12.0, c="black")
        # axnext = self.ClusterFig.add_axes([0.65, 0.02, 0.3, 0.075])
        # BNewCluster = Button(axnext, 'Generate new cluster')
        # print(str(euclidean_distances(K_.cluster_centers_)))
        self.ClusterFig.canvas.mpl_connect('pick_event', self.OnClick)
        self.ClusterFig.canvas.mpl_connect('close_event', self.onClose)

    def GenerateClustersNew(self):
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans

        MotionsNumpy = ConvertGeneratedMotionToNumpy(self.GeneratedMotions)[0]

        MotionLocation = []
        EdgeRotDictCopy = self.EdgeRotDictGeneral
        MotionLocation.append(convert_motions_to_location(list(MotionsNumpy), EdgeRotDictCopy))
        iterator_generated = DataLoader(MotionLocation[0], batch_size=64, shuffle=False, num_workers=8)
        modelpath = "evaluation/checkpoint_0300_mixamo_acc_0.74_train_test_split_smaller_arch.tar"

        # initialize model
        model = initialize_model('cuda', modelpath, 'mixamo')
        # compute features of generated motions
        generated_features, generated_predictions = compute_features(model, iterator_generated)
        generated_stats = calculate_activation_statistics(generated_features)
        results = generated_predictions.max(dim=1).indices

        MotionsNumpy = np.reshape(MotionsNumpy, [len(MotionsNumpy), 4 * 23 * 64])

        K_ = KMeans(n_clusters=8)
        reduced_data_tsne = TSNE(n_components=3, perplexity=50).fit_transform(np.array(MotionsNumpy))
        Labels = K_.fit_predict(reduced_data_tsne)

        AllData = list(zip(np.asarray(results.cpu()), Labels, reduced_data_tsne))
        Colours = ['red', 'green', 'blue', 'olive', 'purple', 'cyan', 'magenta', 'Coral']
        for x in range(len(AllData)):
            self.ClusterAxes.scatter(x=AllData[x][2][0], y=AllData[x][2][1], s=1.0,
                                     c=Colours[AllData[x][1]], picker=1)
        for CenterID in range(len(K_.cluster_centers_)):
            self.ClusterAxes.scatter(x=K_.cluster_centers_[CenterID][0], y=K_.cluster_centers_[CenterID][1], s=12.0,
                                     c="black")

        for x in range(14):
            axnext = self.ClusterFig.add_axes([0.05 * x, 0.02, 0.03, 0.015])
            BNewCluster = Button(axnext, str(x))

    def OnClick(self, event):

        ind = event.ind
        print(ind)
        motion_np, _ = ConvertGeneratedMotionToNumpy(self.GeneratedMotions)
        values = pd.DataFrame(list(zip(motion_np, _))).iloc[ind[0]]

        motion2bvh(np.array(values[0]), osp.join(self.SaveSelectedMotionLocation, f'{values[1]}.bvh'),
                   parents=self.Entity.parents_list, type='sample', entity=self.Entity.str(),
                   edge_rot_dict_general=self.EdgeRotDictGeneral)

        webbrowser.open("http://127.0.0.1:23336/?seed=" + str(values[1]), new=self.NewBrowser)
        self.NewBrowser = False

    def onClose(self, event):
        for filename in os.listdir(self.SaveSelectedMotionLocation):
            file_path = os.path.join(self.SaveSelectedMotionLocation, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def ViewAnimation(self):
        return render_template('ViewAnimation.html')
