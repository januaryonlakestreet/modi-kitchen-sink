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


class ModiViewer:
    def __init__(self, motions, mean_joints, std_joint, entity, edge_rot_dict_general):
        self.GeneratedMotions = pd.Series(motions[0]['motion'])
        self.MeanJoints = mean_joints
        self.StdJoints = std_joint
        self.Entity = entity
        self.EdgeRotDictGeneral = edge_rot_dict_general
        self.SaveSelectedMotionLocation = './clustering/static/animations/'
        self.ClusterFig, self.ClusterAxes = plt.subplots()
        self.ClusterAxes.set_title("Modi clustering")
        # self.ElbowMethod()
        # self.SilhouetteScore()
        self.GenerateClusters()
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

    def GenerateClusters(self):
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans

        MotionsNumpy = ConvertGeneratedMotionToNumpy(self.GeneratedMotions)[0]
        MotionsNumpy = np.reshape(MotionsNumpy, [len(MotionsNumpy), 4 * 23 * 64])

        K_ = KMeans(n_clusters=8)
        reduced_data_tsne = TSNE(n_components=3, perplexity=50).fit_transform(np.array(MotionsNumpy))
        Labels = K_.fit_predict(reduced_data_tsne)

        Colours = ['red', 'green', 'blue', 'olive', 'purple', 'cyan', 'magenta', 'Coral']

        for x in range(8):
            self.ClusterAxes.scatter(x=reduced_data_tsne[Labels == x, 0], y=reduced_data_tsne[Labels == x, 1], s=1.0,
                                     c=Colours[x], picker=True)
            cluster_center = K_.cluster_centers_[x]
            self.ClusterAxes.scatter(x=cluster_center[0], y=cluster_center[1], s=12.0, c="black")
        # axnext = self.ClusterFig.add_axes([0.65, 0.02, 0.3, 0.075])
        # BNewCluster = Button(axnext, 'Generate new cluster')
        self.ClusterFig.canvas.mpl_connect('pick_event', self.OnClick)
        self.ClusterFig.canvas.mpl_connect('close_event', self.onClose)

    def OnClick(self, event):

        ind = event.ind
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



