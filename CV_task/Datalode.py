from plyfile import PlyData
import torch as th
from torch.utils.data import Dataset
import numpy as np
import os
import six
import matplotlib.pylab as plt
snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}

def load_ply(file_name, with_faces=False, with_color=False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']])
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val

def snc_category_to_synth_id():
    d = snc_synth_id_to_category
    inv_map = {v: k for k, v in six.iteritems(d)}
    return inv_map


class PointCloudDataset(Dataset):
    def __init__(self, root):
        pcs = os.listdir(root)

        self.pcs = [os.path.join(root, pc) for pc in pcs]

    def __getitem__(self, index):
        pc_path = self.pcs[index]
        pil_pc = load_ply(pc_path)
        array = np.asarray(pil_pc)
        data = th.from_numpy(array)
        return data

    def __len__(self):
        return len(self.pcs)

class PointCloudDataset_train(Dataset):
    def __init__(self, root):
        pcs = os.listdir(root)
        pcs = [pcs[i] for i in range(0,3*len(pcs)//4)]
        self.pcs = [os.path.join(root, pc) for pc in pcs]

    def __getitem__(self, index):
        pc_path = self.pcs[index]
        pil_pc = load_ply(pc_path)
        array = np.asarray(pil_pc)
        data = th.from_numpy(array)
        return data

    def __len__(self):
        return len(self.pcs)

class PointCloudDataset_test(Dataset):
    def __init__(self, root):
        pcs = os.listdir(root)
        pcs = [pcs[i] for i in range(3 * len(pcs) // 4, len(pcs))]
        self.pcs = [os.path.join(root, pc) for pc in pcs]

    def __getitem__(self, index):
        pc_path = self.pcs[index]
        pil_pc = load_ply(pc_path)
        array = np.asarray(pil_pc)
        data = th.from_numpy(array)
        return data

    def __len__(self):
        return len(self.pcs)

    
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    batchsize, ndataset, dimension = xyz.shape

    centroids = th.zeros(batchsize, npoint, dtype=th.long).to(device)
    distance = th.ones(batchsize, ndataset).to(device) * 1e10

    farthest = th.randint(0, ndataset, (batchsize,), dtype=th.long).to(device)

    batch_indices = th.arange(batchsize, dtype=th.long).to(device)
    for i in range(npoint):
        centroids[:,i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, 3)
        dist = th.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = th.max(distance, -1)[1]
    return centroids

class PointCloudDataset_all(Dataset):
    def __init__(self, list_dir):
        self.pcs = list_dir

    def __getitem__(self, index):
        pc_path = self.pcs[index]
        pil_pc = load_ply(pc_path)
        array = np.asarray(pil_pc)
        data = th.from_numpy(array)
        return data

    def __len__(self):
        return len(self.pcs)

def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False, marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10, azim=240, axis=None, title=None, *args, **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig