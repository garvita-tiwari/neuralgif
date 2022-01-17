"""source: https://github.com/jchibane/ndf"""
from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import ipdb


class SMPLdata(Dataset):


    def __init__(self, mode, data_path, split_file, batch_size, num_workers=12, num_sample_points = 1024,  sample_distribution = [0.5, 0.5],
                 sample_sigmas = [0.01, 0.1], num_parts=24, shape=False,  d_class='smpl' ,**kwargs):


        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)
        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file)[mode]
        self.data = ['/{}'.format(self.split[i]) for i in range(len(self.split)) if
                     os.path.exists(os.path.join(self.path, '{}.npz'.format(self.split[i])))]
        self.n_part = num_parts
        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)
        self.global_scale = 1.0
        self.mode = mode
        self.d_class = d_class
        self.skinning =np.load('./data_files/skinning_body.npy')

        self.tpose_verts = np.load('./data_files/000_tpose.npy')
        self.shape = shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):


        path = self.path  + self.data[idx] +'.npz'
        #load skinning for boundary points
        nasa_data = np.load(path)
        skin_data = nasa_data['gt_skin']
        #trans_root =  nasa_data['trans_root']
        transform =  nasa_data['transform']
        sample_num = int(len(skin_data)/3)  #on the surface points
        all_points = nasa_data['posed_points'][sample_num:]
        all_points_can = nasa_data['can_points'][sample_num:]
        df_Val = nasa_data['sdf'][sample_num:]
        all_labels = df_Val
        all_skin_bp = skin_data[sample_num:]


        pose =  nasa_data['pose']
        #transform_inv = np.array([np.linalg.inv(transform[i]) for i in range(24)])
        scale_data = nasa_data['bounds']
        bottom_cotner = scale_data[0]
        upper_corner = scale_data[1]
        min = np.min(bottom_cotner)
        max = np.max(upper_corner)
        name = self.data[idx]

        n_bbox = int(len(all_labels)/2)
        n_surf = int(len(all_labels)/2)

        #surface points
        can_sample_points0 = all_points_can[:n_bbox] *self.global_scale
        boundary_sample_points0 = all_points[:n_bbox] *self.global_scale
        boundary_sample_occupancies0 = all_labels[:n_bbox]
        skin_0 = all_skin_bp[:n_bbox]
        #110% box points
        can_sample_points1 = all_points_can[n_bbox: 2*n_bbox] *self.global_scale
        boundary_sample_points1 = all_points[n_bbox: 2*n_bbox] *self.global_scale
        boundary_sample_occupancies1 = all_labels[n_bbox: 2*n_bbox]
        skin_1 = all_skin_bp[n_bbox: 2*n_bbox]

        points = []
        occupancies = []
        skin_bp = []
        can_pts = []
        num = int(self.num_sample_points/2)
        subsample_indices = np.random.randint(0, n_bbox, num)
        points.extend(boundary_sample_points0[subsample_indices])
        can_pts.extend(can_sample_points0[subsample_indices])
        occupancies.extend(boundary_sample_occupancies0[subsample_indices])
        skin_bp.extend(skin_0[subsample_indices])

        subsample_indices = np.random.randint(0, n_bbox, num)
        points.extend(boundary_sample_points1[subsample_indices])
        occupancies.extend(boundary_sample_occupancies1[subsample_indices])
        skin_bp.extend(skin_1[subsample_indices])
        can_pts.extend(can_sample_points1[subsample_indices])


        verts = nasa_data['smpl_verts']
        subsample_indices = np.random.randint(0, len(verts), self.num_sample_points)


        smpl_verts = verts[subsample_indices]
        can_smpl = self.tpose_verts[subsample_indices]
        gt_skin = self.skinning[subsample_indices]
        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points


        joints = nasa_data['joint']

        if self.shape:
            beta_id = self.data[idx].split('/')[1]
            beta = np.load('/BS/cloth-anim/static00/tailor_data/shirt_male/shape/beta_{}.npy'.format(beta_id))[:10]
        else:
            beta = None

        return {'path': self.data[idx],
                'gt_skin': np.array(gt_skin, dtype=np.float32),
                'skin_bp': np.array(skin_bp, dtype=np.float32),
                'all_skin': np.array(self.skinning, dtype=np.float32),
                'transform': np.array(transform, dtype=np.float32),
                'all_verts': np.array(verts, dtype=np.float32),
                'smpl_verts': np.array(smpl_verts, dtype=np.float32),
                'points': np.array(points, dtype=np.float32),
                'label': np.array(occupancies, dtype=np.float32),
                'pose': np.array(pose, dtype=np.float32),
                'joints': np.array(joints, dtype=np.float32),
                'can_smpl': np.array(can_smpl, dtype=np.float32),
                'can_pts_gt': np.array(can_pts, dtype=np.float32),
                'min': min, 'max': max,
                'beta': np.array(beta, dtype=np.float32)}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
