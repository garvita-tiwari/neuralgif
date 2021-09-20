"""source: https://github.com/jchibane/ndf
GT data for evaluation"""
import trimesh
import numpy as np
import argparse

import torch
from kaolin.metrics.point import SidedDistance
from psbody.mesh import Mesh, MeshViewer

import os
import pickle as pkl
import igl
import sys
sys.path.append('/BS/garvita/work/code/cloth_static/TailorNet')
import ipdb

from models.torch_smpl4garment import TorchSMPL4Garment
data_dir = '/BS/RVH_3dscan_raw2/static00/neuralGIF_data/smpl_test'

sys.path.append('/BS/garvita/work/code/if-net/data_processing')
import implicit_waterproofing as iw
import mcubes

if __name__ == "__main__":
    all_beta = []
    for i in range(9):
        beta = np.load('/BS/cloth-anim/static00/tailor_data/shirt_male/shape/beta_{:03}.npy'.format(i))
        all_beta.append(beta[:10])
    beta1 = np.zeros_like(all_beta[0])
    beta1[0] = 2.0
    beta1[1] = -2.0
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-frame', type=int)

    args = parser.parse_args()

    cmu_root = '/BS/RVH/work/data/people_completion/poses/CMU'
    all_cmu = sorted(os.listdir(cmu_root))[812:]  #[100:]
    pose_file = '/BS/RVH/work/data/people_completion/poses/SMPL/female.pkl'
    poses = pkl.load(open(pose_file, 'rb'), encoding="latin1")
    smpl_torch = TorchSMPL4Garment('female')
    sample_num = 100000

    sub_folder  =os.path.join(data_dir, '2-2')
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    print(len(poses))
    for j in range(len(poses)):
        if j%10 != 0:
            continue
        frame_num = '{:06}'.format(j)


        #creat smpl
        frame_pose = np.array(poses[j])
        #frame_pose[:3] = 0.0
        pose_torch = torch.from_numpy(frame_pose.astype(np.float32)).unsqueeze(0)
        betas_torch = torch.from_numpy(beta1.astype(np.float32)).unsqueeze(0)

        smpl_verts = smpl_torch.forward(pose_torch, betas_torch)
        transform = smpl_torch.A.detach().numpy()[0]

        transform_inv = np.array([np.linalg.inv(transform[i]) for i in range(24)])
        joints  = smpl_torch.J_transformed.detach().numpy()[0]

        #check the smpl mesh and joint and unposed mesh here
        m1 = Mesh(v=smpl_verts.detach().numpy()[0], f=smpl_torch.faces)
        mesh = trimesh.Trimesh(m1.v, smpl_torch.faces)

        bottom_corner, upper_corner = mesh.bounds


        minimun = np.min(bottom_corner)
        maximum = np.max(upper_corner)
        grid_points = iw.create_grid_points_from_bounds(minimun, maximum, 256)
        logits = igl.signed_distance(grid_points, mesh.vertices, mesh.faces)[0]  # -ve -> inside
        logits = np.reshape(logits, (256,) * 3)

        #logits = 1.0 -logits
        logits = (-1)*logits
        vertices, triangles = mcubes.marching_cubes(logits, 0.0)
        step = (maximum - minimun) / (256 )
        #step = (max - min) / (self.resolution*self.upsampling_steps - 1)
        vertices = np.multiply(vertices, step)
        vertices += [minimun, minimun,minimun]

        m1 = trimesh.Trimesh(vertices, triangles)

        m1.export(os.path.join(sub_folder,  frame_num + '.obj'))



        print('Finished  {} '.format( frame_num))


