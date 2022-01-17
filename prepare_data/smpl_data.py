"""source: https://github.com/jchibane/ndf"""
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
from utils.rotation import normalize_y_rotation
data_dir = '/BS/RVH_3dscan_raw2/static00/neuralGIF_data/smpl_norm'
mesh_dir_dataa = '/BS/cloth3d/static00/neuralGIF_data/smpl_norm'


if __name__ == "__main__":

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
    beta = np.load('/BS/cloth-anim/static00/tailor_data/shirt_male/shape/beta_{:03}.npy'.format(args.frame))
    cmu_sub = '{:03}'.format(args.frame)
    sub_folder  =os.path.join(data_dir, cmu_sub)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    mesh_dir  =os.path.join(mesh_dir_dataa, cmu_sub +'_mesh')
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    print(len(poses))
    for j in range(len(poses)):
        frame_num = '{:06}'.format(j)

        out_file = os.path.join(sub_folder, '{}.npz'.format(frame_num))
        frame_pose = np.array(poses[j])
        frame_pose = normalize_y_rotation(frame_pose)
        print(frame_pose[:3])


        #creat smpl

        #frame_pose[:3] = 0.0
        pose_torch = torch.from_numpy(frame_pose.astype(np.float32)).unsqueeze(0)
        betas_torch = torch.from_numpy(beta[:10].astype(np.float32)).unsqueeze(0)

        smpl_verts = smpl_torch.forward(pose_torch, betas_torch)
        transform = smpl_torch.A.detach().numpy()[0]

        transform_inv = np.array([np.linalg.inv(transform[i]) for i in range(24)])
        joints  = smpl_torch.J_transformed.detach().numpy()[0]

        #check the smpl mesh and joint and unposed mesh here
        m1 = Mesh(v=smpl_verts.detach().numpy()[0], f=smpl_torch.faces)
        #save mesh for rendering
        m1.write_obj(os.path.join(mesh_dir,'{}.obj'.format(frame_num) ))
        if os.path.exists(out_file):
            print('already done:  ', out_file)
            continue
        skinning_pt = smpl_torch.weight.detach().numpy()[0]
        #ipdb.set_trace()
        #sample points near the smpl posed surface
        mesh =  trimesh.Trimesh(m1.v, smpl_torch.faces)

        boundary_points = []
        points = mesh.sample(sample_num)
        boundary_points_1 = points + 0.01 * np.random.randn(sample_num, 3)

        points = mesh.sample(sample_num)
        boundary_points_2 = points + 0.1 * np.random.randn(sample_num, 3)
        boundary_points.extend(points)
        boundary_points.extend(boundary_points_1)
        boundary_points.extend(boundary_points_2)
        boundary_points = np.array(boundary_points)
        #
        points_torch = torch.from_numpy(boundary_points.astype(np.float32)).unsqueeze(0).cuda()
        tn_verts_torch = torch.from_numpy(m1.v.astype(np.float32)).unsqueeze(0).cuda()
        sideddistance = SidedDistance()
        dist1 = sideddistance(points_torch, tn_verts_torch).cpu().detach().numpy()[0]
        skinning_pt = skinning_pt[dist1]
        gt_skin = skinning_pt
        skinning_pt = torch.from_numpy(skinning_pt.astype(np.float32)).unsqueeze(0).cuda()
        transform_np = transform
        transform = torch.from_numpy(transform.astype(np.float32)).unsqueeze(0).cuda()
        W = skinning_pt
        T = torch.matmul(W, transform.view(1, 24, 16)).view(1, -1, 4, 4)

        Tinv = torch.inverse(T)
        verts_homo = torch.cat([ points_torch, torch.ones(1, points_torch.shape[1], 1).cuda()], dim=2)
        v_def = torch.matmul(Tinv, verts_homo.unsqueeze(-1))[:, :, :3, 0]
        transformed_pt = v_def.cpu().detach().numpy()[0]
        #save the first sample_num pts
        # m1 = Mesh(v=transformed_pt[:sample_num])
        # m1.write_obj('/BS/RVH_3dscan_raw2/static00/neuralGIF_data/smpl/000_debug/{}.obj'.format(frame_num))
        df = igl.signed_distance(boundary_points, mesh.vertices, mesh.faces)[0]
        bottom_cotner, upper_corner = mesh.bounds

        np.savez(out_file,gt_skin=gt_skin,  bounds=np.array([bottom_cotner, upper_corner]), posed_points=boundary_points,
                 sdf=df, can_points=transformed_pt, joint=joints, transform=transform_np, pose=frame_pose,
                 org_pose=np.array(poses[j]), smpl_verts=m1.v)

        print('Finished {} {} '.format(cmu_sub, frame_num))


