"""source: https://github.com/jchibane/ndf"""
import trimesh
import numpy as np

import torch.nn as nn
import torch

from kaolin.metrics.point import SidedDistance

from psbody.mesh import Mesh, MeshViewer

import os
import pickle as pkl
import sys
sys.path.append('/BS/garvita/work/code/cloth_static/TailorNet')
sys.path.append('/BS/garvita/work/code/if-net/data_processing')
import ipdb
import igl
import mcubes
from models.torch_smpl4garment import TorchSMPL4Garment
from models.smpl4garment import SMPL4Garment
import cv2


data_dir = '/BS/RVH_3dscan_raw2/static00/neuralGIF_data/clothseq/102'
registration_directory = '/BS/garvita3/static00/FRL_dataset_1/processed/WIDC102/registration'




WIDC102_POSE_CORRECT = [1,8, 14, 16, 18, 19, 20, 21] + \
                       [*range(24, 28)] + \
                       [29, 32, 40, 42, 43, 47, 48, 51, 55, 56, 58, 60, 62, 64, 65, 67 ] +\
                       [71, 73, 76, 77, 80, 81] + [*range(83, 93)] +  [*range(94,162)] +  [*range(163, 190)] + \
                       [*range(191, 209)] + [211] + [*range(213, 224)] + [*range(225, 234)] + \
                       [*range(235, 347)]  + [*range(557, 778)]

def normalize_y_rotation(raw_theta):
    """Rotate along y axis so that root rotation can always face the camera.
    Theta should be a [3] or [72] numpy array.
    """
    only_global = True
    if raw_theta.shape == (72,):
        theta = raw_theta[:3]
        only_global = False
    else:
        theta = raw_theta[:]
    raw_rot = cv2.Rodrigues(theta)[0]
    rot_z = raw_rot[:, 2]
    # we should rotate along y axis counter-clockwise for t rads to make the object face the camera
    if rot_z[2] == 0:
        t = (rot_z[0] / np.abs(rot_z[0])) * np.pi / 2
    elif rot_z[2] > 0:
        t = np.arctan(rot_z[0]/rot_z[2])
    else:
        t = np.arctan(rot_z[0]/rot_z[2]) + np.pi
    cost, sint = np.cos(t), np.sin(t)
    norm_rot = np.array([[cost, 0, -sint],[0, 1, 0],[sint, 0, cost]])
    final_rot = np.matmul(norm_rot, raw_rot)
    final_theta = cv2.Rodrigues(final_rot)[0][:, 0]
    if not only_global:
        return np.concatenate([final_theta, raw_theta[3:]], 0), norm_rot
    else:
        return final_theta, norm_rot


if __name__ == "__main__":
    smpl = SMPL4Garment('male')

    smpl_torch = TorchSMPL4Garment('male')
    sample_num = 1000000
    mesh_dir = os.path.join(data_dir +'_mesh')
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    #template for face
    for idx in WIDC102_POSE_CORRECT:
        frame_name = 'Jacket.{:06}'.format(idx)
        out_file = os.path.join(data_dir, frame_name +'.npz')

        if os.path.exists( out_file):
            print('already done: ', frame_name)
            continue
        # read the original mesh
        out_mesh = os.path.join(mesh_dir, frame_name +'.obj')
        org_mesh = os.path.join(registration_directory,
                                frame_name, '{}_org.obj'.format(frame_name))
        reg_file = os.path.join(registration_directory,
                                frame_name, 'singlemesh_female/{}_params.pkl'.format(frame_name))


        reg_mesh = os.path.join(registration_directory,
                                frame_name, 'singlemesh_female/{}_reg_high.obj'.format(frame_name))

        mesh = trimesh.load(org_mesh, process=False, maintain_order=True)
        reg_mesh = Mesh(filename=reg_mesh)
        with open(reg_file, 'rb') as f:
            reg_data = pkl.load(f, encoding="latin1")
        pose_tmp = reg_data['pose']
        beta = reg_data['betas']
        pose_tmp[60:] = 0.0
        # normalise pose
        theta_normalized, rot_mat = normalize_y_rotation(pose_tmp)
        reg_mesh.v -= reg_data['trans']
        mesh.vertices -= reg_data['trans']
        reg_mesh.v = reg_mesh.v.dot(rot_mat)
        mesh.vertices = mesh.vertices.dot(rot_mat)
        mesh.export(out_mesh)
        # Mesh(v=mesh.vertices,f=mesh.faces).write_obj('/BS/garvita/work/code/tmp.obj')
        # reg_mesh.write_obj('/BS/garvita/work/code/tmp2.obj')

        bottom_corner, upper_corner = mesh.bounds
        pose_torch = torch.from_numpy(theta_normalized.astype(np.float32)).unsqueeze(0)
        betas_torch = torch.from_numpy(beta[:10].astype(np.float32)).unsqueeze(0)
        smpl_verts = smpl_torch.forward(pose_torch, betas_torch)

        minimun = np.min(bottom_corner)
        maximum = np.max(upper_corner)
        skinning_pt = smpl_torch.weight.detach().numpy()[0]

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
        tn_verts_torch = torch.from_numpy(reg_mesh.v.astype(np.float32)).unsqueeze(0).cuda()
        sideddistance = SidedDistance()
        dist1 = sideddistance(points_torch, tn_verts_torch).cpu().detach().numpy()[0]
        skinning_pt = skinning_pt[dist1]
        gt_skin = skinning_pt
        skinning_pt = torch.from_numpy(skinning_pt.astype(np.float32)).unsqueeze(0).cuda()
        transform = smpl_torch.A.detach().numpy()[0]

        transform_np = transform
        joints  = smpl_torch.J_transformed.detach().numpy()[0]


        transform = torch.from_numpy(transform.astype(np.float32)).unsqueeze(0).cuda()
        W = skinning_pt
        T = torch.matmul(W, transform.view(1, 24, 16)).view(1, -1, 4, 4)

        Tinv = torch.inverse(T)
        points_torch = torch.from_numpy(boundary_points[:sample_num].astype(np.float32)).unsqueeze(0).cuda()
        ipdb.set_trace()
        verts_homo = torch.cat([ points_torch, torch.ones(1, points_torch.shape[1], 1).cuda()], dim=2)
        v_def = torch.matmul(Tinv, verts_homo.unsqueeze(-1))[:, :, :3, 0]
        transformed_pt = v_def.cpu().detach().numpy()[0]

        m1 = Mesh(v=transformed_pt[:sample_num])
        m1.write_obj(os.path.join(data_dir + 'debug', frame_name + '.obj'))
        df = igl.signed_distance(boundary_points, mesh.vertices, mesh.faces)[0]
        bottom_cotner, upper_corner = mesh.bounds

        np.savez(out_file,gt_skin=gt_skin,  bounds=np.array([bottom_cotner, upper_corner]), posed_points=boundary_points,
                 sdf=df, can_points=transformed_pt, joint=joints, transform=transform_np, pose=theta_normalized, smpl_verts=smpl_verts.detach().numpy()[0])

        print('Finished {} '.format( frame_name))
