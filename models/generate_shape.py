from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn

import sys

sys.path.append('/BS/garvita/work/code/if-net')
import data_processing.implicit_waterproofing as iw
import mcubes
import trimesh
import ipdb

#todo: create a base trainer
from models.network import  net_modules
class ShapeTrainer(object):

    def __init__(self,   opt, checkpoint):
        self.device = opt['train']['device']

        if opt['model']['CanSDF']['use']:
            self.model_occ = getattr(net_modules, opt['model']['CanSDF']['name'])
            self.model_occ = self.model_occ(opt['model']['CanSDF']).to(self.device)
            self.model_occ.eval()

        if opt['model']['DispPred']['use']:
            self.model_disp = getattr(net_modules, opt['model']['DispPred']['name'])
            self.model_disp = self.model_disp(opt).to(self.device)
            self.model_disp.eval()


        if opt['model']['WeightPred']['use']:
            self.model_wgt = getattr(net_modules, opt['model']['WeightPred']['name'])
            self.model_wgt = self.model_wgt(opt['model']['WeightPred']).to(self.device)
            self.model_wgt.eval()

        self.exp_path = '{}/{}/'.format( opt['experiment']['root_dir'], opt['experiment']['exp_name'])
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( opt['experiment']['exp_name'])
        self.load_checkpoint(checkpoint)

        self.resolution = 32
        self.out_actvn = nn.Sigmoid()
        self.n_part = opt['experiment']['num_part']

        self.padding = 0.25
        self.batch_points = opt['eval']['batch_points']
        self.threshold= 0.0

    def tranform_pts(self, pts, transform, W, trans=None):
        if trans is not None:
            pts = pts - trans.unsqueeze(1)

        #transform = torch.from_numpy(transform.astype(np.float32)).unsqueeze(0).cuda()
        #W = torch.from_numpy(weight_pred.astype(np.float32)).unsqueeze(0).cuda()
        T = torch.matmul(W, transform.view(transform.shape[0], 24, 16)).view(transform.shape[0], -1, 4, 4)
        Tinv = torch.inverse(T)
        verts_homo = torch.cat([pts, torch.ones(pts.shape[0], pts.shape[1], 1).cuda()], dim=2)
        transformed_pts = torch.matmul(Tinv, verts_homo.unsqueeze(-1))[:, :, :3, 0]

        return transformed_pts


    def generate_mesh(self, data):
        device = self.device
        threshold =0.0
        box_size = 1 + self.padding
        #box_size get from min max  #todo: check this
        min = data['min'].detach().numpy()[0]
        max = data['max'].detach().numpy()[0]

        transform = data["transform"].to(device)
        smpl_verts =  data["smpl_verts"].to(device)
        gt_skin =  data["gt_skin"].to(device)
        pose = data['pose'].to(device)
        joints = data['joints'].to(device)


        # split points to handle higher resolution
        grid_values = []
        all_pts = []
        all_logits = []
        logits_list = []
        grid_points = iw.create_grid_points_from_bounds(min -self.padding , max +self.padding, self.resolution)
        grid_coords = torch.from_numpy(grid_points).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)
        canonical_points =[]


        for pointsf in grid_points_split:
            pose_in = pose.unsqueeze(1).repeat(1, pointsf.shape[1], 1)


            all_pts.extend(pointsf[0].detach().cpu().numpy())
            with torch.no_grad():
                pts_in = pointsf.unsqueeze(2).repeat(1, 1, self.n_part, 1)
                body_enc_feat = pts_in - joints
                body_enc_feat = torch.sqrt(torch.sum(body_enc_feat * body_enc_feat, dim=3))
                body_enc_feat = torch.nn.functional.normalize(body_enc_feat, p=1.0, dim=2)

                weight_pred = self.model_wgt(pointsf, body_enc_feat, pose_in)
                weight_vec = weight_pred.unsqueeze(3).repeat(1, 1, 1, int(pose_in.shape[2] / 24))
                weight_vec = weight_vec.reshape(weight_pred.shape[0], weight_pred.shape[1], pose_in.shape[2])
                weighted_pose = weight_vec * pose_in
                can_pt = self.tranform_pts(pointsf, transform, weight_pred)
                #append canonical points here
                can_np = can_pt.detach().cpu().numpy()[0]
                if self.model_disp:
                    disp = self.model_disp(can_pt, weighted_pose, body_enc_feat)
                    can_pt = can_pt + disp
                sdf_pred = self.model_occ(can_pt, weighted_pose, body_enc_feat)[:, :, 0]

                logits = sdf_pred
                #append canonical points which lie inside
                #values = logits[0][0].cpu().detach().numpy()
            grid_values.append(logits[0])
            canonical_points.extend(can_np)
        #     all_logits.extend(values.cpu().detach().numpy())
        #     logits_list.append((-1)*logits.squeeze(0).squeeze(0).detach().cpu())
        #
        # grid_values2 = torch.cat(grid_values, dim=0).cpu().detach().numpy()
        grid_values = torch.cat(grid_values, dim=0).cpu().detach().numpy()
        canonical_points = np.array(canonical_points)
        # subsample_indices = np.random.randint(0, len(canonical_points), 5000)
        # canonical_points = canonical_points[subsample_indices]
        #canonical_points = canonical_points[grid_values > 0.5]
        # if self.dc == 'smpl':
        #     canonical_points = canonical_points[grid_values > 0.5]
        # else:
        #
        canonical_points = canonical_points[grid_values < 0.0]
        #logits = torch.cat(grid_values, dim=0).numpy()
        return  grid_values,  min, max, canonical_points
        #return grid_values2, grid_values, logits,  min, max



    def mesh_from_logits(self, logits, min, max, can_pts):

        #logits = np.reshape(logits, (self.resolution,) * 3)
        # padding to ba able to retrieve object close to bounding box bondary
        #logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        logits = np.reshape(logits, (self.resolution,) * 3)
        # padding to ba able to retrieve object close to bounding box bondary
        #threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # if self.dc == 'smpl':
        #     logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=1)
        #
        #     threshold = 0.0
        # else:
        #     threshold = 0.0
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=1)


        vertices, triangles = mcubes.marching_cubes(logits, self.threshold)

        # remove translation due to padding
        max = max + self.padding
        min = min - self.padding
        #vertices -= 1
        #rescale to original scale
        step = (max - min) / (self.resolution )
        #step = (max - min) / (self.resolution*self.upsampling_steps - 1)
        vertices = np.multiply(vertices, step)
        vertices += [min, min,min]
        #vertices= vertices*self.global_scale
        return trimesh.Trimesh(vertices, triangles), trimesh.Trimesh(can_pts)



    def load_checkpoint(self, checkpoint):
        if checkpoint is None:
            checkpoints = glob(self.checkpoint_path+'/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))

            checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model_occ.load_state_dict(checkpoint['model_state_occ_dict'])

        if self.model_wgt:
            self.model_wgt.load_state_dict(checkpoint['model_state_wgt_dict'])

        if self.model_disp:
            self.model_disp.load_state_dict(checkpoint['model_state_disp_dict'])



    def compute_val_loss(self, ep):

        self.model_occ.eval()
        sum_val_loss = 0

        val_data_loader = self.val_dataset.get_loader()
        for batch in val_data_loader:
            loss, _= self.compute_loss(batch, ep)
            sum_val_loss += loss.item()
        return sum_val_loss /len(val_data_loader)

