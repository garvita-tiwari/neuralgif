"""This is for training tailornet meshes"""
from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn

#todo: create a base trainer
from models.network import  net_modules
class ShapeTrainer(object):

    def __init__(self,  train_dataset, val_dataset, opt):
        self.device = opt['train']['device']

        if opt['model']['CanSDF']:
            self.model_occ = getattr(net_modules, opt['model']['CanSDF'])
            self.model_occ = self.model_occ(opt).to(self.device)
            self.optimizer_occ = getattr(optim, opt['train']['optimizer'])
            self.optimizer_occ = self.optimizer_occ(self.model_occ.parameters(), opt['train']['optimizer_param'])   #adatpive arguments
            #self.optimizer_occ = self.optimizer_occ(self.model_occ.parameters(), lr=1e-4)   #adatpive arguments

        if opt['model']['DispNet']:
            self.model_disp = getattr(net_modules, opt['model']['DispNet'])
            self.model_disp = self.model_disp(opt).to(self.device)
            self.optimizer_disp = getattr(optim, opt['train']['optimizer'])
            self.optimizer_disp = self.optimizer_disp(self.model_occ.parameters(), opt['train']['optimizer_param'])   #adatpive arguments

        if opt['model']['WeightPred']:
            self.model_wgt = getattr(net_modules, opt['model']['WeightPred'])
            self.model_wgt = self.model_wgt(opt['model']['WeightNet']).to(self.device)
            self.optimizer_wgt = getattr(optim, opt['train']['optimizer'])
            self.optimizer_wgt = self.optimizer_wgt(self.model_occ.parameters(), opt['train']['optimizer_param'])   #adatpive arguments


        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.exp_path = '{}/{}/'.format( opt['experiment']['root_dir'], opt['experiment']['exp_name'])
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( opt['experiment']['exp_name'])
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(opt['experiment']['exp_name']))

        self.val_min = None
        self.train_min = None
        self.loss = opt['train']['loss_type']
        self.n_part = 24
        self.loss_mse = torch.nn.MSELoss()

        self.batch_size=  opt['train']['batch_size']
        self.joints = range(opt['experiment']['num_part'])
        self.out_act = nn.Sigmoid()

        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()
        self.clamp_dist =  opt['train']['clamp_dist']
        self.label_w = opt['train']['label_w']
        self.minimal_w  = opt['train']['minimal_w']
        self.body_enc = opt['model']['WeightNet']['body_enc']
        self.disp_reg= opt['train']['disp_reg']
        self.d_class= opt['experiment']['d_class']

        self.train_stage_1 = opt['train']['train_stage']


    def train_step(self,batch, ep=None):

        self.model_occ.train()
        self.optimizer_occ.zero_grad()
        if self.model_wgt and ep <= self.train_stage_1:
            self.model_wgt.train()
            self.optimizer_wgt.zero_grad()   #todo: need this even for not updating steps?
        if self.model_disp and ep > self.train_stage_1:
            self.model_disp.train()
            self.optimizer_disp.zero_grad()

        loss, loss_dict = self.compute_loss(batch, ep)
        loss.backward()
        self.optimizer_occ.step()
        if self.model_wgt and ep <= self.train_stage_1:
            self.optimizer_wgt.step()
        if self.model_disp and ep > self.train_stage_1:
            self.optimizer_disp.step()

        return loss.item(), loss_dict

    def tranform_pts(self, pts, transform, W, trans=None):
        if trans is not None:
            pts = pts - trans.unsqueeze(1)

        #transform = torch.from_numpy(transform.astype(np.float32)).unsqueeze(0).cuda()
        #W = torch.from_numpy(weight_pred.astype(np.float32)).unsqueeze(0).cuda()
        T = torch.matmul(W, transform.view(transform.shape[0], 24, 16)).view(transform.shape[0], -1, 4, 4)
        # TODO: this is not working with GPU, find out why
        Tinv = torch.inverse(T)
        verts_homo = torch.cat([pts, torch.ones(pts.shape[0], pts.shape[1], 1).cuda()], dim=2)
        transformed_pts = torch.matmul(Tinv, verts_homo.unsqueeze(-1))[:, :, :3, 0]

        return transformed_pts

    def compute_loss(self,batch,ep=None):
        device = self.device

        occ_gt = batch.get('label').to(device)
        pts = batch.get("points").to(device)
        skin_bp = batch.get("skin_bp").to(device)
        pose = batch.get("pose").to(device)
        transform = batch.get("transform").to(device)
        pose_in = pose.unsqueeze(1).repeat(1, pts.shape[1], 1)
        smpl_verts = batch.get("smpl_verts").to(device)
        gt_skin = batch.get("gt_skin").to(device)
        joints = batch.get("joints").to(device)
        pts_in = pts.unsqueeze(2).repeat(1, 1, self.n_part, 1)
        joints = joints.unsqueeze(1).repeat(1, smpl_verts.shape[1], 1, 1)
        loss_dict = {}
        #run weight prediction on boundary points and supervise using GT
        if self.body_enc:
            smpl_verts = smpl_verts.unsqueeze(2).repeat(1, 1, self.n_part, 1)

            body_enc_feat = pts_in - joints
            #body_enc_feat = body_enc_feat.reshape(body_enc_feat.shape[0], body_enc_feat.shape[1], 72)
            body_enc_feat = torch.sum(body_enc_feat * body_enc_feat, dim=3)
            body_enc_feat_smpl = smpl_verts - joints
            #body_enc_feat_smpl = body_enc_feat_smpl.reshape(body_enc_feat_smpl.shape[0], body_enc_feat_smpl.shape[1], 72)
            body_enc_feat_smpl = torch.sum(body_enc_feat_smpl * body_enc_feat_smpl, dim=3)

            weight_pred = self.model_wgt(body_enc_feat, pose_in)
            weight_smpl = self.model_wgt(body_enc_feat_smpl, pose_in)
            #save weight as rgb color
            can_pt = self.tranform_pts(batch.get("smpl_verts").to(device), transform, weight_smpl)
            # unpose using predicted smpl weights
            # #map this to 0,255
            # import ipdb
            # ipdb.set_trace()

        else:
            weight_smpl = self.model_wgt(smpl_verts, pose_in)
            weight_pred = self.model_wgt(pts, pose_in)
        w1 = self.loss_l1(weight_smpl, gt_skin)
        w2 =  self.loss_l1(weight_pred, skin_bp)
        weight_loss = w1 + w2
        total_loss = 5.*weight_loss
        loss_dict['wgt'] = weight_loss
        #todo: add nearest neighbour based loss term


        return total_loss, loss_dict,can_pt

    def train_model(self, epochs, eval=True):
        loss = 0
        epoch = self.load_checkpoint()
        val_loss = self.compute_val_loss(epoch)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(489)

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model_occ.load_state_dict(checkpoint['model_state_occ_dict'])
        self.optimizer_occ.load_state_dict(checkpoint['optimizer_occ_state_dict'])

        if self.model_wgt:
            self.model_wgt.load_state_dict(checkpoint['model_state_wgt_dict'])
            self.optimizer_wgt.load_state_dict(checkpoint['optimizer_wgt_state_dict'])
        if self.model_disp:
            self.model_disp.load_state_dict(checkpoint['model_state_disp_dict'])
            self.optimizer_disp.load_state_dict(checkpoint['optimizer_disp_state_dict'])


        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self, ep):

        self.model_occ.eval()
        sum_val_loss = 0
        num_batches = 15

        val_data_loader = self.val_dataset.get_loader()
        all_verst = []
        all_col = []
        for batch in val_data_loader:
            loss, _, verts = self.compute_loss(batch, ep)
            sum_val_loss += loss.item()
            all_verst.append(verts.detach().cpu().numpy())
            #all_col.append(col.detach().cpu().numpy())
        np.save('/BS/RVH_3dscan_raw2/static00/model/neuralgif/smpl/trial_1/debug/vert_can.npy', np.array(all_verst))
        return sum_val_loss /len(val_data_loader)

