"""source: https://github.com/jchibane/ndf"""
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
class MultiShapeTrainer(object):

    def __init__(self,  train_dataset, val_dataset, opt):
        self.device = opt['train']['device']

        if opt['model']['CanSDF']['use']:
            self.model_occ = getattr(net_modules, opt['model']['CanSDF']['name'])
            self.model_occ = self.model_occ(opt['model']['CanSDF']).to(self.device)
            self.optimizer_occ = getattr(optim, opt['train']['optimizer'])
            self.optimizer_occ = self.optimizer_occ(self.model_occ.parameters(), opt['train']['optimizer_param'])

        if opt['model']['DispPred']['use']:
            self.model_disp = getattr(net_modules, opt['model']['DispPred']['name'])
            self.model_disp = self.model_disp(opt).to(self.device)
            self.optimizer_disp = getattr(optim, opt['train']['optimizer'])
            self.optimizer_disp = self.optimizer_disp(self.model_disp.parameters(), opt['train']['optimizer_param'])

        if opt['model']['DispPred_beta']['use']:
            self.model_disp_beta = getattr(net_modules, opt['model']['DispPred_beta']['name'])
            self.model_disp_beta = self.model_disp_beta(opt, input_dim=13).to(self.device) #todo: hardcoded change this
            self.optimizer_disp_beta = getattr(optim, opt['train']['optimizer'])
            self.optimizer_disp_beta = self.optimizer_disp_beta(self.model_disp.parameters(), opt['train']['optimizer_param'])


        if opt['model']['WeightPred']['use']:
            self.model_wgt = getattr(net_modules, opt['model']['WeightPred']['name'])
            self.model_wgt = self.model_wgt(opt['model']['WeightPred']).to(self.device)
            self.optimizer_wgt = getattr(optim, opt['train']['optimizer'])
            self.optimizer_wgt = self.optimizer_wgt(self.model_wgt.parameters(), opt['train']['optimizer_param'])


        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.clamp_dist =  opt['train']['clamp_dist']

        self.disp_reg= opt['train']['disp_reg']
        self.d_class= opt['experiment']['d_class']

        self.train_stage_1 = opt['train']['train_stage_1']
        self.train_stage_2 = opt['train']['train_stage_2']
        self.loss_weight = {'wgt': opt['train']['wgt_wgt'], 'sdf':opt['train']['sdf_wgt'], 'disp': opt['train']['disp_wgt'], 'diff_can':  opt['train']['diff_can'], 'spr_wgt': opt['train']['spr_wgt']}


        #create exp name based on lossfunc
        self.exp_name = '{}_{}_{}_{}_{}'.format(self.loss_weight['wgt'],self.loss_weight['sdf'],self.loss_weight['disp'], self.loss_weight['diff_can'], self.loss_weight['spr_wgt'] )
        self.exp_path = '{}/{}/'.format( opt['experiment']['root_dir'], self.exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( self.exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(self.exp_name))

        self.val_min = None
        self.train_min = None
        self.loss = opt['train']['loss_type']
        self.n_part = opt['experiment']['num_part']
        self.loss_mse = torch.nn.MSELoss()

        self.batch_size=  opt['train']['batch_size']

        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()



    def train_step(self,batch, ep=None):


        if ep > self.train_stage_1:
            self.model_occ.train()
            self.optimizer_occ.zero_grad()
        if self.model_wgt and ep <= self.train_stage_2:
            self.model_wgt.train()
            self.optimizer_wgt.zero_grad()   #todo: need this even for not updating steps?
        if self.model_disp and ep > self.train_stage_2:
            self.model_disp.train()
            self.optimizer_disp.zero_grad()
            self.model_disp_beta.train()
            self.optimizer_disp_beta.zero_grad()

        loss, loss_dict = self.compute_loss(batch, ep)
        loss.backward()
        if ep > self.train_stage_1:
            self.optimizer_occ.step()
        if self.model_wgt and ep <= self.train_stage_2:
            self.optimizer_wgt.step()
        if self.model_disp and ep > self.train_stage_2:
            self.optimizer_disp.step()
            self.optimizer_disp_beta.step()

        return loss.item(), loss_dict

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

    def compute_loss(self,batch,ep=None):
        device = self.device

        sdf_gt = batch.get('label').to(device)
        pts = batch.get("points").to(device)
        skin_bp = batch.get("skin_bp").to(device)
        pose = batch.get("pose").to(device)
        transform = batch.get("transform").to(device)
        pose_in = pose.unsqueeze(1).repeat(1, pts.shape[1], 1)
        smpl_vert = batch.get("smpl_verts").to(device)
        can_smpl = batch.get("can_smpl").to(device)
        can_pts_gt = batch.get("can_pts_gt").to(device)
        gt_skin = batch.get("gt_skin").to(device)
        joints = batch.get("joints").to(device)
        beta_d = batch.get("beta").to(device)

        beta = beta_d.unsqueeze(1).repeat(1, smpl_vert.shape[1], 1)
        pts_in = pts.unsqueeze(2).repeat(1, 1, self.n_part, 1)
        joints = joints.unsqueeze(1).repeat(1, smpl_vert.shape[1], 1, 1)
        loss_dict = {}
        if ep < self.train_stage_1:

            smpl_verts = smpl_vert.unsqueeze(2).repeat(1, 1, self.n_part, 1)
            body_enc_feat = pts_in - joints
            body_enc_feat = torch.sqrt(torch.sum(body_enc_feat * body_enc_feat, dim=3))
            body_enc_feat = torch.nn.functional.normalize(body_enc_feat, p=1.0,dim=2)

            body_enc_feat_smpl = smpl_verts - joints
            body_enc_feat_smpl = torch.sqrt(torch.sum(body_enc_feat_smpl * body_enc_feat_smpl, dim=3))
            body_enc_feat_smpl = torch.nn.functional.normalize(body_enc_feat_smpl, p=1.0,dim=2)

            weight_pred = self.model_wgt(pts, body_enc_feat, beta)
            weight_smpl = self.model_wgt(smpl_vert, body_enc_feat_smpl, beta)
            #
            # import ipdb
            # ipdb.set_trace()
            can_pt  = self.tranform_pts(smpl_vert, transform, weight_smpl)
            can_pt_bp = self.tranform_pts(pts, transform, weight_pred)
            diff_can = can_pt - can_smpl
            diff_can = torch.sqrt(torch.sum(diff_can*diff_can, dim=2)).mean()

            diff_can_bp = can_pt_bp - can_pts_gt
            diff_can_bp = torch.sqrt(torch.sum(diff_can_bp*diff_can_bp, dim=2)).mean()
            w1 = self.loss_l1(weight_smpl, gt_skin)
            w2 =  self.loss_l1(weight_pred, skin_bp)
            spr_wgt =  (weight_pred.abs() + 1e-12).pow(0.8).sum(1).mean() + (
                        weight_smpl.abs() + 1e-12).pow(0.8).sum(1).mean()

            weight_loss = w1 + w2
            loss_dict['wgt'] = weight_loss
            loss_dict['diff_can'] = (diff_can +diff_can_bp)/2.0
            loss_dict['spr_wgt'] = spr_wgt
            #total_loss = weight_loss +  self.loss_weight['diff_can']*loss_dict['diff_can']
            total_loss = weight_loss +  self.loss_weight['diff_can']*loss_dict['diff_can'] +  self.loss_weight['diff_can']**loss_dict['spr_wgt']


        else:
            #todo: move this outside the block
            smpl_verts = smpl_vert.unsqueeze(2).repeat(1, 1, self.n_part, 1)
            body_enc_feat = pts_in - joints
            body_enc_feat = torch.sqrt(torch.sum(body_enc_feat * body_enc_feat, dim=3))
            body_enc_feat = torch.nn.functional.normalize(body_enc_feat, p=1.0,dim=2)
            body_enc_feat_smpl = smpl_verts - joints
            body_enc_feat_smpl = torch.sqrt(torch.sum(body_enc_feat_smpl * body_enc_feat_smpl, dim=3))
            body_enc_feat_smpl = torch.nn.functional.normalize(body_enc_feat_smpl, p=1.0,dim=2)


            weight_pred = self.model_wgt(pts, body_enc_feat, beta)
            weight_smpl = self.model_wgt(smpl_vert, body_enc_feat_smpl, beta)

            w1 = self.loss_l1(weight_smpl, gt_skin)
            w2 =  self.loss_l1(weight_pred, skin_bp)
            weight_loss = w1 + w2
            loss_dict['wgt'] = weight_loss

            can_pt  = self.tranform_pts(pts, transform, weight_pred)

            can_pt_smpl  = self.tranform_pts(smpl_vert, transform, weight_smpl)
            diff_can = can_pt_smpl - can_smpl
            diff_can = torch.sqrt(torch.sum(diff_can*diff_can, dim=2)).mean()

            spr_wgt =  (weight_pred.abs() + 1e-12).pow(0.8).sum(1).mean() + (
                        weight_smpl.abs() + 1e-12).pow(0.8).sum(1).mean()

            weight_vec = weight_pred.unsqueeze(3).repeat(1,1,1,int(pose_in.shape[2]/24))
            weight_vec = weight_vec.reshape(weight_pred.shape[0], weight_pred.shape[1], pose_in.shape[2])
            weighted_pose = weight_vec*pose_in
            if self.model_disp and ep > self.train_stage_2:
                disp = self.model_disp(can_pt, weighted_pose)
                disp_beta = self.model_disp_beta(can_pt, beta)
                can_pt = can_pt + disp + disp_beta

            sdf_pred = self.model_occ(can_pt, weighted_pose, beta)[:,:,0]

            # if self.d_class == 'smpl':
            #     sdf_pred = self.out_act(sdf_pred)
            sdf_loss =self.loss_l1(sdf_pred, sdf_gt)
            loss_dict['sdf'] = sdf_loss
            #total_loss = self.loss_weight['sdf']* sdf_loss + self.loss_weight['wgt']*weight_loss
            #total_loss = self.loss_weight['sdf']* sdf_loss + self.loss_weight['wgt']*weight_loss + self.loss_weight['diff_can']*diff_can
            total_loss = self.loss_weight['sdf']* sdf_loss + self.loss_weight['wgt']*weight_loss +  self.loss_weight['diff_can']*diff_can  + self.loss_weight['spr_wgt']*spr_wgt
            if self.model_disp and self.disp_reg and ep > 10000:
                disp_loss = torch.sum(disp*disp) /(self.batch_size* disp.shape[1]*3)
                loss_dict['disp'] = disp_loss
                total_loss = total_loss + self.loss_weight['disp']*disp_loss

        return total_loss, loss_dict

    def train_model(self, epochs, eval=True):
        loss = 0
        start = self.load_checkpoint()
        for epoch in range(start, epochs):
            sum_loss = 0
            loss_terms = {'wgt': 0, 'sdf': 0, 'disp': 0, 'diff_can': 0, 'spr_wgt': 0}
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            if epoch % 100 == 0:
                self.save_checkpoint(epoch)

            for batch in train_data_loader:
                loss, loss_dict = self.train_step(batch, epoch)
                for k in loss_dict.keys():
                    loss_terms[k] += self.loss_weight[k]*loss_dict[k].item()
                    print("Current loss: {} {}  ".format(k, loss_dict[k].item()))

                sum_loss += loss
            batch_loss = sum_loss / len(train_data_loader)
            print("Current batch_loss: {} {}  ".format(epoch, batch_loss))

            for k in loss_dict.keys():
                loss_terms[k] = loss_dict[k]/ len(train_data_loader)
            if self.train_min is None:
                self.train_min = batch_loss
            if batch_loss < self.train_min:
                self.save_checkpoint(epoch)
                for path in glob(self.exp_path + 'train_min=*'):
                    os.remove(path)
                np.save(self.exp_path + 'train_min={}'.format(epoch), [epoch, batch_loss])


            if eval:
                val_loss = self.compute_val_loss(epoch)
                print('validation loss:   ', val_loss)
                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    self.save_checkpoint(epoch)
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, batch_loss])
                self.writer.add_scalar('val loss batch avg', val_loss, epoch)

            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', batch_loss, epoch)
            for k in loss_dict.keys():
                self.writer.add_scalar('training loss {} avg'.format(k), loss_terms[k] , epoch)

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            if self.model_disp :
                torch.save({'epoch': epoch, 'model_state_occ_dict': self.model_occ.state_dict(),
                            'optimizer_occ_state_dict': self.optimizer_occ.state_dict(),
                            'model_state_wgt_dict': self.model_wgt.state_dict(),
                            'optimizer_wgt_state_dict': self.optimizer_wgt.state_dict(),
                            'model_state_disp_dict': self.model_disp.state_dict(),
                            'optimizer_disp_state_dict': self.optimizer_disp.state_dict(),
                            'model_state_disp_beta_dict': self.model_disp_beta.state_dict(),
                            'optimizer_disp_beta_state_dict': self.optimizer_disp_beta.state_dict() }, path,
                           _use_new_zipfile_serialization=False)

            else:
                torch.save({'epoch':epoch, 'model_state_occ_dict': self.model_occ.state_dict(),
                            'optimizer_occ_state_dict': self.optimizer_occ.state_dict(), 'model_state_wgt_dict': self.model_wgt.state_dict(),
                            'optimizer_wgt_state_dict': self.optimizer_wgt.state_dict()}, path,  _use_new_zipfile_serialization=False)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

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
            self.model_disp_beta.load_state_dict(checkpoint['model_state_disp_beta_dict'])
            self.optimizer_disp_beta.load_state_dict(checkpoint['optimizer_disp_beta_state_dict'])

        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self, ep):

        self.model_occ.eval()
        sum_val_loss = 0

        val_data_loader = self.val_dataset.get_loader()
        for batch in val_data_loader:
            loss, _= self.compute_loss(batch, ep)
            sum_val_loss += loss.item()
        return sum_val_loss /len(val_data_loader)

