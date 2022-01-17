import ipdb
import torch
import torch.nn as nn

from models.network.net_utils import PosEncoder
class WeightPred(nn.Module):

    def __init__(self, opt_weight ):
        super(WeightPred, self).__init__()
        self.num_neuron = opt_weight['total_dim']
        self.num_layers = opt_weight['num_layers']
        self.num_parts = opt_weight['num_parts']

        self.shape = opt_weight['beta']
        self.body_enc = opt_weight['body_enc']
        self.input_dim = 3 + self.num_parts   # (X, theta)
        if self.body_enc:
            self.input_dim = 3 + self.num_parts    # (X- jts, theta)

        self.layers = nn.ModuleList()
        self.pose_enc = opt_weight['pose_enc']
        if self.shape:
            self.input_dim = self.input_dim +10
        ### apply positional encoding on input features
        if self.pose_enc:
            x_freq = opt_weight['x_freq']
            jts_freq = opt_weight['jts_freq']
            self.input_dim = 3 + self.num_parts * 3 + 3 * 2 * x_freq + self.num_parts * 2 * jts_freq # (X, PE(X), PE(theta))
            #todo: add for # (X, PE(X- jts), PE(theta)) and # ( PE(X- jts), PE(theta))
            if self.body_enc:
                self.input_dim = 24 + self.num_parts * 3 +72 * 2 * x_freq  #(theta, PE(X- jts))
                self.input_dim =  72 + 72 * 2 * x_freq  #(PE(|X- jts|))
                self.input_dim =  24 + 24 * 2 * x_freq  #(PE(|X- jts|))
                self.input_dim = 3 + self.num_parts + 3 * 2 * x_freq + self.num_parts * 2 * jts_freq  # (X, PE(X), PE(theta))

            self.x_enc = PosEncoder(x_freq, True)
            self.jts_enc = PosEncoder(jts_freq, True)

        ##### create network
        current_dim = self.input_dim
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(current_dim, self.num_neuron))
            #self.layers.append(nn.Conv1d(current_dim, self.num_neuron, 1))
            current_dim = self.num_neuron
        self.layers.append(nn.Linear(current_dim, 24))
        self.actvn = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, jts,beta=None):
        batch_size = x.shape[0]
        num_pts = x.shape[1]

        if self.pose_enc:  #todo : check this
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            x = self.x_enc.encode(x)
            x = x.reshape(batch_size, num_pts, x.shape[1])

            jts = jts.reshape(jts.shape[0] * jts.shape[1], jts.shape[2])
            jts = self.jts_enc.encode(jts)
            jts = jts.reshape(batch_size, num_pts, jts.shape[1])
        for i in range(self.num_layers - 1):
            if i == 0:
                if self.shape:
                    x_net = torch.cat((x, jts, beta), dim=2)
                else:
                    x_net = torch.cat((x, jts), dim=2)
                #x_net = x
                x_net = self.actvn(self.layers[i](x_net))
                residual = x_net
            else:
                x_net = self.actvn(self.layers[i](x_net) + residual)
                residual = x_net
        x_net = self.softmax(self.layers[-1](x_net))
        return x_net

class CanSDF(nn.Module):

    def __init__(self, opt_can):
        super(CanSDF, self).__init__()
        self.num_neuron = opt_can['total_dim']
        self.num_layers = opt_can['num_layers']
        self.num_parts = opt_can['num_parts']

        self.shape = opt_can['beta']
        self.body_enc = opt_can['body_enc']
        x_freq =  opt_can['x_freq']
        jts_freq =  opt_can['jts_freq']

        self.input_dim = 3 + self.num_parts * 3

        self.layers = nn.ModuleList()
        self.pose_enc =  opt_can['pose_enc']

        if self.pose_enc:
            self.input_dim = 3 + self.num_parts*3 + 3 * 2 * x_freq + self.num_parts*3 * 2 * jts_freq  # (X, PE(X), PE(theta))

            self.x_enc = PosEncoder(x_freq, True)
            self.jts_enc = PosEncoder(jts_freq, True)
        if self.shape:
            self.input_dim = self.input_dim +10
        ##### create network
        current_dim = self.input_dim
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(current_dim, self.num_neuron))
            current_dim = self.num_neuron
        self.layers.append(nn.Linear(current_dim, 1))

        self.actvn = nn.LeakyReLU(0.1)
        self.out_actvn = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, jts,beta=None):
        batch_size = x.shape[0]
        num_pts = x.shape[1]
        if self.pose_enc:  #todo : check this

            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            x = self.x_enc.encode(x)

            x = x.reshape(batch_size, num_pts, x.shape[1])
            jts = jts.reshape(jts.shape[0] * jts.shape[1], jts.shape[2])
            jts = self.jts_enc.encode(jts)
            jts = jts.reshape(batch_size, num_pts, jts.shape[1])

        for i in range(self.num_layers - 1):
            if i == 0:
                if self.shape:
                    x_net = torch.cat((x, jts, beta), dim=2)
                else:
                    x_net = torch.cat((x, jts), dim=2)
                x_net = self.actvn(self.layers[i](x_net))
                residual = x_net
            else:
                x_net = self.actvn(self.layers[i](x_net) + residual)
                residual = x_net

        x_net = self.layers[-1](x_net)
        return x_net



class DispPred(nn.Module):

    def __init__(self,opt, total_dim=960, num_parts=24, pose_enc=False, jts_freq=8, x_freq=16, num_layers=5,pose_str=False, body_enc=False,beta=False, input_dim=None):
        super(DispPred, self).__init__()
        self.num_neuron = total_dim
        self.num_layers = num_layers
        self.num_parts = num_parts
        x_freq = x_freq
        jts_freq = jts_freq
        self.shape = beta

        self.input_dim = 3 + self.num_parts * 3
        if pose_str:
            self.input_dim =  3 + self.num_parts * 6
        self.layers = nn.ModuleList()
        self.pose_enc = pose_enc

        ### apply positional encoding on input features
        if self.pose_enc:
            self.input_dim = 3 + self.num_parts * 3 + 3 * 2 * x_freq + 72 * 2 * jts_freq #todo: check this
            self.input_dim = 3 + self.num_parts * 3 + 3 * 2 * x_freq

            if body_enc:
                self.input_dim = 24 + self.num_parts * 3 + 3 * 2 * x_freq
            if pose_str:
                self.input_dim = 3 + self.num_parts * 6  + 3 * 2 * x_freq
                if body_enc:
                    self.input_dim = 24 + self.num_parts * 6+ 3 * 2 * x_freq
            self.x_enc = PosEncoder(x_freq, True)
            self.jts_enc = PosEncoder(jts_freq, True)
        if self.shape:
            self.input_dim = self.input_dim +10
        if input_dim:
            self.input_dim = input_dim
        ##### create network
        current_dim = self.input_dim
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(current_dim, self.num_neuron))
            #self.layers.append(nn.Conv1d(current_dim, self.num_neuron, 1))
            current_dim = self.num_neuron
        self.layers.append(nn.Linear(current_dim, 3))
        #self.layers.append(nn.Conv1d(current_dim, 1, 1))

        self.actvn = nn.LeakyReLU(0.1)
        self.out_actvn = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, jts,beta=None):

        batch_size = x.shape[0]
        num_pts = x.shape[1]
        if self.pose_enc:  #todo : check this
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            x = self.x_enc.encode(x)

            x = x.reshape(batch_size, num_pts, x.shape[1])

        for i in range(self.num_layers - 1):
            if i == 0:
                if self.shape:
                    x_net = torch.cat((x, jts, beta), dim=2)
                else:
                    x_net = torch.cat((x, jts), dim=2)
                x_net = self.actvn(self.layers[i](x_net))
                residual = x_net
            else:
                x_net = self.actvn(self.layers[i](x_net) + residual)
                residual = x_net

        x_net = self.layers[-1](x_net)
        return x_net


class NormNet(nn.Module):

    def __init__(self,opt, total_dim=960, num_parts=24, pose_enc=False, jts_freq=8, x_freq=16, num_layers=5,pose_str=False, body_enc=False,beta=False):
        super(NormNet, self).__init__()
        self.num_neuron = total_dim
        self.num_layers = num_layers
        self.num_parts = num_parts
        x_freq = x_freq
        jts_freq = jts_freq
        self.shape = beta

        self.input_dim = 3 + self.num_parts * 3
        if pose_str:
            self.input_dim =  3 + self.num_parts * 6
        self.layers = nn.ModuleList()
        self.pose_enc = pose_enc

        ### apply positional encoding on input features
        if self.pose_enc:
            self.input_dim = 3 + self.num_parts * 3 + 3 * 2 * x_freq + 72 * 2 * jts_freq #todo: check this
            self.input_dim = 3 + self.num_parts * 3 + 3 * 2 * x_freq

            if body_enc:
                self.input_dim = 24 + self.num_parts * 3 + 3 * 2 * x_freq
            if pose_str:
                self.input_dim = 3 + self.num_parts * 6  + 3 * 2 * x_freq
                if body_enc:
                    self.input_dim = 24 + self.num_parts * 6+ 3 * 2 * x_freq
            self.x_enc = PosEncoder(x_freq, True)
            self.jts_enc = PosEncoder(jts_freq, True)
        if self.shape:
            self.input_dim = self.input_dim +10
        ##### create network
        current_dim = self.input_dim
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(current_dim, self.num_neuron))
            #self.layers.append(nn.Conv1d(current_dim, self.num_neuron, 1))
            current_dim = self.num_neuron
        self.layers.append(nn.Linear(current_dim, 3))
        #self.layers.append(nn.Conv1d(current_dim, 1, 1))

        self.actvn = nn.LeakyReLU(0.1)
        self.out_actvn = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, jts,beta=None):

        batch_size = x.shape[0]
        num_pts = x.shape[1]
        if self.pose_enc:  #todo : check this
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            x = self.x_enc.encode(x)

            x = x.reshape(batch_size, num_pts, x.shape[1])

        for i in range(self.num_layers - 1):
            if i == 0:
                if self.shape:
                    x_net = torch.cat((x, jts, beta), dim=2)
                else:
                    x_net = torch.cat((x, jts), dim=2)
                x_net = self.actvn(self.layers[i](x_net))
                residual = x_net
            else:
                x_net = self.actvn(self.layers[i](x_net) + residual)
                residual = x_net

        x_net = self.layers[-1](x_net)
        return x_net
