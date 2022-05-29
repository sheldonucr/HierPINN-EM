"""
@author: Wentian Jin
"""

import torch
import numpy as np
import os, sys
import time
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch import optim
# from pyDOE import lhs
from scipy.io import loadmat
from scipy import interpolate
import re
import csv

import matplotlib as mpl
# mpl.use('tkagg')
# mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

import argparse
from datetime import datetime
from shutil import copyfile

import random
import math
from timeit import default_timer as timer

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(*paths):
    if isinstance(paths, list) or isinstance(paths, tuple):
        for path in paths:
            mkdir(path)
    else:
        raise ValueError

class Parser(argparse.ArgumentParser):
  def __init__(self): 
    super(Parser, self).__init__(description='EM 1D Simnet Solver')
    self.add_argument('--sample-dir', type=str, default="./Samples_second_stage_atom_flux_predictor/EMdataset_10seg_1n2/", help='directory to save output') # tested on 8000-10000 cases in the dataset
    # self.add_argument('--data-path', type=str, default="/fermi_data/shared/wentian/hierpinn_em/EMdataset_10seg_1n2/", help='path to data')  # server
    self.add_argument('--data-path', type=str, default="/fermi_data/shared/wentian/hierpinn_em/test_trees/", help='path to data')  # server
    self.add_argument('--model-path', type=str, default="./ckpt/trial_function_mlp_20.pkl", help='path to trained mynet model') # fisrt stage trained on 8000 ten-segment straight lines with layer configuration [7, 256, 512, 1024, 512, 256, 1]    
    
    self.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    self.add_argument('--lr-div', type=float, default=10., help='lr div factor to get the initial lr')
    self.add_argument('--lr-pct', type=float, default=0.3, help='percentage to reach the maximun lr, which is args.lr')
    self.add_argument('--weight-decay', type=float, default=0., help="weight decay")

    self.add_argument('--n-epochs', type=float, default=1000, help='Toatal training epochs')
    self.add_argument('--cuda', type=int, default=0, choices=[0, 1, 2, 3], help='cuda index')
    self.add_argument('--save-freq', type=int, default=100, help='Epochs between plot')
    self.add_argument('--fig-format', type=str, default='1d2d3d', help='Figure plot format, 1d, 2d, 3d or mixed, e.g. 2d3d')

    self.add_argument('--testcase', type=int, default=0, help='Testcase number')
    self.add_argument('--timer', action='store_true', default=False, help='Print the training time')

  def parse(self):
    args = self.parse_args()

    # Create output dir
    dt = datetime.now()
    args.date = dt.strftime("%Y%m%d%H%M%S")
    args.hparams = f'EM1d_Epochs{args.n_epochs}_TreeModel_{args.date}'
    args.run_dir = args.sample_dir + '/' + args.hparams
    args.code_bkup_dir = args.run_dir + '/code_bkup'
    mkdirs(args.run_dir, args.code_bkup_dir)

    # backup the code of current model
    code_path_src = os.path.basename(__file__)
    code_path_dst = args.code_bkup_dir + '/' + code_path_src
    copyfile(code_path_src,code_path_dst)

    # print('Arguments:')
    # pprint(vars(args))
    # with open(args.run_dir + "/args.txt", 'w') as args_file:
    #     json.dump(vars(args), args_file, indent=4)

    return args

args = Parser().parse()

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

########################################################################
########################### Parameters #################################
#########################################################################
time_interval = 1.e4
T_n_points = 101
time_length = time_interval * (T_n_points-1)
e = 1.69e-19
Ea=0.84*e
kB=1.3806e-23
T=353.0
D0 = 7.5e-8
Da=D0*math.exp(-Ea/(kB*T))
B = 1e11
Omega=1.182e-29
Z=10
rou = 3.e-08
kappa = Da*B*Omega/(kB*T)

# [X,Time,L,W,G,k1,k2]
mynet_mlp = [7, 256, 512, 1024, 512, 256, 1] # 7-layer
# mynet_mlp = [7, 256, 512, 1024, 2048, 2048, 2048, 2048, 1024, 512, 256, 1] # 12-layer

#[Number_of_node, Gl, Gu, Gr, Gd, Time]
flux_mlp = [6, 256, 512, 1024, 512, 256, 3]



# read dataset from csv start
if not os.path.exists(args.sample_dir+'/training.csv'):
    copyfile(args.model_path[:args.model_path.rfind('/')]+'/training.csv', args.sample_dir+'/training.csv')
    copyfile(args.model_path[:args.model_path.rfind('/')]+'/testing.csv', args.sample_dir+'/testing.csv')
    copyfile(args.model_path[:args.model_path.rfind('/')]+'/statistics.csv', args.sample_dir+'/statistics.csv')

# reader = csv.reader(open(args.sample_dir+'/training.csv', "r"), delimiter=",")
# training_list = list(reader)
# training_list = [int(file[0]) for file in training_list]
# random.shuffle(training_list)

reader = csv.reader(open(args.sample_dir+'/testing.csv', "r"), delimiter=",")
testing_list = list(reader)
testing_list = [int(file[0]) for file in testing_list]
random.shuffle(testing_list)

reader = csv.reader(open(args.sample_dir+'/statistics.csv', "r"), delimiter=",")
statistics_list = list(reader)
statistics_list = [float(value) for value in statistics_list[1]]
max_time, max_length, max_G, max_k, stress_max, stress_min = statistics_list
# read dataset from csv end

# testing_list=[training_list[0]]
if(args.testcase > -1):
    testing_list=[args.testcase]
    print(args.testcase)

# params for domain
stress_range = stress_max - stress_min

k_x = 1/max_length
k_t = 1/max_time # 3600*24*365*10=315360000



# Validation grid
node_segments_list = []
segment_endopoints_and_direction_list = []

flux_predictor_inputs_Ni_list = []
flux_predictor_inputs_Gl_list = []
flux_predictor_inputs_Gu_list = []
flux_predictor_inputs_Gr_list = []
flux_predictor_inputs_Gd_list = []
flux_predictor_inputs_T_list = []

boundary_X_list = []
boundary_T_list = []
boundary_L_list = []
boundary_W_list = []
boundary_G_list = []

valid_truth_list = []
valid_X_list = []
valid_T_list = []
valid_L_list = []
valid_W_list = []
valid_G_list = []
valid_k1_list = []
valid_k2_list = []
for file in testing_list:
    node_dict_single_case = {}
    node_count = 0
    node_segments_list_single_case = [] # connected segments in four derictions for all nodes [left, up, right, down]
    segment_endopoints_and_direction_single_case = [] # [(x1,y1),(x2,y2),(x3,y3),(x4,y4),horizontal:0 or vertical:1]

    segment_length_list = []
    G_list = []
    idx_MSE_single_case = np.empty([0,7])
    truth_MSE_single_case = np.empty([0,1])
    file_path = args.data_path + str(file)

    # Read .geo file for wire geometries
    with open(file_path+".geo") as f:
        lines = f.readlines()
    for line in lines:
        if line[0:9] == 'Rectangle':
            rect_vertices = re.findall('-*[0-9]+\.*[0-9]*', line)
            current_segment_length = max(abs(float(rect_vertices[4]+"e-6")),abs(float(rect_vertices[5]+"e-6")))
            segment_length_list.append(current_segment_length)

            node_0 = (float(rect_vertices[1]), float(rect_vertices[2]))
            node_1 = (node_0[0] + float(rect_vertices[4]), node_0[1] + float(rect_vertices[5]))

            if abs(float(rect_vertices[4])) > abs(float(rect_vertices[5])): # horizontal segment
                node_0_2nd = (node_0[0], node_0[1] + float(rect_vertices[5]))
                node_1_2nd = (node_0[0] + float(rect_vertices[4]), node_0[1])
                node_0 = ((node_0[0]+node_0_2nd[0])/2, (node_0[1]+node_0_2nd[1])/2)
                node_1 = ((node_1[0]+node_1_2nd[0])/2, (node_1[1]+node_1_2nd[1])/2)
                if(float(rect_vertices[4]) > 0):
                    segment_endopoints_and_direction_single_case.append([node_0,node_1,0])
                else:
                    segment_endopoints_and_direction_single_case.append([node_1,node_0,0])
            else: # vertical segment
                node_0_2nd = (node_0[0] + float(rect_vertices[4]), node_0[1])
                node_1_2nd = (node_0[0], node_0[1] + float(rect_vertices[5]))
                node_0 = ((node_0[0]+node_0_2nd[0])/2, (node_0[1]+node_0_2nd[1])/2)
                node_1 = ((node_1[0]+node_1_2nd[0])/2, (node_1[1]+node_1_2nd[1])/2)
                if(float(rect_vertices[5]) > 0):
                    segment_endopoints_and_direction_single_case.append([node_0,node_1,1])
                else:
                    segment_endopoints_and_direction_single_case.append([node_1,node_0,1])
            
            if (node_0 not in node_dict_single_case):
                node_dict_single_case[node_0] = node_count
                node_count += 1
                node_segments_list_single_case.append([None, None, None, None]) # segment numbers arranged in the order of: [left, up, right, down]

            if (node_1 not in node_dict_single_case):
                node_dict_single_case[node_1] = node_count
                node_count += 1
                node_segments_list_single_case.append([None, None, None, None]) # segment numbers arranged in the order of: [left, up, right, down]

            current_segment_n = len(segment_length_list)-1
            node_0_n = node_dict_single_case[node_0]
            node_1_n = node_dict_single_case[node_1]
            if(abs(float(rect_vertices[4])) > abs(float(rect_vertices[5]))): # horizontal rectangle
                if(float(rect_vertices[4])>0): # pointing right
                    node_segments_list_single_case[node_0_n][2] = current_segment_n # current segment is attached to the right of node_0
                    node_segments_list_single_case[node_1_n][0] = current_segment_n # current segment is attached to the left of node_1
                else:
                    node_segments_list_single_case[node_0_n][0] = current_segment_n
                    node_segments_list_single_case[node_1_n][2] = current_segment_n
            else: # vertical rectangle
                if(float(rect_vertices[5])>0): # pointing up
                    node_segments_list_single_case[node_0_n][1] = current_segment_n # current segment is attached to the upside of node_0
                    node_segments_list_single_case[node_1_n][3] = current_segment_n # current segment is attached to the downside of node_1
                else:
                    node_segments_list_single_case[node_0_n][3] = current_segment_n
                    node_segments_list_single_case[node_1_n][1] = current_segment_n

    # Read .mat file for stress results and current
    data_mat = loadmat(file_path+".mat")
    current = data_mat['J']
    stress = data_mat['sVC']       

    n_segments = len(segment_length_list)
    n_nodes = len(node_segments_list_single_case)
    L_list = [L/max_length for L in segment_length_list]

    # Current density
    for J in current[0]:
        G = e * Z * rou * J / Omega
        G_list.append(G)


    # Generate input for flux predictor net
    flux_predictor_inputs_Ni_single_case_list = []
    flux_predictor_inputs_Gl_single_case_list = []
    flux_predictor_inputs_Gu_single_case_list = []
    flux_predictor_inputs_Gr_single_case_list = []
    flux_predictor_inputs_Gd_single_case_list = []
    flux_predictor_inputs_T_single_case_list = []
    for i in range(n_nodes):
        connected_segments = node_segments_list_single_case[i]
        if(sum(segment is not None for segment in connected_segments) == 1): # this is a boundary node, not an internal node, no need to predict k
            continue
        Ni = (i+1)/n_nodes # Number of internal nodes: 0.1, 0.2, ... , 0.9

        G_l = 0 if connected_segments[0]==None else G_list[connected_segments[0]]/max_G
        G_u = 0 if connected_segments[1]==None else G_list[connected_segments[1]]/max_G
        G_r = 0 if connected_segments[2]==None else G_list[connected_segments[2]]/max_G
        G_d = 0 if connected_segments[3]==None else G_list[connected_segments[3]]/max_G

        T = np.linspace(0,1, num=T_n_points, endpoint=True)

        Ni = Ni*np.ones([T.shape[0], 1])
        G_l = G_l*np.ones([T.shape[0], 1])
        G_u = G_u*np.ones([T.shape[0], 1])
        G_r = G_r*np.ones([T.shape[0], 1])
        G_d = G_d*np.ones([T.shape[0], 1])
        T = np.expand_dims(T, axis=-1)

        Ni = torch.FloatTensor(Ni).to(device)
        G_l = torch.FloatTensor(G_l).to(device)
        G_u = torch.FloatTensor(G_u).to(device)
        G_r = torch.FloatTensor(G_r).to(device)
        G_d = torch.FloatTensor(G_d).to(device)
        T = torch.FloatTensor(T).to(device)

        flux_predictor_inputs_Ni_single_case_list.append(Ni)
        flux_predictor_inputs_Gl_single_case_list.append(G_l)
        flux_predictor_inputs_Gu_single_case_list.append(G_u)
        flux_predictor_inputs_Gr_single_case_list.append(G_r)
        flux_predictor_inputs_Gd_single_case_list.append(G_d)
        flux_predictor_inputs_T_single_case_list.append(T)

    node_segments_list.append(node_segments_list_single_case)
    segment_endopoints_and_direction_list.append(segment_endopoints_and_direction_single_case)

    flux_predictor_inputs_Ni_list.append(flux_predictor_inputs_Ni_single_case_list)
    flux_predictor_inputs_Gl_list.append(flux_predictor_inputs_Gl_single_case_list)
    flux_predictor_inputs_Gu_list.append(flux_predictor_inputs_Gu_single_case_list)
    flux_predictor_inputs_Gr_list.append(flux_predictor_inputs_Gr_single_case_list)
    flux_predictor_inputs_Gd_list.append(flux_predictor_inputs_Gd_single_case_list)
    flux_predictor_inputs_T_list.append(flux_predictor_inputs_T_single_case_list)


    # Generate input for stress net
    boundary_X_single_case_list = []
    boundary_T_single_case_list = []
    boundary_L_single_case_list = []
    boundary_W_single_case_list = []
    boundary_G_single_case_list = []
    for i in range(n_segments):
        truth = stress[i,0] # shape = [L_n_points, T_n_points]

        wire_interval = segment_length_list[i]/(truth.shape[0]-1)

        # Genrate indexes for colocation points [x,t,L,W,G,k1,k2]
        truth = 2*((truth-stress_min) / stress_range)-1 # convert to [-1, 1]
        x = np.linspace(0,1, num=2, endpoint=True) # only generate points on boudaries, i.e. x = 0 and 1
        t = np.linspace(0,1, num=truth.shape[1], endpoint=True)
        X, T = np.meshgrid(x, t) # shape=[T_n_points, L_n_points]
        L = np.tile(L_list[i], [X.shape[0],X.shape[1]])
        W = np.tile(1.0, [X.shape[0],X.shape[1]])
        G = np.tile(G_list[i]/max_G, [X.shape[0],X.shape[1]])

        X = np.expand_dims(X.flatten(), axis=-1)
        T = np.expand_dims(T.flatten(), axis=-1)
        L = np.expand_dims(L.flatten(), axis=-1)
        W = np.expand_dims(W.flatten(), axis=-1)
        G = np.expand_dims(G.flatten(), axis=-1)

        X = torch.FloatTensor(X).to(device)
        T = torch.FloatTensor(T).to(device)
        L = torch.FloatTensor(L).to(device)
        W = torch.FloatTensor(W).to(device)
        G = torch.FloatTensor(G).to(device)

        boundary_X_single_case_list.append(X)
        boundary_T_single_case_list.append(T)
        boundary_L_single_case_list.append(L)
        boundary_W_single_case_list.append(W)
        boundary_G_single_case_list.append(G)

    boundary_X_list.append(boundary_X_single_case_list)
    boundary_T_list.append(boundary_T_single_case_list)
    boundary_L_list.append(boundary_L_single_case_list)
    boundary_W_list.append(boundary_W_single_case_list)
    boundary_G_list.append(boundary_G_single_case_list)


    # Generate validation grid
    valid_truth_single_case_list = []
    valid_X_single_case_list = []
    valid_T_single_case_list = []
    valid_L_single_case_list = []
    valid_W_single_case_list = []
    valid_G_single_case_list = []
    valid_k1_single_case_list = []
    valid_k2_single_case_list = []
    for i in range(n_segments):
        truth = stress[i,0] # shape = [L_n_points, T_n_points]

        wire_interval = segment_length_list[i]/(truth.shape[0]-1)

        # calculate k1 and k2
        k1 = truth[:2, :]
        k2 = truth[-2:, :]
        k1 = (k1[1,:]-k1[0,:])/wire_interval
        k2 = (k2[1,:]-k2[0,:])/wire_interval
        k1 += G_list[i]
        k2 += G_list[i] 
        k1 = k1 / max_k
        k2 = k2 / max_k

        # Genrate indexes for colocation points [x,t,L,W,G,k1,k2]
        truth = 2*((truth-stress_min) / stress_range)-1 # convert to [-1, 1]
        x = np.linspace(0,1, num=truth.shape[0], endpoint=True)
        t = np.linspace(0,1, num=truth.shape[1], endpoint=True)
        X, T = np.meshgrid(x, t) # shape=[T_n_points, L_n_points]
        L = np.tile(L_list[i], [X.shape[0],X.shape[1]])
        W = np.tile(1.0, [X.shape[0],X.shape[1]])
        G = np.tile(G_list[i]/max_G, [X.shape[0],X.shape[1]])
        if(i==0):
            k1[:] = 0
        if(i == n_segments-1):
            k2[:] = 0
        k1 = np.tile(k1,[truth.shape[0],1]).T # shape=[T_n_points, L_n_points]
        k2 = np.tile(k2,[truth.shape[0],1]).T # shape=[T_n_points, L_n_points]

        X = np.expand_dims(X.flatten(), axis=-1)
        T = np.expand_dims(T.flatten(), axis=-1)
        L = np.expand_dims(L.flatten(), axis=-1)
        W = np.expand_dims(W.flatten(), axis=-1)
        G = np.expand_dims(G.flatten(), axis=-1)
        k1 = np.expand_dims(k1.flatten(), axis=-1)
        k2 = np.expand_dims(k2.flatten(), axis=-1)

        X = torch.FloatTensor(X).to(device)
        T = torch.FloatTensor(T).to(device)
        L = torch.FloatTensor(L).to(device)
        W = torch.FloatTensor(W).to(device)
        G = torch.FloatTensor(G).to(device)
        k1 = torch.FloatTensor(k1).to(device)
        k2 = torch.FloatTensor(k2).to(device)

        valid_truth_single_case_list.append(truth)
        valid_X_single_case_list.append(X)
        valid_T_single_case_list.append(T)
        valid_L_single_case_list.append(L)
        valid_W_single_case_list.append(W)
        valid_G_single_case_list.append(G)
        valid_k1_single_case_list.append(k1)
        valid_k2_single_case_list.append(k2)

    valid_truth_list.append(valid_truth_single_case_list)
    valid_X_list.append(valid_X_single_case_list)
    valid_T_list.append(valid_T_single_case_list)
    valid_L_list.append(valid_L_single_case_list)
    valid_W_list.append(valid_W_single_case_list)
    valid_G_list.append(valid_G_single_case_list)
    valid_k1_list.append(valid_k1_single_case_list)
    valid_k2_list.append(valid_k2_single_case_list)



def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

#a very simple torch method to compute derivatives.
def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, create_graph=True, allow_unused=True)[0]
        f = grads
        if grads is None:
            print('bad grad')
            return torch.tensor(0.)
    return grads

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# mynet_mlp = [7, 256, 512, 512, 256, 1]
class Net(nn.Module):
    # initializers
    def __init__(self, mlp_layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(mlp_layers)-1):
            self.layers.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))

        self.weight_init(mean=0.0, std=0.02)

    # weight_init
    def weight_init(self, mean, std):
        # for m in self._modules:
        for m in self.layers:
            normal_init(m, mean, std)

    # forward method
    def forward(self, x,t,L,W,G,k1,k2):
        he = torch.cat((x,t,L,W,G,k1,k2), 1)
        for l, m in enumerate(self.layers):
            he = m(he)
            if l!=len(self.layers)-1:
                he = F.relu(he)
                # he = F.tanh(he)

        return he

class FluxPredictNet(nn.Module):
    # initializers
    def __init__(self, mlp_layers):
        super(FluxPredictNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(mlp_layers)-1):
            self.layers.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))

        self.weight_init(mean=0.0, std=0.02)

    # weight_init
    def weight_init(self, mean, std):
        # for m in self._modules:
        for m in self.layers:
            normal_init(m, mean, std)

    # forward method
    def forward(self, ni,Gl,Gu,Gr,Gd,t):
        he = torch.cat((ni,Gl,Gu,Gr,Gd,t), 1)
        for l, m in enumerate(self.layers):
            he = m(he)
            if l!=len(self.layers)-1:
                # he = F.sigmoid(he)
                # he = F.relu(he)
                he = F.tanh(he)
        
        # output layer
        # tanh
        he = F.tanh(he)

        #sigmoid
        # he = F.sigmoid(he)
        # he = (2*he)-1

        return he

def annealing_linear(start, end, pct):
    return start + pct * (end-start)


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out

class OneCycleScheduler(object):
    """
    (0, pct_start) -- linearly increase lr
    (pct_start, 1) -- cos annealing
    """
    def __init__(self, lr_max, div_factor=25., pct_start=0.3):
        super(OneCycleScheduler, self).__init__()
        self.lr_max = lr_max
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.lr_low = self.lr_max / self.div_factor
    
    def step(self, pct):
        # pct: [0, 1]
        if pct <= self.pct_start:
            return annealing_linear(self.lr_low, self.lr_max, pct / self.pct_start)

        else:
            return annealing_cos(self.lr_max, self.lr_low / 1e4, (
                pct - self.pct_start) / (1 - self.pct_start))

def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]

def rmse_nan_array(array1, array2):
    diff_list = []
    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            if not np.isnan(array1[i,j]):
                diff_list.append(array1[i,j] - array2[i,j])

    rmse = np.sqrt(np.mean(np.square(diff_list)))
    return rmse

def interior_loss(input_batch,kappa=4.0675706e-18, k_x=1, k_t=1, stress_range = 1):
    x=input_batch[:,0:1]
    t=input_batch[:,1:2]
    L=input_batch[:,2:3]
    W=input_batch[:,3:4]
    G=input_batch[:,4:5]
    k1=input_batch[:,5:6]
    k2=input_batch[:,6:7]
    k_x =k_x/L.detach() # input k_x = 1/maxLength, input L = wireLength/maxLength, real k_x = 1/((wireLength/maxLength)*maxLength) = 1/maxLength / (wireLength/maxlength)
    u = mynet(x,t,L,W,G,k1,k2)
    #u = [v[0], v[1]]
    u_t = nth_derivative(flat(u), wrt=t, n=1)
    u_x = nth_derivative(flat(u), wrt=x, n=1)
    u_xx = nth_derivative(flat(u_x), wrt=x, n=1)
    loss = stress_range*u_t*k_t - stress_range*kappa*u_xx*(k_x**2)
    loss = (loss**2).mean()
    # w = torch.tensor(0.01/np.pi)
    # f = u_t + u*u_x - w*u_xx
    return loss

def boundary_loss(input_batch, k_x=1, stress_range = 1, max_G=1, max_k=1, is_left=True):
    x=input_batch[:,0:1]
    t=input_batch[:,1:2]
    L=input_batch[:,2:3]
    W=input_batch[:,3:4]
    G=input_batch[:,4:5]
    k1=input_batch[:,5:6]
    k2=input_batch[:,6:7]
    k_x =k_x/L.detach() # input k_x = 1/maxLength, input L = wireLength/maxLength, real k_x = 1/((wireLength/maxLength)*maxLength) = 1/maxLength / (wireLength/maxlength)
    u = mynet(x,t,L,W,G,k1,k2)
    u_x = nth_derivative(flat(u), wrt=x, n=1)
    G = G.detach()*max_G # G:[-1,1]
    if is_left:
        k = k1.detach()*max_k
    else:
        k = k2.detach()*max_k
    # loss = u_x*(k_x*stress_range)+G - k
    loss = u_x+G/(k_x*stress_range) - k/(k_x*stress_range)
    loss = (loss**2).mean()
    return loss

def initial_loss(input_batch):
    x=input_batch[:,0:1]
    t=input_batch[:,1:2]
    L=input_batch[:,2:3]
    W=input_batch[:,3:4]
    G=input_batch[:,4:5]
    k1=input_batch[:,5:6]
    k2=input_batch[:,6:7]
    u = mynet(x,t,L,W,G,k1,k2)
    loss = u
    loss = (loss**2).mean()
    return loss

def MSE_loss(input_batch, truth_batch):
    x=input_batch[:,0:1]
    t=input_batch[:,1:2]
    L=input_batch[:,2:3]
    W=input_batch[:,3:4]
    G=input_batch[:,4:5]
    k1=input_batch[:,5:6]
    k2=input_batch[:,6:7]
    u = mynet(x,t,L,W,G,k1,k2)
    loss = u - truth_batch
    loss = (loss**2).mean()
    return loss

mynet = Net(mynet_mlp)

mynet.load_state_dict(torch.load(args.model_path, map_location=device))

mynet = mynet.to(device)
mynet.eval()

flux_predictor = FluxPredictNet(flux_mlp)
flux_predictor = flux_predictor.to(device)

optimizer = optim.Adam(flux_predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if(args.lr_pct >= 0):
    scheduler = OneCycleScheduler(lr_max=args.lr, div_factor=args.lr_div, pct_start=args.lr_pct)
# criterion = nn.MSELoss()


# Training begins
last_saved_epoch = 0
loss_list = []
if args.timer:
    total_training_time = 0
for epoch in range(args.n_epochs):
    if args.timer:
        tmr_start = timer()
    mynet.eval()
    flux_predictor.train()
    loss_average = 0
    for case in range(len(testing_list)):
        flux_predictor.zero_grad()
        # Flux prediction
        node_segments_list_single_case = node_segments_list[case]
        flux_predictor_inputs_Ni_single_case_list = flux_predictor_inputs_Ni_list[case]
        flux_predictor_inputs_Gl_single_case_list = flux_predictor_inputs_Gl_list[case]
        flux_predictor_inputs_Gu_single_case_list = flux_predictor_inputs_Gu_list[case]
        flux_predictor_inputs_Gr_single_case_list = flux_predictor_inputs_Gr_list[case]
        flux_predictor_inputs_Gd_single_case_list = flux_predictor_inputs_Gd_list[case]
        flux_predictor_inputs_T_single_case_list = flux_predictor_inputs_T_list[case]

        stress_predictor_inputs_k_single_case_list = [[torch.zeros(T_n_points,1,device=device),torch.zeros(T_n_points,1,device=device)] for _ in range(len(boundary_X_list[case]))] # initialize all [k1,k2] to zeros with shape of [T_n_points,1]
        j = 0
        for i in range(len(flux_predictor_inputs_Ni_single_case_list)): # loop over all internal junctions
            k = flux_predictor(flux_predictor_inputs_Ni_single_case_list[i],
                                flux_predictor_inputs_Gl_single_case_list[i],
                                flux_predictor_inputs_Gu_single_case_list[i],
                                flux_predictor_inputs_Gr_single_case_list[i],
                                flux_predictor_inputs_Gd_single_case_list[i],
                                flux_predictor_inputs_T_single_case_list[i])

            # Fill predicted k values into stress_predictor_inputs_k_single_case_list
            while(sum(segment is not None for segment in node_segments_list_single_case[j]) == 1): # skip boundary nodes which are not internal junctions
                j+=1
            connected_segments = node_segments_list_single_case[j]
            connected_segments_n = sum(segment is not None for segment in connected_segments)
            flux_sum = torch.zeros(T_n_points,1) # kl-ku-kr+kd=0
            flux_sum = flux_sum.to(device)
            if(connected_segments[0] is not None): # left
                stress_predictor_inputs_k_single_case_list[connected_segments[0]][1] = k[:,0:1]
                flux_sum += k[:,0:1]
                connected_segments_n -= 1
            if(connected_segments[1] is not None): # up
                if(connected_segments_n==1):
                    stress_predictor_inputs_k_single_case_list[connected_segments[1]][0] = flux_sum
                else:
                    stress_predictor_inputs_k_single_case_list[connected_segments[1]][0] = k[:,1:2]
                    flux_sum -= k[:,1:2]
                connected_segments_n -= 1
            if(connected_segments[2] is not None): # right
                if(connected_segments_n==1):
                    stress_predictor_inputs_k_single_case_list[connected_segments[2]][0] = flux_sum
                else:
                    stress_predictor_inputs_k_single_case_list[connected_segments[2]][0] = k[:,2:3]
                    flux_sum -= k[:,2:3]
                connected_segments_n -= 1
            if(connected_segments[3] is not None): # down
                stress_predictor_inputs_k_single_case_list[connected_segments[3]][1] = -flux_sum
            
            j+=1
                

            # stress_predictor_inputs_k_single_case_list.append(k)

        # stress continuity
        boundary_X_single_case_list = boundary_X_list[case]
        boundary_T_single_case_list = boundary_T_list[case]
        boundary_L_single_case_list = boundary_L_list[case]
        boundary_W_single_case_list = boundary_W_list[case]
        boundary_G_single_case_list = boundary_G_list[case]

        u_boundary_list = []
        for i in range(len(boundary_X_single_case_list)): # loop over all segments
            k1, k2 = stress_predictor_inputs_k_single_case_list[i]

            # if(i==0):
            #     k1 = torch.zeros_like(stress_predictor_inputs_k_single_case_list[0])
            #     k2 = stress_predictor_inputs_k_single_case_list[0]
            # elif(i == len(boundary_X_single_case_list)-1):
            #     k1 = stress_predictor_inputs_k_single_case_list[-1]
            #     k2 = torch.zeros_like(stress_predictor_inputs_k_single_case_list[-1])
            # else:
            #     k1 = stress_predictor_inputs_k_single_case_list[i-1]
            #     k2 = stress_predictor_inputs_k_single_case_list[i]

            # X_n_points = 2
            X_n_points = int(boundary_X_single_case_list[i].shape[0]/T_n_points)
            k1_cat = [k1 for _ in range(X_n_points)]
            k2_cat = [k2 for _ in range(X_n_points)]

            k1 = torch.cat(k1_cat,dim=1) # shape=[T_n_points, L_n_points]
            k2 =  torch.cat(k2_cat,dim=1) # shape=[T_n_points, L_n_points]

            k1 = torch.reshape(k1, [-1,1])
            k2 = torch.reshape(k2, [-1,1])

            u = mynet(boundary_X_single_case_list[i],
                    boundary_T_single_case_list[i],
                    boundary_L_single_case_list[i],
                    boundary_W_single_case_list[i],
                    boundary_G_single_case_list[i],
                    k1,
                    k2)
            u = u.reshape([T_n_points, -1])
            u_boundary_list.append(u)


        loss_single_case = []
        j = 0
        for i in range(len(flux_predictor_inputs_Ni_single_case_list)): # loop over all internal junctions
            while(sum(segment is not None for segment in node_segments_list_single_case[j]) == 1): # skip boundary nodes which are not internal junctions
                j+=1
            connected_segments = node_segments_list_single_case[j]
            junction_stress_list = []
            for l,segment in enumerate(connected_segments):
                if(segment is not None):
                    if(l==0 or l==3): # left or down, append stress on right end to the stress_list
                        junction_stress_list.append(u_boundary_list[segment][:,1])
                    elif(l==1 or l==2): # up or right, append stress on left end to the stress_list
                        junction_stress_list.append(u_boundary_list[segment][:,0])
            for l in range(len(junction_stress_list)-1):
                loss = junction_stress_list[l] - junction_stress_list[l+1]
                loss = (loss**2).mean()
                loss_single_case.append(loss)
            j+=1

        loss = loss_single_case[0]
        for l in loss_single_case[1:]:
            loss += l

        loss_average += loss.item()

        loss.backward()
        # print(mynet.il.weight.grad)

        optimizer.step()

    if args.timer:
        tmr_end = timer()
        total_training_time += tmr_end - tmr_start
        print(f" Epoch time: {str(tmr_end - tmr_start)}")
        print(f" Total time: {str(total_training_time)}")

    # lr scheduling
    step = epoch
    pct = step / args.n_epochs
    if(args.lr_pct >= 0):
        lr = scheduler.step(pct)
        adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr

    loss_average /= len(testing_list)
    loss_list.append([loss_average])
    if not args.timer:
        with open(args.run_dir+'/traning_loss_list.csv', "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(loss_list)

    print(f"Epoch {epoch}:"\
            f" mean training loss: {loss_average:.6f}, lr: {lr:.6f}")

    if epoch !=0 and (epoch % args.save_freq == 0 or epoch == args.n_epochs-1):
        for case in range(len(testing_list)):
            # Flux prediction
            node_segments_list_single_case = node_segments_list[case]
            segment_endopoints_and_direction_single_case = segment_endopoints_and_direction_list[case]
            flux_predictor_inputs_Ni_single_case_list = flux_predictor_inputs_Ni_list[case]
            flux_predictor_inputs_Gl_single_case_list = flux_predictor_inputs_Gl_list[case]
            flux_predictor_inputs_Gu_single_case_list = flux_predictor_inputs_Gu_list[case]
            flux_predictor_inputs_Gr_single_case_list = flux_predictor_inputs_Gr_list[case]
            flux_predictor_inputs_Gd_single_case_list = flux_predictor_inputs_Gd_list[case]
            flux_predictor_inputs_T_single_case_list = flux_predictor_inputs_T_list[case]

            stress_predictor_inputs_k_single_case_list = [[torch.zeros(T_n_points,1,device=device),torch.zeros(T_n_points,1,device=device)] for _ in range(len(boundary_X_list[case]))] # initialize all [k1,k2] to zeros with shape of [T_n_points,1]
            j = 0
            for i in range(len(flux_predictor_inputs_Ni_single_case_list)):
                k = flux_predictor(flux_predictor_inputs_Ni_single_case_list[i],
                                    flux_predictor_inputs_Gl_single_case_list[i],
                                    flux_predictor_inputs_Gu_single_case_list[i],
                                    flux_predictor_inputs_Gr_single_case_list[i],
                                    flux_predictor_inputs_Gd_single_case_list[i],
                                    flux_predictor_inputs_T_single_case_list[i])

                # Fill predicted k values into stress_predictor_inputs_k_single_case_list
                while(sum(segment is not None for segment in node_segments_list_single_case[j]) == 1): # skip boundary nodes which are not internal junctions
                    j+=1;
                connected_segments = node_segments_list_single_case[j]
                connected_segments_n = sum(segment is not None for segment in connected_segments)
                flux_sum = torch.zeros(T_n_points,1) # kl-ku-kr+kd=0
                flux_sum = flux_sum.to(device)
                if(connected_segments[0] is not None): # left
                    stress_predictor_inputs_k_single_case_list[connected_segments[0]][1] = k[:,0:1]
                    flux_sum += k[:,0:1]
                    connected_segments_n -= 1
                if(connected_segments[1] is not None): # up
                    if(connected_segments_n==1):
                        stress_predictor_inputs_k_single_case_list[connected_segments[1]][0] = flux_sum
                    else:
                        stress_predictor_inputs_k_single_case_list[connected_segments[1]][0] = k[:,1:2]
                        flux_sum -= k[:,1:2]
                    connected_segments_n -= 1
                if(connected_segments[2] is not None): # right
                    if(connected_segments_n==1):
                        stress_predictor_inputs_k_single_case_list[connected_segments[2]][0] = flux_sum
                    else:
                        stress_predictor_inputs_k_single_case_list[connected_segments[2]][0] = k[:,2:3]
                        flux_sum -= k[:,2:3]
                    connected_segments_n -= 1
                if(connected_segments[3] is not None): # down
                    stress_predictor_inputs_k_single_case_list[connected_segments[3]][1] = -flux_sum
                
                j+=1

            valid_X_single_case_list = valid_X_list[case]
            valid_T_single_case_list = valid_T_list[case]
            valid_L_single_case_list = valid_L_list[case]
            valid_W_single_case_list = valid_W_list[case]
            valid_G_single_case_list = valid_G_list[case]
            valid_k1_single_case_list = valid_k1_list[case]
            valid_k2_single_case_list = valid_k2_list[case]
            valid_truth_single_case_list = valid_truth_list[case]

            u_valid_list = []
            if args.timer:
                total_inference_time = 0
            for i in range(len(valid_X_single_case_list)):
                k1, k2 = stress_predictor_inputs_k_single_case_list[i]

                # X_n_points = 2
                X_n_points = int(valid_X_single_case_list[i].shape[0]/T_n_points)
                k1_cat = [k1 for _ in range(X_n_points)]
                k2_cat = [k2 for _ in range(X_n_points)]

                k1 = torch.cat(k1_cat,dim=1) # shape=[T_n_points, L_n_points]
                k2 =  torch.cat(k2_cat,dim=1) # shape=[T_n_points, L_n_points]

                k1 = torch.reshape(k1, [-1,1])
                k2 = torch.reshape(k2, [-1,1])
                if args.timer:
                    tmr_start = timer()
                u = mynet(valid_X_single_case_list[i],
                        valid_T_single_case_list[i],
                        valid_L_single_case_list[i],
                        valid_W_single_case_list[i],
                        valid_G_single_case_list[i],
                        # valid_k1_single_case_list[i],
                        # valid_k2_single_case_list[i])
                        k1,
                        k2)

                if args.timer:
                    tmr_end = timer()
                    total_inference_time += tmr_end - tmr_start
                    print(f" Segment time: {str(tmr_end - tmr_start)}")
                    print(f" Total time: {str(total_inference_time)}")

                u = to_numpy(u).reshape([T_n_points, -1]).T
                u_valid_list.append(u)

            # RMSE calculation
            u = np.vstack(u_valid_list)
            truth = np.vstack(valid_truth_single_case_list)

            u_orig = (u+1)/2 * stress_range + stress_min
            truth_orig = (truth+1)/2 * stress_range + stress_min

            rmse = np.sqrt(np.mean((u_orig-truth_orig)**2))
            with open(args.run_dir+'/testcase_rmse_list.csv', 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerows([[args.data_path, testing_list[case], rmse]])


            if('1d' in args.fig_format):
                # u = np.vstack(u_valid_list)
                # truth = np.vstack(valid_truth_single_case_list)

                font_size = 24
                fig = plt.figure(figsize=(60, 20))
                gs = gridspec.GridSpec(3, 1)

                # Subplot 1: ground truth
                ax = plt.subplot(gs[0])
                ax.imshow(truth, cmap='rainbow', origin='upper', aspect='auto',  vmin = -1, vmax = 1)
                ax.set_title('Ground Truth',  fontsize = font_size)


                # Subplot 2: Prediction
                ax = plt.subplot(gs[1])
                ax.imshow(u, cmap='rainbow',  origin='upper', aspect='auto', vmin = -1, vmax = 1)
                ax.set_title('Prediction',  fontsize = font_size)


                # Subplot 3: 3 time instances stress
                ax = plt.subplot(gs[2])
                stress_end_truth = truth[:,1]
                stress_end_pred = u[:,1]
                ax.plot(stress_end_truth,'-', label = 'FEM: t=1E4')
                ax.plot(stress_end_pred,'--', label='Proposed: t=1E4')

                stress_end_truth = truth[:,10]
                stress_end_pred = u[:,10]
                ax.plot(stress_end_truth,'-', label = 'FEM: t=1E5')
                ax.plot(stress_end_pred,'--', label='Proposed: t=1E5')

                stress_end_truth = truth[:,100]
                stress_end_pred = u[:,100]
                ax.plot(stress_end_truth,'-', label = 'FEM: t=1E6')
                ax.plot(stress_end_pred,'--', label='Proposed: t=1E6')

                # ax.legend(loc='upper right')
                ax.legend(loc='lower right')
                ax.set_title('FEM vs Proposed',  fontsize = font_size)

                fig.suptitle(args.hparams)

                # plt.show()
                plt.savefig('{}/truth_vs_pred_{}_epoch_{}_1d.png'.format(args.run_dir, str(testing_list[case]), str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)
            
            if('2d' in args.fig_format):
                v_max = 1
                v_min = -1
                max_size = 256
                max_size = int(max_size * 2)
                true_stress_map = np.ones([max_size,max_size]) * float("nan")
                pred_stress_map = np.ones([max_size,max_size]) * float("nan")
                for segment in range(len(u_valid_list)):
                    stress_pred = u_valid_list[segment][:,-1]
                    stress_true = valid_truth_single_case_list[segment][:,-1]

                    start_point = segment_endopoints_and_direction_single_case[segment][0]
                    end_point = segment_endopoints_and_direction_single_case[segment][1]
                    
                    start_point = (np.array(start_point) * 2).astype(np.int)
                    end_point = (np.array(end_point) * 2).astype(np.int)
                    
                    segment_length = max(end_point-start_point)
                    segment_width = 1
                    segment_width = int(segment_width * 2)

                    x = np.linspace(0, segment_length, len(stress_true))
                    interp_points = np.linspace(0,segment_length, segment_length+1)

                    f = interpolate.interp1d(x,stress_true)
                    stress_true_interp = [f(point).item() for point in interp_points]

                    f = interpolate.interp1d(x,stress_pred)
                    stress_pred_interp = [f(point).item() for point in interp_points]

                    if(start_point[0]==end_point[0]): # wire is vertical
                        for n_width in range(segment_width):
                            true_stress_map[start_point[0] - int(segment_width/2) + n_width, start_point[1]:end_point[1]+1] = stress_true_interp
                            pred_stress_map[start_point[0] - int(segment_width/2) + n_width, start_point[1]:end_point[1]+1] = stress_pred_interp
                    else: # wire is horizontal
                        for n_width in range(segment_width):
                            true_stress_map[start_point[0]:end_point[0]+1, start_point[1] - int(segment_width/2) + n_width] = stress_true_interp
                            pred_stress_map[start_point[0]:end_point[0]+1, start_point[1] - int(segment_width/2) + n_width] = stress_pred_interp
                
                # 2D image RMSE calculation
                pred_stress_map_orig = (pred_stress_map+1)/2 * stress_range + stress_min
                true_stress_map_orig = (true_stress_map+1)/2 * stress_range + stress_min

                rmse = rmse_nan_array(pred_stress_map_orig, true_stress_map_orig)
                with open(args.run_dir+'/testcase_2d_image_rmse_list.csv', 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows([[args.data_path, testing_list[case], rmse]])

                fig = plt.figure(figsize=(60, 60))
                ax = plt.subplot(111)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)
                im = ax.imshow(pred_stress_map, cmap='rainbow', vmax = v_max, vmin = v_min)
                # im = ax.imshow(pred_stress_map, cmap='rainbow', interpolation='nearest')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                plt.colorbar(im, cax=cax)
                # plt.show()
                plt.savefig('{}/truth_vs_pred_{}_epoch_{}_pred_2d.png'.format(args.run_dir, str(testing_list[case]), str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)

                fig = plt.figure(figsize=(60, 60))
                ax = plt.subplot(111)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)
                im = ax.imshow(true_stress_map, cmap='rainbow', vmax = v_max, vmin = v_min)
                # im = ax.imshow(true_stress_map, cmap='rainbow', interpolation='nearest')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                plt.colorbar(im, cax=cax)
                # plt.show()
                plt.savefig('{}/truth_vs_pred_{}_epoch_{}_true_2d.png'.format(args.run_dir, str(testing_list[case]), str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)


            if('3d' in args.fig_format):
                # t=1e6
                font_size = 24
                fig = plt.figure(figsize=(60, 20))
                ax = plt.axes(projection='3d')
                for segment in range(len(u_valid_list)):
                    # if(segment not in [6,8,10,12,14,17,16,18]):
                    #     continue
                    z_pred_value = u_valid_list[segment][:,-1]
                    z_truth_value = valid_truth_single_case_list[segment][:,-1]

                    start_point = segment_endopoints_and_direction_single_case[segment][0]
                    end_point = segment_endopoints_and_direction_single_case[segment][1]

                    x_idx = np.linspace(start_point[0], end_point[0], num=z_pred_value.shape[0], endpoint=True)
                    y_idx = np.linspace(start_point[1], end_point[1], num=z_pred_value.shape[0], endpoint=True)

                    if segment == 0:
                        label_r = 'FEM: t=1E6'
                        label_p = 'Proposed: t=1E6'
                    else:
                        label_r = None
                        label_p = None
                    # ax.plot3D(x_idx, y_idx, z_truth_value, '-', color = 'dodgerblue', label = label_r, linewidth=2)
                    ax.plot3D(x_idx, y_idx, z_truth_value, 'r-', label = label_r)
                    # ax.scatter(x_idx, y_idx, z_truth_value, c=z_pred_value, cmap='Blues', alpha=1, label='Prediction')

                    # ax.plot3D(x_idx, y_idx, z_pred_value, '--', color='orange', label = label_p, linewidth=2)
                    ax.scatter(x_idx, y_idx, z_pred_value, color = 'b', marker = 'o', alpha=0.2, label=label_p)
                    # ax.plot3D(x_idx, y_idx, z_pred_value, 'b.', alpha=0.1, label='Prediction')
                    # ax.scatter(x_idx, y_idx, z_pred_value, c=z_pred_value, marker = 'o', cmap='Blues', alpha=0.5, label='Prediction')

                ax.legend(loc='upper right', fontsize = font_size)
                ax.set_title('FEM vs Proposed', fontsize = font_size)

                fig.suptitle(args.hparams)

                # plt.show()
                plt.savefig('{}/truth_vs_pred_{}_epoch_{}_3d.png'.format(args.run_dir, str(testing_list[case]), str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)


