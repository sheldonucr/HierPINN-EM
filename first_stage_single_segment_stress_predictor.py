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
from scipy.io import loadmat
import re
import csv

import matplotlib as mpl
# mpl.use('tkagg')
# mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

import argparse
from datetime import datetime
from shutil import copyfile

import random
import math

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
    self.add_argument('--sample-dir', type=str, default="./Samples_first_stage_single_segment_stress_predictor/EMdataset_10seg_1n2", help='directory to save output')
    self.add_argument('--data-path', type=str, default="/fermi_data/shared/wentian/hierpinn_em/EMdataset_10seg_1n2/", help='path to data') 

    self.add_argument('--read-csv', type=bool, default=False, help='Read training and testing data from .csv file')    
    self.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    self.add_argument('--weight-decay', type=float, default=0., help="weight decay")

    self.add_argument('--w-mse', type=float, default=1, help='Weight of MSE loss. (Ground Truth)')

    self.add_argument('--n-epochs', type=float, default=10000, help='Total training epochs')
    self.add_argument('--n-batches', type=int, default=10000, help='Number of batches in every epoch')
    self.add_argument('--cuda', type=int, default=0, choices=[0, 1, 2, 3], help='cuda index')
    self.add_argument('--save-freq', type=int, default=5, help='Model save frequency, i.e. epochs between model save and plot')

  def parse(self):
    args = self.parse_args()

    # Create output dir
    dt = datetime.now()
    args.date = dt.strftime("%Y%m%d%H%M%S")
    args.hparams = f'EM1d_Epochs{args.n_epochs}_{args.date}'
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

mynet_mlp = [7, 256, 512, 1024, 512, 256, 1]
# mynet_mlp = [7, 256, 512, 1024, 2048, 2048, 2048, 2048, 1024, 512, 256, 1]
total_cases = 10000
training_cases = 8000

# Generate trainig and testing datasets
if args.read_csv:
    # read dataset from csv start
    reader = csv.reader(open(args.sample_dir+'/training.csv', "r"), delimiter=",")
    training_list = list(reader)
    training_list = [int(file[0]) for file in training_list]
    random.shuffle(training_list)

    reader = csv.reader(open(args.sample_dir+'/testing.csv', "r"), delimiter=",")
    testing_list = list(reader)
    testing_list = [int(file[0]) for file in testing_list]
    random.shuffle(testing_list)

    reader = csv.reader(open(args.sample_dir+'/statistics.csv', "r"), delimiter=",")
    statistics_list = list(reader)
    statistics_list = [float(value) for value in statistics_list[1]]
    max_time, max_length, max_G, max_k, stress_max, stress_min = statistics_list
    # read dataset from csv end
else:
    files_list = [i for i in range(total_cases)]
    # random.shuffle(files_list)
    training_list = files_list[:training_cases]
    testing_list = files_list[training_cases:]
    np.savetxt(args.run_dir+"/training.csv", training_list, fmt='%i', delimiter=',')
    np.savetxt(args.run_dir+"/testing.csv", testing_list, fmt='%i', delimiter=',')

    max_G = 0.0
    max_k = 0.0
    max_length = 0.0
    max_time = time_length
    stress_max = 0.0
    stress_min = 0.0
    All_k = np.empty(1)
    for file in files_list:
        segment_length_list = []
        G_list = []
        file_path = args.data_path + str(file)

        # Read .geo file for wire geometries
        with open(file_path+".geo") as f:
            lines = f.readlines()
        for line in lines:
            if line[0:9] == 'Rectangle':
                # rect_vertices = re.findall('-*[0-9]+', line)
                # current_segment_length = float(rect_vertices[4]+"e-6")
                rect_vertices = re.findall('-*[0-9]+\.*[0-9]*', line)
                current_segment_length = max(abs(float(rect_vertices[4]+"e-6")),abs(float(rect_vertices[5]+"e-6")))
                segment_length_list.append(current_segment_length)
                max_length = max(max_length, current_segment_length)

        # Read .mat file for stress results and current
        data_mat = loadmat(file_path+".mat")
        current = data_mat['J']
        stress = data_mat['sVC']

        # Current density
        for J in current[0]:
            G = e * Z * rou * J / Omega
            G_list.append(G)
            max_G = max(max_G, abs(G))

        # Stress
        for i in range(len(segment_length_list)):
            truth = stress[i,0] # shape = [L_n_points, T_n_points]
            stress_max = max(stress_max, truth.max())
            stress_min = min(stress_min, truth.min())

            wire_interval = segment_length_list[i]/(truth.shape[0]-1)

            # calculate k1 and k2
            k1 = truth[:2, :]
            k2 = truth[-2:, :]
            k1 = (k1[1,:]-k1[0,:])/wire_interval
            k2 = (k2[1,:]-k2[0,:])/wire_interval
            k1 += G_list[i]
            k2 += G_list[i]
            All_k = np.hstack([All_k, k1, k2])

            max_k =  max(max_k, abs(k1).max(), abs(k2).max())

    # Plot all k values from lowest to largest
    # All_k.sort()
    # plt.plot(All_k, label = f"All k values")
    # plt.show()

    with open(args.run_dir+"/statistics.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["max_time", "max_length", "max_G", "max_k", "stress_max", "stress_min"])
        writer.writerow([max_time, max_length, max_G, max_k, stress_max, stress_min])

# training_list = training_list[:10]

# params for domain
stress_range = stress_max - stress_min

# Read training data and convert to input arrays
idx_MSE_list = []
truth_MSE_list = []
idx_MSE = np.empty([0,7])
truth_MSE = np.empty([0,1])
# file_num = 0
for file in training_list:
    # file_num += 1
    # print(file_num)
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
    n_segments = len(segment_length_list)
    L_list = [L/max_length for L in segment_length_list]

    # Read .mat file for stress results and current
    data_mat = loadmat(file_path+".mat")
    current = data_mat['J']
    stress = data_mat['sVC']

    # Current density
    for J in current[0]:
        G = e * Z * rou * J / Omega
        G_list.append(G)

    # Stress
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
        # k1_list.append(k1)
        # k2_list.append(k2)

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
        # plt.plot(k1, label = f"k1_{i}")
        # plt.plot(k2, label = f"k2_{i}")
        k1 = np.tile(k1,[truth.shape[0],1]).T # shape=[T_n_points, L_n_points]
        k2 = np.tile(k2,[truth.shape[0],1]).T # shape=[T_n_points, L_n_points]

        idx = [X,T,L,W,G,k1,k2]

        idx_MSE_single_segment = [np.expand_dims(col[:,:].flatten(), axis=-1) for col in idx] # ALL POINTS
        idx_MSE_single_segment = np.hstack(idx_MSE_single_segment)
        truth_MSE_single_segment = np.expand_dims(truth.T.flatten(), axis=-1)

        idx_MSE_single_case = np.vstack([idx_MSE_single_case, idx_MSE_single_segment])
        truth_MSE_single_case = np.vstack([truth_MSE_single_case, truth_MSE_single_segment])

    idx_MSE_list.append(idx_MSE_single_case)
    truth_MSE_list.append(truth_MSE_single_case)

    # plt.show()
    # mpl.use('Agg')

    # # Plot k values for all segments
    # for i in range(n_segments):
    #     plt.plot(k1_list[i], label = f"k1_{i}")
    #     plt.plot(k2_list[i], label = f"k2_{i}")
    # plt.legend()
    # plt.show()
idx_MSE = np.vstack(idx_MSE_list)
truth_MSE = np.vstack(truth_MSE_list)


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

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
print(mynet)
mynet = mynet.to(device)

optimizer = optim.Adam(mynet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
MSE_batches = list(range(idx_MSE.shape[0]))
MSE_bz = int(idx_MSE.shape[0]/args.n_batches)
random.shuffle(MSE_batches)

# Validation grid
if(len(testing_list)>5):
    testing_list = testing_list[-5:]
# testing_list.append(1)
valid_truth_list = []
valid_X_list = []
valid_T_list = []
valid_L_list = []
valid_W_list = []
valid_G_list = []
valid_k1_list = []
valid_k2_list = []
for file in testing_list:
    valid_truth_single_case_list = []
    valid_X_single_case_list = []
    valid_T_single_case_list = []
    valid_L_single_case_list = []
    valid_W_single_case_list = []
    valid_G_single_case_list = []
    valid_k1_single_case_list = []
    valid_k2_single_case_list = []
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
            # rect_vertices = re.findall('-*[0-9]+', line)
            # current_segment_length = float(rect_vertices[4]+"e-6")
            rect_vertices = re.findall('-*[0-9]+\.*[0-9]*', line)
            current_segment_length = max(abs(float(rect_vertices[4]+"e-6")),abs(float(rect_vertices[5]+"e-6")))
            segment_length_list.append(current_segment_length)
    n_segments = len(segment_length_list)
    L_list = [L/max_length for L in segment_length_list]

    # Read .mat file for stress results and current
    data_mat = loadmat(file_path+".mat")
    current = data_mat['J']
    stress = data_mat['sVC']

    # Current density
    for J in current[0]:
        G = e * Z * rou * J / Omega
        G_list.append(G)

    # Stress
    idx_MSE_list = []
    truth_MSE_list = []
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
        # k1_list.append(k1)
        # k2_list.append(k2)

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
        # plt.plot(k1, label = f"k1_{i}")
        # plt.plot(k2, label = f"k2_{i}")
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

# Training begins
last_saved_epoch = 0
loss_list = []
for epoch in range(args.n_epochs):
    mynet.train()
    random.shuffle(MSE_batches)
    loss_average = 0
    for batch in range(args.n_batches):
        # print(f"Epoch : {epoch}, Batch: {batch}")
        MSE_btch = idx_MSE[MSE_batches[batch*MSE_bz:(batch+1)*MSE_bz]]
        truth_MSE_btch = truth_MSE[MSE_batches[batch*MSE_bz:(batch+1)*MSE_bz]]

        MSE_btch = torch.FloatTensor(MSE_btch).to(device)
        truth_MSE_btch = torch.FloatTensor(truth_MSE_btch).to(device)

        mynet.zero_grad()

        loss_mse = MSE_loss(MSE_btch, truth_MSE_btch)

        loss = args.w_mse*loss_mse

        loss_average += loss.item()

        loss.backward()
        # print(mynet.il.weight.grad)
        optimizer.step()

    loss_average /= args.n_batches
    loss_list.append([loss_average])
    with open(args.run_dir+'/image_rmse_list.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(loss_list)

    print(f"Epoch {epoch}:"\
            f" mean training loss: {loss_average:.6f},"\
            f" training loss: {loss:.6f}, MSE: {loss_mse.item():.6f}")

    if epoch !=0 and (epoch % args.save_freq == 0 or epoch == args.n_epochs-1):
        if os.path.exists(args.run_dir + '/trial_function_mlp_{}.pkl'.format(last_saved_epoch)):
            os.remove(args.run_dir + '/trial_function_mlp_{}.pkl'.format(last_saved_epoch))

        torch.save(mynet.state_dict(), args.run_dir + '/trial_function_mlp_{}.pkl'.format(epoch))
        print('Model saved')
        
        last_saved_epoch = epoch

        for k in range(len(testing_list)):
            valid_X_single_case_list = valid_X_list[k]
            valid_T_single_case_list = valid_T_list[k]
            valid_L_single_case_list = valid_L_list[k]
            valid_W_single_case_list = valid_W_list[k]
            valid_G_single_case_list = valid_G_list[k]
            valid_k1_single_case_list = valid_k1_list[k]
            valid_k2_single_case_list = valid_k2_list[k]
            valid_truth_single_case_list = valid_truth_list[k]

            u_valid_list = []
            for i in range(len(valid_X_single_case_list)):
                u = mynet(valid_X_single_case_list[i],
                        valid_T_single_case_list[i],
                        valid_L_single_case_list[i],
                        valid_W_single_case_list[i],
                        valid_G_single_case_list[i],
                        valid_k1_single_case_list[i],
                        valid_k2_single_case_list[i])
                u = to_numpy(u).reshape([T_n_points, -1]).T
                u_valid_list.append(u)

            u = np.vstack(u_valid_list)
            truth = np.vstack(valid_truth_single_case_list)

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

            # Subplot 3: Final stress
            ax = plt.subplot(gs[2])
            stress_end_truth = truth[:,-1]
            stress_end_pred = u[:,-1]
            ax.plot(stress_end_truth,'r-', label = 'Ground Truth')
            ax.plot(stress_end_pred,'b.', label='Prediction')
            ax.legend(loc='upper right')
            ax.set_title('Final stress',  fontsize = font_size)


            fig.suptitle(args.hparams)

            # plt.show()
            plt.savefig('{}/truth_vs_pred_{}_epoch_{}.png'.format(args.run_dir, str(testing_list[k]), str(epoch).zfill(4)), bbox_inches='tight')
            plt.close(fig)

