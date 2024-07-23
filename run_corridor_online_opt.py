import torch
import matplotlib.pyplot as plt
import os
import pickle
import logging
from datetime import datetime

import casadi as cd
import cvxpy as cp
import numpy as np

from src.models import SystemRobots, Controller
from src.plots import plot_trajectories, plot_traj_vs_time
from src.loss_functions import f_loss_states, f_loss_u, f_loss_ca, f_loss_obst, f_loss_side, f_loss_barrier_up
from src.utils import calculate_collisions, set_params, generate_data
from src.utils import WrapLogger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

random_seed = 3
torch.manual_seed(random_seed)

sys_model = "corridor_online_opt"
prefix = ''

# # # # # # # # Parameters and hyperparameters # # # # # # # #
params = set_params("corridor")
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, \
alpha_u, alpha_ca, alpha_obst, n_xi, l, n_traj, std_ini = params

epoch_print = 20
n_train = 100
n_test = 1000 - n_train
validation = True
validation_period = 50
n_validation = 100

std_ini = 0.5

show_plots = False

t_ext = t_end * 4

# # # # # # # # Load data # # # # # # # #
file_path = os.path.join(BASE_DIR, 'data', sys_model)
filename = 'data_' + sys_model + '_stdini' + str(std_ini) + '_agents' + str(n_agents)
filename += '_RS' + str(random_seed) + '.pkl'
filename = os.path.join(file_path, filename)
if not os.path.isfile(filename):
    generate_data(sys_model, t_end*4, n_agents, random_seed, std_ini=std_ini)
assert os.path.isfile(filename)
filehandler = open(filename, 'rb')
data_saved = pickle.load(filehandler)
filehandler.close()
assert data_saved['t_end'] >= t_end and data_saved['t_end'] >= t_ext
train_x0 = data_saved['data_x0'][:n_train, :]
assert train_x0.shape[0] == n_train
test_x0 = data_saved['data_x0'][n_train:, :]
assert test_x0.shape[0] == n_test
validation_x0 = data_saved['data_x0'][n_train:n_train+n_validation, :]
assert validation_x0.shape[0] == n_validation
# ------------ 1. Dataset ------------
train_points = train_x0 + x0.detach().repeat(n_train,1)

# # # # # # # # Set up logger # # # # # # # #
log_name = sys_model + prefix
now = datetime.now().strftime("%m_%d_%H_%Ms")
filename_log = os.path.join(BASE_DIR, 'log')
if not os.path.exists(filename_log):
    os.makedirs(filename_log)
filename_log = os.path.join(filename_log, log_name+'_log_' + now)

logging.basicConfig(filename=filename_log, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger(sys_model)
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# # # # # # # # Define models # # # # # # # #
sys = SystemRobots(xbar, linear)

# # # # # # # # Figures # # # # # # # #
fig_path = os.path.join(BASE_DIR, 'figures', 'temp')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
filename_figure = 'fig_' + log_name
filename_figure = os.path.join(fig_path, filename_figure)

# # # # # # # # Training # # # # # # # #
msg = "\n---------------- Begin ----------------\n"
msg += "Problem: " + sys_model + " -- t_end: %i\n" % t_end
msg += "--------- --------- ---------  ---------"
logger.info(msg)

gamma = 0.9

# Casadi
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 5000}
z = cd.SX.sym('x',8)
sq = cd.Function('f',[z], [z*z], ['x'],['x_sq'])

# Plots
do_plots = False
tp = 35  # 40 50
test_x0_iter = test_x0[100:120, :]

m1 = torch.zeros(sys.m, sys.m)
m1[0:2, 0:2] = torch.eye(2)
m2 = torch.zeros(sys.m, sys.m)
m2[2:4, 0:2] = torch.eye(2)
mask = torch.cat((m1, m2), dim=1)

loss_x, loss_u, loss_ca, loss_obst = 0, 0, 0, 0

print("Total number of datapoints: %i" % test_x0_iter.shape[0])
for idx, x0_iter in enumerate(test_x0_iter):

    w_in = torch.zeros(t_end, sys.n)
    x = x0.detach() + x0_iter

    x_log = torch.zeros(t_end, sys.n)
    u_log = torch.zeros(t_end, sys.m)

    u = 0

    exception = False

    print("Starting %i..." % idx)

    for t in range(t_end):
        # Casadi
        opti = cd.Opti()
        x_cd = opti.variable(sys.n,2)
        u_cd = opti.variable(sys.m,2)

        opti.minimize(u_cd[:,0].T @ u_cd[:,0] )

        x0_eps = np.ones(sys.n) * 1e-9

        # # # # # # # # # CONSTRAINT 1: states # # # # # # # # #

        # # One step ahead:
        fx1 = torch.nn.functional.linear(x - sys.xbar, sys.A).detach().numpy()
        fx2 = (sys.h * sys.b2 / sys.m * sys.mask_nl * torch.tanh(x - sys.xbar)).detach().numpy()
        fx3 = sys.B.detach().numpy() @ u_cd[:,0]
        fx = fx1 + fx2 + fx3
        x_ = (fx + sys.xbar.detach().numpy())
        x_lb_ = x_ - x0_eps
        x_ub_ = x_ + x0_eps
        opti.subject_to(x_lb_ <= x_cd[:,0])
        opti.subject_to(x_cd[:,0] <= x_ub_)

        # # Two step ahead:
        fx1 = sys.A.detach().numpy() @ (x_cd[:, 0] - sys.xbar.detach().numpy())
        fx2 = (sys.h * sys.b2 / sys.m * sys.mask_nl).detach().numpy() * cd.tanh(x_cd[:,0] - sys.xbar.detach().numpy())
        fx3 = sys.B.detach().numpy() @ u_cd[:, 1]
        fx = fx1 + fx2 + fx3
        x__ = (fx + sys.xbar.detach().numpy())
        x_lb__ = x__ - x0_eps
        x_ub__ = x__ + x0_eps
        opti.subject_to(x_lb__ <= x_cd[:, 1])
        opti.subject_to(x_cd[:, 1] <= x_ub__)

        # # # # # # # # # CONSTRAINT 2: Collision avoidance between agents # # # # # # # # #
        delta_x = (x[0] - x[4]).detach().numpy()
        delta_y = (x[1] - x[5]).detach().numpy()
        h_x = np.square(delta_x) + np.square(delta_y) - np.square(2*sys.radius)
        delta_x_ = x_cd[0,0] - x_cd[4,0]
        delta_y_ = x_cd[1,0] - x_cd[5,0]
        h_x_ = sq(delta_x_) + sq(delta_y_) - np.square(2*sys.radius)
        delta_x__ = x_cd[0,1] - x_cd[4,1]
        delta_y__ = x_cd[1,1] - x_cd[5,1]
        h_x__ = sq(delta_x__) + sq(delta_y__) - np.square(2 * sys.radius)
        # opti.subject_to((h_x_ - h_x + gamma*h_x) >= 0)
        opti.subject_to((h_x__ - h_x_ + gamma * h_x_) >= 0)

        # # # # # # # # # CONSTRAINT 3: Collision avoidance with obstacles # # # # # # # # #
        for pos in [1.5,2.5]:
            delta_x1_o1 = (x[0] - pos).detach().numpy()
            delta_x2_o1 = (x[4] - pos).detach().numpy()
            delta_x1_o2 = (x[0] - (-pos)).detach().numpy()
            delta_x2_o2 = (x[4] - (-pos)).detach().numpy()
            delta_y1 = (x[1] - 0).detach().numpy()
            delta_y2 = (x[5] - 0).detach().numpy()
            h_x1_o1 = np.square(delta_x1_o1) + np.square(delta_y1) - np.square(sys.radius_obstacle)
            h_x2_o1 = np.square(delta_x2_o1) + np.square(delta_y2) - np.square(sys.radius_obstacle)
            h_x1_o2 = np.square(delta_x1_o2) + np.square(delta_y1) - np.square(sys.radius_obstacle)
            h_x2_o2 = np.square(delta_x2_o2) + np.square(delta_y2) - np.square(sys.radius_obstacle)
            delta_x1_o1_ = x_cd[0,0] - pos
            delta_x2_o1_ = x_cd[4,0] - pos
            delta_x1_o2_ = x_cd[0,0] - (-pos)
            delta_x2_o2_ = x_cd[4,0] - (-pos)
            delta_y1_ = x_cd[1,0] - 0
            delta_y2_ = x_cd[5,0] - 0
            h_x1_o1_ = sq(delta_x1_o1_) + sq(delta_y1_) - np.square(sys.radius_obstacle)
            h_x2_o1_ = sq(delta_x2_o1_) + sq(delta_y2_) - np.square(sys.radius_obstacle)
            h_x1_o2_ = sq(delta_x1_o2_) + sq(delta_y1_) - np.square(sys.radius_obstacle)
            h_x2_o2_ = sq(delta_x2_o2_) + sq(delta_y2_) - np.square(sys.radius_obstacle)
            delta_x1_o1__ = x_cd[0,1] - pos
            delta_x2_o1__ = x_cd[4,1] - pos
            delta_x1_o2__ = x_cd[0,1] - (-pos)
            delta_x2_o2__ = x_cd[4,1] - (-pos)
            delta_y1__ = x_cd[1,1] - 0
            delta_y2__ = x_cd[5,1] - 0
            h_x1_o1__ = sq(delta_x1_o1__) + sq(delta_y1__) - np.square(sys.radius_obstacle)
            h_x2_o1__ = sq(delta_x2_o1__) + sq(delta_y2__) - np.square(sys.radius_obstacle)
            h_x1_o2__ = sq(delta_x1_o2__) + sq(delta_y1__) - np.square(sys.radius_obstacle)
            h_x2_o2__ = sq(delta_x2_o2__) + sq(delta_y2__) - np.square(sys.radius_obstacle)
            # opti.subject_to((h_x1_o1_ - h_x1_o1 + gamma * h_x1_o1) >= 0)
            # opti.subject_to((h_x2_o1_ - h_x2_o1 + gamma * h_x2_o1) >= 0)
            # opti.subject_to((h_x1_o2_ - h_x1_o2 + gamma * h_x1_o2) >= 0)
            # opti.subject_to((h_x1_o2_ - h_x1_o2 + gamma * h_x1_o2) >= 0)
            opti.subject_to((h_x1_o1__ - h_x1_o1_ + gamma * h_x1_o1_) >= 0)
            opti.subject_to((h_x2_o1__ - h_x2_o1_ + gamma * h_x2_o1_) >= 0)
            opti.subject_to((h_x1_o2__ - h_x1_o2_ + gamma * h_x1_o2_) >= 0)
            opti.subject_to((h_x2_o2__ - h_x2_o2_ + gamma * h_x2_o2_) >= 0)

        opti.solver('ipopt', opts)

        try:
            sol = opti.solve()
            u_cd_sol = sol.value(u_cd[:,0])
            x_sol = sol.value(x_cd[:,0])
        except:
            print("Something went wrong at iteration %i of datapoint %i" % (t,idx))
            u_cd_sol = opti.debug.value(u_cd[:,0])
            x_sol = opti.debug.value(x_cd[:,0])
            exception = True

        h_agent = sys.h_agents(t,x,u, w_in[t, :])
        h_obstacle = sys.h_obstacles(t, x, u, w_in[t, :])

        # apply the input
        u = torch.tensor(u_cd_sol, dtype=torch.float)

        x, _ = sys(t, x, u, w_in[t, :])

        x_log[t,:] = x
        u_log[t,:] = u
        if exception:
            break

        # # Calculate the loss:
        loss_x = loss_x + f_loss_states(t, x_log[t, :], sys, Q)
        loss_u = loss_u + alpha_u * f_loss_u(t, u_log[t, :])
        loss_ca = loss_ca + alpha_ca * f_loss_ca(x_log[t, :], sys, min_dist)
        if alpha_obst != 0:
            loss_obst = loss_obst + alpha_obst * f_loss_obst(x_log[t, :], sys)

if t_end == t_ext:
    print("Caution! Loss is calculated over the wrong time horizon (t_ext). It should be calculated over t_end")
loss = (loss_x + loss_u + loss_ca + loss_obst) / test_x0_iter.shape[0]
loss_xQx = loss_x / test_x0_iter.shape[0]
loss_uRu = loss_u / test_x0_iter.shape[0]
print('Original loss: %.2f' % (loss / t_end))
print('xQx loss: %.2f' % (loss_xQx / t_end))
print('uRu loss: %.4f' % (loss_uRu / t_end))
print('\tLoss x: %.2f --- Loss u: %.2f --- Loss ca: %.2f --- Loss obst: %.2f' % (loss_x, loss_u, loss_ca, loss_obst))

if do_plots:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    # fig = plot_trajectories(x_log, xbar, sys.n_agents, text="CL - after training", T=t-1, obst=2)
    fig = plot_trajectories(x_log, xbar, sys.n_agents, text="", T=tp, obst=1, circles=True, axis=True)
    # plt.show()
    # plot nominal initial condition
    plt.plot(-2, -2, 'x', color='tab:orange', alpha=0.9)
    plt.plot(2, -2, 'x', color='tab:blue', alpha=0.9)
    # adjust the figure
    fig = plt.gcf()
    fig.set_size_inches(6,7)
    plt.axis('equal')
    plt.tight_layout()
    ax = plt.gca()
    ax.set_xlim([-3.05, 3.05])
    ax.set_ylim([-3.05, 4.05])
    text = r'$(c)$'
    plt.text(0., -2.9, text, dict(size=25), ha='center')
    plt.savefig(filename_figure + '_2' + '.pdf', format='pdf')
    plt.close()

    fig2 = plot_traj_vs_time(t_end, n_agents, x_log, u_log)
