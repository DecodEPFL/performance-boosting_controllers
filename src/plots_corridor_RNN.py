import torch
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.utils import set_params, count_parameters
from src.models import SystemRobots, ControllerRNN
from src.plots import plot_trajectories, plot_traj_vs_time
from src.loss_functions import f_loss_states, f_loss_u, f_loss_ca, f_loss_obst, f_loss_side, f_loss_barrier_up


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rectangle = False
mass = 1.0
model_mismatch = False

# ------ IMPORTANT ------
plot_zero_c = False
plot_c = True
plot_gif = False
calculate_loss = True

prefix = '_RNNnonstab'

# ------------------------

sys_model = 'corridor'

is_linear = False
t_end = 100
std_ini = 0.5
n_train = 100
random_seed = 3
use_sp = False

t_ext = t_end * 4
# ------------------------
torch.manual_seed(random_seed)
exp_name = sys_model + prefix
f_name = exp_name + '_T' + str(t_end) + '_stdini' + str(std_ini) + '_RS' + str(random_seed) +'.pt'
params = set_params(sys_model)
min_dist, t_end, n_agents, x0, xbar, _, _, _, Q, alpha_u, alpha_ca, alpha_obst, n_xi, l, _, _ = params
n_xi, l = 10, 11

# ------------ 0. Load ------------
# load data
file_path = os.path.join(BASE_DIR, 'data', sys_model)
f_data = 'data_' + sys_model + '_stdini' + str(std_ini) + '_agents' + str(n_agents)
f_data += '_RS' + str(random_seed) + '.pkl'
print("Loading data from %s ..." % f_data)
f_data = os.path.join(file_path, f_data)
filehandler = open(f_data, 'rb')
data_saved = pickle.load(filehandler)
filehandler.close()
# load model
if plot_c or calculate_loss:
    fname = exp_name + '_T' + str(t_end) + '_stdini' + str(std_ini) + '_RS' + str(random_seed)
    fname += '.pt'
    print("Loading model data from %s ..." % fname)
    filename = os.path.join(BASE_DIR, 'trained_models', fname)
    model_data = torch.load(filename)
    assert model_data['n_xi'] == n_xi
    assert model_data['l'] == l
    assert model_data['use_sp'] == use_sp
    assert model_data['linear'] == is_linear

# ------------ 1. Dataset ------------
# assert data_saved['t_end'] >= t_end and data_saved['t_end'] >= t_ext
train_x0 = data_saved['data_x0'][:n_train, :]
assert train_x0.shape[0] == n_train
test_x0 = data_saved['data_x0'][n_train:, :]
train_points = train_x0 + x0.detach().repeat(n_train,1)

# ------------ 2. Models ------------
sys = SystemRobots(xbar, is_linear)
if plot_c or calculate_loss:
    ctl = ControllerRNN(sys.f, sys.n, sys.m, n_xi, l, use_sp=use_sp, t_end_sp=t_end, std_ini_param=0.1, stab=False)
    ctl.psi_u.load_state_dict(model_data['psi_u'])
    ctl.psi_u.eval()
    if use_sp:
        ctl.sp.load_state_dict(model_data['sp'])
        ctl.sp.eval()
    ctl.psi_u.set_model_param()
    print("The model has %i parameters" % count_parameters(ctl))

# ------------ 3. Plots ------------
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
# Simulate trajectory for zero controller
if plot_zero_c:
    print("Generating plot for zero controller...")
    x_zero1, x_zero2, x_zero3 = torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n)
    w_in1, w_in2, w_in3 = torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n)
    w_in1[0, :] = (x0.detach() - sys.xbar) + test_x0[4]
    w_in2[0, :] = (x0.detach() - sys.xbar) + test_x0[5]
    w_in3[0, :] = (x0.detach() - sys.xbar) + test_x0[6]
    u = torch.zeros(sys.m)
    x_1, x_2, x_3 = sys.xbar, sys.xbar, sys.xbar
    for t in range(t_ext):
        x_1, _ = sys(t, x_1, u, w_in1[t, :])
        x_2, _ = sys(t, x_2, u, w_in2[t, :])
        x_3, _ = sys(t, x_3, u, w_in3[t, :])
        x_zero1[t, :], x_zero2[t, :], x_zero3[t, :] = x_1.detach(), x_2.detach(), x_3.detach()
    # plot trajectory
    tp = 26
    plot_trajectories(x_zero1, xbar, sys.n_agents, text="", obst=1, circles=False, axis=False, T=0)
    plot_trajectories(x_zero2, xbar, sys.n_agents, text="", obst=False, circles=False, axis=False, T=0)
    plot_trajectories(x_zero3, xbar, sys.n_agents, text="", obst=False, circles=True, axis=True, T=tp)
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
    plt.text(0., 4., r'Pre-stabilized system', dict(size=25), ha='center', va='top')
    plt.text(2.9, -2.9, r'$\tau = %d$' % tp, dict(size=25), ha='right')
    # save figure
    f_figure = 'c_OL' + prefix
    if model_mismatch:
        f_figure += '_mass' + str(mass)
    f_figure += '_T' + str(t_end) + '_S' + str(n_train) + '_stdini' + str(std_ini) + '_RS' + str(random_seed)
    f_figure += '_tp' + str(tp) + '.pdf'
    filename_figure = os.path.join(BASE_DIR, 'figures', f_figure)
    plt.savefig(filename_figure, format='pdf')
    plt.close()

# Simulate trajectories for the NN controller
if plot_c:
    print("Generating plot for trained controller...")
    x_log1, x_log2, x_log3 = torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n)
    u_log1 = torch.zeros(t_ext, sys.m)
    w_in1, w_in2, w_in3 = torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n), torch.zeros(t_ext, sys.n)
    w_in1[0, :] = (x0.detach() - sys.xbar) + test_x0[4]
    w_in2[0, :] = (x0.detach() - sys.xbar) + test_x0[5]
    w_in3[0, :] = (x0.detach() - sys.xbar) + test_x0[6]
    u_1, u_2, u_3 = torch.zeros(sys.m), torch.zeros(sys.m), torch.zeros(sys.m)
    x_1, x_2, x_3 = sys.xbar, sys.xbar, sys.xbar
    xi_1, xi_2, xi_3 = torch.zeros(ctl.psi_u.n_xi), torch.zeros(ctl.psi_u.n_xi), torch.zeros(ctl.psi_u.n_xi)
    omega_1, omega_2, omega_3 = (x_1, u_1), (x_2, u_2), (x_3, u_3)
    for t in range(t_ext):
        x_1, _ = sys(t, x_1, u_1, w_in1[t, :])
        x_2, _ = sys(t, x_2, u_2, w_in2[t, :])
        x_3, _ = sys(t, x_3, u_3, w_in3[t, :])
        u_1, xi_1, omega_1 = ctl(t, x_1, xi_1, omega_1)
        u_2, xi_2, omega_2 = ctl(t, x_2, xi_2, omega_2)
        u_3, xi_3, omega_3 = ctl(t, x_3, xi_3, omega_3)
        x_log1[t, :], x_log2[t, :], x_log3[t, :] = x_1.detach(), x_2.detach(), x_3.detach()
        u_log1[t, :] = u_1.detach()
    for idx,x in enumerate([x_log1,x_log2,x_log3]):
        tp = 99
        # plot trajectories
        plot_trajectories(x, xbar, sys.n_agents, text="", obst=1, circles=True, axis=True, T=tp)
        # plot points of initial conditions
        plt.plot(train_points[:, 0], train_points[:, 1], 'o', color='tab:blue', alpha=0.1)
        plt.plot(train_points[:, 4], train_points[:, 5], 'o', color='tab:orange', alpha=0.1)
        # plot nominal initial condition
        plt.plot(-2, -2, 'x', color='tab:orange', alpha=0.9)
        plt.plot(2, -2, 'x', color='tab:blue', alpha=0.9)
        # adjust the figure
        fig = plt.gcf()
        fig.set_size_inches(6, 7)
        plt.axis('equal')
        plt.tight_layout()
        ax = plt.gca()
        if rectangle:
            rect = Rectangle((-4, 2.55), 8, 3, linewidth=0, fill=True, facecolor=(0.5,0.5,0.5,0.2))
            ax.add_patch(rect)
        ax.set_xlim([-3.05, 3.05])
        ax.set_ylim([-3.05, 4.05])
        # plt.text(0., 4., r'75\% trained controller', dict(size=25), ha='center', va='top')
        # plt.text(0., 4., r'Trained controller', dict(size=25), ha='center', va='top')
        plt.text(2.9, -2.9, r'$\tau = %d$' % tp, dict(size=25), ha='right')
        # text = r'$(c)$'
        # plt.text(0., -2.9, text, dict(size=25), ha='center')
        # save figure
        f_figure = 'c_CL' + prefix
        if model_mismatch:
            f_figure += '_mass' + str(mass)
        f_figure += '_T' + str(t_end) + '_S' + str(n_train) + '_stdini' + str(std_ini) + '_RS' + str(random_seed)
        f_figure += 'tp' + str(tp) + 'idx' + str(idx) + '.pdf'
        filename_figure = os.path.join(BASE_DIR, 'figures', f_figure)
        plt.savefig(filename_figure, format='pdf')
        plt.close()

    fig2 = plot_traj_vs_time(t_ext, n_agents, x_log1, u_log1)

# ------------ 5. Cost ------------
if calculate_loss:
    test_x0_iter = test_x0[100:120, :]
    loss_x, loss_u, loss_ca, loss_obst = 0, 0, 0, 0

    print("Total number of datapoints: %i" % test_x0_iter.shape[0])
    for idx, x0_iter in enumerate(test_x0_iter):
        x_log = torch.zeros(t_end, sys.n)
        u_log = torch.zeros(t_end, sys.m)
        w_in = torch.zeros(t_end, sys.n)
        w_in[0, :] = (x0.detach() - sys.xbar) + x0_iter
        u = torch.zeros(sys.m)
        x = sys.xbar
        xi = torch.zeros(ctl.psi_u.n_xi)
        omega = (x, u)
        for t in range(t_end):
            x, _ = sys(t, x, u, w_in[t, :])
            u, xi, omega = ctl(t, x, xi, omega)
            x_log[t, :] = x.detach()
            u_log[t, :] = u.detach()

            # # Calculate the loss:
            loss_x = loss_x + f_loss_states(t, x_log[t, :], sys, Q)
            loss_u = loss_u + alpha_u * f_loss_u(t, u_log[t, :])
            loss_ca = loss_ca + alpha_ca * f_loss_ca(x_log[t, :], sys, min_dist)
            if alpha_obst != 0:
                loss_obst = loss_obst + alpha_obst * f_loss_obst(x_log[t, :], sys)

    loss = (loss_x + loss_u + loss_ca + loss_obst) / test_x0_iter.shape[0]
    loss_xQx = loss_x / test_x0_iter.shape[0]
    loss_uRu = loss_u / test_x0_iter.shape[0]
    print('Original loss: %.2f' % (loss / t_end))
    print('xQx loss: %.2f' % (loss_xQx / t_end))
    print('uRu loss: %.4f' % (loss_uRu / t_end))
    print('\tLoss x: %.2f --- Loss u: %.2f --- Loss ca: %.2f --- Loss obst: %.2f' % (loss_x,loss_u,loss_ca,loss_obst))
