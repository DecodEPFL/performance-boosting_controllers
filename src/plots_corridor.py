import torch
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.utils import set_params
from src.models import SystemRobots, Controller
from src.plots import plot_trajectories


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------ IMPORTANT ------
plot_zero_c = False
plot_c = True
plot_gif = False

time_plot = [16, 26, 100]

prefix = ''
mass = 1.0
model_mismatch = False
rectangle = False

# prefix = ''
# mass = 1.05
# model_mismatch = True
# rectangle = False


# prefix = '_barrier1'  # RS = 0
# mass = 1.0
# model_mismatch = False
# rectangle = True

# prefix = '_train75'
# mass = 1.0
# model_mismatch = False
# rectangle = False

sys_model = 'corridor'

is_linear = False
t_end = 100
std_ini = 0.5
n_agents = 2
n_train = 100
random_seed = 3
use_sp = False

t_ext = t_end * 4
# ------------------------
torch.manual_seed(random_seed)
exp_name = sys_model + prefix
f_name = exp_name + '_T' + str(t_end) + '_stdini' + str(std_ini) + '_RS' + str(random_seed) +'.pt'
params = set_params(sys_model)
min_dist, t_end, n_agents, x0, xbar, _, _, _, _, _, _, _, n_xi, l, _, _ = params
n_xi, l = 8,8

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
if plot_c:
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
if plot_c:
    if model_mismatch:
        sys_big_m = SystemRobots(xbar, is_linear, mass=mass)  # Here we do model mismatch
        ctl = Controller(sys_big_m.f, sys.n, sys.m, n_xi, l, use_sp=use_sp, t_end_sp=t_end)
    else:
        ctl = Controller(sys.f, sys.n, sys.m, n_xi, l, use_sp=use_sp, t_end_sp=t_end)
    ctl.psi_u.load_state_dict(model_data['psi_u'])
    ctl.psi_u.eval()
    if use_sp:
        ctl.sp.load_state_dict(model_data['sp'])
        ctl.sp.eval()
    ctl.psi_u.set_model_param()

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
    if rectangle:
        rect = Rectangle((-4,2.05),8,3,linewidth=0,fill=True, facecolor='tab:gray', alpha=0.5)
        ax.add_patch(rect)
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
    for idx,tp in enumerate(time_plot):
        # plot trajectories
        plot_trajectories(x_log1, xbar, sys.n_agents, text="", obst=1, circles=False, axis=False, T=0)
        plot_trajectories(x_log2, xbar, sys.n_agents, text="", obst=False, circles=False, axis=False, T=0)
        plot_trajectories(x_log3, xbar, sys.n_agents, text="", obst=False, circles=True, axis=True, T=tp)
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
        plt.text(0., 4., r'Trained controller', dict(size=25), ha='center', va='top')
        plt.text(2.9, -2.9, r'$\tau = %d$' % tp, dict(size=25), ha='right')
        # save figure
        f_figure = 'c_CL' + prefix
        if model_mismatch:
            f_figure += '_mass' + str(mass)
        f_figure += '_T' + str(t_end) + '_S' + str(n_train) + '_stdini' + str(std_ini) + '_RS' + str(random_seed)
        f_figure += 'tp' + str(tp) + '.pdf'
        filename_figure = os.path.join(BASE_DIR, 'figures', f_figure)
        plt.savefig(filename_figure, format='pdf')
        plt.close()

# ------------ 5. GIFs ------------

# Base controller
if plot_zero_c and plot_gif:
    for idx, x in enumerate([x_zero1,x_zero2,x_zero3]):
        print("Generating figures for OL trajectory %d..." % (idx+1))
        for tp in range(1, t_end):
            ob_print = True
            if idx > 0:
                plot_trajectories(x_zero1, xbar, sys.n_agents, text="", obst=ob_print, circles=False, T=1)
                ob_print = False
            if idx == 2:
                plot_trajectories(x_zero2, xbar, sys.n_agents, text="", circles=False, T=1)
            plot_trajectories(x, xbar, sys.n_agents, text="", obst=ob_print, circles=True, T=tp)
            # plot points of initial conditions
            # plt.plot(train_points[:, 0], train_points[:, 1], 'o', color='tab:blue', alpha=0.1)
            # plt.plot(train_points[:, 4], train_points[:, 5], 'o', color='tab:orange', alpha=0.1)
            # adjust the figure
            fig = plt.gcf()
            fig.set_size_inches(6, 7)
            plt.axis('equal')
            plt.tight_layout()
            ax = plt.gca()
            ax.set_xlim([-3.05, 3.05])
            ax.set_ylim([-3.05, 4.05])
            plt.text(0., 4., r'Pre-stabilized system', dict(size=25), ha='center', va='top')
            plt.text(1.85, -2.9, r'$\tau = %02d$' % tp, dict(size=25), ha='left')
            f_gif = "ol-%03i" % (tp + 100*idx) + ".png"
            filename_gif = os.path.join(BASE_DIR, 'gif', f_gif)
            plt.savefig(filename_gif)
            plt.close(fig)
    print("Generating OL gif...")
    filename_figs = os.path.join(BASE_DIR, 'gif', "ol-*.png")
    filename_gif = os.path.join(BASE_DIR, 'gif', "ol.gif")
    command = "convert -delay 4 -loop 0 " + filename_figs + " " + filename_gif
    os.system(command)
    print("Gif saved at %s" % filename_gif)
    print("Deleting figures...")
    command = "rm " + filename_figs
    os.system(command)

# Empirical controller
if plot_c and plot_gif:
    for idx, x in enumerate([x_log1,x_log2,x_log3]):
        print("Generating figures for emp trajectory %d..." % (idx+1))
        for tp in range(1, t_end):
            ob_print = True
            if idx > 0:
                plot_trajectories(x_log1, xbar, sys.n_agents, text="", obst=ob_print, circles=False, T=1)
                ob_print = False
            if idx == 2:
                plot_trajectories(x_log2, xbar, sys.n_agents, text="", circles=False, T=1)
            plot_trajectories(x, xbar, sys.n_agents, text="", obst=ob_print, circles=True, T=tp)
            # plot points of initial conditions
            plt.plot(train_points[:, 0], train_points[:, 1], 'o', color='tab:blue', alpha=0.1)
            plt.plot(train_points[:, 4], train_points[:, 5], 'o', color='tab:orange', alpha=0.1)
            # adjust the figure
            fig = plt.gcf()
            fig.set_size_inches(6, 7)
            plt.axis('equal')
            plt.tight_layout()
            ax = plt.gca()
            if rectangle:
                rect = Rectangle((-4, 2.55), 8, 3, linewidth=0, fill=True, facecolor=(0.5, 0.5, 0.5, 0.2))
                ax.add_patch(rect)
            ax.set_xlim([-3.05, 3.05])
            ax.set_ylim([-3.05, 4.05])
            plt.text(0., 4., r'Trained controller', dict(size=25), ha='center', va='top')
            plt.text(1.85, -2.9, r'$\tau = %02d$' % tp, dict(size=25), ha='left')
            f_gif = "cl" + prefix + "-%03i" % (tp + 100*idx) + ".png"
            filename_gif = os.path.join(BASE_DIR, 'gif', f_gif)
            plt.savefig(filename_gif)
            plt.close(fig)
    print("Generating cl gif...")
    filename_figs = os.path.join(BASE_DIR, 'gif', "cl" + prefix + "-*.png")
    filename_gif = os.path.join(BASE_DIR, 'gif', "cl" + prefix + ".gif")
    command = "convert -delay 4 -loop 0 " + filename_figs + " " + filename_gif
    os.system(command)
    print("Gif saved at %s" % filename_gif)
    print("Deleting figures...")
    command = "rm " + filename_figs
    os.system(command)

