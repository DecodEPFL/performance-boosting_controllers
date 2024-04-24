import torch
import matplotlib.pyplot as plt
import os
import pickle
import logging
from datetime import datetime

from src.models import SystemRobots, Controller
from src.plots import plot_trajectories, plot_traj_vs_time
from src.loss_functions import f_loss_states, f_loss_u, f_loss_ca, f_loss_obst, f_loss_side, f_loss_barrier_up
from src.utils import calculate_collisions, set_params, generate_data
from src.utils import WrapLogger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

random_seed = 3
torch.manual_seed(random_seed)

sys_model = "corridor"
prefix = ''

with_barrier = False

# # # # # # # # Parameters and hyperparameters # # # # # # # #
params = set_params(sys_model)
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, \
alpha_u, alpha_ca, alpha_obst, n_xi, l, n_traj, std_ini = params

epochs = 5000
n_traj = 1
std_ini = 0.5
l,n_xi = 8,8

learning_rate = 1e-4  # *0.5

alpha_barrier = 0
# alpha_barrier = 5  # 250
alpha_side = 500
std_ini_param = 0.005
use_sp = False

epoch_print = 20
n_train = 100
n_test = 1000 - n_train
validation = True
validation_period = 50
n_validation = 100

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
ctl = Controller(sys.f, sys.n, sys.m, n_xi, l, use_sp=use_sp, t_end_sp=t_end, std_ini_param=std_ini_param)

# # # # # # # # Define optimizer and parameters # # # # # # # #
optimizer = torch.optim.Adam(ctl.parameters(), lr=learning_rate)

# # # # # # # # Figures # # # # # # # #
fig_path = os.path.join(BASE_DIR, 'figures', 'temp')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
filename_figure = 'fig_' + log_name
filename_figure = os.path.join(fig_path, filename_figure)

# # # # # # # # Training # # # # # # # #
msg = "\n------------ Begin training ------------\n"
msg += "Problem: " + sys_model + " -- t_end: %i" % t_end + " -- lr: %.2e" % learning_rate
msg += " -- epochs: %i" % epochs + " -- n_traj: %i" % n_traj + " -- std_ini: %.2f\n" % std_ini
msg += " -- alpha_u: %.4f" % alpha_u + " -- alpha_ca: %i" % alpha_ca + " -- alpha_obst: %.1e\n" % alpha_obst
msg += "REN info -- n_xi: %i" % n_xi + " -- l: %i " % l + "use_sp: %r\n" % use_sp
msg += "--------- --------- ---------  ---------"
logger.info(msg)
best_valid_loss = 1e9
best_params = None
best_params_sp = None
for epoch in range(epochs):
    # batch data
    if n_traj == 1:
        train_x0_batch = train_x0[epoch % n_train:epoch % n_train + 1, :]
    else:
        inds = torch.randperm(n_train)[:n_traj]
        train_x0_batch = train_x0[inds, :]
    optimizer.zero_grad()
    loss_x, loss_u, loss_ca, loss_obst, loss_side, loss_barrier = 0, 0, 0, 0, 0, 0
    for kk in range(n_traj):
        w_in = torch.zeros(t_end, sys.n)
        w_in[0, :] = (x0.detach() - sys.xbar) + train_x0_batch[kk]
        u = torch.zeros(sys.m)
        x = sys.xbar
        xi = torch.zeros(ctl.psi_u.n_xi)
        omega = (x, u)
        for t in range(t_end):
            x_prev = x
            x, _ = sys(t, x, u, w_in[t, :])
            u, xi, omega = ctl(t, x, xi, omega)
            loss_x = loss_x + f_loss_states(t, x, sys, Q) / n_traj
            loss_u = loss_u + alpha_u * f_loss_u(t, u) / n_traj
            loss_ca = loss_ca + alpha_ca * f_loss_ca(x, sys, min_dist) / n_traj
            loss_side = loss_side + alpha_side * f_loss_side(x) / n_traj
            if alpha_obst != 0:
                loss_obst = loss_obst + alpha_obst * f_loss_obst(x, sys) / n_traj
            if with_barrier and t>0:
                loss_barrier = loss_barrier + alpha_barrier * f_loss_barrier_up(x,x_prev) / n_traj
    loss = loss_x + loss_u + loss_ca + loss_obst + loss_side + loss_barrier
    msg = "Epoch: {:>4d} --- Loss: {:>9.4f} ---||--- Loss x: {:>9.2f}".format(epoch, loss/t_end, loss_x)
    msg += " --- Loss u: {:>9.4f} --- Loss ca: {:>9.2f} --- Loss obst: {:>9.2f}".format(loss_u,loss_ca,loss_obst)
    msg += " --- Loss side: {:>9.2f}--- Loss barrier: {:>9.2f}".format(loss_side, loss_barrier)
    loss.backward()
    optimizer.step()
    ctl.psi_u.set_model_param()
    # record state dict if best on valid
    if validation and epoch % validation_period == 0 and epoch > 0:
        with torch.no_grad():
            loss_x, loss_u, loss_ca, loss_obst, loss_side, loss_barrier = 0, 0, 0, 0, 0, 0
            for kk in range(n_validation):
                w_in = torch.zeros(t_end, sys.n)
                w_in[0, :] = (x0.detach() - sys.xbar) + validation_x0[kk]
                u = torch.zeros(sys.m)
                x = sys.xbar
                xi = torch.zeros(ctl.psi_u.n_xi)
                omega = (x, u)
                for t in range(t_end):
                    x_prev = x
                    x, _ = sys(t, x, u, w_in[t, :])
                    u, xi, omega = ctl(t, x, xi, omega)
                    loss_x = loss_x + f_loss_states(t, x, sys, Q) / n_validation
                    loss_u = loss_u + alpha_u * f_loss_u(t, u) / n_validation
                    loss_ca = loss_ca + alpha_ca * f_loss_ca(x, sys, min_dist) / n_validation
                    loss_side = loss_side + alpha_side * f_loss_side(x) / n_validation
                    if alpha_obst != 0:
                        loss_obst = loss_obst + alpha_obst * f_loss_obst(x, sys) / n_validation
                    if with_barrier and t > 0:
                        loss_barrier = loss_barrier + alpha_barrier * f_loss_barrier_up(x, x_prev) / n_validation
            loss = loss_x + loss_u + loss_ca + loss_obst + loss_side + loss_barrier
        msg += ' ---||--- Original validation loss: %.2f' % (loss / t_end)
        # compare with the best valid loss
        if loss < best_valid_loss:
            best_valid_loss = loss
            best_params = ctl.psi_u.state_dict()
            if use_sp:
                best_params_sp = ctl.sp.state_dict()
            msg += ' (best so far)'
    logger.info(msg)
    if (epoch < epoch_print and epoch % 2 == 0) or epoch % validation_period == 0:
        # Extended time
        x_log = torch.zeros(t_ext, sys.n)
        u_log = torch.zeros(t_ext, sys.m)
        w_in = torch.zeros(t_ext, sys.n)
        w_in[0, :] = (x0.detach() - sys.xbar) + train_x0_batch[0]
        u = torch.zeros(sys.m)
        x = sys.xbar
        xi = torch.zeros(ctl.psi_u.n_xi)
        omega = (x, u)
        for t in range(t_ext):
            x, _ = sys(t, x, u, w_in[t, :])
            u, xi, omega = ctl(t, x, xi, omega)
            x_log[t, :] = x.detach()
            u_log[t, :] = u.detach()
        plot_trajectories(x_log, xbar, sys.n_agents, text="CL at epoch %i" % epoch, T=t_end, obst=False)
        if show_plots:
            plt.show()
        else:
            plt.savefig(filename_figure + '_during_' + '%i_epoch' % epoch + '.png', format='png')
            plt.close()

# Set parameters to the best seen during training
if validation and best_params is not None:
    ctl.psi_u.load_state_dict(best_params)
    ctl.psi_u.eval()
    if use_sp:
        ctl.sp.load_state_dict(best_params_sp)
        ctl.sp.eval()
    ctl.psi_u.set_model_param()

# # # # # # # # Save trained model # # # # # # # #
fname = log_name + '_T' + str(t_end) + '_stdini' + str(std_ini) + '_RS' + str(random_seed)
fname += '.pt'
filename = os.path.join(BASE_DIR, 'trained_models')
if not os.path.exists(filename):
    os.makedirs(filename)
filename = os.path.join(filename, fname)
save_dict = {'psi_u': ctl.psi_u.state_dict(),
             'Q': Q,
             'alpha_u': alpha_u,
             'alpha_ca': alpha_ca,
             'alpha_obst': alpha_obst,
             'n_xi': n_xi,
             'l': l,
             'n_traj': n_traj,
             'epochs': epochs,
             'std_ini_param': std_ini_param,
             'use_sp': use_sp,
             'linear': linear
             }
if use_sp:
    save_dict['sp'] = ctl.sp.state_dict()
torch.save(save_dict, filename)
logger.info('[INFO] Saved trained model as: %s' % fname)

# # # # # # # # Print & plot results # # # # # # # #
x_log = torch.zeros(t_end, sys.n)
u_log = torch.zeros(t_end, sys.m)
w_in = torch.zeros(t_end, sys.n)
w_in[0, :] = (x0.detach() - sys.xbar)
u = torch.zeros(sys.m)
x = sys.xbar
xi = torch.zeros(ctl.psi_u.n_xi)
omega = (x, u)
for t in range(t_end):
    x, _ = sys(t, x, u, w_in[t, :])
    u, xi, omega = ctl(t, x, xi, omega)
    x_log[t, :] = x.detach()
    u_log[t, :] = u.detach()
plot_traj_vs_time(t_end, sys.n_agents, x_log, u_log)
# Number of collisions
n_coll = calculate_collisions(x_log, sys, min_dist)
msg = 'Number of collisions after training: %.1f.' % n_coll
logger.info(msg)

# Extended time
x_log = torch.zeros(t_ext, sys.n)
u_log = torch.zeros(t_ext, sys.m)
w_in = torch.zeros(t_ext, sys.n)
w_in[0, :] = (x0.detach() - sys.xbar)
u = torch.zeros(sys.m)
x = sys.xbar
xi = torch.zeros(ctl.psi_u.n_xi)
omega = (x, u)
for t in range(t_ext):
    x, _ = sys(t, x, u, w_in[t, :])
    u, xi, omega = ctl(t, x, xi, omega)
    x_log[t, :] = x.detach()
    u_log[t, :] = u.detach()
plot_trajectories(x_log, xbar, sys.n_agents, text="CL - after training - extended t", T=t_end, obst=alpha_obst)
if show_plots:
    plt.show()
else:
    plt.savefig(filename_figure + '_trained' + '.png', format='png')
    plt.close()

print("Hola!")
