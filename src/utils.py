import torch
import os
import pickle


# For saving logs when running
class WrapLogger():
    def __init__(self, logger, verbose=True):
        self.can_log = not (logger is None)
        self.logger = logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)


def calculate_collisions(x, sys, min_dist):
    deltax = x[:, 0::4].repeat(sys.n_agents, 1, 1) - x[:, 0::4].repeat(sys.n_agents, 1, 1).transpose(0, 2)
    deltay = x[:, 1::4].repeat(sys.n_agents, 1, 1) - x[:, 1::4].repeat(sys.n_agents, 1, 1).transpose(0, 2)
    distance_sq = (deltax ** 2 + deltay ** 2)
    n_coll = ((0.0001 < distance_sq) * (distance_sq < min_dist**2)).sum()
    return n_coll/2

def set_params(sys_model):
    if sys_model == "corridor":
        # # # # # # # # Parameters # # # # # # # #
        min_dist = 1.  # min distance for collision avoidance
        t_end = 100
        n_agents = 2
        x0, xbar = get_ini_cond(n_agents)
        linear = False
        # # # # # # # # Hyperparameters # # # # # # # #
        learning_rate = 1e-3
        epochs = 500
        Q = torch.kron(torch.eye(n_agents), torch.diag(torch.tensor([1, 1, 1, 1.])))
        alpha_u = 0.1/400  # Regularization parameter for penalizing the input
        alpha_ca = 100
        alpha_obst = 5e3
        n_xi = 32  # \xi dimension -- number of states of REN
        l = 32  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
        n_traj = 5  # number of trajectories collected at each step of the learning
        std_ini = 0.2  # standard deviation of initial conditions
    elif sys_model == "corridor_wp":
        # # # # # # # # Parameters # # # # # # # #
        min_dist = 1.  # min distance for collision avoidance
        t_end = 100
        n_agents = 2
        x0 = torch.tensor([0., 0., 0, 0,
                           0., -2, 0, 0,
                           ])
        xbar = torch.tensor([2., -2, 0, 0,
                             -2, -2, 0, 0,
                             ])
        linear = False
        # # # # # # # # Hyperparameters # # # # # # # #
        learning_rate = 1e-3
        epochs = 1000
        Q = torch.kron(torch.eye(n_agents), torch.diag(torch.tensor([1,1,1, 1.])))
        alpha_u = 0.1  # Regularization parameter for penalizing the input
        alpha_ca = 100
        alpha_obst = 5e3
        alpha_wall = 5e0  # 1e-4
        n_xi = 32  # \xi dimension -- number of states of REN
        l = 32  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
        n_traj = 1  # number of trajectories collected at each step of the learning
        std_ini = 0.2  # standard deviation of initial conditions
    else:  # sys_model == "robots"
        # # # # # # # # Parameters # # # # # # # #
        min_dist = 0.5  # min distance for collision avoidance
        t_end = 100
        n_agents = 12
        x0, xbar = get_ini_cond(n_agents)
        linear = True
        # # # # # # # # Hyperparameters # # # # # # # #
        learning_rate = 2e-3
        epochs = 1500
        Q = torch.kron(torch.eye(n_agents), torch.diag(torch.tensor([1, 1, 1, 1.])))
        alpha_u = 0.1  # Regularization parameter for penalizing the input
        alpha_ca = 1000
        alpha_obst = 0
        n_xi = 8 * 12  # \xi dimension -- number of states of REN
        l = 24  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
        n_traj = 1  # number of trajectories collected at each step of the learning
        std_ini = 0  # standard deviation of initial conditions
    return min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, \
           l, n_traj, std_ini


def get_ini_cond(n_agents):
    # Corridor problem
    if n_agents == 2:
        x0 = torch.tensor([2., -2, 0, 0,
                           -2, -2, 0, 0,
                           ])
        xbar = torch.tensor([-2, 2, 0, 0,
                             2., 2, 0, 0,
                             ])
    # Robots problem
    elif n_agents == 12:
        x0 = torch.tensor([-3, 5, 0, 0.5,
                           -3, 3, 0, 0.5,
                           -3, 1, 0, 0.5,
                           -3, -1, 0, -0.5,
                           -3, -3, 0, -0.5,
                           -3, -5, 0, -0.5,
                           # second column
                           3, 5, -0, 0.5,
                           3, 3, -0, 0.5,
                           3, 1, -0, 0.5,
                           3, -1, 0, -0.5,
                           3, -3, 0, -0.5,
                           3, -5, 0, -0.5,
                           ])
        xbar = torch.tensor([3, -5, 0, 0,
                             3, -3, 0, 0,
                             3, -1, 0, 0,
                             3, 1, 0, 0,
                             3, 3, 0, 0,
                             3, 5, 0, 0,
                             # second column
                             -3, -5, 0, 0,
                             -3, -3, 0, 0,
                             -3, -1, 0, 0,
                             -3, 1, 0, 0,
                             -3, 3, 0, 0,
                             -3, 5.0, 0, 0,
                             ])
    else:
        x0 = torch.randn(4*n_agents)
        xbar = torch.zeros(4*n_agents)
    return x0, xbar


def generate_data(sys_model, t_end, n_agents, random_seed, std_ini=None, std_dist=None):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    torch.manual_seed(random_seed)

    file_path = os.path.join(BASE_DIR, 'data', sys_model)
    path_exist = os.path.exists(file_path)
    if not path_exist:
        os.makedirs(file_path)

    filename = 'data_' + sys_model
    filename += '_stdini' + str(std_ini) if std_ini is not None else ''
    filename += '_stddist' + str(std_dist) if std_dist is not None else ''
    filename += '_agents' + str(n_agents) + '_RS' + str(random_seed) + '.pkl'
    filename = os.path.join(file_path, filename)

    if std_ini is None:
        std_ini = 0
    if std_dist is None:
        std_dist = 0

    # generate data
    n_data_total = 1000
    n_states = 4 * n_agents
    n_w = n_states
    mask = torch.cat([torch.ones(1, t_end, 2),
                      torch.zeros(1, t_end, 2)], dim=-1).repeat(n_data_total, 1, n_agents)
    data_w = std_dist * torch.randn(n_data_total, t_end, n_w)  # * mask[:,0,:]
    data_x0 = std_ini * torch.randn(n_data_total, n_states) * mask[:, 0, :]

    data = {'data_x0': data_x0, 'data_w': data_w, 't_end': t_end}

    filehandler = open(filename, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()
    print("Data saved at %s" % filename)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
