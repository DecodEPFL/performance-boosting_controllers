import torch
import torch.nn.functional as F

from src.loss_functions import normpdf


def _loss_TL_then(p1, p2, one_value=False, moving_end_of_interval=True):
    # p1 and p2 are vectors of dimension t_end with the values of, for instance (0.2-d_{g_{blue}})
    t_end = p1.shape[0]
    mask = torch.tril(torch.ones([t_end, t_end]))
    aux_p1_positivity = torch.min(p1).detach()
    p1 = p1 - aux_p1_positivity
    max_on_p1 = torch.max(p1.unsqueeze(0).repeat(t_end, 1) * mask, dim=1).values
    max_on_p1 = max_on_p1 + aux_p1_positivity
    for_max_on_interval = torch.minimum(p2, max_on_p1)
    if one_value:
        return torch.max(for_max_on_interval, dim=0).values
    if moving_end_of_interval:
        aux_positivity = torch.min(for_max_on_interval).detach()
        for_max_on_interval = for_max_on_interval - aux_positivity
        max_out = torch.max(for_max_on_interval.unsqueeze(0).repeat(t_end, 1) * mask, dim=1).values
        max_out = max_out + aux_positivity
        return max_out
    return for_max_on_interval


def _loss_TL_until(p1, p2, one_value=False, moving_end_of_interval=True):
    # p1 and p2 are vectors of dimension t_end with the values of, for instance (0.2-d_{g_{blue}})
    t_end = p1.shape[0]
    mask = torch.tril(torch.ones([t_end, t_end]))
    aux_p1_negativity = torch.max(p1).detach()
    p1 = p1 - aux_p1_negativity
    min_on_p1 = torch.min(p1.unsqueeze(0).repeat(t_end, 1) * mask, dim=1).values
    min_on_p1 = min_on_p1 + aux_p1_negativity
    for_max_on_interval = torch.minimum(min_on_p1, p2)
    if one_value:
        return torch.max(for_max_on_interval, dim=0).values
    if moving_end_of_interval:
        aux_positivity = torch.min(for_max_on_interval).detach()
        for_max_on_interval = for_max_on_interval - aux_positivity
        max_out = torch.max(for_max_on_interval.unsqueeze(0).repeat(t_end, 1) * mask, dim=1).values
        max_out = max_out + aux_positivity
        return max_out
    return


def _loss_TL_always(p, one_value=False, moving_end_of_interval=True):
    t_end = p.shape[0]
    if one_value:
        return torch.min(p, dim=0).values
    if moving_end_of_interval:
        mask = torch.tril(torch.ones([t_end, t_end]))
        aux_negativity = torch.max(p).detach()
        p = p - aux_negativity
        min_on_p = torch.min(p.unsqueeze(0).repeat(t_end, 1) * mask, dim=1).values
        min_on_p = min_on_p + aux_negativity
        return min_on_p
    return


def _loss_TL_always_implies_next_always_not(p, one_value=True):
    t_end = p.shape[0]
    minus_p_next = -p[1:]
    mask = torch.triu(torch.ones([t_end-1, t_end-1]))
    aux_negativity = torch.max(minus_p_next).detach()
    minus_p_next = minus_p_next - aux_negativity
    min_on_minus_p_next = torch.min(minus_p_next.unsqueeze(0).repeat(t_end-1, 1) * mask, dim=1).values
    min_on_minus_p_next = min_on_minus_p_next + aux_negativity
    for_min_on_interval = torch.maximum(-p, torch.cat((min_on_minus_p_next, -p[-1:])))
    if one_value:
        out = torch.min(for_min_on_interval, dim=0).values
        return out
    return


def _f_tl_goal(x, u, sys):
    t_end = x.shape[0]
    pos = x.reshape(-1, 4)[:, 0:2].reshape(-1, sys.n_agents * 2)
    posbar = sys.xbar.reshape(-1, 4)[:, 0:2].reshape(sys.n_agents * 2)
    target_distance = torch.norm(pos - posbar.unsqueeze(0).repeat(t_end, 1), dim=1)
    mask = torch.triu(torch.ones([t_end, t_end]))
    min_on_goal = 0.05 + torch.min((- target_distance).repeat(t_end, 1) * mask, dim=1).values
    maxmin_on_goal = torch.max(min_on_goal, dim=0).values
    return maxmin_on_goal.unsqueeze(0)


def _f_tl_input(u, sys, u_max):
    t_end = u.shape[0]
    min_on_u = 0.1*torch.min(-torch.relu(u - u_max))
    return min_on_u.unsqueeze(0)


def _f_tl_obstacle(x, u, sys, obstacle_pos, obstacle_radius):
    n_obstacles = obstacle_pos.shape[0]
    t_end = x.shape[0]
    stack_for_min = torch.tensor([])
    pos = x.reshape(-1, 4)[:, 0:2].reshape(-1, sys.n_agents * 2)
    for i in range(n_obstacles):
        radius = obstacle_radius[i,0]
        o = obstacle_pos[i,:].repeat(sys.n_agents)
        o_distance = torch.norm((pos - o.unsqueeze(0).repeat(t_end, 1)).reshape(-1, 2), dim=1)
        min_on_obst = torch.min(o_distance - radius, dim=0).values.unsqueeze(0)
        stack_for_min = torch.cat([stack_for_min, min_on_obst])
    min_on_obst = torch.min(stack_for_min, dim=0).values
    return min_on_obst.unsqueeze(0)


def _f_tl_collision_avoidance_2(x, u, sys, ca_distance):
    dist_min = ca_distance
    t_end = x.shape[0]
    pos = x.reshape(-1, 4)[:, 0:2].reshape(-1, sys.n_agents*2)
    dist_ca = torch.norm(pos[:,0:2] - pos[:,2:4], dim=1)
    min_on_ca = torch.min(dist_ca - dist_min, dim=0).values
    return min_on_ca.unsqueeze(0)


def _f_tl_wall_up(x, sys):
    t_end = x.shape[0]
    qy = x.reshape(-1, 4)[:, 1].reshape(-1, sys.n_agents)
    min_on_wall = torch.min(-torch.relu(qy-2.1), dim=0).values.sum()
    return min_on_wall.unsqueeze(0)
def _f_tl_wall_up_left(x, sys):
    t_end = x.shape[0]
    qy = x.reshape(-1, 4)[:, 1].reshape(-1, sys.n_agents)
    qx = x.reshape(-1, 4)[:, 0].reshape(-1, sys.n_agents)
    min_on_wall = torch.min(torch.min(-torch.relu(qy-2.1), dim=0).values.sum(),
                            torch.min(-torch.relu(-2.5-qx), dim=0).values.sum(),)
    return min_on_wall.unsqueeze(0)


def _f_loss_states(x, sys, Q=None):
    t_end = x.shape[0]
    if Q is None:
        Q = torch.eye(sys.n)
    xbar = sys.xbar.unsqueeze(0).repeat(t_end,1)
    dx = x - xbar
    xQx = (F.linear(dx, Q) * dx).sum(dim=1)
    return xQx.sum().unsqueeze(0)

def _f_loss_u(u, sys, R=None):
    if R is None:
        R = torch.eye(sys.m)
    uRu = (F.linear(u, R) * u).sum(dim=1)
    return uRu.sum().unsqueeze(0)

def _f_loss_obst(x):
    # TODO: is it the best way to define the loss wrt obstacles?
    # TODO: probably a relu thing to not penalize anymore when the agent is some_dist far away or more
    t_end = x.shape[0]
    loss = 0
    for t in range(t_end):
        qx = x[t, ::4].unsqueeze(1)
        qy = x[t, 1::4].unsqueeze(1)
        q = torch.cat((qx,qy), dim=1).view(1,-1).squeeze()
        mu1 = torch.tensor([[-2.5, 0]])
        mu2 = torch.tensor([[2.5, 0.0]])
        mu3 = torch.tensor([[-1.5, 0.0]])
        mu4 = torch.tensor([[1.5, 0.0]])
        cov = torch.tensor([[0.2, 0.2]])
        Q1 = normpdf(q, mu=mu1, cov=cov)
        Q2 = normpdf(q, mu=mu2, cov=cov)
        Q3 = normpdf(q, mu=mu3, cov=cov)
        Q4 = normpdf(q, mu=mu4, cov=cov)
        loss = loss + (Q1 + Q2 + Q3 + Q4).sum()
    return loss.unsqueeze(0)


def _f_loss_ca(x, sys, min_dist=0.5):
    min_sec_dist = min_dist + 0.2
    t_end = x.shape[0]
    qx = x[:, 0::4].unsqueeze(-1).repeat(1, 1, sys.n_agents)
    qy = x[:, 1::4].unsqueeze(-1).repeat(1, 1, sys.n_agents)
    deltaqx = qx - qx.transpose(1, 2)
    deltaqy = qy - qy.transpose(1, 2)
    distance_sq = deltaqx ** 2 + deltaqy ** 2
    mask_itself = torch.logical_not(torch.eye(sys.n // 4)).repeat(t_end, 1, 1)
    mask_far = distance_sq.detach() < (min_sec_dist**2)
    loss_ca = 0.5 * (1/(distance_sq + 1e-3) * mask_far * mask_itself).sum()
    return loss_ca.unsqueeze(0)


def _f_loss_up(x):
    qy = x[:,1::4]
    side_up = torch.relu(qy - 2.1)
    return side_up.sum().unsqueeze(0)


def _f_loss_barrier_up(x):
    t_end = x.shape[0]
    qy = x[:,1::4]
    barrier_up = 0
    gamma = 0.5
    alpha = 1  # useless?
    h = alpha*(2.05-qy)
    for i in range(t_end-1):
        barrier_up = barrier_up + torch.relu((1-gamma)*h[i, :] - h[i+1, :]).sum()
    return barrier_up.unsqueeze(0)


def loss_TL_waypoints(x):
    t_end = x.shape[0]
    n_agents = x.shape[1] // 4
    dist_error = 0.05
    stack_for_min = torch.tensor([])
    center_red = torch.tensor([-2, -2.])
    center_green = torch.tensor([0, 2.])
    center_blue = torch.tensor([2, -2.])
    dist_x1_red = torch.norm(x[:,0:2] - center_red.unsqueeze(0).repeat(t_end, 1), dim=1)
    dist_x1_green = torch.norm(x[:,0:2] - center_green.unsqueeze(0).repeat(t_end, 1), dim=1)
    dist_x1_blue = torch.norm(x[:,0:2] - center_blue.unsqueeze(0).repeat(t_end, 1), dim=1)
    psi_goal_r1 = dist_error - dist_x1_red
    psi_goal_g1 = dist_error - dist_x1_green
    psi_goal_b1 = dist_error - dist_x1_blue

    # Loss "then" car 1
    loss_then_1 = _loss_TL_then(psi_goal_r1, _loss_TL_then(psi_goal_g1, psi_goal_b1,
                                                           one_value=False, moving_end_of_interval=True),
                                one_value=True, moving_end_of_interval=False).unsqueeze(0)
    stack_for_min = torch.cat([stack_for_min, loss_then_1])
    # Loss "until" car 1
    loss_until_1_1 = _loss_TL_until(-torch.maximum(psi_goal_g1, psi_goal_b1), psi_goal_r1,
                                    one_value=True, moving_end_of_interval=False).unsqueeze(0)
    loss_until_1_2 = _loss_TL_until(-psi_goal_b1, psi_goal_g1, one_value=True, moving_end_of_interval=False).unsqueeze(0)
    stack_for_min = torch.cat([stack_for_min, loss_until_1_1, loss_until_1_2])
    # Loss AINAN car 1
    loss_ainan_r1 = _loss_TL_always_implies_next_always_not(psi_goal_r1).unsqueeze(0)
    loss_ainan_g1 = _loss_TL_always_implies_next_always_not(psi_goal_g1).unsqueeze(0)
    loss_ainan_b1 = _loss_TL_always_implies_next_always_not(psi_goal_b1).unsqueeze(0)
    stack_for_min = torch.cat([stack_for_min, loss_ainan_r1, loss_ainan_g1, loss_ainan_b1])
    print("\t CAR 1---||--- Loss_then: %.4f --- Loss_until_1: %.4f " % (loss_then_1, loss_until_1_1) +
          "--- Loss_until_2: %.4f --- Loss_r: %.4f --- Loss_g: %.4f --- Loss_b: %.4f ---"
          % (loss_until_1_2, loss_ainan_r1, loss_ainan_g1, loss_ainan_b1))

    if n_agents == 2:
        dist_x2_red = torch.norm(x[:, 4:6] - center_red.unsqueeze(0).repeat(t_end, 1), dim=1)
        dist_x2_green = torch.norm(x[:, 4:6] - center_green.unsqueeze(0).repeat(t_end, 1), dim=1)
        dist_x2_blue = torch.norm(x[:, 4:6] - center_blue.unsqueeze(0).repeat(t_end, 1), dim=1)
        psi_goal_r2 = dist_error - dist_x2_red
        psi_goal_g2 = dist_error - dist_x2_green
        psi_goal_b2 = dist_error - dist_x2_blue

        # Loss "then" car 2
        loss_then_2 = _loss_TL_then(psi_goal_b2, _loss_TL_then(psi_goal_g2, psi_goal_r2,
                                                               one_value=False, moving_end_of_interval=True),
                                    one_value=True, moving_end_of_interval=False).unsqueeze(0)
        stack_for_min = torch.cat([stack_for_min, loss_then_2])
        # Loss "until" car 2
        loss_until_2_1 = _loss_TL_until(-torch.maximum(psi_goal_g2, psi_goal_r2), psi_goal_b2,
                                        one_value=True, moving_end_of_interval=False).unsqueeze(0)
        loss_until_2_2 = _loss_TL_until(-psi_goal_r2, psi_goal_g2, one_value=True,
                                        moving_end_of_interval=False).unsqueeze(0)
        stack_for_min = torch.cat([stack_for_min, loss_until_2_1, loss_until_2_2])
        # Loss AINAN car 2
        loss_ainan_b2 = _loss_TL_always_implies_next_always_not(psi_goal_b2).unsqueeze(0)
        loss_ainan_g2 = _loss_TL_always_implies_next_always_not(psi_goal_g2).unsqueeze(0)
        loss_ainan_r2 = _loss_TL_always_implies_next_always_not(psi_goal_r2).unsqueeze(0)
        stack_for_min = torch.cat([stack_for_min, loss_ainan_b2, loss_ainan_g2, loss_ainan_r2])
        print("\t CAR 2---||--- Loss_then: %.4f --- Loss_until_1: %.4f " % (loss_then_2, loss_until_2_1) +
              "--- Loss_until_2: %.4f --- Loss_b: %.4f --- Loss_g: %.4f --- Loss_r: %.4f ---"
              % (loss_until_2_2, loss_ainan_b2, loss_ainan_g2, loss_ainan_r2))

    return -torch.min(stack_for_min)


def f_loss_tl(x, u, sys, dict_tl):
    text_to_print = ""
    stack_for_min = torch.tensor([])
    if dict_tl["goal"]:
        maxmin_on_goal = _f_tl_goal(x, u, sys)
        text_to_print = text_to_print + ("\t Goal %.4f ---" % maxmin_on_goal)
        stack_for_min = torch.cat([stack_for_min, maxmin_on_goal])
    if dict_tl["input"]:
        min_on_u = _f_tl_input(u, sys, dict_tl["max_u"])
        text_to_print = text_to_print + ("\t Input %.4f ---" % min_on_u)
        stack_for_min = torch.cat([stack_for_min, min_on_u])
    if dict_tl["obstacle"]:
        min_on_obst = _f_tl_obstacle(x, u, sys, dict_tl["obstacle_pos"], dict_tl["obstacle_radius"])
        text_to_print = text_to_print + ("\t Obstacle %.4f ---" % min_on_obst)
        stack_for_min = torch.cat([stack_for_min, min_on_obst])
    if dict_tl["collision_avoidance"]:
        min_on_ca = _f_tl_collision_avoidance_2(x, u, sys, dict_tl["ca_distance"])
        text_to_print = text_to_print + ("\t Coll_avoidance %.4f ---" % min_on_ca)
        stack_for_min = torch.cat([stack_for_min, min_on_ca])
    if dict_tl["wall_up"]:
        min_on_wall = _f_tl_wall_up(x, sys)
        text_to_print = text_to_print + ("\t Wall_up %.4f ---" % min_on_wall)
        stack_for_min = torch.cat([stack_for_min, min_on_wall])
    if not stack_for_min.shape[0] == 0:
        print(text_to_print)
    else:
        stack_for_min = torch.zeros(1)
    return -torch.min(stack_for_min)


def f_loss_sum(xx, uu, sys, dict_sum_loss):
    text_to_print = ""
    stack_for_sum = torch.tensor([])
    alphas = torch.tensor([])
    t_end = xx.shape[0]
    if dict_sum_loss["state"]:
        loss_x = _f_loss_states(xx, sys, dict_sum_loss["Q"])
        stack_for_sum = torch.cat([stack_for_sum, loss_x])
        alphas = torch.cat([alphas, dict_sum_loss["alpha_x"]])
        text_to_print = text_to_print + ("\t Loss_x %.4f ---" % (alphas[-1]*loss_x))
    if dict_sum_loss["input"]:
        loss_u = _f_loss_u(uu, sys)
        stack_for_sum = torch.cat([stack_for_sum, loss_u])
        alphas = torch.cat([alphas, dict_sum_loss["alpha_u"]])
        text_to_print = text_to_print + ("\t Loss_u %.4f ---" % (alphas[-1] * loss_u))
    if dict_sum_loss["obstacle"]:
        loss_obst = _f_loss_obst(xx)
        stack_for_sum = torch.cat([stack_for_sum, loss_obst])
        alphas = torch.cat([alphas, dict_sum_loss["alpha_obst"]])
        text_to_print = text_to_print + ("\t Loss_obst %.4f ---" % (alphas[-1] * loss_obst))
    if dict_sum_loss["collision_avoidance"]:
        loss_ca = _f_loss_ca(xx, sys, dict_sum_loss["ca_distance"])
        stack_for_sum = torch.cat([stack_for_sum, loss_ca])
        alphas = torch.cat([alphas, dict_sum_loss["alpha_ca"]])
        text_to_print = text_to_print + ("\t Loss_ca %.4f ---" % (alphas[-1] * loss_ca))
    if dict_sum_loss["wall_up"]:
        loss_wall = _f_loss_up(xx)
        stack_for_sum = torch.cat([stack_for_sum, loss_wall])
        alphas = torch.cat([alphas, dict_sum_loss["alpha_wall"]])
        text_to_print = text_to_print + ("\t Loss_wall %.4f ---" % (alphas[-1] * loss_wall))
    if dict_sum_loss["wall_up_barrier"]:
        loss_barrier = _f_loss_barrier_up(xx)
        stack_for_sum = torch.cat([stack_for_sum, loss_barrier])
        alphas = torch.cat([alphas, dict_sum_loss["alpha_barrier"]])
        text_to_print = text_to_print + ("\t Loss_barrier %.4f ---" % (alphas[-1] * loss_barrier))
    if not stack_for_sum.shape[0] == 0:
        print(text_to_print)
    else:
        stack_for_sum = torch.zeros(1)
        alphas = torch.zeros(1)
    return (alphas * stack_for_sum).sum()
