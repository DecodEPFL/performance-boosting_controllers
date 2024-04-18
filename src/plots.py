import torch
import matplotlib.pyplot as plt
import numpy as np

from src.loss_functions import f_loss_obst


def plot_trajectories(x, xbar, n_agents, text="", save=False, filename=None, T=100, obst=False, dots=False,
                      circles=False, axis=False, min_dist=1, f=5):
    fig = plt.figure(f)
    ax = plt.gca()
    if obst:
        if obst == 2:
            r = 1
            circle1 = plt.Circle((-2,0), r, color='tab:gray', alpha=0.5, zorder=10)
            circle2 = plt.Circle((2, 0), r, color='tab:gray', alpha=0.5, zorder=10)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
        else:
            yy, xx = np.meshgrid(np.linspace(-3, 4.5, 125), np.linspace(-3, 3, 100))
            zz = xx * 0
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    zz[i, j] = f_loss_obst(torch.tensor([xx[i, j], yy[i, j], 0.0, 0.0]))
            z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
            c = ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max)
    plt.title(text)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    for i in range(n_agents):
        plt.plot(x[:T+1,4*i].detach(), x[:T+1,4*i+1].detach(), color=colors[i%12], linewidth=1)
        # plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color=colors[i%12], linestyle='dotted', linewidth=0.5)
        plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color='k', linewidth=0.125, linestyle='dotted')
    for i in range(n_agents):
        plt.plot(x[0,4*i].detach(), x[0,4*i+1].detach(), color=colors[i%12], marker='o', fillstyle='none')
        plt.plot(xbar[4*i].detach(), xbar[4*i+1].detach(), color=colors[i%12], marker='*')
    if dots:
        for i in range(n_agents):
            for j in range(T):
                plt.plot(x[j, 4*i].detach(), x[j, 4*i+1].detach(), color=colors[i%12], marker='o')
    if circles:
        for i in range(n_agents):
            r = min_dist/2
            # if obst:
            #     circle = plt.Circle((x[T-1, 4*i].detach(), x[T-1, 4*i+1].detach()), r, color='tab:purple', fill=False)
            # else:
            circle = plt.Circle((x[T, 4*i].detach(), x[T, 4*i+1].detach()), r, color=colors[i%12], alpha=0.5,
                                zorder=10)
            ax.add_patch(circle)
    ax.axes.xaxis.set_visible(axis)
    ax.axes.yaxis.set_visible(axis)
    if save:
        plt.savefig('figures/' + filename+'_'+text+'_trajectories.eps', format='eps')
    return fig


def plot_traj_vs_time(t_end, n_agents, x, u=None, text="", save=False, filename=None):
    t = torch.linspace(0,t_end-1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i])
        plt.plot(t, x[:,4*i+1])
    plt.xlabel(r'$t$')
    plt.title(r'$x(t)$')
    plt.subplot(1, p, 2)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i+2])
        plt.plot(t, x[:,4*i+3])
    plt.xlabel(r'$t$')
    plt.title(r'$v(t)$')
    plt.suptitle(text)
    if p == 3:
        plt.subplot(1, 3, 3)
        for i in range(n_agents):
            plt.plot(t, u[:, 2*i])
            plt.plot(t, u[:, 2*i+1])
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig('figures/' + filename + '_' + text + '_x_u.eps', format='eps')
    else:
        plt.show()
