
import h5py

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


from shapely import box
from shapely.plotting import plot_polygon, plot_line

from cvx_elmap import *
from utils import *
from downsampling import *


import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 14})

cmap = mpl.cm.get_cmap('Reds')


def load_h5(filename):
    hf = h5py.File(filename, 'r')
    cart = hf.get('cartesian_info')
    pos = np.array(cart.get('positions'))
    rot = np.array(cart.get('orientations'))
    return pos, rot
    
def get_min_manhattan(shape, point):
    obs_min_x, obs_min_y, obs_max_x, obs_max_y = shape
    point_x, point_y = point
    if point_x > obs_min_x and point_x < obs_max_x: #x within shape bounds
        if point_y < obs_min_y: #below
            return obs_min_y - point_y
        if point_y > obs_max_y: #above
            return point_y - obs_max_y
    if point_x < obs_min_x: #to the left
        if point_y > obs_max_y: #above
            return (point_y - obs_max_y) + (obs_min_x - point_x)
        if point_y < obs_min_y: #below
            return (obs_min_y - point_y) + (obs_min_x - point_x)
        return obs_min_x - point_x
    if point_x > obs_max_x: #to the right
        if point_y > obs_max_y: #above
            return (point_y - obs_max_y) + (point_x - obs_max_x)
        if point_y < obs_min_y: #below
            return (obs_min_y - point_y) + (point_x - obs_max_x)
        return  point_x - obs_max_x
    return 0 #should only get here if point within shape bounds


def main():

    ## Demonstration

    name = '../h5 files/jaco_box_pushing.h5'
    pos, rot = load_h5(name)
    traj, inds = DouglasPeuckerPoints2(pos, 50)
    
    plt.rcParams['figure.figsize'] = (8, 6)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'k', lw=3, label='Demonstration')
    min_x = -0.2
    max_x = 0.2
    min_y = -1.0
    max_y = -0.57
    min_z = 0.5
    max_z = 1.0
    plot_cube(min_x, max_x, min_y, max_y, min_z, max_z, ax)
    
    ## Setup & solve for varying confidence
    
    PA = ElMap_Perturbation_Analysis(traj, stretch=1.0, bend=1.0)
    
    sol0 = None
    sol1 = None
    sol5 = None
    
    for sf in [0.0, 0.5, 1.0]:
        x_prob = PA.setup_problem()
        constraints = []
        for i in range(len(traj)):
            manh = 0.05
            constraints.append(cp.abs(x_prob[i] - traj[i, 0]) + cp.abs(x_prob[PA.n_pts+i] - traj[i, 1]) - sf*manh <= 0)
        sol = PA.solve_problem(constraints, disp=False)
        if sf == 0.0:
            sol0 = sol
            repro, = ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], 'r', lw=3, label='Reproduction $\sigma_s=1.0$')
        if sf == 0.5:
            sol5 = sol
            repro, = ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], 'tab:olive', lw=3, label='Reproduction $\sigma_s=0.5$')
        if sf == 1.0:
            sol1 = sol
            repro, = ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], 'tab:green', lw=3, label='Reproduction $\sigma_s=0.0$')
        
    
    plt.legend()
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=49, elev=25)
    plt.tight_layout()
    #mysavefig(plt.gcf(), '../pictures/real_world_experiment/box_opening_confidence')
    
    plt.figure()
    mpl.rcParams.update({'font.size': 40})
    plt.xlabel('t')
    plt.ylabel('x')
    plt.xticks([])
    plt.yticks([])
    plt.plot(traj[:, 0], 'k', lw=7)
    plt.plot(sol0[:, 0], 'r', lw=7)
    plt.plot(sol5[:, 0], 'tab:olive', lw=7)
    plt.plot(sol1[:, 0], 'tab:green', lw=7)
    plt.tight_layout()
    #mysavefig(plt.gcf(), '../pictures/real_world_experiment/box_opening_xt')
    
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('y')
    plt.xticks([])
    plt.yticks([])
    plt.plot(traj[:, 1], 'k', lw=7)
    plt.plot(sol0[:, 1], 'r', lw=7)
    plt.plot(sol5[:, 1], 'tab:olive', lw=7)
    plt.plot(sol1[:, 1], 'tab:green', lw=7)
    plt.tight_layout()
    #mysavefig(plt.gcf(), '../pictures/real_world_experiment/box_opening_yt')
    
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('z')
    plt.xticks([])
    plt.yticks([])
    plt.plot(traj[:, 2], 'k', lw=7)
    plt.plot(sol0[:, 2], 'r', lw=7)
    plt.plot(sol5[:, 2], 'tab:olive', lw=7)
    plt.plot(sol1[:, 2], 'tab:green', lw=7)
    plt.tight_layout()
    #mysavefig(plt.gcf(), '../pictures/real_world_experiment/box_opening_zt')
    
    plt.show()
    

if __name__ == '__main__':
    main()