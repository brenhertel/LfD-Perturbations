import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from shapely import box
from shapely.plotting import plot_polygon, plot_line

from cvx_elmap import *
from utils import *
from downsampling import *

import screen_capture_rev2 as scr2

import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 14})

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

    ## Demonstration & Obstacle

    fnames = '../h5 files/box2.h5'
    [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5(fnames, 1)
    data = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(norm_y, (len(norm_y), 1))))
    data = data - data[-1, :]
    data = data * 100
    data[:, 0] = -1 * abs(data[:, 0])
    data[:, 1] = abs(data[:, 1])
    traj = DouglasPeuckerPoints(data, 100)
    
    b = box(-48.5, 12.5, -31.5, 19.5)
    
    ## Problem setup & solve without obstacle avoidance
    
    PA = ElMap_Perturbation_Analysis(traj, stretch=150.0, bend=40.0)
    x_prob = PA.setup_problem()
    constraints = [cp.abs(x_prob[0] - traj[0, 0]) <= 0, cp.abs(x_prob[PA.n_pts] - traj[0, 1]) <= 0, cp.abs(x_prob[PA.n_pts-1] - traj[-1, 0]) <= 0, cp.abs(x_prob[-1] - traj[-1, 1]) <= 0]
    sol = PA.solve_problem(constraints)
    
    plt.rcParams['figure.figsize'] = (6.5, 4.5)

    plt.figure()
    demo, = plt.plot(traj[:, 0], traj[:, 1], 'k', lw=3, label='Demonstration')
    repro, = plt.plot(sol[:, 0], sol[:, 1], 'b', lw=3, label='Unconstrained Reproduction')
    
    ## Solve problem with obstacle avoidance
    
    x_prob = PA.setup_problem()
    constraints = [cp.abs(x_prob[0] - traj[0, 0]) <= 0, cp.abs(x_prob[PA.n_pts] - traj[0, 1]) <= 0, cp.abs(x_prob[PA.n_pts-1] - traj[-1, 0]) <= 0, cp.abs(x_prob[-1] - traj[-1, 1]) <= 0]
    for i in range(len(traj)):
        manh = get_min_manhattan([-48.5, 12.5, -31.5, 19.5], traj[i])
        constraints.append(cp.abs(x_prob[i] - traj[i, 0]) + cp.abs(x_prob[PA.n_pts+i] - traj[i, 1]) - manh <= 0)
    sol = PA.solve_problem(constraints)
    
    repro, = plt.plot(sol[:, 0], sol[:, 1], 'r', lw=3, label='Constrained Reproduction')
    init, = plt.plot(sol[0, 0], sol[0, 1], 'k.', ms=12)
    end, = plt.plot(sol[-1, 0], sol[-1, 1], 'k.', ms=12, label='Endpoint Constraints')
    plot_polygon(b, ax=plt.gca(), add_points=False, color='g', alpha=0.5, label='Obstacle')
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    #mysavefig(plt.gcf(), '../pictures/obstacle_experiment/optimal_reproductions')
    
    ## determine unnecessary constraints (dual value == 0)
    
    duals = []
    for i in range(len(constraints)):
        duals.append(constraints[i].dual_value[0])
    
    ## solve with sensitivity analysis
    
    plt.figure()
    demo, = plt.plot(traj[:, 0], traj[:, 1], 'k', lw=3, label='Demonstration')
    for sf in [0.0, 0.1, 0.4, 0.7, 1.0]:
        x_prob = PA.setup_problem()
        constraints = [cp.abs(x_prob[0] - traj[0, 0]) <= 0, cp.abs(x_prob[PA.n_pts] - traj[0, 1]) <= 0, cp.abs(x_prob[PA.n_pts-1] - traj[-1, 0]) <= 0, cp.abs(x_prob[-1] - traj[-1, 1]) <= 0]
        for i in range(len(traj)):
            if abs(duals[i+4]) > 1e-3:
                manh = get_min_manhattan([-48.5, 12.5, -31.5, 19.5], traj[i])
                constraints.append(cp.abs(x_prob[i] - traj[i, 0]) + cp.abs(x_prob[PA.n_pts+i] - traj[i, 1]) - sf*manh <= 0)
        print(len(constraints))
        sol = PA.solve_problem(constraints, disp=False)
        if sf == 0.0:
            repro, = plt.plot(sol[:, 0], sol[:, 1], 'r', lw=3, alpha=(1-sf), label='Reproductions')
        else:
            plt.plot(sol[:, 0], sol[:, 1], 'r', lw=3, alpha=(1-sf))
        
    init, = plt.plot(sol[0, 0], sol[0, 1], 'k.', ms=12)
    end, = plt.plot(sol[-1, 0], sol[-1, 1], 'k.', ms=12, label='Endpoint Constraints')
    plot_polygon(b, ax=plt.gca(), add_points=False, color='g', alpha=0.5, label='Obstacle')
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    #mysavefig(plt.gcf(), '../pictures/obstacle_experiment/varying_confidence')
    plt.show()

if __name__ == '__main__':
    main()