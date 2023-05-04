import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from cvx_elmap import *
from utils import *
from downsampling import *

import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 18})


ind = 22
    
## Constraints generator for sensitivity analysis
def simple_constraint_gen(PA, u, cst_args):
    x_demo = cst_args[0]
    initial_freedom = cst_args[1]
    viapoint_freedom = cst_args[2]
    endpoint_freedom = cst_args[3]
    x_prob = PA.x
    constraints = [cp.abs(x_prob[0] - x_demo[0]) - initial_freedom <= 0, cp.abs(x_prob[ind] - x_demo[ind]) - viapoint_freedom <= u, cp.abs(x_prob[-1] - x_demo[-1]) - endpoint_freedom <= 0]
    return constraints
    
def main():

    ## Demonstration

    num_points = 50
    t = np.linspace(0, 10, num_points).reshape((num_points, 1))
    x_demo = np.sin(t) + 0.01 * t**2 - 0.05 * (t-5)**2
    
    ## Perturbation Analysis Setup
    
    PA = ElMap_Perturbation_Analysis(x_demo, stretch=8.0, bend=8.0)
    x_prob = PA.setup_problem()
    initial_freedom = 0.0
    viapoint_freedom = 0.01
    endpoint_freedom = 0.0
    constraints = [cp.abs(x_prob[0] - x_demo[0]) - initial_freedom <= 0, cp.abs(x_prob[ind] - x_demo[ind]) - viapoint_freedom <= 0, cp.abs(x_prob[-1] - x_demo[-1]) - endpoint_freedom <= 0]
    sol = PA.solve_problem(constraints)
    lamda = constraints[1].dual_value
    p_star = PA.problem.value
    
    umax = 0.25 #experimentally determined
    
    vals, u_vals, sols = generate_sensitivity(PA, simple_constraint_gen, min_u=0.0, max_u=umax, n_u=50, cst_args=[x_demo, initial_freedom, viapoint_freedom, endpoint_freedom])
    
    
    plt.rcParams['figure.figsize'] = (6.5, 5)

    plt.figure()
    demo, = plt.plot(t, x_demo, 'k', lw=3)
    repro, = plt.plot(t, sol, 'r', lw=3)
    via, = plt.plot([t[ind], t[ind]], [x_demo[ind] - viapoint_freedom, x_demo[ind] + viapoint_freedom], 'g.-', lw=2, ms=12)
    plt.plot(t[0], x_demo[0], 'k.', ms=12)
    end, = plt.plot(t[-1], x_demo[-1], 'k.', ms=12)
    plt.legend((demo, end, repro, via), ('Demonstration', 'Fixed Constraints', 'Reproduction', 'Via-Point Constraint'), ncol=1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('t')
    plt.ylabel('x')
    #for i in range(len(sols)):
    #    plt.plot(t, sols[i])
    plt.tight_layout()
    #mysavefig(plt.gcf(), '../pictures/via_point_1d_experiment/via_point_1d_original')
    
    plt.figure()
    lowbound, = plt.plot(u_vals, p_star - lamda * u_vals, 'k', lw=3)
    act_vals, = plt.plot(u_vals, vals, 'c.', ms=12)
    u0, = plt.plot([0, 0], [min(p_star - lamda * u_vals), max(vals)], 'k--', lw=2, alpha=0.5)
    plt.legend((lowbound, act_vals, u0), ('Lower Bound', 'Optimal Results', 'u = 0'))
    plt.xlabel('u')
    plt.ylabel('p*(u)')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    #mysavefig(plt.gcf(), '../pictures/via_point_1d_experiment/via_point_1d_sensitivity')
    
    plt.figure()
    demo, = plt.plot(t, x_demo, 'k', lw=3)
    for i in reversed(range(len(sols))):
        #print(u_vals[i], my_map(u_vals[i], 0, umax, 1, 0))
        repro, = plt.plot(t, sols[i], 'r', lw=3, alpha=my_map(u_vals[i], 0, umax, 1, 0))
    via, = plt.plot([t[ind], t[ind]], [x_demo[ind] - viapoint_freedom, x_demo[ind] + viapoint_freedom + umax], 'b.-', lw=2, ms=12)
    end, = plt.plot(t[0], x_demo[0], 'k.', ms=12)
    end, = plt.plot(t[-1], x_demo[-1], 'k.', ms=12)
    plt.legend((demo, end, repro, via), ('Demonstration', 'Fixed Constraints', 'Perturbed Reproductions', 'Via-Point Perturbations'), ncol=1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('t')
    plt.ylabel('x')
    plt.tight_layout()
    #mysavefig(plt.gcf(), '../pictures/via_point_1d_experiment/via_point_1d_perturbations')
    
    plt.show()

    
if __name__ == '__main__':
    main()