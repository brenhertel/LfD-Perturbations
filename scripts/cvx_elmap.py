import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

class ElMap_Perturbation_Analysis(object):

    def __init__(self, traj, stretch=0.01, bend=0.001):
        self.tgt_traj = traj
        self.n_pts, self.n_dims = np.shape(self.tgt_traj)
        self.num_points = np.size(self.tgt_traj)
        if self.n_dims == 1:
            self.traj_x = self.tgt_traj
            self.traj_stacked = self.traj_x
        elif self.n_dims == 2:
            self.traj_x = np.reshape(self.tgt_traj[:, 0], (self.n_pts, 1))
            self.traj_y = np.reshape(self.tgt_traj[:, 1], (self.n_pts, 1))
            self.traj_stacked = np.vstack((self.traj_x, self.traj_y))
        elif self.n_dims == 3:
            self.traj_x = np.reshape(self.tgt_traj[:, 0], (self.n_pts, 1))
            self.traj_y = np.reshape(self.tgt_traj[:, 1], (self.n_pts, 1))
            self.traj_z = np.reshape(self.tgt_traj[:, 2], (self.n_pts, 1))
            self.traj_stacked = np.vstack((self.traj_x, self.traj_y, self.traj_z))
            self.n_pts2 = 2*self.n_pts
        else:
            print("Too many dimensions! n_dims must be <= 3")
            exit()
        self.stretch_const = stretch
        self.bend_const = bend
        
    def setup_problem(self):
        K = np.diag(np.ones(self.num_points))
        I = np.eye(self.num_points)
        e1 = np.diag(-1*np.ones(self.num_points-1))
        e2 = np.diag(np.ones(self.num_points-1))
        E = np.zeros((self.num_points-1, self.num_points))
        E[:,0:self.num_points-1]+= e1
        E[:,1:self.num_points] += e2
        if self.n_dims >= 2:
            E[self.n_pts-1, self.n_pts-1] = 0
            E[self.n_pts-1, self.n_pts] = 0
        if self.n_dims == 3:
            E[self.n_pts2-1, self.n_pts2-1] = 0
            E[self.n_pts2-1, self.n_pts2] = 0
        r1 = np.diag(np.ones(self.num_points-2))
        r2 = -2*np.diag(np.ones(self.num_points-2))
        R = np.zeros((self.num_points-2, self.num_points))
        R[:,0:self.num_points-2] += r1
        R[:,1:self.num_points-1] += r2
        R[:,2:self.num_points] += r1
        if self.n_dims >= 2:
            R[self.n_pts-2, self.n_pts-2] = 0
            R[self.n_pts-2, self.n_pts-1] = 0
            R[self.n_pts-2, self.n_pts] = 0
            R[self.n_pts-1, self.n_pts-1] = 0
            R[self.n_pts-1, self.n_pts] = 0
            R[self.n_pts-1, self.n_pts+1] = 0
        if self.n_dims == 3:
            R[self.n_pts2-2, self.n_pts2-2] = 0
            R[self.n_pts2-2, self.n_pts2-1] = 0
            R[self.n_pts2-2, self.n_pts2] = 0
            R[self.n_pts2-1, self.n_pts2-1] = 0
            R[self.n_pts2-1, self.n_pts2] = 0
            R[self.n_pts2-1, self.n_pts2+1] = 0  
        
        #self.K = K
        #self.I = I
        #self.E = E
        #self.R = R
        
        #print(R[self.n_pts-3:self.n_pts+3, self.n_pts-3:self.n_pts+3])
           
        #ex = E @ self.traj_stacked
        #rx = R @ self.traj_stacked
            
        #print(ex[self.n_pts-3:self.n_pts+3])
        #print(rx[self.n_pts-3:self.n_pts+3])
            
        #print(np.shape(self.traj_stacked))
        self.x = cp.Variable(np.shape(self.traj_stacked))
        #self.objective = cp.Minimize(cp.norm(cp.square(K @ self.traj_stacked - self.x), 2) 
        #                            + self.stretch_const * cp.norm(cp.square(E @ self.x), 2)
        #                            + self.bend_const * cp.norm(cp.square(R @ self.x), 2))
        self.objective = cp.Minimize(cp.sum_squares(I @ self.x - K @ self.traj_stacked) 
                                    + self.stretch_const * cp.sum_squares(E @ self.x)
                                    + self.bend_const * cp.sum_squares(R @ self.x))
        return self.x
        
    def solve_problem(self, constraints, disp=True):
        self.consts = constraints
        
        self.problem = cp.Problem(self.objective, self.consts)
        self.problem.solve(verbose=disp)
        
        if disp:
            print("status:", self.problem.status)
            print("optimal value", self.problem.value)
            #print("optimal var", x.value)
            for i in range(len(self.consts)):
                print("dual value for constraint " + str(i), ": ", constraints[i].dual_value)
            
        if self.n_dims == 1:
            self.sol = self.x.value
        elif self.n_dims == 2:
            self.sol = np.hstack((self.x.value[:self.n_pts], self.x.value[self.n_pts:]))
        elif self.n_dims == 3:
            self.sol = np.hstack((self.x.value[:self.n_pts], self.x.value[self.n_pts:self.n_pts2], self.x.value[self.n_pts2:]))
        
        return self.sol
        
    def plot_solved_problem(self):
        if self.n_dims == 1:
            fig = plt.figure()
            plt.plot(self.tgt_traj, 'k', lw=5, label="Demo")
            plt.plot(self.sol, 'r', lw=5, label="Repro")
            return fig
        if self.n_dims == 2:
            fig = plt.figure()
            plt.plot(self.traj_x, self.traj_y, 'k', lw=3, label="Demo")
            plt.plot(self.sol[:, 0], self.sol[:, 1], 'r', lw=3, label="Repro")
            return fig
        if self.n_dims == 3:
            print("3D PLOTTING NOT IMPLEMENTED YET")
        return
        
## Same as above, but with slightly different (but equivalent) formulation
class ElMap_Perturbation_Analysis2(object):

    def __init__(self, traj, stretch=0.01, bend=0.001):
        self.tgt_traj = traj
        self.n_pts, self.n_dims = np.shape(self.tgt_traj)
        self.num_points = np.size(self.tgt_traj)
        if self.n_dims == 1:
            self.traj_x = self.tgt_traj
            self.traj_stacked = self.traj_x
        elif self.n_dims == 2:
            self.traj_x = np.reshape(self.tgt_traj[:, 0], (self.n_pts, 1))
            self.traj_y = np.reshape(self.tgt_traj[:, 1], (self.n_pts, 1))
            self.traj_stacked = np.vstack((self.traj_x, self.traj_y))
        elif self.n_dims == 3:
            self.traj_x = np.reshape(self.tgt_traj[:, 0], (self.n_pts, 1))
            self.traj_y = np.reshape(self.tgt_traj[:, 1], (self.n_pts, 1))
            self.traj_z = np.reshape(self.tgt_traj[:, 2], (self.n_pts, 1))
            self.traj_stacked = np.vstack((self.traj_x, self.traj_y, self.traj_z))
            self.n_pts2 = 2*self.n_pts
        else:
            print("Too many dimensions! n_dims must be <= 3")
            exit()
        self.stretch_const = stretch
        self.bend_const = bend
        
    def setup_problem(self):
        I = np.eye(self.num_points)
        K = np.diag(np.ones(self.num_points))
        e1 = np.diag(-1*np.ones(self.num_points-1))
        e2 = np.diag(np.ones(self.num_points-1))
        E = np.zeros((self.num_points-1, self.num_points))
        E[:,0:self.num_points-1]+= e1
        E[:,1:self.num_points] += e2
        if self.n_dims >= 2:
            E[self.n_pts-1, self.n_pts-1] = 0
            E[self.n_pts-1, self.n_pts] = 0
        if self.n_dims == 3:
            E[self.n_pts2-1, self.n_pts2-1] = 0
            E[self.n_pts2-1, self.n_pts2] = 0
        r1 = np.diag(np.ones(self.num_points-2))
        r2 = -2*np.diag(np.ones(self.num_points-2))
        R = np.zeros((self.num_points-2, self.num_points))
        R[:,0:self.num_points-2] += r1
        R[:,1:self.num_points-1] += r2
        R[:,2:self.num_points] += r1
        if self.n_dims >= 2:
            R[self.n_pts-2, self.n_pts-2] = 0
            R[self.n_pts-2, self.n_pts-1] = 0
            R[self.n_pts-2, self.n_pts] = 0
            R[self.n_pts-1, self.n_pts-1] = 0
            R[self.n_pts-1, self.n_pts] = 0
            R[self.n_pts-1, self.n_pts+1] = 0
        if self.n_dims == 3:
            R[self.n_pts2-2, self.n_pts2-2] = 0
            R[self.n_pts2-2, self.n_pts2-1] = 0
            R[self.n_pts2-2, self.n_pts2] = 0
            R[self.n_pts2-1, self.n_pts2-1] = 0
            R[self.n_pts2-1, self.n_pts2] = 0
            R[self.n_pts2-1, self.n_pts2+1] = 0  
        
        #print(R[self.n_pts-3:self.n_pts+3, self.n_pts-3:self.n_pts+3])
           
        #ex = E @ self.traj_stacked
        #rx = R @ self.traj_stacked
            
        #print(ex[self.n_pts-3:self.n_pts+3])
        #print(rx[self.n_pts-3:self.n_pts+3])
            
        #print(np.shape(self.traj_stacked))
        Center = I + self.stretch_const * np.transpose(E) @ E + self.bend_const * np.transpose(R) @ R
        self.x = cp.Variable(np.shape(self.traj_stacked))
        #print(np.shape(self.x))
        #print(np.shape(Center))
        #print(self.x)
        #print(Center)
        #print(np.shape(self.x.T @ Center))
        #self.objective = cp.Minimize(self.x.T @ Center @ self.x - np.transpose(K @ self.traj_stacked) @ (2 * (I @ self.x) - (K @ self.traj_stacked)))
        
        self.objective = cp.Minimize(cp.square(cp.norm(I @ self.x - K @ self.traj_stacked, 2)) 
                                    + self.stretch_const * cp.square(cp.norm(E @ self.x, 2))
                                    + self.bend_const * cp.square(cp.norm(R @ self.x, 2)))
        #self.objective = cp.Minimize(self.x.T @ np.transpose(I) @ I @ self.x + np.transpose(K @ self.traj_stacked) @ (K @ self.traj_stacked) - 2 * np.transpose(K @ self.traj_stacked) @ (I @ self.x)
        #                            + self.stretch_const * cp.norm(cp.square(E @ self.x), 2)
        #                            + self.bend_const * cp.norm(cp.square(R @ self.x), 2))
        return self.x
        
    def solve_problem(self, constraints, disp=True):
        self.consts = constraints
        
        self.problem = cp.Problem(self.objective, self.consts)
        self.problem.solve(verbose=disp)
        
        if disp:
            print("status:", self.problem.status)
            print("optimal value", self.problem.value)
            #print("optimal var", x.value)
            for i in range(len(self.consts)):
                print("dual value for constraint " + str(i), ": ", constraints[i].dual_value)
            
        if self.n_dims == 1:
            self.sol = self.x.value
        elif self.n_dims == 2:
            self.sol = np.hstack((self.x.value[:self.n_pts], self.x.value[self.n_pts:]))
        elif self.n_dims == 3:
            self.sol = np.hstack((self.x.value[:self.n_pts], self.x.value[self.n_pts:self.n_pts2], self.x.value[self.n_pts2:]))
        
        return self.sol
        
    def plot_solved_problem(self):
        if self.n_dims == 1:
            fig = plt.figure()
            plt.plot(self.tgt_traj, 'k', lw=5, label="Demo")
            plt.plot(self.sol, 'r', lw=5, label="Repro")
            return fig
        if self.n_dims == 2:
            fig = plt.figure()
            plt.plot(self.traj_x, self.traj_y, 'k', lw=3, label="Demo")
            plt.plot(self.sol[:, 0], self.sol[:, 1], 'r', lw=3, label="Repro")
            return fig
        if self.n_dims == 3:
            print("3D PLOTTING NOT IMPLEMENTED YET")
        return

def generate_interpolator(PA, constraint_gen, lamda, min_val=float('-inf'), max_val=float('inf'), alpha=0.95, cst_args=[]):
    
    interpx = [max(min_val, -2*alpha/lamda), max(min_val, -alpha/lamda), 0, min(max_val, alpha/lamda), min(max_val, 2*alpha/lamda)]
    print(interpx[0])
    print(interpx[1])
    print(interpx[0] == interpx[1])
    while interpx[0] == interpx[1]:
        interpx.pop(0)
    while interpx[-2] == interpx[-1]:
        interpx.pop(-1)
    print(interpx)
    interpy = []
    sols = []
    for u in interpx:
        x_prob = PA.setup_problem()
        constraints = constraint_gen(PA, u, cst_args)
        sol = PA.solve_problem(constraints, disp=False)
        interpy.append(PA.problem.value)
        sols.append(sol)
        
    print(interpx)
    print(interpy)
    f = interpolate.interp1d(interpx, interpy, kind='cubic', fill_value='extrapolate')
    return f, interpx, interpy, sols
        
def generate_sensitivity(PA, constraint_gen, min_u, max_u, n_u=50, cst_args=[]):
    
    u_vals = np.linspace(min_u, max_u, n_u)
    
    vals = []
    sols = []
    for u in u_vals:
        x_prob = PA.setup_problem()
        constraints = constraint_gen(PA, u, cst_args)
        sol = PA.solve_problem(constraints, disp=False)
        #print(sol[0, :])
        vals.append(PA.problem.value)
        sols.append(sol)
        
    return vals, u_vals, sols
        
if __name__ == '__main__':
    
    # demonstration
    num_points = 50
    t = np.linspace(0, 10, num_points).reshape((num_points, 1))
    x_demo = np.sin(t) + 0.01 * t**2 - 0.05 * (t-5)**2
    PA = ElMap_Perturbation_Analysis(x_demo, stretch=5.0, bend=2.0)
    x_prob = PA.setup_problem()
    initial_freedom = 0.5
    endpoint_freedom = 0.1
    constraints = [cp.abs(x_prob[0] - x_demo[0]) - initial_freedom <= 0, cp.abs(x_prob[-1] - x_demo[-1]) - endpoint_freedom <= 0]
    sol = PA.solve_problem(constraints)
    fig = PA.plot_solved_problem()
    plt.plot([0, 0], [x_demo[0] - initial_freedom, x_demo[0] + initial_freedom], 'g.-', ms=12, label="Initial Freedom")
    plt.plot([num_points-1, num_points-1], [x_demo[-1] - endpoint_freedom, x_demo[-1] + endpoint_freedom], 'g.-', ms=12, label="Endpoint Freedom")
    plt.legend()
    plt.show()