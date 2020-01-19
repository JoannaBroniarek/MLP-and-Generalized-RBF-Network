import numpy as np
from scipy.optimize import minimize
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA

class DM_MLP(object):
    """
    Extreme learning MultiLayer Perceptron with ONE hidden layer.
    
    Parameters
    -----------
    X : {array-like}; shape = (number_of_samples, number_of_features)
        Training vectors.
    
    y : {array-like}; shape = (number_of_samples)
        Target vector.
    
    N : {int}
        Optimal number of neuron.
    
    ro : {float}
        Optimal regularization parameter.
        
    sigma : {float}
        Optimal spread.
    
    best_v : {array-like}; shape = N 
        Best parameters obtained during a training phase.
        
    best_u : {array-like}; 
        Best parameters (b, W) obtained during a training phase.
    
    nfev, njev : {int}
        Number of functions and gradient evaluations.
    
    optimizing_time : {float}
        Time for optimizing the network.
    
    num_feat : {int},
        Number_of_features.
    
    """

    def __init__(self, X, y, N, ro, sigma):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.N = N
        self.ro = ro
        self.sigma = sigma
        self.best_v = None
        self.best_u = None
        self.nfev = None
        self.njev = None
        self.optimizing_time = None
        self.P, self.num_feat = X.shape

    
    def init_v(self):
        """  
        Initialize v as a vector. 
        """
        
        np.random.seed(1843896)
        return np.random.normal(0,3, size = (self.N,))
    
    
    def init_u(self):
        
        np.random.seed(1843896)
        b = np.random.normal(0,0.2, size = (self.N,))
        np.random.seed(1843896)
        W = np.random.normal(0,3,size = (self.N,self.num_feat))
        
        return np.concatenate([b,W.reshape(-1,)])
    
    
    @staticmethod
    def tanh(t,sigma):
        return 1/(1+1/np.exp(2*sigma*t)) - 1/(np.exp(2*sigma*t)+1)
    
    
    def f_predict(self, X, u, v): 
        """ 
        Predicts the target values for a feature matrix X.
        Activation function is a hyperbolic tangent.
        """
        X = np.concatenate((X, -1*np.ones(X.shape[0]).reshape(-1,1)), axis=1)
        
        W = u[self.N:].reshape(self.N, self.num_feat)
        b = u[:self.N].reshape(-1,1)
        U = np.concatenate((W,b), axis=1)
        
        linear_part = U.dot(X.T)
        activated = DM_MLP.tanh(linear_part, self.sigma)
        return (v.T).dot(activated)
    
    
    def Error_train(self, u, v):
        """  Regularized training error. """
        y_pred = self.f_predict(self.X, u, v)
        error = np.mean((y_pred - self.y)**2) / 2
        error += self.ro*(np.inner(u,u)+np.inner(v,v))
        return error
    
    def switched_Error_train(self, v, u):
        """  Regularized training error. """
        
        y_pred = self.f_predict(self.X, u, v)
        error = np.mean((y_pred - self.y)**2) / 2
        error += self.ro*(np.inner(u,u)+np.inner(v,v))
        return error
    
    def MSE(self, X_val, y_val, u, v):
        """  Mean Square Error. """
        
        y_pred = self.f_predict(X_val, u, v)
        return np.mean((y_pred - y_val)**2)
    

    def gradient_E_v(self, v, u):
        W = u[self.N:].reshape(self.N, self.num_feat)
        b = u[:self.N]
        
        grad_f_v = DM_MLP.tanh(W.dot(self.X.T) - b.reshape(-1,1), self.sigma)
        gradient_v = (grad_f_v.dot((self.f_predict(self.X,u,v)-self.y).reshape(-1,1)))/self.X.shape[0] + 2*self.ro*v.reshape(-1,1)
        return gradient_v.reshape(-1,)
    
    
    def gradient_E_u(self, u, v):
        
        W = u[self.N:].reshape(self.N, self.num_feat)
        b = u[:self.N].reshape(-1,1)
        U = np.concatenate((W,b), axis=1)
        
        # Add to data matrix X the last column with minus ones.
        X = np.concatenate((self.X, -1*np.ones(self.P).reshape(-1,1)), axis=1)
        
        # Sequential stepd to calculate gradient E wrt. u
        activated_pow2 = (DM_MLP.tanh(U.dot(X.T), self.sigma))**2
        
        minus_act2 = self.sigma*(1 - activated_pow2)

        minus_act2_times_X = minus_act2.reshape(self.N,self.P,1) * X
        
        grad_f = minus_act2_times_X.transpose(1, 2, 0) * v.reshape(-1,)
        
        pred_err =  self.f_predict(self.X, u, v).reshape(self.P, 1) - self.y.reshape(self.P, 1)
        
        grad_part1 = grad_f.T.dot(pred_err).reshape(self.N, self.num_feat+1)
        
        grad_E = grad_part1/self.P + 2*self.ro*U
        
        # Flatten 
        W_order = grad_E[:,:self.num_feat].flatten(order='C')
        b_order = grad_E.flatten(order='F')[-self.N:]
        return np.concatenate([b_order, W_order])

    
    def train(self):
        """  The training phase. """
                
        v = self.init_v()
        u = self.init_u()
        
        epsilon1 = 10**-5
        epsilon2 = 5*10**-5
        
        start_time = time()
        
        stopping_criteria = False
        while not stopping_criteria:
            minimization_result = minimize(self.switched_Error_train, v, method = 'CG', args = (u), jac = self.gradient_E_v)
            v = minimization_result.x
            
            minimization_result = minimize(self.Error_train, u, method = 'BFGS', args = (v), jac = self.gradient_E_u)
            u = minimization_result.x
            
            gradient_u = self.gradient_E_u(u, v).reshape(1,-1)
            gradient_u_norm = LA.norm(gradient_u)

            gradient_v = self.gradient_E_v(v, u).reshape(1,-1)
            gradient_v_norm = LA.norm(gradient_v)
            
            stopping_criteria = gradient_u_norm < epsilon1 and gradient_v_norm < epsilon2
            
        opt_time = time()- start_time
    
        self.best_v = v
        self.best_u = u
        self.nfev = minimization_result.nfev
        self.njev = minimization_result.njev
        self.optimizing_time = opt_time
        return self


    def plot_3D(self, space):
        """ 
        Plots the surface of predicted function.  
        
        Parameters
        ----------
        space : {int}
            A number that specifies the density of a mesh.
            
        """
        
        fig = plt.figure()
        ax = Axes3D(fig) 
       
        x = np.linspace(-2, 2, space).reshape(space,1) 
        y = np.linspace(-2, 2, space).reshape(space,1)
        X, Y = np.meshgrid(x, y)
        
        # Build XY as a matrix with all (x1, x2) coordinates.
        # dim(XY) = (space**2, number_of_features)
        XY = np.array([X.flatten(),Y.flatten()]).T 
        XY = np.matrix(XY)
        Z = self.f_predict(XY, self.best_u, self.best_v)
        Z = Z.reshape(X.shape) 
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title('Predicted Function Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("Predicted_Function_Plot3.png")  