import numpy as np
from scipy.optimize import minimize
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EL_MLP(object):
    """
    Extreme learning MultiLayer Perceptron with ONE hidden layer.
    
    Parameters
    -----------
    X : {array-like}; shape = (number_of_samples, number_of_features)
        Training vectors.
    
    y : {array-like}; shape = (number_of_samples)
        Target vector.
        
    num_feat : {int},
        Number_of_features.
        
    method : {string}; 
        The solver for optimization. Default method is "BFGS".
    
    N : {int}
        Optimal number of neuron.
    
    ro : {float}
        Optimal regularization parameter.
        
    sigma : {float}
        Optimal spread.
            
    b : {array-like}; shape = N
        Randomly generated weight parameters.
        
    W : {array-like} shape = (N,number_of_features)
    
    best_v : {array-like}; shape = N 
        Best parameters obtained during a training phase.
    
    nfev, njev : {int}
        Number of functions and gradient evaluations.
    
    optimizing_time : {float}
        Time for optimizing the network.
    
    """

    def __init__(self, X, y, N, ro, sigma, method = 'BFGS'):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.num_feat = X.shape[1]
        self.method = method
        self.N = N
        self.ro = ro
        self.sigma = sigma
        np.random.seed(1843896) 
        self.b = np.random.normal(0,0.2, size = (self.N,1))
        np.random.seed(1843896) #TODO : must it be twice ?
        self.W = np.random.normal(0,3,size = (self.N, self.num_feat))
        self.best_v = None
        self.nfev = None
        self.njev = None
        self.optimizing_time = None

    def init_v(self):
        """  
        Initialize v as a vector. 
        """
        np.random.seed(1843896)
        return np.random.normal(0,3, size = (self.N,1))
    
    @staticmethod
    def tanh(t,sigma):
        return 1/(1+1/np.exp(2*sigma*t)) - 1/(np.exp(2*sigma*t)+1)
    
    
    def f_predict(self, X, v): 
        """ 
        Predicts the target values for a feature matrix X.
        Activation function is a hyperbolic tangent.
        """
        
        linear_part = self.W.dot(X.T) - self.b.reshape(self.N,1)
        activated = EL_MLP.tanh(linear_part, self.sigma)
        
        return (v.T).dot(activated)
    

    def Error_train(self, v):
        """  Regularized training error. """
       
        y_pred = self.f_predict(self.X, v)
        error = np.mean((y_pred - self.y)**2) / 2
        omega = np.concatenate([self.b.reshape(self.N,), self.W.flatten(), v])
        error += self.ro*np.inner(omega,omega)
        return error
    
    def MSE(self, X_val, y_val, v):
        """  Mean Square Error. """
        
        y_pred = self.f_predict(X_val, v)
        return np.mean((y_pred - y_val)**2)
    
    
    def train(self):
        """  The training phase. """
        
        v_0 = self.init_v()
        
        start_time = time()
        minimization_result = minimize(self.Error_train, v_0, method = self.method)
        opt_time = time()- start_time
        min_v = minimization_result.x
    
        self.best_v = min_v
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
        Z = self.f_predict(XY,self.best_v)
        Z = Z.reshape(X.shape) 
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title('Predicted Function Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("Predicted_Function_Plot21.png")  