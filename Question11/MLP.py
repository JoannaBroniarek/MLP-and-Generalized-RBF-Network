import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import itertools
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

class MLP(object):
    """
    MultiLayer Perceptron with ONE hidden layer.
    
    Parameters
    -----------
    X : {array-like}; shape = (number_of_samples, number_of_features)
        Training vectors.
    
    y : {array-like}; shape = (number_of_samples)
        Target vector.
        
    method : {string}; 
        The solver for optimization. Default method is "BFGS".
    
    best_hyperparams : {[N, rho, sigma]} 
        Best hyperparameters obtained during a training phase.

    best_omega : {[b, W, v]} 
        Best parameters obtained during a training phase.
    
    nfev, njev : {int}
        Number of functions and gradient evaluations.
    
    optimizing_time : {float}
        Time for optimizing the network.
    
    num_feat : {int},
        Number_of_features.
    
    """

    def __init__(self, X, y, method='BFGS'):
        self.X = X.to_numpy() #TODO: chceck if we still need to keep it as attribute
        self.y = y.to_numpy()
        self.method = method
        self.best_hyperparams = []
        self.best_omega = None
        self.nfev = None
        self.njev = None
        self.optimizing_time = None
        self.num_feat = X.shape[1]
        self.history = None
    
    def init_omega(self, N):
        """  
        Initialize omega as a flattened vector of [b, W, v]. 
        
        Parameters
        -----------
        N : {int} - number of neurons in a hidden layer
        """
        
        np.random.seed(1843896)
        b = np.random.normal(0,0.2, size = (N,1))
        np.random.seed(1843896)
        W = np.random.normal(0,3,size = (N,self.num_feat))
        np.random.seed(1843896)
        v = np.random.normal(0,3, size = (N,1))

        return np.concatenate([b,W.reshape(-1,1),v])
    
    @staticmethod
    def tanh(t,sigma):
        return 1/(1+1/np.exp(2*sigma*t)) - 1/(np.exp(2*sigma*t)+1)
    
    
    def f_predict(self, X, omega, hyperparams): 
        """ 
        Predicts the target values for a feature matrix X.
        Activation function is a hyperbolic tangent.
        """
        
        N, ro, sigma = hyperparams        
        b = omega[:N]
        W = omega[N:N*self.num_feat+N].reshape(N, self.num_feat)
        v = omega[N*self.num_feat+N: ]

        linear_part = W.dot(X.T) - b.reshape(N,1)
        activated = MLP.tanh(linear_part, sigma)
        
        return (v.T).dot(activated)
    

    def Error_train(self, omega, X, y, hyperparams):
        """  Regularized training error. """
        
        N, ro , sigma = hyperparams
    
        y_pred = self.f_predict(X, omega, hyperparams)
        error = np.mean((y_pred - y)**2) / 2
        error += ro*np.inner(omega,omega)
        return error
    
    
    def MSE(self, X_val, y_val, omega, hyperparams):
        """  Mean Square Error. """
        
        y_pred = self.f_predict(X_val, omega, hyperparams)
        return np.mean((y_pred - y_val)**2)
    
    
    def train(self, grid_hyperparams, n_cv=5):
        """  
        The training phase. 
        
        Parameters
        ----------
        
        grid_hyperparams : {nested-list} ; 
            Specified grid for hyperparameters in following order [N, rho, sigma].

        n_cv : {int} ; default = 5
            Number of splits for cross validation into train and validation sets.
        
        """

        actual_error = np.inf
        
        grid = list(itertools.product(*grid_hyperparams))
        
        self.history = pd.DataFrame(index=np.arange(0, len(grid)),
                                    columns=('N', 'ro', 'sigma', 'validation_error', 'gradient_norm', 'nfev', 'njac'))
        
        counter = -1
        for N, ro, sigma in tqdm(grid):
            
            counter += 1
            
            omega_0 = self.init_omega(N)
            
            # n-fold Cross Validation
            
            Cv_errors_list = []
            kf = KFold(n_splits=n_cv, random_state=1868264)
            
            for train_index, val_index in kf.split(self.X):
                X_train, X_val = self.X[train_index], self.X[val_index]
                y_train, y_val = self.y[train_index], self.y[val_index]
                
                minimization_result = minimize(self.Error_train, omega_0, method = self.method, 
                                           args = (X_train, y_train, [N, ro, sigma]))
                min_omega = minimization_result.x  # best omega 
                val_error = self.MSE(X_val, y_val, min_omega, [N, ro, sigma])
                Cv_errors_list.append(val_error)
            
            Cv_error = np.mean(Cv_errors_list)
            
            if Cv_error < actual_error:
                actual_error = Cv_error
                self.best_hyperparams = [N,ro,sigma]
            
            actual_minimization = minimize(self.Error_train, omega_0, method = self.method, 
                                           args = (self.X, self.y, [N, ro, sigma]))
            
            gradient = minimization_result.jac
            gradient_norm = np.sqrt(np.inner(gradient,gradient))
            self.history.loc[counter] = [N, ro, sigma, Cv_error, gradient_norm, actual_minimization.nfev, actual_minimization.njev]
            
        # The final minimization is done with respect to the whole Training Set (together with validation set)
        omega_0 = self.init_omega(self.best_hyperparams[0])
        
        start_time = time()
        final_minimization = minimize(self.Error_train, omega_0, method = self.method, 
                                       args = (self.X, self.y, self.best_hyperparams))
        self.optimizing_time = time() - start_time
        self.best_omega = final_minimization.x
        self.nfev = final_minimization.nfev
        self.njev = final_minimization.njev

        # TODO : check convergence! + try with gradient (to discussion)
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
        Z = self.f_predict(XY,self.best_omega,self.best_hyperparams)
        Z = Z.reshape(X.shape) 
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title('Predicted Function Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("Predicted_Function_Plot11.png")
    
    
    
    
    
    
    