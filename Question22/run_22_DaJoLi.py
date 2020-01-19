import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import random

# our external file:
from EL_RBF import EL_RBF


if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Load data & Split into Train and Test set:
        file_path = "dataPoints.xlsx"
        dataset = pd.read_excel(file_path)
        X_train, X_test, y_train, y_test = train_test_split(dataset[['x1','x2']],dataset['y'], train_size = 0.85, random_state = 1868264)
        
        # We fix the hyperparameters
        N = 15
        sigma = 0.5
        ro = 0.00001        
        
        # Initialize EL_MLP
        el_rbf1 = EL_RBF(X_train, y_train,
                         N, ro, sigma, 
                         method='BFGS')
      
        # Train model 
        el_rbf1.train()
        
        # Plot estimated function
        el_rbf1.plot_3D(100)
        
        # 1. Number of neurons:
        print(N)
        # 2. Sigma :
        print(sigma)
        # 3. Rho:
        print(ro)
        # 4. Other hyperparameters:
        print() 
        # 5. Optimization solver:
        print(el_rbf1.method)
        # 6. Number of function evaluations:
        print(el_rbf1.nfev)
        # 7. Number of gradient evaluations: 
        print(el_rbf1.njev)
        # 8. Time for optimizing the network
        print(el_rbf1.optimizing_time)
        # 9. Train error:
        print(el_rbf1.MSE(X_train, y_train, el_rbf1.best_v))
        # 10. Test error:
        print(el_rbf1.MSE(X_test, y_test, el_rbf1.best_v))