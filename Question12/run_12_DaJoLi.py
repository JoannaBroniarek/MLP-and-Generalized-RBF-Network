import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

# our external file:
from RBF import RBF


if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Load data & Split into Train and Test set:
        file_path = "dataPoints.xlsx"
        dataset = pd.read_excel(file_path)
        X_train, X_test, y_train, y_test = train_test_split(dataset[['x1','x2']],dataset['y'], train_size = 0.85, random_state = 1868264)
        
        # Initialize RBF
        rbf1 = RBF(X_train, y_train, method='CG')
        
        # Specify grid for hyper-parameters : [N, ro, sigma]
        grid = [range(15,16), 
                np.linspace(10**(-5), 10**(-3), 1), 
                np.linspace(0.5,1.5,1)]
        
        # Train model 
        rbf1.train(grid_hyperparams=grid, n_cv=5)
        
        # Best hyperparameters:
        N, ro , sigma = rbf1.best_hyperparams
        
        # Best omega = [b, W, v]
        omega = rbf1.best_omega
        
        # Plot estimated function
        rbf1.plot_3D(100)
        
        # 1. Number of neurons:
        print(N)
        # 2. Sigma :
        print(sigma)
        # 3. Rho:
        print(ro)
        # 4. Other hyperparameters:
        print() 
        # 5. Optimization solver:
        print(rbf1.method)
        # 6. Number of function evaluations:
        print(rbf1.nfev)
        # 7. Number of gradient evaluations: 
        print(rbf1.njev)
        # 8. Time for optimizing the network
        print(rbf1.optimizing_time)
        # 9. Train error:
        print(rbf1.MSE(X_train, y_train, rbf1.best_omega, rbf1.best_hyperparams))
        # 10. Test error:
        print(rbf1.MSE(X_test, y_test, rbf1.best_omega, rbf1.best_hyperparams))
        
        # we save the result in a pickle file
        import pickle
        pickle.dump( mlp1.history, open( "mlp_history.p", "wb" ) )
        