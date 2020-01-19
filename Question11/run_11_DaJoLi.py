import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

# our external file:
from MLP import MLP


if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Load data & Split into Train and Test set:
        file_path = "dataPoints.xlsx"
        dataset = pd.read_excel(file_path)
        X_train, X_test, y_train, y_test = train_test_split(dataset[['x1','x2']],dataset['y'], train_size = 0.85, random_state = 1868264)
        
        # Initialize MLP
        mlp1 = MLP(X_train, y_train, method = 'BFGS')
        
        # Specify grid for hyper-parameters : [N, ro, sigma]
        grid = [range(5,21), 
                np.linspace(10**(-8), 10**(-6), 3), 
                np.linspace(1,2.5, 4)]
        
        # Train model 

        mlp1.train(grid_hyperparams=grid, n_cv=5)
        
        # Best hyperparameters:
        N, ro , sigma = mlp1.best_hyperparams
        
        # Best omega = [b, W, v]
        omega = mlp1.best_omega
        
        # Plot estimated function
        mlp1.plot_3D(100)
        
        # 1. Number of neurons:
        print(N)
        # 2. Sigma :
        print(sigma)
        # 3. Rho:
        print(ro)
        # 4. Other hyperparameters:
        print() 
        # 5. Optimization solver:
        print(mlp1.method)
        # 6. Number of function evaluations:
        print(mlp1.nfev)
        # 7. Number of gradient evaluations: 
        print(mlp1.njev)
        # 8. Time for optimizing the network
        print(mlp1.optimizing_time)
        # 9. Train error:
        print(mlp1.MSE(X_train, y_train, mlp1.best_omega, mlp1.best_hyperparams))
        # 10. Test error:
        print(mlp1.MSE(X_test, y_test, mlp1.best_omega, mlp1.best_hyperparams))
        
        # we save the result in a pickle file
        import pickle
        pickle.dump( mlp1.history, open( "mlp_history.p", "wb" ) )
        
        
    
    
    
    
    
    
    
    
    