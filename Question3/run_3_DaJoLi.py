import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

# our external file:
from DM_MLP import DM_MLP


if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Load data & Split into Train and Test set:
        file_path = "dataPoints.xlsx"
        dataset = pd.read_excel(file_path)
        X_train, X_test, y_train, y_test = train_test_split(dataset[['x1','x2']],dataset['y'], train_size = 0.85, random_state = 1868264)
        
        # We fix the hyperparameters
        N = 12  
        sigma = 0.342857
        ro = 8.07143e-08
        
        # Initialize EL_MLP
        dm_mlp1 = DM_MLP(X_train, y_train,
                         N, ro, sigma)
        
        # Train model 
        dm_mlp1.train()
        
        # Plot estimated function
        dm_mlp1.plot_3D(100)
        
        # 1. Number of neurons:
        print(N)
        # 2. Sigma :
        print(sigma)
        # 3. Rho:
        print(ro)
        # 4. Other hyperparameters:
        print() 
        # 5. Optimization solver:
        print('CG ','BFGS')
        # 6. Number of function evaluations:
        print(dm_mlp1.nfev)
        # 7. Number of gradient evaluations: 
        print(dm_mlp1.njev)
        # 8. Time for optimizing the network
        print(dm_mlp1.optimizing_time)
        # 9. Train error:
        print(dm_mlp1.MSE(X_train, y_train, dm_mlp1.best_u, dm_mlp1.best_v))
        # 10. Test error:
        print(dm_mlp1.MSE(X_test, y_test, dm_mlp1.best_u, dm_mlp1.best_v))