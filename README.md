# MLP and Generalized RBF Network

University Project with the scope of the Optimization Methods for Machine Learning Course.

The goal of this project was to implement neural networks to solve a regression problem. It was needed to reconstruct in the region [−2,2] × [−1,1] a two dimensional function, having only a data set obtained by randomly sampling 300 points. The image of original function was provided.

Project consists of three following questions:

* **Question 1 - Full minimization**

In this part, two shallow **Feedforward Neural Networks** (FNN with one hidden layer) were implemented: **a MLP and RBF networks**. The hyper-parameters  $\pi$ of the network were settled by means of an heuristic procedure and the parameters $\omega$ were settled by optimizing the regularized training error:
$$
E(\omega; \pi) = \frac{1}{2P} \sum_{p=1}^P (f(x^p) - y^p)^2 + \rho \parallel \omega \parallel^2
$$

* **Question 2 - Two block methods**

In this part the same as before two learning machines were considered:  **MLP and RBF networks**. Now the hyperparameters are fixed as the best ones found with the Grid-Search (**Extreme Learning**). Additionally, the parameters are divided in two block.

* **Question 3 - Decomposition method**

This time, only the **MLP is considered with the parameters divided in two blocks** as the question before. However, in this case there is **no parameters fixed**. The current methodology is to find the optimal value for one block fixing the other alternatively.

More detailed description of the work done and results can be found in the report pdf file **report.pdf**.

## Code organization

Organization of the code follows the professor's rules where for each question there had to be created a different folder with two files (example for a question **1.1**): 

* a file called run_ **11** _DaJoLi.py that must include code executed for solving the specific question
* a file with the complete code written that includes all the functions that are called from the run_ **11** _DaJoLi.py file.

## Short summary of results

![summary_table](/home/jb/Desktop/summary_table.png)

where the hyper-parameters are: 

N - the number of neurons N of the hidden layer

σ - the spread in the activation function 

ρ - the regularization parameter.



**Image of the best prediction (left) & Image of the original function (right)**

![]()

![]()

