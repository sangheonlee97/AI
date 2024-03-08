param_bounds = {'X1' : (-1, 5),
                'X2' : (0, 4)}

def y_function(X1, X2):
    return -X1**2 - (X2 - 2)**2 + 10

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,
    pbounds=param_bounds,
    random_state=42,
)

optimizer.maximize(
    init_points=5,
    n_iter=20)
print(optimizer.max)