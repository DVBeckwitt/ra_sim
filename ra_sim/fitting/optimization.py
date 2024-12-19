from bayes_opt import BayesianOptimization

def run_bayesian_optimization(objective_function, pbounds, initial_params):
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    optimizer.probe(params=initial_params, lazy=True)
    optimizer.maximize(init_points=5, n_iter=50)
    return optimizer
