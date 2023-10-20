from gplearn.genetic import SymbolicRegressor
import numpy as np

# Generate some example data
x = np.random.uniform(-10, 10, 1000)
y = np.sin(x) + np.cos(2 * x) + 0.5 * x + np.random.normal(0, 0.2, 1000)

# Create a symbolic regressor
est_gp = SymbolicRegressor(population_size=5000, generations=20, function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
                           const_range=(-10, 10), verbose=1)

# Fit the regressor to the data
est_gp.fit(x.reshape(-1, 1), y)

# Print the best symbolic expression
print(est_gp._program)

# Predict with the best expression
y_pred = est_gp.predict(x.reshape(-1, 1))