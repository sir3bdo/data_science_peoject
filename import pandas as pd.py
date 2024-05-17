import pandas as pd
from scipy import stats
import numpy as np

import statsmodels.api as sm
#load the dataset
data = pd.read_csv('satgpa.csv')

#display the dataset
print(data)
# Generate some example data
np.random.seed(0)
X = np.random.randn(100, 2)
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = X @ np.array([5, 3, -2]) + np.random.randn(100)

# Fit the model
model = sm.OLS(y, X).fit()

# Extract coefficients and standard errors
coefficients = model.params
standard_errors = model.bse

# Calculate the Z-values
z_values = coefficients / standard_errors

# Calculate the p-values from the Z-values
p_values = 2 * (1 - stats.norm.cdf(np.abs(z_values)))

# Print the results
for i, (coef, se, z, p) in enumerate(zip(coefficients, standard_errors, z_values, p_values)):
    print(f"Coefficient {i}:")
    print(f"  Estimate: {coef}")
    print(f"  Std Error: {se}")
    print(f"  Z-value: {z}")
    print(f"  p-value: {p}")
    print()

# Alternatively, you can print the model summary
print(model.summary())