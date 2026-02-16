import statsmodels.api as sm
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# Load the dataset
House_Price = pd.read_csv('house_price_regression_dataset.csv')

# Define the independent variable (X) and the dependent variable (y)
X = House_Price[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
y = House_Price['House_Price']

# Add a constant to the independent variables (for the intercept)
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X)

results = model.fit()

# Print the summary of the regression results
print(results.summary())