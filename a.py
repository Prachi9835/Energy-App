
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
# Replace 'energy_data.csv' with the actual path to your dataset
df = pd.read_csv('Normalized_Energy_Consumption.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Check for missing values and handle them if any
df.fillna(method='ffill', inplace=True)

# Feature Selection
# Let's assume you are using all features except 'Standby_kWh' as input features
X = df[['Minimum_kW', 'Summer_kWh', 'Winter_kWh', 'Rainy_kWh']]  # Input features
y = df['Standby_kWh']  # Target feature

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R² Score: {r2}')

# Plot the predicted vs actual energy consumption for the test set
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Standby_kWh')
plt.plot(y_pred, label='Predicted Standby_kWh', linestyle='--')
plt.xlabel('Samples')
plt.ylabel('Standby_kWh')
plt.title('Actual vs Predicted Standby_kWh')
plt.legend()
plt.show()