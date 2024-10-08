import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('Normalized_Energy_Consumption.csv')  # Adjust path if necessary

# Display the first few rows of the dataset to verify the structure
print(df.head())

# Check for missing values and handle them if any
df.fillna(method='ffill', inplace=True)

# Select features and target variable
X = df[['Minimum_kW', 'Summer_kWh', 'Winter_kWh', 'Rainy_kWh']]
y = df['Standby_kWh']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}

# Store predictions for each model
predictions = {}

# Evaluate models
best_model = None
best_mae = float('inf')

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred  # Store predictions
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² Score: {r2}\n")
    
    if mae < best_mae:
        best_mae = mae
        best_model = model

print(f"The best model is {type(best_model)} with MAE: {best_mae}")

# Plotting actual vs predicted values for all models
plt.figure(figsize=(10, 6))

# Scatter plot for actual vs predicted for each model
for model_name, y_pred in predictions.items():
    plt.scatter(y_test, y_pred, alpha=0.5, label=model_name)

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')

# Set labels and title
plt.xlabel('Actual Standby Consumption (kWh)')
plt.ylabel('Predicted Standby Consumption (kWh)')
plt.title('Actual vs Predicted Standby Consumption for Different Models')
plt.legend()
plt.grid(True)
plt.show()