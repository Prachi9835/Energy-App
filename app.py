import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Map seasons to numerical values
season_mapping = {'Summer': 0, 'Winter': 1, 'Rainy': 2}

# Function to train a Random Forest model for cost prediction
def train_random_forest_model(df, cost_per_kwh):
    features = []
    targets = []

    for _, row in df.iterrows():
        for season in ['Summer', 'Winter', 'Rainy']:
            active_power = row[f'{season}_kWh']
            standby_power = row['Standby_kWh']
            season_numeric = season_mapping[season]  # Convert season to numeric

            # Iterate over different usage hours and prepare training data
            for hours in range(1, 25):
                daily_consumption = (active_power * hours) + (standby_power * (24 - hours))
                monthly_consumption = daily_consumption * 30
                estimated_cost = monthly_consumption * cost_per_kwh

                # Feature vector: [active_power, standby_power, usage_hours, season_numeric]
                features.append([active_power, standby_power, hours, season_numeric])
                targets.append(estimated_cost)

    # Convert to numpy arrays
    features = np.array(features)
    targets = np.array(targets)

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, targets)
    
    return model

# Updated optimize_usage function to use Random Forest
def optimize_usage_with_rf(appliance_name, season, cost_per_kwh, df, model):
    appliance_data = df[df['Appliance'] == appliance_name].iloc[0]
    active_power = appliance_data[f'{season}_kWh']
    standby_power = appliance_data['Standby_kWh']
    season_numeric = season_mapping[season]  # Convert season to numeric
    
    optimal_usage = None
    min_cost = float('inf')

    for usage_hours in range(1, 25):
        # Use the Random Forest model to predict the cost
        estimated_cost = model.predict([[active_power, standby_power, usage_hours, season_numeric]])[0]
        
        if estimated_cost < min_cost:
            min_cost = estimated_cost
            optimal_usage = usage_hours
    
    return optimal_usage, min_cost

# Function to predict electricity cost for one appliance
def predict_cost(appliance_name, usage_hours_per_day, season, cost_per_kwh, df):
    appliance_data = df[df['Appliance'] == appliance_name].iloc[0]
    active_power = appliance_data[f'{season}_kWh']
    standby_power = appliance_data['Standby_kWh']

    daily_consumption = (active_power * usage_hours_per_day) + (standby_power * (24 - usage_hours_per_day))
    monthly_consumption = daily_consumption * 30
    estimated_cost = monthly_consumption * cost_per_kwh
    return estimated_cost

# Streamlit Web Interface
st.title("Monthly Electricity Bill Estimation and Optimization in ₹")

def load_data():
    file_path = 'Normalized_Energy_Consumption.csv'
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' not found. Please make sure it is in the correct directory.")
        return None
    else:
        return pd.read_csv(file_path)

# Load dataset and extract appliance names
df = load_data()
if df is not None:
    if 'Appliance' not in df.columns:
        st.error("The 'Appliance' column is missing from the CSV file.")
    else:
        appliance_list = df['Appliance'].unique().tolist()
        selected_appliances = st.multiselect("Select appliances:", appliance_list)
        
        usage_hours = {}
        for appliance in selected_appliances:
            usage_hours[appliance] = st.number_input(f"Enter usage hours per day for {appliance}:", min_value=1, max_value=24, key=appliance)

        season = st.selectbox("Select season:", ['Summer', 'Winter', 'Rainy'])
        cost_per_kwh = st.number_input("Enter cost per kWh in ₹ (e.g., 8):", min_value=0.0, step=0.01)

        model = train_random_forest_model(df, cost_per_kwh)
        
        if st.button('Estimate Total Monthly Bill'):
            total_cost = 0
            for appliance in selected_appliances:
                hours_per_day = usage_hours[appliance]
                cost = predict_cost(appliance, hours_per_day, season, cost_per_kwh, df)
                total_cost += cost
                st.write(f"Estimated Monthly Bill for {appliance}: ₹{cost:.2f}")
            
            st.write(f"Total Estimated Monthly Bill: ₹{total_cost:.2f}")
        
        if st.button('Optimize Usage to Reduce Bill'):
            st.write("Optimizing usage hours for selected appliances...")
            total_optimized_cost = 0
            
            for appliance in selected_appliances:
                optimal_hours, optimized_cost = optimize_usage_with_rf(appliance, season, cost_per_kwh, df, model)
                total_optimized_cost += optimized_cost
                st.write(f"Optimal usage hours for {appliance}: {optimal_hours} hours/day")
                st.write(f"Optimized Monthly Bill for {appliance}: ₹{optimized_cost:.2f}")
            
            st.write(f"Total Optimized Monthly Bill: ₹{total_optimized_cost:.2f}")
