import pandas as pd
import numpy as np

print('TASK 1: EXPLORATORY DATA ANALYSIS')
print('=' * 50)

# Create sample data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Province': np.random.choice(['Gauteng', 'Western Cape', 'KZN'], n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'TotalPremium': np.random.normal(5000, 1000, n_samples),
    'TotalClaims': np.random.exponential(800, n_samples),
    'VehicleType': np.random.choice(['Sedan', 'SUV', 'Truck'], n_samples),
    'Age': np.random.randint(20, 70, n_samples),
    'CarValue': np.random.normal(150000, 50000, n_samples)
})

# Ensure positive values
data['TotalPremium'] = data['TotalPremium'].clip(1000)
data['TotalClaims'] = data['TotalClaims'].clip(0, 10000)
data['CarValue'] = data['CarValue'].clip(50000)

print('Dataset Shape:', data.shape)
print()
print('Data Summary:')
print(data.describe())

# Calculate loss ratio
data['LossRatio'] = data['TotalClaims'] / data['TotalPremium']
print()
print('Average Loss Ratio:', data['LossRatio'].mean())

# Save to CSV
data.to_csv('data/insurance_data.csv', index=False)
print()
print('Data saved to: data/insurance_data.csv')
print('Task 1 Complete''')
