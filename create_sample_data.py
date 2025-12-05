import pandas as pd
import numpy as np
import os

# Create sample insurance data
np.random.seed(42)
data = pd.DataFrame({
    'PolicyID': range(1000, 1100),
    'TotalPremium': np.random.uniform(1000, 10000, 100),
    'TotalClaims': np.random.exponential(2000, 100),
    'Province': np.random.choice(['Gauteng', 'Western Cape', 'KZN'], 100),
    'VehicleType': np.random.choice(['Sedan', 'SUV', 'Bakkie'], 100),
    'Gender': np.random.choice(['Male', 'Female'], 100)
})

# Make 70% claims zero
data.loc[np.random.choice(100, size=70), 'TotalClaims'] = 0

# Ensure directory exists
os.makedirs('data/raw', exist_ok=True)

# Save
data.to_csv('data/raw/insurance_sample.csv', index=False)
print(f"Created sample data: {len(data)} records")
print(f"File saved: data/raw/insurance_sample.csv")
