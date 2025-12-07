import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print('TASK 4: MACHINE LEARNING MODELS')
print('=' * 50)

# Load data
df = pd.read_csv('data/insurance_data.csv')

# Prepare features
features = ['Age', 'CarValue']
categorical = ['Province', 'Gender', 'VehicleType']

# One-hot encoding
X = pd.get_dummies(df[features + categorical], drop_first=True)
y = df['TotalClaims']

print('Features:', X.shape[1])
print('Samples:', X.shape[0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print('Training samples:', X_train.shape[0])
print('Test samples:', X_test.shape[0])

# Train model
print()
print('Training Random Forest Model...')
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print()
print('Model Performance:')
print('RMSE:', rmse)
print('R-squared:', r2)

# Feature Importance
print()
print('Top 5 Features:')
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(5)

for idx, row in feature_importance.iterrows():
    print('  ', row['feature'], ':', row['importance'])

print()
print('Task 4 Complete''')
