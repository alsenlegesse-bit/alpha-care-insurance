import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Task 4: Simple ML Model")
print("="*50)

# Create sample data
np.random.seed(42)
n = 1000

data = {
    'Age': np.random.randint(20, 70, n),
    'CarAge': np.random.randint(0, 15, n),
    'CarValue': np.random.normal(150000, 50000, n),
    'HasClaim': np.random.choice([0, 1], n, p=[0.8, 0.2]),
    'TotalPremium': np.random.normal(5000, 1000, n),
    'TotalClaims': np.random.exponential(1000, n)
}

df = pd.DataFrame(data)

# Simple feature engineering
df['RiskScore'] = df['CarAge'] * 0.1 + df['CarValue']/100000 * 0.2
df['TotalClaims'] = df['TotalClaims'] * (1 + df['RiskScore'])

# 1. Claim Severity Model
print("\n1. Claim Severity Model (Linear Regression):")
df_claims = df[df['TotalClaims'] > 0]

if len(df_claims) > 10:
    X = df_claims[['Age', 'CarAge', 'CarValue']]
    y = df_claims['TotalClaims']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   RMSE: R{rmse:,.2f}")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Top feature: {'CarValue' if abs(model.coef_[2]) > abs(model.coef_[0]) else 'Age'}")
else:
    print("   Not enough claim data")

# 2. Premium Optimization
print("\n2. Premium Optimization:")
current_avg_premium = df['TotalPremium'].mean()
predicted_risk = df['RiskScore'].mean() * 1000  # Simplified risk calculation
optimal_premium = current_avg_premium * (1 + predicted_risk/10000)

print(f"   Current avg premium: R{current_avg_premium:,.2f}")
print(f"   Risk-adjusted premium: R{optimal_premium:,.2f}")
print(f"   Suggested increase: {((optimal_premium/current_avg_premium)-1)*100:.1f}%")

print("\n" + "="*50)
print("Task 4 Complete!")
print("Business Recommendation: Adjust premiums based on car age and value")
