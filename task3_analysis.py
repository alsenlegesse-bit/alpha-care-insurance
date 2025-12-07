import pandas as pd
import numpy as np
import scipy.stats as stats

print("Task 3: Hypothesis Testing Started")
print("="*50)

# Try to load data
try:
    df = pd.read_csv('insurance_data.csv')
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except:
    # Create sample data if file doesn't exist
    print("Creating sample data for testing...")
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Province': np.random.choice(['GP', 'WC', 'KZN', 'EC'], n),
        'PostalCode': np.random.randint(1000, 2000, n),
        'Gender': np.random.choice(['M', 'F'], n),
        'TotalPremium': np.random.normal(5000, 1000, n),
        'TotalClaims': np.random.exponential(500, n)
    })

# Create metrics
df['Has_Claim'] = (df['TotalClaims'] > 0).astype(int)
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

# Test 1: Province differences (Chi-square)
print("\n1. Testing Province Risk Differences:")
contingency = pd.crosstab(df['Province'], df['Has_Claim'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"   Chi2 = {chi2:.3f}, p-value = {p:.4f}")
print(f"   Result: {'REJECT' if p < 0.05 else 'FAIL TO REJECT'} H₀")

# Test 2: Gender differences (T-test)
print("\n2. Testing Gender Risk Differences:")
male_claims = df[df['Gender']=='M']['Has_Claim']
female_claims = df[df['Gender']=='F']['Has_Claim']
t_stat, p = stats.ttest_ind(male_claims, female_claims)
print(f"   T-stat = {t_stat:.3f}, p-value = {p:.4f}")
print(f"   Result: {'REJECT' if p < 0.05 else 'FAIL TO REJECT'} H₀")

print("\n" + "="*50)
print("Task 3 Complete!")
