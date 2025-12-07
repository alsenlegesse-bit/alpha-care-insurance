import pandas as pd
import numpy as np
import scipy.stats as stats

print('TASK 3: HYPOTHESIS TESTING')
print('=' * 50)

# Load data
df = pd.read_csv('data/insurance_data.csv')

# Create metrics
df['HasClaim'] = (df['TotalClaims'] > 500).astype(int)
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

print('Total Policies:', len(df))
print('Claim Rate:', df['HasClaim'].mean())
print('Average Margin:', df['Margin'].mean())

print()
print('=' * 40)
print('HYPOTHESIS TEST RESULTS')
print('=' * 40)

# Test 1: Province differences
print()
print('Test 1: Province Risk Differences')
province_claims = df.groupby('Province')['HasClaim'].mean()
print('Claim rates by province:')
for prov, rate in province_claims.items():
    print('  ', prov, ':', rate)

# Chi-square test
contingency = pd.crosstab(df['Province'], df['HasClaim'])
chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
print('Chi-square test: chi2 =', chi2, 'p =', p_val)
if p_val < 0.05:
    print('Result: REJECT H0')
else:
    print('Result: FAIL TO REJECT H0')

# Test 2: Gender differences
print()
print('Test 2: Gender Risk Differences')
gender_claims = df.groupby('Gender')['HasClaim'].mean()
print('Claim rates by gender:')
for gender, rate in gender_claims.items():
    print('  ', gender, ':', rate)

# T-test
male_claims = df[df['Gender']=='Male']['TotalClaims']
female_claims = df[df['Gender']=='Female']['TotalClaims']
t_stat, p_val = stats.ttest_ind(male_claims, female_claims)
print('T-test: t =', t_stat, 'p =', p_val)
if p_val < 0.05:
    print('Result: REJECT H0')
else:
    print('Result: FAIL TO REJECT H0')

print()
print('Task 3 Complete''')
