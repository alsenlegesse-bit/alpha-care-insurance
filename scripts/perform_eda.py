import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

class SimpleEDA:
    def __init__(self):
        self.df = None
        
    def load_sample_data(self):
        """Create sample data if real data isn't available yet"""
        print("Creating sample data for demonstration...")
        
        # Create sample insurance data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'PolicyID': range(1, n_samples + 1),
            'TotalPremium': np.random.uniform(500, 5000, n_samples),
            'TotalClaims': np.random.uniform(0, 10000, n_samples),
            'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal', 
                                         'Eastern Cape', 'Free State'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'VehicleType': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Truck', 'Motorcycle'], n_samples),
            'Make': np.random.choice(['Toyota', 'Volkswagen', 'Ford', 'BMW', 'Mercedes'], n_samples),
            'SumInsured': np.random.uniform(50000, 500000, n_samples),
            'TransactionMonth': pd.date_range('2014-02-01', periods=n_samples, freq='D'),
        }
        
        self.df = pd.DataFrame(data)
        
        # Make some claims zero (no claim scenario)
        self.df.loc[np.random.choice(n_samples, size=700, replace=False), 'TotalClaims'] = 0
        
        print(f"Created sample data with {n_samples} records")
        print("Columns:", list(self.df.columns))
        
        return self.df
    
    def basic_analysis(self):
        """Perform basic EDA"""
        print("\n" + "="*50)
        print("BASIC EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print(f"\n1. Dataset Shape: {self.df.shape}")
        print(f"   Rows: {self.df.shape[0]}")
        print(f"   Columns: {self.df.shape[1]}")
        
        print("\n2. Data Types:")
        print(self.df.dtypes)
        
        print("\n3. Missing Values:")
        print(self.df.isnull().sum())
        
        print("\n4. Basic Statistics:")
        print(self.df.describe())
    
    def calculate_loss_ratio(self):
        """Calculate loss ratio"""
        print("\n" + "="*50)
        print("LOSS RATIO ANALYSIS")
        print("="*50)
        
        total_claims = self.df['TotalClaims'].sum()
        total_premium = self.df['TotalPremium'].sum()
        loss_ratio = (total_claims / total_premium) * 100
        
        print(f"Total Claims: R{total_claims:,.2f}")
        print(f"Total Premium: R{total_premium:,.2f}")
        print(f"Loss Ratio: {loss_ratio:.2f}%")
        
        # By Province
        print("\nLoss Ratio by Province:")
        province_stats = self.df.groupby('Province').agg({
            'TotalClaims': 'sum',
            'TotalPremium': 'sum'
        })
        province_stats['LossRatio'] = (province_stats['TotalClaims'] / province_stats['TotalPremium']) * 100
        print(province_stats.sort_values('LossRatio', ascending=False))
        
        # By Gender
        print("\nLoss Ratio by Gender:")
        gender_stats = self.df.groupby('Gender').agg({
            'TotalClaims': 'sum',
            'TotalPremium': 'sum'
        })
        gender_stats['LossRatio'] = (gender_stats['TotalClaims'] / gender_stats['TotalPremium']) * 100
        print(gender_stats)
        
        return loss_ratio
    
    def create_visualizations(self):
        """Create basic visualizations"""
        print("\n" + "="*50)
        print("DATA VISUALIZATIONS")
        print("="*50)
        
        # Create reports directory
        os.makedirs('reports/figures', exist_ok=True)
        
        # 1. Loss Ratio by Province
        plt.figure(figsize=(10, 6))
        province_stats = self.df.groupby('Province').agg({
            'TotalClaims': 'sum',
            'TotalPremium': 'sum'
        })
        province_stats['LossRatio'] = (province_stats['TotalClaims'] / province_stats['TotalPremium']) * 100
        
        bars = plt.bar(province_stats.index, province_stats['LossRatio'])
        plt.xlabel('Province')
        plt.ylabel('Loss Ratio (%)')
        plt.title('Loss Ratio by Province')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('reports/figures/loss_ratio_province.png', dpi=300)
        plt.show()
        
        # 2. Premium vs Claims Scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['TotalPremium'], self.df['TotalClaims'], alpha=0.5)
        plt.xlabel('Total Premium')
        plt.ylabel('Total Claims')
        plt.title('Premium vs Claims Scatter Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/figures/premium_vs_claims.png', dpi=300)
        plt.show()
        
        # 3. Vehicle Type Distribution
        plt.figure(figsize=(10, 6))
        vehicle_counts = self.df['VehicleType'].value_counts()
        plt.pie(vehicle_counts.values, labels=vehicle_counts.index, autopct='%1.1f%%')
        plt.title('Vehicle Type Distribution')
        plt.tight_layout()
        plt.savefig('reports/figures/vehicle_distribution.png', dpi=300)
        plt.show()
        
        print("\nVisualizations saved to 'reports/figures/'")
    
    def run(self):
        """Run complete EDA"""
        print("Starting EDA for Insurance Data...")
        
        # Load data
        self.load_sample_data()
        
        # Perform analysis
        self.basic_analysis()
        self.calculate_loss_ratio()
        self.create_visualizations()
        
        print("\n" + "="*50)
        print("EDA COMPLETED!")
        print("="*50)

if __name__ == "__main__":
    eda = SimpleEDA()
    eda.run()
