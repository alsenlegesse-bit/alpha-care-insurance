import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class InsuranceEDA:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        
    def load_data(self):
        """Load data - try real data first, then create sample"""
        print("Attempting to load insurance data...")
        
        # Try to load real data if path exists
        if self.data_path and os.path.exists(self.data_path):
            try:
                self.df = pd.read_csv(self.data_path, low_memory=False)
                print(f"✓ Loaded real data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
                return True
            except Exception as e:
                print(f"Could not load real data: {e}")
                print("Creating sample data for analysis...")
        
        # Create sample data for demonstration
        print("Creating sample insurance data...")
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'PolicyID': range(100001, 100001 + n_samples),
            'TotalPremium': np.random.uniform(1000, 15000, n_samples),
            'TotalClaims': np.random.exponential(2000, n_samples),
            'SumInsured': np.random.uniform(50000, 300000, n_samples),
            'CalculatedPremiumPerTerm': np.random.uniform(500, 5000, n_samples),
            'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal', 
                                         'Eastern Cape', 'Free State', 'Mpumalanga',
                                         'Limpopo', 'North West', 'Northern Cape'], 
                                         n_samples, p=[0.35, 0.25, 0.15, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02]),
            'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'VehicleType': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Bakkie', 
                                            'Motorcycle', 'Commercial'], n_samples),
            'Make': np.random.choice(['Toyota', 'Volkswagen', 'Ford', 'BMW', 'Mercedes',
                                     'Nissan', 'Hyundai', 'Kia'], n_samples),
            'Model': np.random.choice(['Corolla', 'Polo', 'Ranger', '3 Series', 'C-Class',
                                      'NP200', 'i20', 'Sportage'], n_samples),
            'RegistrationYear': np.random.randint(1995, 2023, n_samples),
            'Cubiccapacity': np.random.choice([1200, 1400, 1600, 1800, 2000, 2200, 2500, 3000], n_samples),
            'PostalCode': np.random.choice(['0001', '2000', '4000', '6000', '8000'], n_samples),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
            'TransactionMonth': pd.date_range('2014-02-01', periods=n_samples, freq='D'),
            'CoverType': np.random.choice(['Comprehensive', 'Third Party', 'Third Party Fire & Theft'], n_samples),
            'Bodytype': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Double Cab', 'Single Cab'], n_samples),
        }
        
        self.df = pd.DataFrame(data)
        
        # Make 70% of claims zero (no claim)
        no_claim_idx = np.random.choice(n_samples, size=int(n_samples * 0.7), replace=False)
        self.df.loc[no_claim_idx, 'TotalClaims'] = 0
        
        # Add some outliers
        outlier_idx = np.random.choice(n_samples, size=50, replace=False)
        self.df.loc[outlier_idx, 'TotalClaims'] = np.random.uniform(50000, 200000, 50)
        
        print(f"✓ Created sample data with {n_samples} records")
        print("  Columns available:", list(self.df.columns))
        
        return True
    
    def explore_data_structure(self):
        """Basic data exploration"""
        print("\n" + "="*60)
        print("DATA STRUCTURE EXPLORATION")
        print("="*60)
        
        print(f"\n📊 Dataset Shape: {self.df.shape}")
        print(f"   Rows: {self.df.shape[0]:,}")
        print(f"   Columns: {self.df.shape[1]}")
        
        # Data types
        print("\n📝 Data Types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Categorize columns
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\n🔢 Numerical Columns ({len(self.numerical_cols)}):")
        print(f"   {self.numerical_cols[:10]}")  # Show first 10
        
        print(f"\n🏷️ Categorical Columns ({len(self.categorical_cols)}):")
        print(f"   {self.categorical_cols[:10]}")  # Show first 10
        
        return self.df.info()
    
    def check_data_quality(self):
        """Check for data quality issues"""
        print("\n" + "="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Missing values
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percentage
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("\n❓ Missing Values:")
        missing_cols = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_cols) > 0:
            print(f"   Columns with missing values: {len(missing_cols)}")
            print(missing_cols.head(10))
        else:
            print("   ✓ No missing values found")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\n🔍 Duplicate Rows: {duplicates}")
        
        return missing_df
    
    def calculate_loss_ratio(self):
        """Calculate loss ratio - KEY BUSINESS METRIC"""
        print("\n" + "="*60)
        print("LOSS RATIO ANALYSIS (Key Business Metric)")
        print("="*60)
        
        # Overall loss ratio
        total_claims = self.df['TotalClaims'].sum()
        total_premium = self.df['TotalPremium'].sum()
        overall_loss_ratio = (total_claims / total_premium) * 100 if total_premium > 0 else 0
        
        print(f"\n💰 Overall Portfolio:")
        print(f"   Total Claims: R{total_claims:,.2f}")
        print(f"   Total Premium: R{total_premium:,.2f}")
        print(f"   Loss Ratio: {overall_loss_ratio:.2f}%")
        
        # Loss ratio by province
        if 'Province' in self.df.columns:
            print("\n🗺️ Loss Ratio by Province:")
            province_loss = self.df.groupby('Province').agg(
                TotalClaims=('TotalClaims', 'sum'),
                TotalPremium=('TotalPremium', 'sum'),
                PolicyCount=('PolicyID', 'count')
            )
            province_loss['LossRatio'] = (province_loss['TotalClaims'] / province_loss['TotalPremium']) * 100
            province_loss = province_loss.sort_values('LossRatio', ascending=False)
            print(province_loss[['PolicyCount', 'TotalPremium', 'TotalClaims', 'LossRatio']])
            
            # Visualize
            plt.figure(figsize=(12, 6))
            bars = plt.bar(province_loss.index, province_loss['LossRatio'], color='skyblue')
            plt.axhline(y=overall_loss_ratio, color='red', linestyle='--', label=f'Overall: {overall_loss_ratio:.1f}%')
            plt.xlabel('Province')
            plt.ylabel('Loss Ratio (%)')
            plt.title('Loss Ratio by Province (Higher = More Risky)', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('reports/figures/loss_ratio_by_province.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Loss ratio by gender
        if 'Gender' in self.df.columns:
            print("\n👥 Loss Ratio by Gender:")
            gender_loss = self.df.groupby('Gender').agg(
                TotalClaims=('TotalClaims', 'sum'),
                TotalPremium=('TotalPremium', 'sum'),
                PolicyCount=('PolicyID', 'count')
            )
            gender_loss['LossRatio'] = (gender_loss['TotalClaims'] / gender_loss['TotalPremium']) * 100
            print(gender_loss[['PolicyCount', 'TotalPremium', 'TotalClaims', 'LossRatio']])
        
        # Loss ratio by vehicle type
        if 'VehicleType' in self.df.columns:
            print("\n🚗 Loss Ratio by Vehicle Type (Top 10):")
            vehicle_loss = self.df.groupby('VehicleType').agg(
                TotalClaims=('TotalClaims', 'sum'),
                TotalPremium=('TotalPremium', 'sum'),
                PolicyCount=('PolicyID', 'count')
            )
            vehicle_loss['LossRatio'] = (vehicle_loss['TotalClaims'] / vehicle_loss['TotalPremium']) * 100
            vehicle_loss = vehicle_loss.sort_values('LossRatio', ascending=False).head(10)
            print(vehicle_loss[['PolicyCount', 'TotalPremium', 'TotalClaims', 'LossRatio']])
        
        return overall_loss_ratio
    
    def analyze_distributions(self):
        """Analyze distributions of key variables"""
        print("\n" + "="*60)
        print("DISTRIBUTION ANALYSIS")
        print("="*60)
        
        key_columns = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
        available_cols = [col for col in key_columns if col in self.df.columns]
        
        if not available_cols:
            print("No key columns found")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(available_cols[:4]):  # Show first 4
            ax = axes[idx]
            
            # Plot histogram
            ax.hist(self.df[col], bins=50, alpha=0.7, edgecolor='black', density=True)
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'Mean: R{self.df[col].mean():,.0f}\nStd: R{self.df[col].std():,.0f}\nSkew: {self.df[col].skew():.2f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('reports/figures/distributions_key_variables.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Box plots for outlier detection
        print("\n📦 Outlier Detection (Box Plots):")
        fig, axes = plt.subplots(1, len(available_cols), figsize=(5*len(available_cols), 6))
        axes = axes.flatten() if len(available_cols) > 1 else [axes]
        
        for idx, col in enumerate(available_cols):
            ax = axes[idx]
            self.df.boxplot(column=col, ax=ax, grid=False)
            ax.set_title(f'Box Plot: {col}', fontweight='bold')
            ax.set_ylabel('Value (R)')
            
            # Calculate outliers
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)]
            ax.text(0.05, 0.95, f'Outliers: {len(outliers)}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('reports/figures/outliers_boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_temporal_trends(self):
        """Analyze trends over time"""
        print("\n" + "="*60)
        print("TEMPORAL TREND ANALYSIS")
        print("="*60)
        
        if 'TransactionMonth' in self.df.columns:
            try:
                # Extract month from date
                self.df['Month'] = pd.to_datetime(self.df['TransactionMonth']).dt.to_period('M')
                
                # Group by month
                monthly_data = self.df.groupby('Month').agg({
                    'TotalClaims': 'sum',
                    'TotalPremium': 'sum',
                    'PolicyID': 'count'
                }).reset_index()
                
                monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
                monthly_data['LossRatio'] = (monthly_data['TotalClaims'] / monthly_data['TotalPremium']) * 100
                
                print("\n📅 Monthly Performance:")
                print(monthly_data)
                
                # Plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Total Claims
                axes[0, 0].plot(monthly_data['Month'], monthly_data['TotalClaims'], 'r-o', linewidth=2)
                axes[0, 0].set_title('Total Claims Over Time', fontweight='bold')
                axes[0, 0].set_xlabel('Month')
                axes[0, 0].set_ylabel('Total Claims (R)')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Total Premium
                axes[0, 1].plot(monthly_data['Month'], monthly_data['TotalPremium'], 'b-o', linewidth=2)
                axes[0, 1].set_title('Total Premium Over Time', fontweight='bold')
                axes[0, 1].set_xlabel('Month')
                axes[0, 1].set_ylabel('Total Premium (R)')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Loss Ratio
                axes[1, 0].plot(monthly_data['Month'], monthly_data['LossRatio'], 'g-o', linewidth=2)
                axes[1, 0].set_title('Loss Ratio Over Time', fontweight='bold')
                axes[1, 0].set_xlabel('Month')
                axes[1, 0].set_ylabel('Loss Ratio (%)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Number of Policies
                axes[1, 1].bar(monthly_data['Month'], monthly_data['PolicyID'], alpha=0.7)
                axes[1, 1].set_title('Number of Policies Over Time', fontweight='bold')
                axes[1, 1].set_xlabel('Month')
                axes[1, 1].set_ylabel('Number of Policies')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig('reports/figures/temporal_trends.png', dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"Could not analyze temporal trends: {e}")
    
    def create_creative_visualizations(self):
        """Create 3 creative visualizations for insights"""
        print("\n" + "="*60)
        print("CREATIVE VISUALIZATIONS")
        print("="*60)
        
        # Ensure reports directory exists
        os.makedirs('reports/figures', exist_ok=True)
        
        # Visualization 1: Risk vs Reward Scatter Plot
        print("\n🎯 Visualization 1: Risk vs Reward Analysis")
        
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            plt.figure(figsize=(12, 8))
            
            # Create claim categories
            claim_labels = ['No Claim', 'Small Claim (<R5k)', 'Medium Claim (R5k-20k)', 'Large Claim (>R20k)']
            claim_bins = [-1, 0, 5000, 20000, float('inf')]
            self.df['ClaimCategory'] = pd.cut(self.df['TotalClaims'], bins=claim_bins, labels=claim_labels)
            
            # Scatter plot with colors by claim category
            colors = {'No Claim': 'green', 'Small Claim (<R5k)': 'yellow', 
                     'Medium Claim (R5k-20k)': 'orange', 'Large Claim (>R20k)': 'red'}
            
            for category, color in colors.items():
                subset = self.df[self.df['ClaimCategory'] == category]
                plt.scatter(subset['TotalPremium'], subset['TotalClaims'], 
                           alpha=0.6, color=color, label=category, s=30)
            
            plt.xlabel('Total Premium (R)', fontweight='bold')
            plt.ylabel('Total Claims (R)', fontweight='bold')
            plt.title('Risk vs Reward: Premium vs Claims Analysis', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Claim Severity')
            
            # Add reference lines
            plt.axhline(y=5000, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(y=20000, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=self.df['TotalPremium'].median(), color='blue', linestyle='--', alpha=0.5, label='Median Premium')
            
            plt.tight_layout()
            plt.savefig('reports/figures/risk_vs_reward.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Visualization 2: Vehicle Make Risk Heatmap
        print("\n🚙 Visualization 2: Vehicle Make Risk Analysis")
        
        if 'Make' in self.df.columns and 'VehicleType' in self.df.columns:
            # Calculate average claim per make and vehicle type
            risk_matrix = self.df.groupby(['Make', 'VehicleType']).agg({
                'TotalClaims': 'mean',
                'PolicyID': 'count'
            }).reset_index()
            
            # Pivot for heatmap
            pivot_data = risk_matrix.pivot(index='Make', columns='VehicleType', values='TotalClaims')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                       linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Average Claims (R)'})
            plt.title('Heatmap: Average Claims by Vehicle Make and Type', fontsize=14, fontweight='bold')
            plt.xlabel('Vehicle Type')
            plt.ylabel('Vehicle Make')
            plt.tight_layout()
            plt.savefig('reports/figures/vehicle_risk_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Visualization 3: Geographic Risk Profile
        print("\n🗺️ Visualization 3: Geographic Risk Profile")
        
        if 'Province' in self.df.columns:
            # Calculate multiple metrics per province
            province_stats = self.df.groupby('Province').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count',
                'SumInsured': 'mean'
            }).reset_index()
            
            province_stats['LossRatio'] = (province_stats['TotalClaims'] / province_stats['TotalPremium']) * 100
            province_stats['AvgClaim'] = province_stats['TotalClaims'] / province_stats['PolicyID']
            
            # Create radar-like visualization (parallel coordinates)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Loss Ratio by Province
            axes[0, 0].barh(province_stats['Province'], province_stats['LossRatio'], color='lightcoral')
            axes[0, 0].set_xlabel('Loss Ratio (%)')
            axes[0, 0].set_title('Risk Level by Province', fontweight='bold')
            axes[0, 0].axvline(x=province_stats['LossRatio'].mean(), color='red', linestyle='--', alpha=0.7)
            
            # 2. Number of Policies
            axes[0, 1].bar(province_stats['Province'], province_stats['PolicyID'], color='lightblue')
            axes[0, 1].set_xlabel('Province')
            axes[0, 1].set_ylabel('Number of Policies')
            axes[0, 1].set_title('Policy Distribution', fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Average Sum Insured
            axes[1, 0].bar(province_stats['Province'], province_stats['SumInsured']/1000, color='lightgreen')
            axes[1, 0].set_xlabel('Province')
            axes[1, 0].set_ylabel('Avg Sum Insured (R thousands)')
            axes[1, 0].set_title('Average Coverage Value', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Scatter: Volume vs Risk
            axes[1, 1].scatter(province_stats['PolicyID'], province_stats['LossRatio'], 
                              s=province_stats['SumInsured']/1000, alpha=0.7, color='purple')
            axes[1, 1].set_xlabel('Number of Policies')
            axes[1, 1].set_ylabel('Loss Ratio (%)')
            axes[1, 1].set_title('Volume vs Risk Trade-off', fontweight='bold')
            
            # Add labels to scatter points
            for i, row in province_stats.iterrows():
                axes[1, 1].annotate(row['Province'][:3], 
                                   (row['PolicyID'], row['LossRatio']),
                                   textcoords="offset points",
                                   xytext=(0,5), ha='center', fontsize=8)
            
            plt.suptitle('Geographic Risk Profile Analysis', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig('reports/figures/geographic_risk_profile.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Clean up
            if 'ClaimCategory' in self.df.columns:
                self.df.drop('ClaimCategory', axis=1, inplace=True)
    
    def generate_insights_report(self):
        """Generate key insights from EDA"""
        print("\n" + "="*60)
        print("KEY INSIGHTS REPORT")
        print("="*60)
        
        insights = {
            'business_metrics': {},
            'risk_findings': [],
            'recommendations': []
        }
        
        # Calculate key metrics
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            total_claims = self.df['TotalClaims'].sum()
            total_premium = self.df['TotalPremium'].sum()
            loss_ratio = (total_claims / total_premium) * 100 if total_premium > 0 else 0
            
            insights['business_metrics'] = {
                'total_premium_collected': total_premium,
                'total_claims_paid': total_claims,
                'overall_loss_ratio': loss_ratio,
                'average_premium': self.df['TotalPremium'].mean(),
                'average_claim': self.df[self.df['TotalClaims'] > 0]['TotalClaims'].mean() if len(self.df[self.df['TotalClaims'] > 0]) > 0 else 0
            }
        
        # Risk findings
        if 'Province' in self.df.columns:
            province_loss = self.df.groupby('Province').apply(
                lambda x: (x['TotalClaims'].sum() / x['TotalPremium'].sum() * 100) 
                if x['TotalPremium'].sum() > 0 else 0
            )
            highest_risk = province_loss.idxmax()
            lowest_risk = province_loss.idxmin()
            
            insights['risk_findings'].append(
                f"Highest risk province: {highest_risk} (Loss Ratio: {province_loss.max():.1f}%)"
            )
            insights['risk_findings'].append(
                f"Lowest risk province: {lowest_risk} (Loss Ratio: {province_loss.min():.1f}%)"
            )
        
        if 'Gender' in self.df.columns:
            gender_stats = self.df.groupby('Gender').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            })
            gender_stats['LossRatio'] = (gender_stats['TotalClaims'] / gender_stats['TotalPremium']) * 100
            
            for gender, data in gender_stats.iterrows():
                insights['risk_findings'].append(
                    f"{gender} drivers: Loss Ratio = {data['LossRatio']:.1f}%"
                )
        
        # Recommendations
        insights['recommendations'].append(
            "Consider premium adjustments for high-risk provinces identified in the analysis"
        )
        insights['recommendations'].append(
            "Target marketing campaigns in low-risk regions to attract profitable customers"
        )
        insights['recommendations'].append(
            "Review vehicle makes with consistently high claim amounts for potential rate adjustments"
        )
        
        # Save insights to JSON
        with open('reports/eda_insights.json', 'w') as f:
            json.dump(insights, f, indent=4, default=str)
        
        # Print summary
        print("\n📈 BUSINESS METRICS:")
        for key, value in insights['business_metrics'].items():
            if 'ratio' in key or 'average' in key:
                if 'premium' in key or 'claim' in key:
                    print(f"  • {key.replace('_', ' ').title()}: R{value:,.2f}")
                else:
                    print(f"  • {key.replace('_', ' ').title()}: {value:.2f}%")
            else:
                print(f"  • {key.replace('_', ' ').title()}: R{value:,.2f}")
        
        print("\n⚠️ RISK FINDINGS:")
        for finding in insights['risk_findings']:
            print(f"  • {finding}")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in insights['recommendations']:
            print(f"  • {rec}")
        
        return insights
    
    def run_complete_analysis(self):
        """Run the complete EDA pipeline"""
        print("="*60)
        print("STARTING COMPREHENSIVE EDA FOR ALPHACARE INSURANCE")
        print("="*60)
        
        # Create necessary directories
        os.makedirs('reports/figures', exist_ok=True)
        os.makedirs('reports/data', exist_ok=True)
        
        # Execute analysis steps
        steps = [
            ('📂 Loading Data', self.load_data),
            ('🔍 Data Structure', self.explore_data_structure),
            ('✅ Data Quality', self.check_data_quality),
            ('💰 Loss Ratio Analysis', self.calculate_loss_ratio),
            ('📊 Distributions & Outliers', self.analyze_distributions),
            ('📅 Temporal Trends', self.analyze_temporal_trends),
            ('🎨 Creative Visualizations', self.create_creative_visualizations),
            ('📈 Insights Report', self.generate_insights_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*60}")
            print(f"STEP: {step_name}")
            print('='*60)
            try:
                step_func()
            except Exception as e:
                print(f"⚠️ Error in {step_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("✅ EDA COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📁 Outputs saved in:")
        print("   • reports/figures/ - All visualizations")
        print("   • reports/eda_insights.json - Key insights")
        print("\n📋 Next steps:")
        print("   • Review insights for business recommendations")
        print("   • Proceed to Task 2: Data Version Control (DVC)")
        
        return self.df

# Main execution
if __name__ == "__main__":
    # Initialize EDA - try real data first, then sample
    data_path = "data/raw/insurance_data.csv"
    
    print("🔧 Initializing Insurance EDA...")
    eda = InsuranceEDA(data_path)
    
    # Run complete analysis
    result_df = eda.run_complete_analysis()
    
    # Save processed data
    if result_df is not None:
        result_df.to_csv('data/processed/insurance_data_eda_processed.csv', index=False)
        print("\n💾 Processed data saved to: data/processed/insurance_data_eda_processed.csv")
