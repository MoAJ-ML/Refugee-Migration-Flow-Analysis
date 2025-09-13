#!/usr/bin/env python3
"""
Refugee Migration Flow Analysis
===============================

A comprehensive data science project analyzing UNHCR refugee data to forecast
migration flows and generate policy-relevant insights.

Author: Data Science Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("REFUGEE MIGRATION FLOW ANALYSIS")
print("=" * 60)

# Step 1: Data Loading and Initial Exploration
print("\n1. LOADING AND EXPLORING DATA")
print("-" * 40)

# Load the dataset
df = pd.read_csv('download/persons_of_concern.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nLast 5 rows:")
print(df.tail())

# Check for unique values in key columns
print(f"\nUnique years: {sorted(df['Year'].unique())}")
print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")

# Check the data for countries
print(f"\nCountries of Asylum (non-null): {df[df['Country of Asylum'] != '-']['Country of Asylum'].nunique()}")
print(f"Countries of Origin (non-null): {df[df['Country of Origin'] != '-']['Country of Origin'].nunique()}")

# Display basic statistics
print(f"\nBasic Statistics for Refugee Numbers:")
print(df['Refugees'].describe())

# Step 2: Data Cleaning and Preprocessing
print("\n\n2. DATA CLEANING AND PREPROCESSING")
print("-" * 40)

# Convert Year to datetime for better time series handling
df['Date'] = pd.to_datetime(df['Year'], format='%Y')
df = df.sort_values('Date').reset_index(drop=True)

# Create a comprehensive dataset with all population types
print("Creating comprehensive migration flow dataset...")

# Calculate total persons of concern (POC)
df['Total_POC'] = (df['Refugees'] + df['Asylum Seekers'] + df['IDPs'] + 
                   df['Returned Refugees'] + df['Returned IDPs'] + 
                   df['Stateless'] + df['HST'] + df['OOC'])

# Calculate net migration flows (new arrivals minus returns)
df['Net_Refugee_Flow'] = df['Refugees'] - df['Returned Refugees']
df['Net_IDP_Flow'] = df['IDPs'] - df['Returned IDPs']

# Calculate growth rates
df['Refugee_Growth_Rate'] = df['Refugees'].pct_change() * 100
df['Total_POC_Growth_Rate'] = df['Total_POC'].pct_change() * 100

print(f"Data cleaning completed. New columns added:")
print(f"- Date: {df['Date'].dtype}")
print(f"- Total_POC: Total persons of concern")
print(f"- Net_Refugee_Flow: Net refugee movements")
print(f"- Net_IDP_Flow: Net IDP movements")
print(f"- Growth rates calculated")

# Check for any remaining missing values
print(f"\nMissing values after cleaning:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Display cleaned dataset info
print(f"\nCleaned dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min().year} - {df['Date'].max().year}")

# Step 3: Exploratory Data Analysis (EDA)
print("\n\n3. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Global Refugee Migration Trends (2000-2024)', fontsize=16, fontweight='bold')

# 1. Total Refugees over time
axes[0, 0].plot(df['Date'], df['Refugees'], linewidth=2, color='#2E86AB', marker='o')
axes[0, 0].set_title('Global Refugee Population Over Time', fontweight='bold')
axes[0, 0].set_ylabel('Number of Refugees')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Total Persons of Concern
axes[0, 1].plot(df['Date'], df['Total_POC'], linewidth=2, color='#A23B72', marker='s')
axes[0, 1].set_title('Total Persons of Concern Over Time', fontweight='bold')
axes[0, 1].set_ylabel('Total POC')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Net Migration Flows
axes[1, 0].plot(df['Date'], df['Net_Refugee_Flow'], linewidth=2, color='#F18F01', label='Net Refugee Flow', marker='^')
axes[1, 0].plot(df['Date'], df['Net_IDP_Flow'], linewidth=2, color='#C73E1D', label='Net IDP Flow', marker='v')
axes[1, 0].set_title('Net Migration Flows', fontweight='bold')
axes[1, 0].set_ylabel('Net Flow')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Growth Rates
axes[1, 1].plot(df['Date'][1:], df['Refugee_Growth_Rate'][1:], linewidth=2, color='#7209B7', marker='o')
axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1, 1].set_title('Refugee Population Growth Rate', fontweight='bold')
axes[1, 1].set_ylabel('Growth Rate (%)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/global_refugee_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# Create interactive Plotly visualizations
print("\nCreating interactive visualizations...")

# Interactive time series plot
fig_interactive = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Global Refugee Population', 'Total Persons of Concern', 
                   'Net Migration Flows', 'Growth Rates'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Add traces
fig_interactive.add_trace(
    go.Scatter(x=df['Date'], y=df['Refugees'], mode='lines+markers', 
               name='Refugees', line=dict(color='#2E86AB', width=3)),
    row=1, col=1
)

fig_interactive.add_trace(
    go.Scatter(x=df['Date'], y=df['Total_POC'], mode='lines+markers', 
               name='Total POC', line=dict(color='#A23B72', width=3)),
    row=1, col=2
)

fig_interactive.add_trace(
    go.Scatter(x=df['Date'], y=df['Net_Refugee_Flow'], mode='lines+markers', 
               name='Net Refugee Flow', line=dict(color='#F18F01', width=3)),
    row=2, col=1
)

fig_interactive.add_trace(
    go.Scatter(x=df['Date'], y=df['Net_IDP_Flow'], mode='lines+markers', 
               name='Net IDP Flow', line=dict(color='#C73E1D', width=3)),
    row=2, col=1
)

fig_interactive.add_trace(
    go.Scatter(x=df['Date'][1:], y=df['Refugee_Growth_Rate'][1:], mode='lines+markers', 
               name='Growth Rate', line=dict(color='#7209B7', width=3)),
    row=2, col=2
)

# Update layout
fig_interactive.update_layout(
    title_text="Interactive Global Refugee Migration Analysis (2000-2024)",
    title_x=0.5,
    height=800,
    showlegend=True
)

fig_interactive.write_html('visualizations/interactive_refugee_analysis.html')
print("Interactive visualization saved as 'visualizations/interactive_refugee_analysis.html'")

# Statistical Analysis
print("\n\n4. STATISTICAL ANALYSIS")
print("-" * 40)

# Key statistics
print("Key Statistics (2024 vs 2000):")
print(f"Refugees: {df.iloc[-1]['Refugees']:,} (2024) vs {df.iloc[0]['Refugees']:,} (2000)")
print(f"Total POC: {df.iloc[-1]['Total_POC']:,} (2024) vs {df.iloc[0]['Total_POC']:,} (2000)")
print(f"Growth: {((df.iloc[-1]['Refugees'] / df.iloc[0]['Refugees']) - 1) * 100:.1f}% for refugees")
print(f"Growth: {((df.iloc[-1]['Total_POC'] / df.iloc[0]['Total_POC']) - 1) * 100:.1f}% for total POC")

# Peak years analysis
peak_refugee_year = df.loc[df['Refugees'].idxmax()]
peak_poc_year = df.loc[df['Total_POC'].idxmax()]

print(f"\nPeak Analysis:")
print(f"Peak refugee year: {peak_refugee_year['Year']} with {peak_refugee_year['Refugees']:,} refugees")
print(f"Peak POC year: {peak_poc_year['Year']} with {peak_poc_year['Total_POC']:,} total POC")

# Recent trends (last 5 years)
recent_data = df.tail(5)
print(f"\nRecent Trends (2020-2024):")
print(f"Average annual refugee growth: {recent_data['Refugee_Growth_Rate'].mean():.2f}%")
print(f"Average annual POC growth: {recent_data['Total_POC_Growth_Rate'].mean():.2f}%")

# Step 5: Time Series Preprocessing for Forecasting
print("\n\n5. TIME SERIES PREPROCESSING")
print("-" * 40)

# Prepare time series data for forecasting
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create time series for main variables
ts_data = df[['Date', 'Refugees', 'Total_POC', 'Net_Refugee_Flow']].copy()
ts_data = ts_data.set_index('Date')

print("Time series data prepared:")
print(f"Shape: {ts_data.shape}")
print(f"Date range: {ts_data.index.min()} to {ts_data.index.max()}")

# Train-test split (use last 5 years for testing)
train_size = len(ts_data) - 5
train_data = ts_data.iloc[:train_size]
test_data = ts_data.iloc[train_size:]

print(f"\nTrain-Test Split:")
print(f"Training period: {train_data.index.min().year} - {train_data.index.max().year} ({len(train_data)} years)")
print(f"Testing period: {test_data.index.min().year} - {test_data.index.max().year} ({len(test_data)} years)")

# Stationarity tests
print(f"\nStationarity Tests (Augmented Dickey-Fuller):")
for col in ['Refugees', 'Total_POC', 'Net_Refugee_Flow']:
    result = adfuller(train_data[col].dropna())
    print(f"{col}: ADF Statistic = {result[0]:.4f}, p-value = {result[1]:.4f}")
    if result[1] <= 0.05:
        print(f"  -> Series is stationary (reject null hypothesis)")
    else:
        print(f"  -> Series is non-stationary (fail to reject null hypothesis)")

# Step 6: ARIMA/SARIMAX Forecasting Models
print("\n\n6. ARIMA/SARIMAX FORECASTING MODELS")
print("-" * 40)

def evaluate_forecast(y_true, y_pred, model_name):
    """Calculate evaluation metrics for forecasting models"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"RMSE: {rmse:,.0f}")
    print(f"MAE: {mae:,.0f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# ARIMA model for Refugees
print("Fitting ARIMA model for Refugee population...")

# Auto ARIMA parameter selection (simplified)
def find_best_arima_params(ts_data, max_p=3, max_d=2, max_q=3):
    """Find best ARIMA parameters using AIC"""
    best_aic = float('inf')
    best_params = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts_data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                except:
                    continue
    
    return best_params

# Find best parameters for Refugees
best_params_refugees = find_best_arima_params(train_data['Refugees'])
print(f"Best ARIMA parameters for Refugees: {best_params_refugees}")

# Fit ARIMA model
arima_model_refugees = ARIMA(train_data['Refugees'], order=best_params_refugees)
arima_fitted_refugees = arima_model_refugees.fit()

# Make predictions
arima_forecast_refugees = arima_fitted_refugees.forecast(steps=len(test_data))
arima_forecast_refugees = pd.Series(arima_forecast_refugees, index=test_data.index)

# Evaluate ARIMA model
arima_metrics_refugees = evaluate_forecast(test_data['Refugees'], arima_forecast_refugees, "ARIMA (Refugees)")

# ARIMA model for Total POC
print("\nFitting ARIMA model for Total POC...")
best_params_poc = find_best_arima_params(train_data['Total_POC'])
print(f"Best ARIMA parameters for Total POC: {best_params_poc}")

arima_model_poc = ARIMA(train_data['Total_POC'], order=best_params_poc)
arima_fitted_poc = arima_model_poc.fit()

arima_forecast_poc = arima_fitted_poc.forecast(steps=len(test_data))
arima_forecast_poc = pd.Series(arima_forecast_poc, index=test_data.index)

arima_metrics_poc = evaluate_forecast(test_data['Total_POC'], arima_forecast_poc, "ARIMA (Total POC)")

# Step 7: Prophet Forecasting Model
print("\n\n7. PROPHET FORECASTING MODEL")
print("-" * 40)

try:
    from prophet import Prophet
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    def prepare_prophet_data(ts_data, column_name):
        prophet_data = pd.DataFrame({
            'ds': ts_data.index,
            'y': ts_data[column_name]
        })
        return prophet_data
    
    # Prophet model for Refugees
    print("Fitting Prophet model for Refugee population...")
    prophet_data_refugees = prepare_prophet_data(train_data, 'Refugees')
    
    prophet_model_refugees = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    prophet_model_refugees.fit(prophet_data_refugees)
    
    # Make future predictions
    future_refugees = prophet_model_refugees.make_future_dataframe(periods=len(test_data), freq='Y')
    prophet_forecast_refugees = prophet_model_refugees.predict(future_refugees)
    
    # Extract test period predictions
    prophet_pred_refugees = prophet_forecast_refugees['yhat'].iloc[-len(test_data):]
    prophet_pred_refugees.index = test_data.index
    
    prophet_metrics_refugees = evaluate_forecast(test_data['Refugees'], prophet_pred_refugees, "Prophet (Refugees)")
    
    # Prophet model for Total POC
    print("\nFitting Prophet model for Total POC...")
    prophet_data_poc = prepare_prophet_data(train_data, 'Total_POC')
    
    prophet_model_poc = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    prophet_model_poc.fit(prophet_data_poc)
    
    future_poc = prophet_model_poc.make_future_dataframe(periods=len(test_data), freq='Y')
    prophet_forecast_poc = prophet_model_poc.predict(future_poc)
    
    prophet_pred_poc = prophet_forecast_poc['yhat'].iloc[-len(test_data):]
    prophet_pred_poc.index = test_data.index
    
    prophet_metrics_poc = evaluate_forecast(test_data['Total_POC'], prophet_pred_poc, "Prophet (Total POC)")
    
    prophet_available = True
    
except ImportError:
    print("Prophet not available. Skipping Prophet models.")
    prophet_available = False
    prophet_metrics_refugees = None
    prophet_metrics_poc = None

# Step 8: Model Evaluation and Comparison
print("\n\n8. MODEL EVALUATION AND COMPARISON")
print("-" * 40)

# Create comparison visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Forecasting Model Comparison (2020-2024)', fontsize=16, fontweight='bold')

# Refugees forecasting comparison
axes[0, 0].plot(train_data.index, train_data['Refugees'], label='Training Data', color='blue', linewidth=2)
axes[0, 0].plot(test_data.index, test_data['Refugees'], label='Actual', color='red', linewidth=2, marker='o')
axes[0, 0].plot(test_data.index, arima_forecast_refugees, label='ARIMA Forecast', color='green', linewidth=2, linestyle='--')
if prophet_available:
    axes[0, 0].plot(test_data.index, prophet_pred_refugees, label='Prophet Forecast', color='orange', linewidth=2, linestyle='--')
axes[0, 0].set_title('Refugee Population Forecasting', fontweight='bold')
axes[0, 0].set_ylabel('Number of Refugees')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Total POC forecasting comparison
axes[0, 1].plot(train_data.index, train_data['Total_POC'], label='Training Data', color='blue', linewidth=2)
axes[0, 1].plot(test_data.index, test_data['Total_POC'], label='Actual', color='red', linewidth=2, marker='o')
axes[0, 1].plot(test_data.index, arima_forecast_poc, label='ARIMA Forecast', color='green', linewidth=2, linestyle='--')
if prophet_available:
    axes[0, 1].plot(test_data.index, prophet_pred_poc, label='Prophet Forecast', color='orange', linewidth=2, linestyle='--')
axes[0, 1].set_title('Total POC Forecasting', fontweight='bold')
axes[0, 1].set_ylabel('Total POC')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Model performance comparison (RMSE)
models = ['ARIMA (Refugees)', 'ARIMA (POC)']
rmse_values = [arima_metrics_refugees['RMSE'], arima_metrics_poc['RMSE']]

if prophet_available:
    models.extend(['Prophet (Refugees)', 'Prophet (POC)'])
    rmse_values.extend([prophet_metrics_refugees['RMSE'], prophet_metrics_poc['RMSE']])

axes[1, 0].bar(models, rmse_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(models)])
axes[1, 0].set_title('Model Performance Comparison (RMSE)', fontweight='bold')
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].tick_params(axis='x', rotation=45)

# Model performance comparison (MAPE)
mape_values = [arima_metrics_refugees['MAPE'], arima_metrics_poc['MAPE']]
if prophet_available:
    mape_values.extend([prophet_metrics_refugees['MAPE'], prophet_metrics_poc['MAPE']])

axes[1, 1].bar(models, mape_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(models)])
axes[1, 1].set_title('Model Performance Comparison (MAPE)', fontweight='bold')
axes[1, 1].set_ylabel('MAPE (%)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/forecasting_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print model comparison summary
print("\nModel Performance Summary:")
print("=" * 50)
print(f"{'Model':<20} {'RMSE':<15} {'MAE':<15} {'MAPE':<10}")
print("-" * 50)
print(f"{'ARIMA (Refugees)':<20} {arima_metrics_refugees['RMSE']:<15,.0f} {arima_metrics_refugees['MAE']:<15,.0f} {arima_metrics_refugees['MAPE']:<10.2f}%")
print(f"{'ARIMA (POC)':<20} {arima_metrics_poc['RMSE']:<15,.0f} {arima_metrics_poc['MAE']:<15,.0f} {arima_metrics_poc['MAPE']:<10.2f}%")

if prophet_available:
    print(f"{'Prophet (Refugees)':<20} {prophet_metrics_refugees['RMSE']:<15,.0f} {prophet_metrics_refugees['MAE']:<15,.0f} {prophet_metrics_refugees['MAPE']:<10.2f}%")
    print(f"{'Prophet (POC)':<20} {prophet_metrics_poc['RMSE']:<15,.0f} {prophet_metrics_poc['MAE']:<15,.0f} {prophet_metrics_poc['MAPE']:<10.2f}%")

# Step 9: Future Forecasting and Policy Insights
print("\n\n9. FUTURE FORECASTING AND POLICY INSIGHTS")
print("-" * 40)

# Generate forecasts for next 10 years
forecast_years = 10
future_dates = pd.date_range(start=ts_data.index.max() + pd.DateOffset(years=1), 
                            periods=forecast_years, freq='Y')

print(f"Generating forecasts for {forecast_years} years ahead ({future_dates[0].year}-{future_dates[-1].year})...")

# Use the best performing model for future forecasts
# For simplicity, we'll use ARIMA for both variables
future_arima_refugees = arima_fitted_refugees.forecast(steps=forecast_years)
future_arima_poc = arima_fitted_poc.forecast(steps=forecast_years)

# Create future forecast dataframe
future_forecasts = pd.DataFrame({
    'Date': future_dates,
    'Refugees_Forecast': future_arima_refugees,
    'Total_POC_Forecast': future_arima_poc
})

# Calculate growth rates for future forecasts
current_refugees = ts_data['Refugees'].iloc[-1]
current_poc = ts_data['Total_POC'].iloc[-1]

future_forecasts['Refugee_Growth'] = ((future_forecasts['Refugees_Forecast'] / current_refugees) - 1) * 100
future_forecasts['POC_Growth'] = ((future_forecasts['Total_POC_Forecast'] / current_poc) - 1) * 100

print(f"\nFuture Forecasts Summary:")
print(f"{'Year':<6} {'Refugees':<15} {'Growth%':<10} {'Total POC':<15} {'Growth%':<10}")
print("-" * 60)
for _, row in future_forecasts.iterrows():
    print(f"{row['Date'].year:<6} {row['Refugees_Forecast']:<15,.0f} {row['Refugee_Growth']:<10.1f} {row['Total_POC_Forecast']:<15,.0f} {row['POC_Growth']:<10.1f}")

# Create future forecasting visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Future Refugee Migration Forecasts (2025-2034)', fontsize=16, fontweight='bold')

# Refugees forecast
axes[0].plot(ts_data.index, ts_data['Refugees'], label='Historical Data', color='blue', linewidth=2)
axes[0].plot(future_forecasts['Date'], future_forecasts['Refugees_Forecast'], 
             label='Forecast', color='red', linewidth=2, linestyle='--', marker='o')
axes[0].axvline(x=ts_data.index.max(), color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
axes[0].set_title('Refugee Population Forecast', fontweight='bold')
axes[0].set_ylabel('Number of Refugees')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Total POC forecast
axes[1].plot(ts_data.index, ts_data['Total_POC'], label='Historical Data', color='blue', linewidth=2)
axes[1].plot(future_forecasts['Date'], future_forecasts['Total_POC_Forecast'], 
             label='Forecast', color='red', linewidth=2, linestyle='--', marker='o')
axes[1].axvline(x=ts_data.index.max(), color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
axes[1].set_title('Total Persons of Concern Forecast', fontweight='bold')
axes[1].set_ylabel('Total POC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/future_forecasts.png', dpi=300, bbox_inches='tight')
plt.show()

# Policy Insights and Recommendations
print("\n\n10. POLICY INSIGHTS AND RECOMMENDATIONS")
print("-" * 40)

print("Key Findings:")
print("=" * 30)

# Historical trends analysis
total_growth_refugees = ((ts_data['Refugees'].iloc[-1] / ts_data['Refugees'].iloc[0]) - 1) * 100
total_growth_poc = ((ts_data['Total_POC'].iloc[-1] / ts_data['Total_POC'].iloc[0]) - 1) * 100

print(f"1. Historical Growth (2000-2024):")
print(f"   - Refugee population increased by {total_growth_refugees:.1f}%")
print(f"   - Total POC increased by {total_growth_poc:.1f}%")

# Recent acceleration
recent_growth_refugees = ((ts_data['Refugees'].iloc[-1] / ts_data['Refugees'].iloc[-6]) - 1) * 100
recent_growth_poc = ((ts_data['Total_POC'].iloc[-1] / ts_data['Total_POC'].iloc[-6]) - 1) * 100

print(f"\n2. Recent Acceleration (2019-2024):")
print(f"   - Refugee population increased by {recent_growth_refugees:.1f}% in 5 years")
print(f"   - Total POC increased by {recent_growth_poc:.1f}% in 5 years")

# Future projections
future_growth_refugees = future_forecasts['Refugee_Growth'].iloc[-1]
future_growth_poc = future_forecasts['POC_Growth'].iloc[-1]

print(f"\n3. Future Projections (by 2034):")
print(f"   - Refugee population projected to grow by {future_growth_refugees:.1f}%")
print(f"   - Total POC projected to grow by {future_growth_poc:.1f}%")

print(f"\n4. Policy Recommendations:")
print(f"   - Prepare for continued growth in displacement numbers")
print(f"   - Strengthen international cooperation and burden-sharing")
print(f"   - Invest in early warning systems for conflict prevention")
print(f"   - Enhance integration programs for host communities")
print(f"   - Develop sustainable solutions for protracted displacement")

# Export results
print(f"\n\n11. EXPORTING RESULTS")
print("-" * 40)

# Save forecasts to CSV
future_forecasts_export = future_forecasts.copy()
future_forecasts_export['Date'] = future_forecasts_export['Date'].dt.year
future_forecasts_export.to_csv('refugee_forecasts_2025_2034.csv', index=False)
print("Future forecasts saved to 'refugee_forecasts_2025_2034.csv'")

# Save model performance metrics
performance_metrics = pd.DataFrame({
    'Model': models,
    'RMSE': rmse_values,
    'MAPE': mape_values
})
performance_metrics.to_csv('model_performance_metrics.csv', index=False)
print("Model performance metrics saved to 'model_performance_metrics.csv'")

print(f"\n" + "=" * 60)
print("REFUGEE MIGRATION FLOW ANALYSIS COMPLETED")
print("=" * 60)
print(f"All visualizations saved in 'visualizations/' folder")
print(f"Forecast data exported to CSV files")
print(f"Analysis complete with policy insights and recommendations")
