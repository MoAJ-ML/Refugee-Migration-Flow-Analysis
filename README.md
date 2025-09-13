# Refugee Migration Flow Analysis

A comprehensive data science project analyzing UNHCR refugee data to forecast migration flows and generate policy-relevant insights using advanced time-series forecasting techniques.

## 🎯 Project Overview

This project analyzes global refugee migration patterns from 2000-2024 and provides forecasts for the next 10 years (2025-2034) using multiple forecasting models including ARIMA and Prophet. The analysis reveals critical trends in global displacement and provides actionable policy recommendations.

## 📊 Key Findings

### Historical Trends (2000-2024)
- **Refugee population increased by 155.2%** (from 12.1M to 31.0M)
- **Total Persons of Concern increased by 598.5%** (from 21.9M to 152.7M)
- **Peak refugee year**: 2023 with 31.6M refugees
- **Peak POC year**: 2024 with 152.7M total POC

### Recent Acceleration (2019-2024)
- Refugee population increased by **51.6%** in just 5 years
- Total POC increased by **81.6%** in 5 years
- Average annual growth rates: 9.56% (refugees), 13.33% (POC)

### Future Projections (2025-2034)
- Models predict a **decline** in displacement numbers
- Refugee population projected to decrease by **11.1%** by 2034
- Total POC projected to decrease by **16.5%** by 2034

## 🏗️ Project Structure

```
Refugee Migration Flow Analysis/
├── download/                                    # Original UNHCR datasets
│   ├── persons_of_concern.csv                  # Main dataset (2000-2024)
│   ├── footnotes.csv                           # Data footnotes and explanations
│   └── UNHCR_RefugeeDataFinder_Copyright.pdf  # Copyright information
├── visualizations/                             # Generated visualizations
│   ├── enhanced_forecasts.png                 # Enhanced forecasts with confidence intervals
│   ├── enhanced_model_analysis.png            # Enhanced model analysis
│   └── enhanced_trends_analysis.png           # Enhanced trends with annotations
├── refugee_migration_analysis.py              # Main analysis script
├── create_visualizations.py                   # Visualization creation script
├── requirements.txt                           # Python dependencies
├── refugee_forecasts_2025_2034.csv           # Future forecasts export
├── model_performance_metrics.csv             # Model evaluation metrics
└── README.md                                  # This documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd "Refugee Migration Flow Analysis"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main analysis**
   ```bash
   python refugee_migration_analysis.py
   ```

4. **Create enhanced visualizations**
   ```bash
   python create_visualizations.py
   ```

### Expected Output
The scripts will generate:
- Comprehensive analysis in the terminal
- 3 visualization files in the `visualizations/` folder
- 2 CSV files with forecasts and model metrics

## 📈 Analysis Components

### 1. Data Exploration & Cleaning
- **Dataset**: 25 years of global refugee statistics (2000-2024)
- **Variables**: Refugees, Asylum Seekers, IDPs, Stateless persons, and more
- **Processing**: Time series conversion, growth rate calculations, net flow analysis

### 2. Exploratory Data Analysis (EDA)
- **Historical trends visualization** with multiple time series
- **Interactive dashboard** using Plotly for detailed exploration
- **Statistical analysis** of growth patterns and peak years
- **Net migration flow analysis** (arrivals minus returns)

### 3. Time Series Forecasting Models

#### ARIMA Models
- **Auto-parameter selection** using AIC optimization
- **Best parameters**: ARIMA(2,2,1) for refugees, ARIMA(2,2,3) for POC
- **Performance**: 15.34% MAPE for refugees, 17.55% MAPE for POC

#### Prophet Models
- **Seasonality handling** with yearly patterns
- **Multiplicative seasonality** mode for better trend capture
- **Performance**: 30.93% MAPE for refugees, 36.14% MAPE for POC

### 4. Model Evaluation
- **Metrics**: RMSE, MAE, MAPE
- **Train-Test Split**: 20 years training (2000-2019), 5 years testing (2020-2024)
- **ARIMA outperformed Prophet** in both accuracy and reliability

### 5. Future Forecasting
- **10-year projections** (2025-2034)
- **Declining trend** predicted for both refugees and total POC
- **Uncertainty considerations** and policy implications

## 📊 Visualizations

### 1. Global Refugee Trends (`global_refugee_trends.png`)
- Historical refugee population over time
- Total Persons of Concern trends
- Net migration flows (refugees and IDPs)
- Growth rate analysis
- **Fixed**: All x-axis labels properly formatted, no cutoff issues

### 2. Model Performance Comparison (`model_performance_comparison.png`)
- ARIMA vs Prophet performance comparison
- Actual vs predicted values for test period
- RMSE and MAPE comparison charts
- Model reliability assessment

### 4. Future Forecasts (`future_forecasts.png`)
- 10-year projections with confidence intervals
- Historical data with forecast overlay
- Clear visualization of predicted trends
- Policy-relevant time horizons

### 5. Enhanced Analysis Charts
- **Enhanced Trends Analysis**: Advanced trend analysis with annotations
- **Enhanced Model Analysis**: Detailed model performance metrics
- **Enhanced Forecasts**: Forecasts with confidence intervals and uncertainty bands

## 🔬 Technical Details

### Data Sources
- **UNHCR Refugee Statistics Database**
- **Population Statistics Database → Persons of Concern**
- **Global aggregated data** (2000-2024)
- **Annual frequency** with comprehensive coverage

### Methodology
1. **Stationarity Testing**: Augmented Dickey-Fuller tests
2. **Parameter Optimization**: Grid search with AIC minimization
3. **Cross-Validation**: Time series split with holdout period
4. **Model Selection**: Performance-based comparison
5. **Forecast Generation**: Multi-step ahead predictions

### Model Performance Summary
| Model | Variable | RMSE | MAE | MAPE |
|-------|----------|------|-----|------|
| ARIMA | Refugees | 6,018,000 | 4,688,721 | 15.34% |
| ARIMA | Total POC | 31,902,917 | 25,102,199 | 17.55% |
| Prophet | Refugees | 9,814,438 | 8,823,269 | 30.93% |
| Prophet | Total POC | 52,543,128 | 47,648,124 | 36.14% |

## 🎯 Policy Insights & Recommendations

### Key Findings
1. **Exponential Growth**: Displacement has grown dramatically over 24 years
2. **Recent Acceleration**: Growth rates have increased significantly since 2019
3. **Model Predictions**: Forecasts suggest potential decline in future displacement
4. **Uncertainty**: High variability in recent years requires cautious interpretation

### Policy Recommendations
1. **Prepare for Continued Growth**: Despite model predictions, prepare for potential continued displacement
2. **Strengthen International Cooperation**: Enhanced burden-sharing mechanisms needed
3. **Invest in Early Warning Systems**: Conflict prevention and early intervention
4. **Enhance Integration Programs**: Support for host communities and displaced populations
5. **Develop Sustainable Solutions**: Address root causes of protracted displacement

### Critical Considerations
- **Model Limitations**: Forecasting displacement is inherently uncertain
- **External Factors**: Wars, climate change, and policy changes can dramatically alter trends
- **Data Quality**: Global aggregated data may mask regional variations
- **Policy Impact**: Government interventions can significantly affect migration flows

## 📁 Output Files

### CSV Exports
- **`refugee_forecasts_2025_2034.csv`**: Detailed 10-year forecasts with growth rates
- **`model_performance_metrics.csv`**: Comprehensive model evaluation metrics

### Visualization Files
- **`global_refugee_trends.png`**: Main trends analysis (high-resolution)
- **`model_performance_comparison.png`**: Model performance comparison
- **`future_forecasts.png`**: Future projections visualization
- **`enhanced_trends_analysis.png`**: Enhanced trends with annotations
- **`enhanced_model_analysis.png`**: Enhanced model analysis
- **`enhanced_forecasts.png`**: Enhanced forecasts with confidence intervals

## 🛠️ Dependencies

```
pandas>=1.5.0          # Data manipulation and analysis
numpy>=1.21.0          # Numerical computing
matplotlib>=3.5.0      # Static plotting
seaborn>=0.11.0        # Statistical visualization
plotly>=5.0.0          # Interactive visualizations
scikit-learn>=1.1.0    # Machine learning utilities
statsmodels>=0.13.0    # Statistical models (ARIMA)
prophet>=1.1.0         # Facebook Prophet forecasting
tensorflow>=2.10.0     # Deep learning (optional)
jupyter>=1.0.0         # Jupyter notebook support
notebook>=6.4.0        # Notebook interface
```

## 🔧 Customization

### Modifying Forecast Horizon
```python
# In refugee_migration_analysis.py, line 473
forecast_years = 10  # Change to desired number of years
```

### Adding New Variables
```python
# Add new columns to ts_data in line 239
ts_data = df[['Date', 'Refugees', 'Total_POC', 'Net_Refugee_Flow', 'Your_New_Variable']].copy()
```

### Adjusting Model Parameters
```python
# Modify ARIMA parameter search in find_best_arima_params function
max_p=3, max_d=2, max_q=3  # Adjust search ranges
```

## 📚 References

- **UNHCR Refugee Statistics**: https://www.unhcr.org/refugee-statistics/
- **Population Statistics Database**: UNHCR Data Finder
- **ARIMA Methodology**: Box-Jenkins approach to time series forecasting
- **Prophet Documentation**: Facebook's forecasting tool for business metrics

## ⚠️ Important Notes

1. **Data Interpretation**: Global aggregated data may not reflect regional variations
2. **Forecast Uncertainty**: Displacement forecasting is inherently uncertain due to external factors
3. **Model Limitations**: Statistical models cannot predict sudden geopolitical changes
4. **Policy Implications**: Use forecasts as guidance, not absolute predictions
5. **Regular Updates**: Re-run analysis with new data for updated insights

## 📞 Support

For questions about this analysis or to request modifications:
- Review the code comments in `refugee_migration_analysis.py`
- Check the generated CSV files for detailed results
- Examine the interactive HTML dashboard for data exploration

## 📄 License

This project uses UNHCR data which is subject to their copyright terms. Please refer to `download/UNHCR_RefugeeDataFinder_Copyright.pdf` for usage guidelines.

---

**Analysis completed**: September 2024  
**Data period**: 2000-2024  
**Forecast period**: 2025-2034  
**Models used**: ARIMA, Prophet  
**Status**: ✅ Complete with enhanced visualizations and policy insights