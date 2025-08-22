"""
Core Data Processing Engine for Remittance Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemittanceAnalyzer:
    """Advanced analytics engine for remittance data"""
    
    def __init__(self, csv_path: str):
        """Initialize with data loading and preprocessing"""
        self.data = self.load_and_validate(csv_path)
        self.insights_cache = {}
        self.processed_data = self.preprocess_data()
        logger.info(f"Loaded data: {len(self.data)} years of remittance data")
    
    def load_and_validate(self, csv_path: str) -> pd.DataFrame:
        """Load and validate the CSV data"""
        try:
            df = pd.read_csv(csv_path)
            
            # Clean the data - remove any empty rows
            df = df.dropna(subset=['Year', 'Remittances (million USD)'])
            
            # Convert Year to integer, handling any string formatting
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df = df.dropna(subset=['Year'])
            df['Year'] = df['Year'].astype(int)
            
            # Ensure Remittances column is numeric
            df['Remittances (million USD)'] = pd.to_numeric(df['Remittances (million USD)'], errors='coerce')
            df = df.dropna(subset=['Remittances (million USD)'])
            
            # Sort by year and reset index
            df = df.sort_values('Year').reset_index(drop=True)
            
            logger.info(f"Successfully loaded data for years {df['Year'].min()}-{df['Year'].max()}")
            logger.info(f"Total data points: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self) -> Dict:
        """Advanced data preprocessing and feature engineering"""
        processed = {}
        
        # Basic metrics
        processed['remittances_usd'] = self.data['Remittances (million USD)'].values
        processed['years'] = self.data['Year'].values
        processed['n_years'] = len(self.data)
        
        # Growth rates and changes
        processed['yoy_changes'] = np.diff(processed['remittances_usd']) / processed['remittances_usd'][:-1] * 100
        processed['absolute_changes'] = np.diff(processed['remittances_usd'])
        
        # Moving averages
        processed['ma_3yr'] = pd.Series(processed['remittances_usd']).rolling(3, center=True).mean().values
        processed['ma_5yr'] = pd.Series(processed['remittances_usd']).rolling(5, center=True).mean().values
        
        # Volatility measures
        processed['volatility_3yr'] = pd.Series(processed['yoy_changes']).rolling(3).std().values
        processed['volatility_5yr'] = pd.Series(processed['yoy_changes']).rolling(5).std().values
        
        return processed
    
    def calculate_trend_slope(self, window: int = 5) -> np.ndarray:
        """Calculate rolling trend slopes"""
        slopes = []
        years = self.data['Year'].values
        values = self.data['Remittances (million USD)'].values
        
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx >= 3:  # Need at least 3 points
                x = years[start_idx:end_idx]
                y = values[start_idx:end_idx]
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        
        return np.array(slopes)
    
    def detect_trend_changes(self) -> List[Dict]:
        """Detect significant trend changes"""
        slopes = self.calculate_trend_slope()
        changes = []
        
        for i in range(1, len(slopes) - 1):
            if not (np.isnan(slopes[i-1]) or np.isnan(slopes[i+1])):
                # Check for significant slope change
                slope_change = slopes[i+1] - slopes[i-1]
                if abs(slope_change) > np.std(slopes[~np.isnan(slopes)]) * 1.5:
                    changes.append({
                        'year': int(self.data['Year'].iloc[i]),
                        'type': 'acceleration' if slope_change > 0 else 'deceleration',
                        'magnitude': abs(slope_change),
                        'significance': 'high' if abs(slope_change) > np.std(slopes[~np.isnan(slopes)]) * 2 else 'medium'
                    })
        
        return changes
    
    def detect_anomalies(self, method: str = 'iqr', threshold: float = 2.0) -> List[Dict]:
        """Detect anomalies in the data using multiple methods"""
        anomalies = []
        values = self.data['Remittances (million USD)'].values
        years = self.data['Year'].values
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            anomaly_mask = z_scores > threshold
            
        elif method == 'iqr':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomaly_mask = (values < lower_bound) | (values > upper_bound)
            
        else:  # Modified Z-score
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            anomaly_mask = np.abs(modified_z_scores) > threshold
        
        # Create anomaly records
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly:
                anomalies.append({
                    'year': int(years[i]),
                    'value': float(values[i]),
                    'type': 'high' if values[i] > np.median(values) else 'low',
                    'severity': self._calculate_anomaly_severity(values[i], values),
                    'method': method
                })
        
        return sorted(anomalies, key=lambda x: x['year'])
    
    def _calculate_anomaly_severity(self, value: float, all_values: np.ndarray) -> str:
        """Calculate severity of anomaly"""
        z_score = abs(stats.zscore(all_values)[np.where(all_values == value)[0][0]])
        if z_score > 3:
            return 'extreme'
        elif z_score > 2:
            return 'high'
        else:
            return 'moderate'
    
    def analyze_growth_patterns(self) -> Dict:
        """Analyze growth patterns and phases"""
        yoy_changes = self.processed_data['yoy_changes']
        years = self.processed_data['years'][1:]  # Skip first year (no growth rate)
        
        # Growth phases
        growth_phases = []
        current_phase = None
        phase_start = years[0]
        
        for i, (year, growth) in enumerate(zip(years, yoy_changes)):
            phase_type = 'growth' if growth > 5 else 'decline' if growth < -5 else 'stable'
            
            if current_phase != phase_type:
                if current_phase is not None:
                    growth_phases.append({
                        'phase': current_phase,
                        'start_year': int(phase_start),
                        'end_year': int(years[i-1]) if i > 0 else int(year),
                        'duration': int(years[i-1] - phase_start + 1) if i > 0 else 1
                    })
                current_phase = phase_type
                phase_start = year
        
        # Add final phase
        if current_phase:
            growth_phases.append({
                'phase': current_phase,
                'start_year': int(phase_start),
                'end_year': int(years[-1]),
                'duration': int(years[-1] - phase_start + 1)
            })
        
        return {
            'average_growth': float(np.mean(yoy_changes)),
            'growth_volatility': float(np.std(yoy_changes)),
            'max_growth': float(np.max(yoy_changes)),
            'max_decline': float(np.min(yoy_changes)),
            'growth_phases': growth_phases,
            'compound_annual_growth': float(((self.data['Remittances (million USD)'].iloc[-1] / 
                                           self.data['Remittances (million USD)'].iloc[0]) ** 
                                          (1 / (len(self.data) - 1)) - 1) * 100)
        }
    
    def calculate_advanced_metrics(self) -> Dict:
        """Calculate comprehensive statistical metrics"""
        values = self.data['Remittances (million USD)'].values
        years = self.data['Year'].values
        
        # Basic statistics
        metrics = {
            'descriptive_stats': {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std_dev': float(np.std(values)),
                'min_value': float(np.min(values)),
                'max_value': float(np.max(values)),
                'range': float(np.max(values) - np.min(values)),
                'coefficient_of_variation': float(np.std(values) / np.mean(values))
            }
        }
        
        # Time series properties
        adf_result = adfuller(values)
        metrics['time_series'] = {
            'is_stationary': adf_result[1] < 0.05,
            'adf_pvalue': float(adf_result[1]),
            'trend_strength': float(abs(np.corrcoef(years, values)[0, 1]))
        }
        
        # Growth analysis
        metrics['growth_analysis'] = self.analyze_growth_patterns()
        
        # Anomalies
        metrics['anomalies'] = {
            'iqr_anomalies': self.detect_anomalies('iqr'),
            'zscore_anomalies': self.detect_anomalies('zscore', 2.5)
        }
        
        # Trend analysis (calculate after processed_data is available)
        metrics['trend_analysis'] = {
            'trend_slopes': self.calculate_trend_slope().tolist(),
            'trend_changes': self.detect_trend_changes()
        }
        
        return metrics
    
    def get_period_analysis(self, start_year: int, end_year: int) -> Dict:
        """Get detailed analysis for a specific period"""
        mask = (self.data['Year'] >= start_year) & (self.data['Year'] <= end_year)
        period_data = self.data[mask]
        
        if len(period_data) == 0:
            return {'error': 'No data available for specified period'}
        
        period_values = period_data['Remittances (million USD)'].values
        period_years = period_data['Year'].values
        
        analysis = {
            'period': f"{start_year}-{end_year}",
            'data_points': len(period_data),
            'total_remittances': float(np.sum(period_values)),
            'average_annual': float(np.mean(period_values)),
            'growth_rate': float((period_values[-1] / period_values[0] - 1) * 100) if len(period_values) > 1 else 0,
            'volatility': float(np.std(period_values)),
            'trend': 'increasing' if period_values[-1] > period_values[0] else 'decreasing',
            'key_statistics': {
                'peak_year': int(period_years[np.argmax(period_values)]),
                'peak_value': float(np.max(period_values)),
                'trough_year': int(period_years[np.argmin(period_values)]),
                'trough_value': float(np.min(period_values))
            }
        }
        
        # Year-over-year changes within period
        if len(period_values) > 1:
            yoy_changes = np.diff(period_values) / period_values[:-1] * 100
            analysis['yoy_analysis'] = {
                'average_yoy_growth': float(np.mean(yoy_changes)),
                'max_yoy_growth': float(np.max(yoy_changes)),
                'min_yoy_growth': float(np.min(yoy_changes)),
                'yoy_volatility': float(np.std(yoy_changes))
            }
        
        return analysis
    
    def compare_periods(self, period1: Tuple[int, int], period2: Tuple[int, int]) -> Dict:
        """Compare two time periods"""
        analysis1 = self.get_period_analysis(period1[0], period1[1])
        analysis2 = self.get_period_analysis(period2[0], period2[1])
        
        if 'error' in analysis1 or 'error' in analysis2:
            return {'error': 'Invalid period data'}
        
        comparison = {
            'period1': analysis1,
            'period2': analysis2,
            'comparison': {
                'average_difference': analysis2['average_annual'] - analysis1['average_annual'],
                'growth_rate_difference': analysis2['growth_rate'] - analysis1['growth_rate'],
                'volatility_difference': analysis2['volatility'] - analysis1['volatility'],
                'relative_performance': 'better' if analysis2['average_annual'] > analysis1['average_annual'] else 'worse'
            }
        }
        
        return comparison

if __name__ == "__main__":
    # Test the analyzer
    try:
        analyzer = RemittanceAnalyzer("data/Bangladesh Remittances Dataset (19952025).csv")
        metrics = analyzer.calculate_advanced_metrics()
        print("Analytics engine loaded successfully!")
        print(f"Data spans: {analyzer.data['Year'].min()}-{analyzer.data['Year'].max()}")
        print(f"Average annual remittances: ${metrics['descriptive_stats']['mean']:.2f}M")
    except Exception as e:
        print(f"Error testing analyzer: {e}")