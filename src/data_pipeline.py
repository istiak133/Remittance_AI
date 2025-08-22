"""
Data Pipeline for Remittance AI Core
Handles data ingestion, cleaning, validation, and database storage
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)

class DataPipeline:
    """Comprehensive data pipeline for remittance data management"""
    
    def __init__(self, db_path: str = "data/remittance_core.db"):
        """Initialize the data pipeline"""
        self.db_path = db_path
        self.db_connection = None
        self.data_validators = self._setup_validators()
        self.cleaning_rules = self._setup_cleaning_rules()
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Data pipeline initialized with database: {db_path}")
    
    def _setup_validators(self) -> Dict:
        """Setup data validation rules"""
        return {
            'required_columns': ['Year', 'Remittances (million USD)'],
            'year_range': (1990, 2030),
            'remittance_min': 0,
            'remittance_max': 100000,  # 100 billion USD max
            'data_types': {
                'Year': 'int',
                'Remittances (million USD)': 'float'
            }
        }
    
    def _setup_cleaning_rules(self) -> Dict:
        """Setup data cleaning rules"""
        return {
            'remove_duplicates': True,
            'fill_missing_years': True,
            'outlier_threshold': 3.0,  # Standard deviations
            'smooth_extreme_values': True,
            'validate_continuity': True
        }
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            self.db_connection = sqlite3.connect(self.db_path)
            cursor = self.db_connection.cursor()
            
            # Create main data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS remittance_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER NOT NULL,
                    remittances_usd REAL NOT NULL,
                    remittances_bdt REAL,
                    yoy_change REAL,
                    cumulative_growth REAL,
                    data_source TEXT,
                    upload_timestamp TEXT,
                    data_hash TEXT,
                    quality_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create data quality log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upload_timestamp TEXT,
                    filename TEXT,
                    records_processed INTEGER,
                    records_cleaned INTEGER,
                    quality_score REAL,
                    validation_errors TEXT,
                    cleaning_actions TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create data sources table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT UNIQUE,
                    source_type TEXT,
                    last_updated TEXT,
                    record_count INTEGER,
                    quality_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.db_connection.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def upload_file(self, file_path: str, file_type: str = 'auto', source_name: str = None) -> Dict:
        """Upload and process a data file"""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Auto-detect file type if not specified
            if file_type == 'auto':
                file_type = self._detect_file_type(file_path)
            
            # Load data based on file type
            if file_type == 'csv':
                data = pd.read_csv(file_path)
            elif file_type == 'excel':
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Validate data structure
            validation_result = self._validate_data_structure(data)
            if not validation_result['is_valid']:
                return {
                    'success': False,
                    'error': 'Data validation failed',
                    'details': validation_result['errors']
                }
            
            # Clean and process data
            cleaned_data = self._clean_data(data)
            
            # Store in database
            storage_result = self._store_data(cleaned_data, source_name or os.path.basename(file_path))
            
            # Log quality metrics
            self._log_data_quality(file_path, len(data), len(cleaned_data), storage_result['quality_score'])
            
            return {
                'success': True,
                'records_processed': len(data),
                'records_stored': len(cleaned_data),
                'quality_score': storage_result['quality_score'],
                'data_hash': storage_result['data_hash'],
                'message': f"Successfully processed {len(cleaned_data)} records"
            }
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _detect_file_type(self, file_path: str) -> str:
        """Auto-detect file type based on extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return 'csv'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        else:
            raise ValueError(f"Cannot detect file type for: {file_path}")
    
    def _validate_data_structure(self, data: pd.DataFrame) -> Dict:
        """Validate the structure and content of uploaded data"""
        errors = []
        
        # Check required columns
        missing_columns = set(self.data_validators['required_columns']) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        for col, expected_type in self.data_validators['data_types'].items():
            if col in data.columns:
                if expected_type == 'int':
                    if not pd.api.types.is_integer_dtype(data[col]):
                        errors.append(f"Column {col} should be integer type")
                elif expected_type == 'float':
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        errors.append(f"Column {col} should be numeric type")
        
        # Check for empty dataframe
        if data.empty:
            errors.append("Dataframe is empty")
        
        # Check year range if Year column exists
        if 'Year' in data.columns:
            year_data = pd.to_numeric(data['Year'], errors='coerce')
            invalid_years = year_data[(year_data < self.data_validators['year_range'][0]) | 
                                    (year_data > self.data_validators['year_range'][1])]
            if not invalid_years.empty:
                errors.append(f"Years outside valid range {self.data_validators['year_range']}: {invalid_years.unique()}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the data"""
        cleaned_data = data.copy()
        
        # Remove duplicates
        if self.cleaning_rules['remove_duplicates']:
            initial_count = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            logger.info(f"Removed {initial_count - len(cleaned_data)} duplicate records")
        
        # Clean Year column
        if 'Year' in cleaned_data.columns:
            cleaned_data['Year'] = pd.to_numeric(cleaned_data['Year'], errors='coerce')
            cleaned_data = cleaned_data.dropna(subset=['Year'])
            cleaned_data['Year'] = cleaned_data['Year'].astype(int)
        
        # Clean Remittances column
        if 'Remittances (million USD)' in cleaned_data.columns:
            cleaned_data['Remittances (million USD)'] = pd.to_numeric(
                cleaned_data['Remittances (million USD)'], errors='coerce'
            )
            cleaned_data = cleaned_data.dropna(subset=['Remittances (million USD)'])
            
            # Handle outliers
            if self.cleaning_rules['smooth_extreme_values']:
                cleaned_data = self._handle_outliers(cleaned_data, 'Remittances (million USD)')
        
        # Sort by year
        cleaned_data = cleaned_data.sort_values('Year').reset_index(drop=True)
        
        # Fill missing years if requested
        if self.cleaning_rules['fill_missing_years']:
            cleaned_data = self._fill_missing_years(cleaned_data)
        
        # Validate data continuity
        if self.cleaning_rules['validate_continuity']:
            cleaned_data = self._validate_data_continuity(cleaned_data)
        
        return cleaned_data
    
    def _handle_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Handle outliers using statistical methods"""
        values = data[column].values
        mean = np.mean(values)
        std = np.std(values)
        
        # Calculate z-scores
        z_scores = np.abs((values - mean) / std)
        outlier_mask = z_scores > self.cleaning_rules['outlier_threshold']
        
        if outlier_mask.any():
            logger.info(f"Found {outlier_mask.sum()} outliers in {column}")
            # Replace outliers with rolling median
            rolling_median = data[column].rolling(window=3, center=True, min_periods=1).median()
            data.loc[outlier_mask, column] = rolling_median[outlier_mask]
        
        return data
    
    def _fill_missing_years(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing years with interpolated values"""
        if 'Year' not in data.columns:
            return data
        
        year_range = range(data['Year'].min(), data['Year'].max() + 1)
        missing_years = set(year_range) - set(data['Year'])
        
        if missing_years:
            logger.info(f"Filling {len(missing_years)} missing years: {sorted(missing_years)}")
            
            # Create complete year range
            complete_years = pd.DataFrame({'Year': list(year_range)})
            
            # Merge with existing data
            merged_data = complete_years.merge(data, on='Year', how='left')
            
            # Interpolate missing values
            if 'Remittances (million USD)' in merged_data.columns:
                merged_data['Remittances (million USD)'] = merged_data['Remittances (million USD)'].interpolate(method='linear')
            
            return merged_data
        
        return data
    
    def _validate_data_continuity(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data continuity issues"""
        if 'Year' not in data.columns or 'Remittances (million USD)' not in data.columns:
            return data
        
        # Check for large gaps in data
        year_diffs = data['Year'].diff()
        large_gaps = year_diffs[year_diffs > 5]  # Gaps larger than 5 years
        
        if not large_gaps.empty:
            logger.warning(f"Found large gaps in data: {large_gaps.unique()}")
        
        return data
    
    def _store_data(self, data: pd.DataFrame, source_name: str) -> Dict:
        """Store cleaned data in the database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Calculate data hash for deduplication
            data_hash = self._calculate_data_hash(data)
            
            # Check if data already exists
            cursor.execute("SELECT id FROM remittance_data WHERE data_hash = ?", (data_hash,))
            if cursor.fetchone():
                logger.info("Data already exists in database")
                return {
                    'quality_score': 1.0,
                    'data_hash': data_hash,
                    'message': 'Data already exists'
                }
            
            # Prepare data for insertion
            records_to_insert = []
            for _, row in data.iterrows():
                record = {
                    'year': int(row['Year']),
                    'remittances_usd': float(row['Remittances (million USD)']),
                    'remittances_bdt': float(row.get('Remittances (billion BDT)', 0)),
                    'yoy_change': float(row.get('YoY Change (%)', 0)),
                    'cumulative_growth': float(row.get('Cumulative Growth vs. 1995-1996 (%)', 0)),
                    'data_source': source_name,
                    'upload_timestamp': datetime.now().isoformat(),
                    'data_hash': data_hash,
                    'quality_score': self._calculate_quality_score(data)
                }
                records_to_insert.append(record)
            
            # Insert data one by one to avoid constraint issues
            for record in records_to_insert:
                try:
                    cursor.execute('''
                        INSERT INTO remittance_data 
                        (year, remittances_usd, remittances_bdt, yoy_change, cumulative_growth, 
                         data_source, upload_timestamp, data_hash, quality_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record['year'], record['remittances_usd'], record['remittances_bdt'],
                        record['yoy_change'], record['cumulative_growth'], record['data_source'],
                        record['upload_timestamp'], record['data_hash'], record['quality_score']
                    ))
                except sqlite3.IntegrityError as e:
                    if "UNIQUE constraint failed" in str(e):
                        logger.warning(f"Record for year {record['year']} already exists, skipping")
                        continue
                    else:
                        raise
            
            # Update data sources table
            self._update_data_source(source_name, len(records_to_insert))
            
            self.db_connection.commit()
            
            logger.info(f"Successfully stored {len(records_to_insert)} records")
            
            return {
                'quality_score': self._calculate_quality_score(data),
                'data_hash': data_hash,
                'message': f'Stored {len(records_to_insert)} records'
            }
            
        except Exception as e:
            logger.error(f"Data storage failed: {e}")
            self.db_connection.rollback()
            raise
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate a hash of the data for deduplication"""
        data_string = data.to_string()
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate a quality score for the data"""
        score = 1.0
        
        # Penalize for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_ratio * 0.3
        
        # Penalize for outliers
        if 'Remittances (million USD)' in data.columns:
            values = data['Remittances (million USD)'].values
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            outlier_ratio = (z_scores > 3).sum() / len(values)
            score -= outlier_ratio * 0.2
        
        # Penalize for data gaps
        if 'Year' in data.columns:
            year_diffs = data['Year'].diff()
            gap_ratio = (year_diffs > 1).sum() / len(year_diffs)
            score -= gap_ratio * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _update_data_source(self, source_name: str, record_count: int):
        """Update or create data source record"""
        cursor = self.db_connection.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_sources 
            (source_name, source_type, last_updated, record_count, quality_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (source_name, 'file_upload', datetime.now().isoformat(), record_count, 1.0))
    
    def _log_data_quality(self, filename: str, processed: int, cleaned: int, quality_score: float):
        """Log data quality metrics"""
        cursor = self.db_connection.cursor()
        
        cursor.execute('''
            INSERT INTO data_quality_log 
            (upload_timestamp, filename, records_processed, records_cleaned, quality_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), filename, processed, cleaned, quality_score))
        
        self.db_connection.commit()
    
    def get_data_summary(self) -> Dict:
        """Get summary of stored data"""
        try:
            cursor = self.db_connection.cursor()
            
            # Get basic stats
            cursor.execute("SELECT COUNT(*) FROM remittance_data")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(year), MAX(year) FROM remittance_data")
            year_range = cursor.fetchone()
            
            cursor.execute("SELECT AVG(quality_score) FROM remittance_data")
            avg_quality = cursor.fetchone()[0] or 0.0
            
            # Get data sources
            cursor.execute("SELECT source_name, record_count FROM data_sources")
            sources = cursor.fetchall()
            
            return {
                'total_records': total_records,
                'year_range': f"{year_range[0]}-{year_range[1]}" if year_range[0] and year_range[1] else "N/A",
                'average_quality_score': round(avg_quality, 3),
                'data_sources': [{'name': s[0], 'records': s[1]} for s in sources],
                'database_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {}
    
    def export_data(self, format: str = 'csv', output_path: str = None) -> str:
        """Export data from database"""
        try:
            query = "SELECT year, remittances_usd, remittances_bdt, yoy_change, cumulative_growth FROM remittance_data ORDER BY year"
            data = pd.read_sql_query(query, self.db_connection)
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"data/export_remittance_data_{timestamp}.{format}"
            
            if format == 'csv':
                data.to_csv(output_path, index=False)
            elif format == 'excel':
                data.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Data exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise
    
    def clear_database(self):
        """Clear all data from database (for testing purposes)"""
        try:
            cursor = self.db_connection.cursor()
            
            # Clear all tables
            cursor.execute("DELETE FROM remittance_data")
            cursor.execute("DELETE FROM data_quality_log")
            cursor.execute("DELETE FROM data_sources")
            
            # Reset auto-increment counters
            cursor.execute("DELETE FROM sqlite_sequence")
            
            self.db_connection.commit()
            logger.info("Database cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            self.db_connection.rollback()
            raise
    
    def close(self):
        """Close database connection"""
        if self.db_connection:
            self.db_connection.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    # Test the data pipeline
    try:
        pipeline = DataPipeline()
        
        # Test with existing data
        result = pipeline.upload_file("data/Bangladesh Remittances Dataset (19952025).csv")
        print("Upload result:", result)
        
        # Get data summary
        summary = pipeline.get_data_summary()
        print("Data summary:", summary)
        
        pipeline.close()
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
