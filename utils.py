import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from datetime import datetime, timezone, timedelta
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from functools import lru_cache
import json
import io
import scipy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class F1DataAPI:
    BASE_URL = "https://api.openf1.org/v1"
    MAX_RETRIES = 5  # Increased from 3 to 5
    RETRY_DELAY = 2  # Increased from 1 to 2 seconds
    TIMEOUT = 30  # Increased from 10 to 30 seconds
    CACHE_TTL = 300  # Increased from 60 to 300 seconds (5 minutes)
    
    class APIError(Exception):
        """Custom exception for API errors"""
        pass
    
    @staticmethod
    def _create_session() -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=F1DataAPI.MAX_RETRIES,
            backoff_factor=F1DataAPI.RETRY_DELAY,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
            raise_on_status=False  # Don't raise an exception on status
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    @staticmethod
    def _cache_key(*args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key = f"{args}_{sorted(kwargs.items())}"
        return hash(key).__str__()
    
    @staticmethod
    def _is_cache_valid(cache_time: datetime) -> bool:
        """Check if cached data is still valid."""
        return (datetime.now() - cache_time) < timedelta(seconds=F1DataAPI.CACHE_TTL)
    
    @staticmethod
    def _make_request(url: str, params: Optional[Dict] = None, endpoint: str = "") -> Tuple[Optional[Union[List, Dict]], Optional[str]]:
        """Make HTTP request with retries and error handling."""
        session = F1DataAPI._create_session()
        error_msg = None
        retry_count = 0
        max_attempts = F1DataAPI.MAX_RETRIES + 1  # +1 for initial attempt
        
        while retry_count < max_attempts:
            try:
                response = session.get(
                    url,
                    params=params,
                    timeout=F1DataAPI.TIMEOUT
                )
                
                # Check if we got a 500 error
                if response.status_code == 500:
                    retry_count += 1
                    if retry_count < max_attempts:
                        sleep_time = F1DataAPI.RETRY_DELAY * (2 ** (retry_count - 1))  # Exponential backoff
                        logger.warning(f"Got 500 error from {endpoint}, attempt {retry_count}/{max_attempts-1}, waiting {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                
                # For other error status codes
                if not response.ok:
                    error_msg = f"HTTP {response.status_code} error from {endpoint}: {response.text}"
                    logger.error(error_msg)
                    return None, error_msg
                
                try:
                    data = response.json()
                except ValueError as e:
                    error_msg = f"Invalid JSON response from {endpoint}: {e}"
                    logger.error(error_msg)
                    return None, error_msg
                
                # Validate response is a list or dict
                if not isinstance(data, (list, dict)):
                    error_msg = f"Invalid response format from {endpoint}: {type(data)}"
                    logger.error(error_msg)
                    return None, error_msg
                
                # If we got here, we have valid data
                if retry_count > 0:
                    logger.info(f"Successfully retrieved data from {endpoint} after {retry_count} retries")
                return data, None
                
            except requests.exceptions.Timeout:
                error_msg = f"Request to {endpoint} timed out after {F1DataAPI.TIMEOUT} seconds"
                logger.error(error_msg)
                break  # Don't retry on timeout
            except requests.exceptions.ConnectionError as e:
                retry_count += 1
                if retry_count < max_attempts:
                    sleep_time = F1DataAPI.RETRY_DELAY * (2 ** (retry_count - 1))
                    logger.warning(f"Connection error to {endpoint}, attempt {retry_count}/{max_attempts-1}, waiting {sleep_time}s: {e}")
                    time.sleep(sleep_time)
                    continue
                error_msg = f"Connection error to {endpoint} after {retry_count} attempts: {e}"
                logger.error(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = f"Request to {endpoint} failed: {e}"
                logger.error(error_msg)
                break  # Don't retry on other errors
            
        session.close()
        return None, error_msg

    @staticmethod
    def get_available_sessions() -> Tuple[List[Dict], Optional[str]]:
        """Fetch list of available sessions from OpenF1 API with caching."""
        data, error = F1DataAPI._make_request(
            f"{F1DataAPI.BASE_URL}/sessions",
            endpoint="sessions"
        )
        if not data:
            return [], error
            
        if not isinstance(data, list):
            return [], "Sessions data is not a list"
            
        valid_sessions = []
        invalid_sessions = 0
        required_fields = {'session_key', 'date_start', 'session_name', 'session_type'}
        
        for session in data:
            if not isinstance(session, dict):
                invalid_sessions += 1
                continue
                
            # Check required fields exist and have non-empty values
            missing_fields = set()
            for field in required_fields:
                if field not in session or not session[field]:
                    missing_fields.add(field)
                    
            if missing_fields:
                invalid_sessions += 1
                # Log only once per batch of invalid sessions
                if invalid_sessions == 1:
                    logger.warning(
                        f"Found sessions with missing or empty required fields. Example missing fields: {missing_fields}"
                    )
                continue
                
            # Validate session_key format
            try:
                session_id = int(session['session_key'])
                if session_id <= 0:
                    invalid_sessions += 1
                    continue
            except (ValueError, TypeError):
                invalid_sessions += 1
                continue
                
            # Validate date format
            try:
                datetime.fromisoformat(session['date_start'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                invalid_sessions += 1
                continue
                
            # Map API fields to our expected format
            session_mapped = {
                'session_id': session['session_key'],
                'date': session['date_start'],
                'type': session['session_name'],
                'meeting_name': session.get('circuit_short_name', session.get('location', '')),
                'country_name': session.get('country_name', ''),
                'session_type': session['session_type']
            }
            
            valid_sessions.append(session_mapped)
            
        if not valid_sessions and invalid_sessions > 0:
            logger.error(f"All {invalid_sessions} sessions were invalid. No valid sessions found.")
            return [], "No valid sessions found in API response"
            
        if invalid_sessions > 0:
            logger.info(f"Filtered out {invalid_sessions} invalid sessions. Found {len(valid_sessions)} valid sessions.")
            
        return sorted(valid_sessions, key=lambda x: x.get('date', ''), reverse=True), None

    @staticmethod
    @lru_cache(maxsize=128)
    def get_session_data(session_id: int) -> Tuple[Optional[Dict], Optional[str]]:
        """Fetch session information from OpenF1 API with caching."""
        if not F1DataAPI._validate_session_id(session_id):
            return None, f"Invalid session_id: {session_id}"
            
        data, error = F1DataAPI._make_request(
            f"{F1DataAPI.BASE_URL}/sessions",
            endpoint="sessions"
        )
        if not data:
            return None, error
            
        # Find the session with matching session_key
        session_data = None
        if isinstance(data, list):
            for session in data:
                if str(session.get('session_key')) == str(session_id):
                    session_data = session
                    break
        else:
            session_data = data

        if not session_data:
            return None, f"Session {session_id} not found in response"

        required_fields = {'session_key', 'session_name', 'session_type', 'date_start'}
        missing_fields = required_fields - set(session_data.keys())
        if missing_fields:
            return None, f"Missing required fields: {missing_fields}"
            
        # Map API fields to our expected format
        session_mapped = {
            'session_id': session_data['session_key'],
            'session_type': session_data['session_type'],
            'track_name': session_data.get('circuit_short_name', session_data.get('location', 'Unknown')),
            'date': session_data['date_start'],
            'country_name': session_data.get('country_name', ''),
            'type': session_data['session_name']
        }
            
        return session_mapped, None

    @staticmethod
    @lru_cache(maxsize=128)
    def get_driver_list(session_id: int) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """Get list of drivers for a session."""
        if not F1DataAPI._validate_session_id(session_id):
            return None, f"Invalid session_id: {session_id}"
            
        data, error = F1DataAPI._make_request(
            f"{F1DataAPI.BASE_URL}/drivers",
            params={"session_key": session_id},
            endpoint="drivers"
        )
        if not data:
            return None, error
            
        if not isinstance(data, list):
            return None, "Expected list response from drivers endpoint"
            
        required_fields = {'driver_number', 'full_name', 'team_name'}
        drivers = []
        for driver in data:
            missing_fields = required_fields - set(driver.keys())
            if missing_fields:
                continue
                
            drivers.append({
                'driver_number': driver['driver_number'],
                'driver_name': driver['full_name'],
                'team_name': driver['team_name']
            })
            
        if not drivers:
            return None, "No valid driver data found"
            
        return drivers, None

    @staticmethod
    @lru_cache(maxsize=128)
    def get_lap_times(session_id: int, driver_number: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get lap times for a session, optionally filtered by driver."""
        if not F1DataAPI._validate_session_id(session_id):
            return None, f"Invalid session_id: {session_id}"
            
        params = {"session_key": session_id}
        if driver_number is not None:
            if not isinstance(driver_number, int) or driver_number <= 0:
                return None, f"Invalid driver_number: {driver_number}"
            params["driver_number"] = driver_number
            
        data, error = F1DataAPI._make_request(
            f"{F1DataAPI.BASE_URL}/laps",
            params=params,
            endpoint="laps"
        )
        if not data:
            return None, error
            
        if not isinstance(data, list):
            return None, "Expected list response from laps endpoint"
            
        required_fields = {'lap_number', 'driver_number', 'lap_duration'}
        try:
            df = pd.DataFrame(data)
            missing_fields = required_fields - set(df.columns)
            if missing_fields:
                return None, f"Missing required fields in lap data: {missing_fields}"
                
            # Map lap_duration to lap_time for compatibility
            df['lap_time'] = df['lap_duration']
            
            return df, None
            
        except Exception as e:
            return None, f"Error creating DataFrame: {str(e)}"

    @staticmethod
    def get_telemetry(session_id: int, driver_number: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Fetch telemetry data for a driver."""
        if not F1DataAPI._validate_session_id(session_id):
            return None, f"Invalid session_id: {session_id}"
            
        if not isinstance(driver_number, int) or driver_number <= 0:
            return None, f"Invalid driver_number: {driver_number}"
            
        data, error = F1DataAPI._make_request(
            f"{F1DataAPI.BASE_URL}/car_data",
            params={
                "session_key": session_id,
                "driver_number": driver_number
            },
            endpoint="car_data"
        )
        if not data:
            return None, error
            
        if not isinstance(data, list):
            return None, "Expected list response from car_data endpoint"
            
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return df, "No telemetry data available for this driver"
            return df, None
        except Exception as e:
            return None, f"Error creating DataFrame: {str(e)}"

    @staticmethod
    def get_weather_data(session_id: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Fetch weather data for a session."""
        if not F1DataAPI._validate_session_id(session_id):
            return None, f"Invalid session_id: {session_id}"
            
        data, error = F1DataAPI._make_request(
            f"{F1DataAPI.BASE_URL}/weather",
            params={"session_key": session_id},
            endpoint="weather"
        )
        if not data:
            return None, error
            
        if not isinstance(data, list):
            return None, "Expected list response from weather endpoint"
            
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return df, "No weather data available for this session"
            return df, None
        except Exception as e:
            return None, f"Error creating DataFrame: {str(e)}"

    @staticmethod
    def _validate_session_id(session_id: Any) -> bool:
        """Validate session ID."""
        if not isinstance(session_id, (int, str)):
            return False
        try:
            session_id = int(session_id)
            return session_id > 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _validate_date(date_str: str) -> bool:
        """Validate date string format."""
        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return True
        except (ValueError, AttributeError):
            return False

    @staticmethod
    @lru_cache(maxsize=128)
    def get_session_results(session_id: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get session results with position, points, and status."""
        if not F1DataAPI._validate_session_id(session_id):
            return None, f"Invalid session_id: {session_id}"
            
        # First get lap times to calculate fastest laps
        lap_times_df, error = F1DataAPI.get_lap_times(session_id)
        if error:
            return None, f"Error getting lap times: {error}"
            
        # Get driver list
        drivers_list, error = F1DataAPI.get_driver_list(session_id)
        if error:
            return None, f"Error getting driver list: {error}"
            
        if not drivers_list:
            return None, "No driver data available"
            
        # Convert to DataFrame
        drivers_df = pd.DataFrame(drivers_list)
        
        # If we have lap times, calculate fastest lap for each driver
        if lap_times_df is not None and not lap_times_df.empty:
            # Process lap times
            lap_times_df = process_lap_times(lap_times_df)
            
            # Get fastest lap per driver
            fastest_laps = lap_times_df.groupby('driver_number')['lap_time_seconds'].min().reset_index()
            fastest_laps.columns = ['driver_number', 'fastest_lap']
            
            # Merge with drivers
            results_df = pd.merge(drivers_df, fastest_laps, on='driver_number', how='left')
            
            # Calculate total laps per driver
            lap_counts = lap_times_df.groupby('driver_number')['lap_number'].max().reset_index()
            lap_counts.columns = ['driver_number', 'total_laps']
            
            # Merge with results
            results_df = pd.merge(results_df, lap_counts, on='driver_number', how='left')
            
            # Sort by fastest lap (ascending)
            results_df = results_df.sort_values('fastest_lap').reset_index(drop=True)
            
            # Add position column
            results_df['position'] = results_df.index + 1
            
            # Reorder columns
            cols = ['position', 'driver_number', 'driver_name', 'team_name', 'fastest_lap', 'total_laps']
            results_df = results_df[cols]
            
            return results_df, None
        else:
            # If no lap times, just return driver list with position as None
            drivers_df['position'] = None
            drivers_df['fastest_lap'] = None
            drivers_df['total_laps'] = None
            
            # Reorder columns
            cols = ['position', 'driver_number', 'driver_name', 'team_name', 'fastest_lap', 'total_laps']
            results_df = drivers_df[cols]
            
            return results_df, None

def process_lap_times(lap_data: pd.DataFrame) -> pd.DataFrame:
    """Process and clean lap time data."""
    if lap_data is None or lap_data.empty:
        return pd.DataFrame()
        
    # Make a copy to avoid modifying the original
    processed_data = lap_data.copy()
    
    # Convert time strings to seconds and handle NaN values
    if 'lap_time' in processed_data.columns:
        # Convert lap times to seconds
        processed_data['lap_time_seconds'] = pd.to_numeric(processed_data['lap_time'], errors='coerce')
        
        # Remove invalid lap times (NaN, negative, or extremely large values)
        processed_data = processed_data[
            (processed_data['lap_time_seconds'].notna()) &
            (processed_data['lap_time_seconds'] > 0) &
            (processed_data['lap_time_seconds'] < 1000)  # Assuming no lap takes more than 1000 seconds
        ]
        
        # If we have very few valid laps, return empty DataFrame
        if len(processed_data) < 3:  # Need at least 3 laps for meaningful statistics
            logger.warning("Not enough valid lap times for analysis")
            return pd.DataFrame()
    
    # Ensure required columns exist
    required_columns = ['driver_number', 'lap_number', 'lap_time_seconds']
    missing_columns = set(required_columns) - set(processed_data.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return pd.DataFrame()
    
    # Convert driver_number and lap_number to numeric, handling errors
    processed_data['driver_number'] = pd.to_numeric(processed_data['driver_number'], errors='coerce')
    processed_data['lap_number'] = pd.to_numeric(processed_data['lap_number'], errors='coerce')
    
    # Remove rows with invalid driver or lap numbers
    processed_data = processed_data[
        (processed_data['driver_number'].notna()) &
        (processed_data['lap_number'].notna()) &
        (processed_data['driver_number'] > 0) &
        (processed_data['lap_number'] > 0)
    ]
    
    # Sort by driver and lap number
    processed_data = processed_data.sort_values(['driver_number', 'lap_number'])
    
    # Reset index after all the filtering
    processed_data = processed_data.reset_index(drop=True)
    
    return processed_data

def calculate_race_statistics(lap_data: pd.DataFrame) -> Dict:
    """Calculate basic race statistics."""
    if lap_data.empty:
        return {}
        
    # Drop NaN values before calculating statistics
    lap_times = lap_data['lap_time_seconds'].dropna()
    
    if len(lap_times) == 0:
        return {}
        
    stats = {
        'fastest_lap': lap_times.min(),
        'average_lap': lap_times.mean(),
        'total_laps': lap_data['lap_number'].max(),
        'drivers_count': lap_data['driver_number'].nunique()
    }
    
    return stats

def predict_lap_times(historical_laps: pd.DataFrame, driver_number: int) -> Dict:
    """Predict future lap times based on historical data."""
    if historical_laps.empty:
        return {}
        
    driver_laps = historical_laps[historical_laps['driver_number'] == driver_number].copy()
    
    # Drop rows with NaN values in lap_time_seconds
    driver_laps = driver_laps.dropna(subset=['lap_time_seconds'])
    
    if len(driver_laps) < 5:  # Need minimum data points
        return {}
        
    X = driver_laps[['lap_number']].values
    y = driver_laps['lap_time_seconds'].values
    
    # Additional check for NaN values
    if np.isnan(y).any():
        logger.warning(f"NaN values found in lap times for driver {driver_number} after dropna. Filtering them out.")
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(y) < 5:  # Recheck minimum data points after filtering
            return {}
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_lap = driver_laps['lap_number'].max() + 1
    predicted_time = model.predict([[next_lap]])[0]
    
    trend = 'improving' if model.coef_[0] < 0 else 'declining'
    
    return {
        'next_lap_prediction': predicted_time,
        'trend': trend,
        'confidence': model.score(X, y)
    }

def analyze_sector_performance(lap_data: pd.DataFrame) -> Dict:
    """Analyze performance across different sectors."""
    if lap_data.empty:
        return {}
    
    # Check if we have sector data in the DataFrame
    sector_columns = [col for col in lap_data.columns if 'sector' in col.lower()]
    
    # If we have direct sector columns
    if any(col in ['sector1_time', 'sector2_time', 'sector3_time'] for col in lap_data.columns):
        try:
            analysis = {}
            
            # Process each sector if available
            if 'sector1_time' in lap_data.columns:
                sector1_times = pd.to_numeric(lap_data['sector1_time'], errors='coerce').dropna()
                if not sector1_times.empty:
                    analysis['sector1_avg'] = sector1_times.mean()
                    analysis['sector1_std'] = sector1_times.std()
                    analysis['sector1_min'] = sector1_times.min()
                    
            if 'sector2_time' in lap_data.columns:
                sector2_times = pd.to_numeric(lap_data['sector2_time'], errors='coerce').dropna()
                if not sector2_times.empty:
                    analysis['sector2_avg'] = sector2_times.mean()
                    analysis['sector2_std'] = sector2_times.std()
                    analysis['sector2_min'] = sector2_times.min()
                    
            if 'sector3_time' in lap_data.columns:
                sector3_times = pd.to_numeric(lap_data['sector3_time'], errors='coerce').dropna()
                if not sector3_times.empty:
                    analysis['sector3_avg'] = sector3_times.mean()
                    analysis['sector3_std'] = sector3_times.std()
                    analysis['sector3_min'] = sector3_times.min()
                    
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing sector times: {e}")
            return {}
    
    # If we have a sector_times column that contains JSON or dict data
    elif 'sector_times' in lap_data.columns:
        try:
            # Try to normalize the sector_times column which might contain JSON or dict data
            sectors_df = pd.json_normalize(lap_data['sector_times'].apply(lambda x: 
                                          json.loads(x) if isinstance(x, str) else x))
            
            analysis = {}
            
            # Check for different possible column names
            sector1_col = next((col for col in sectors_df.columns if col in ['1', 'sector1', 'S1']), None)
            sector2_col = next((col for col in sectors_df.columns if col in ['2', 'sector2', 'S2']), None)
            sector3_col = next((col for col in sectors_df.columns if col in ['3', 'sector3', 'S3']), None)
            
            if sector1_col and not sectors_df[sector1_col].empty:
                sector1_times = pd.to_numeric(sectors_df[sector1_col], errors='coerce').dropna()
                if not sector1_times.empty:
                    analysis['sector1_avg'] = sector1_times.mean()
                    analysis['sector1_std'] = sector1_times.std()
                    analysis['sector1_min'] = sector1_times.min()
            
            if sector2_col and not sectors_df[sector2_col].empty:
                sector2_times = pd.to_numeric(sectors_df[sector2_col], errors='coerce').dropna()
                if not sector2_times.empty:
                    analysis['sector2_avg'] = sector2_times.mean()
                    analysis['sector2_std'] = sector2_times.std()
                    analysis['sector2_min'] = sector2_times.min()
            
            if sector3_col and not sectors_df[sector3_col].empty:
                sector3_times = pd.to_numeric(sectors_df[sector3_col], errors='coerce').dropna()
                if not sector3_times.empty:
                    analysis['sector3_avg'] = sector3_times.mean()
                    analysis['sector3_std'] = sector3_times.std()
                    analysis['sector3_min'] = sector3_times.min()
                
            return analysis
        except Exception as e:
            logger.error(f"Error processing sector_times column: {e}")
            return {}
    
    # No sector data found
    logger.warning("No sector data found in lap_data DataFrame")
    return {}

def export_data(data: pd.DataFrame, format: str = 'csv') -> bytes:
    """Export data in various formats."""
    if data.empty:
        return b''
        
    if format == 'csv':
        return data.to_csv(index=False).encode('utf-8')
    elif format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=False, sheet_name='F1Data')
            # Auto-adjust columns' width
            worksheet = writer.sheets['F1Data']
            for i, col in enumerate(data.columns):
                # Find the maximum length of the column
                max_len = max(
                    data[col].astype(str).map(len).max(),  # Length of largest item
                    len(str(col))  # Length of column name
                ) + 2  # Add a little extra space
                worksheet.set_column(i, i, max_len)  # Set column width
        
        output.seek(0)
        return output.getvalue()
    elif format == 'json':
        return data.to_json(orient='records').encode('utf-8')
    return b''

def calculate_advanced_statistics(lap_data: pd.DataFrame) -> Dict:
    """Calculate advanced race statistics."""
    if lap_data is None or lap_data.empty:
        return {}
        
    # Ensure we have the required column
    if 'lap_time_seconds' not in lap_data.columns:
        logger.error("lap_time_seconds column not found in data")
        return {}
        
    # Drop NaN values and get valid lap times
    lap_times = lap_data['lap_time_seconds'].dropna()
    
    # Remove outliers (times that are too short or too long)
    lap_times = lap_times[
        (lap_times > 0) &
        (lap_times < 1000)  # Assuming no lap takes more than 1000 seconds
    ]
    
    if len(lap_times) == 0:
        logger.warning("No valid lap times for statistics calculation")
        return {}
        
    try:
        stats = {
            'fastest_lap': float(lap_times.min()),
            'average_lap': float(lap_times.mean()),
            'median_lap': float(lap_times.median()),
            'std_dev': float(lap_times.std()),
            'total_laps': int(lap_data['lap_number'].max()),
            'drivers_count': int(lap_data['driver_number'].nunique())
        }
        
        # Calculate consistency score only if we have valid standard deviation
        if stats['std_dev'] > 0:
            stats['consistency_score'] = float(1 / (stats['std_dev'] / stats['average_lap']))
        else:
            stats['consistency_score'] = 0
            
        # Only calculate percentiles if we have enough data points
        if len(lap_times) >= 5:
            stats.update({
                'percentile_95': float(np.percentile(lap_times, 95)),
                'percentile_5': float(np.percentile(lap_times, 5))
            })
            
        # Only calculate advanced statistics if we have enough data points
        if len(lap_times) >= 3:
            stats.update({
                'skewness': float(scipy.stats.skew(lap_times)),
                'kurtosis': float(scipy.stats.kurtosis(lap_times))
            })
            
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        return {}

def format_lap_time(seconds: float) -> str:
    """Format lap time in seconds to a readable format.
    
    For times over 1 minute, displays as '1m 23.456s' format.
    For times under 1 minute, displays as '45.678s' format.
    """
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.3f}s"
    else:
        return f"{seconds:.3f}s" 