from typing import Dict, Any, List, Optional, Union, Callable
import math
import sys
import traceback
from loguru import logger

# Data validation and processing utility functions
def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary, avoiding KeyError
    
    Args:
        data: Data dictionary
        key: Key name
        default: Default value returned when key doesn't exist
        
    Returns:
        Data value or default value
    """
    if not isinstance(data, dict):
        return default
    return data.get(key, default)

def safe_number(value: Any, default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely convert value to number, handling None and exception cases
    
    Args:
        value: Value to convert
        default: Default value when conversion fails
        
    Returns:
        Converted number or default value
    """
    if value is None:
        return default
    try:
        num = float(value)
        # Check if it's a valid number (not NaN or infinity)
        if math.isnan(num) or math.isinf(num):
            return default
        return num
    except (ValueError, TypeError):
        return default

def safe_list(value: Any) -> List:
    """
    Ensure value is list type, handling None and non-list values
    
    Args:
        value: Input value
        
    Returns:
        Value in list form
    """
    if value is None:
        return []
    if not isinstance(value, list):
        return [value]
    return value

def safe_sum(values: List, default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely calculate list sum, handling None values and empty lists
    
    Args:
        values: Value list
        default: Default value when list is empty or calculation fails
        
    Returns:
        List sum or default value
    """
    if not values:
        return default
    try:
        # Filter out None values and convert to numbers
        valid_values = [safe_number(v) for v in values if v is not None]
        return sum(valid_values) if valid_values else default
    except Exception as e:
        logger.error(f"Error calculating sum: {e}")
        return default

def safe_avg(values: List, default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely calculate list average, handling None values, empty lists and division by zero
    
    Args:
        values: Value list
        default: Default value when list is empty or calculation fails
        
    Returns:
        List average or default value
    """
    if not values:
        return default
    try:
        # Filter out None values and convert to numbers
        valid_values = [safe_number(v) for v in values if v is not None]
        return sum(valid_values) / len(valid_values) if valid_values else default
    except Exception as e:
        logger.error(f"Error calculating average: {e}")
        return default

def safe_max(values: List, default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely calculate list maximum, handling None values and empty lists
    
    Args:
        values: Value list
        default: Default value when list is empty or calculation fails
        
    Returns:
        List maximum or default value
    """
    if not values:
        return default
    try:
        # Filter out None values and convert to numbers
        valid_values = [safe_number(v) for v in values if v is not None]
        return max(valid_values) if valid_values else default
    except Exception as e:
        logger.error(f"Error calculating maximum: {e}")
        return default

def safe_min(values: List, default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely calculate list minimum, handling None values and empty lists
    
    Args:
        values: Value list
        default: Default value when list is empty or calculation fails
        
    Returns:
        List minimum or default value
    """
    if not values:
        return default
    try:
        # Filter out None values and convert to numbers
        valid_values = [safe_number(v) for v in values if v is not None]
        return min(valid_values) if valid_values else default
    except Exception as e:
        logger.error(f"Error calculating minimum: {e}")
        return default

def safe_count(values: List, predicate: Callable = None) -> int:
    """
    Safely calculate number of elements in list that meet condition
    
    Args:
        values: Value list
        predicate: Predicate function, defaults to counting non-None elements
        
    Returns:
        Number of elements meeting condition
    """
    if not values:
        return 0
    if predicate is None:
        return len([v for v in values if v is not None])
    try:
        return len([v for v in values if v is not None and predicate(v)])
    except Exception as e:
        logger.error(f"Error calculating count: {e}")
        return 0

def log_metric_error(metric_name: str, error: Exception, context: Dict = None):
    """
    Log metric calculation error
    
    Args:
        metric_name: Metric name
        error: Exception object
        context: Context information, such as input data
    """
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb:
        # Get the last frame of the traceback (where the error occurred)
        last_frame = traceback.extract_tb(exc_tb)[-1]
        filename = last_frame.filename
        lineno = last_frame.lineno
        func_name = last_frame.name
        location_info = f"in function '{func_name}' (file '{filename}', line {lineno})"
    else:
        location_info = "at an unknown location"

    error_msg = f"Error calculating metric '{metric_name}': {error} {location_info}"

    if context:
        # Limit context size to avoid log explosion
        context_str = str(context)
        if len(context_str) > 500:
            context_str = context_str[:500] + "..."
        error_msg += f"\nContext: {context_str}"
    logger.error(error_msg)


def create_line_chart_option(title: str, x_axis_data: List[str], series_data: List[Any], series_name: str = "Value") -> Dict[str, Any]:
    """Creates a minimal ECharts line chart option with only essential data."""
    x_axis_data = x_axis_data if x_axis_data is not None else []
    series_data = series_data if series_data is not None else []
    
    return {
        "xAxis": {"data": x_axis_data},
        "series": [{"name": series_name, "type": "line", "data": series_data}]
    }

def create_pie_chart_option(title: str, series_data: List[Dict[str, Any]], series_name: str = "Distribution") -> Dict[str, Any]:
    """Creates a minimal ECharts pie chart option with only essential data."""
    series_data = series_data if series_data is not None else []
    
    return {
        "series": [{
            "name": series_name,
            "type": "pie",
            "data": series_data
        }]
    }

def create_bar_chart_option(title: str, x_axis_data: List[str], series_data: List[Any], series_name: str = "Value") -> Dict[str, Any]:
    """Creates a minimal ECharts bar chart option with only essential data."""
    x_axis_data = x_axis_data if x_axis_data is not None else []
    series_data = series_data if series_data is not None else []
    
    if not series_data or not isinstance(series_data[0], dict):
        series_list = [{
            "name": series_name,
            "type": "bar",
            "data": series_data
        }]
    else:
        series_list = series_data
        
    return {
        "xAxis": {"data": x_axis_data},
        "series": series_list
    }

def create_time_series_chart_option(title: str, series_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Creates a minimal ECharts time-series chart option with only essential data."""
    series_list = series_list if series_list is not None else []
    
    for s in series_list:
        s["type"] = "line"
        
    return {
        "xAxis": {"type": 'time'},
        "series": series_list
    }


# --- End ECharts Option Helpers --- 