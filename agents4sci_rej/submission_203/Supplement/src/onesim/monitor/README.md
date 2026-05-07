# OneSim Monitor System

The Monitor system is a core component of the OneSim framework, designed to collect, compute, visualize, and export various performance metrics for multi-agent systems. It provides a unified interface and flexible configuration options for both real-time API monitoring and static chart exports.

## 1. Overview

The Monitor module offers the following core functionalities:

- **Metric Definition & Registration**: Define monitoring metrics with `MetricDefinition` and register them with the monitoring system
- **Data Collection**: Collect runtime data from environments and agents
- **Metric Computation**: Execute calculation functions based on registered metric definitions
- **Data Visualization**: Support for line charts, bar charts, and pie charts
- **API Interface**: Provide real-time monitoring data through API endpoints
- **Static Export**: Export monitoring data as static charts or JSON files

The Monitor system supports dynamic update frequencies, asynchronous data collection, and multiple data format conversions, ensuring efficient operation in various scenarios.

## 2. Core Components

### 2.1 Data Structures

- **`MetricDefinition`**: Defines a metric, including name, description, visualization type, and required variables
- **`VariableSpec`**: Specifies data sources required for metric calculations
- **`MetricResult`**: Contains calculation results with both raw data and visualization-ready data
- **`TimeSeriesMetricData`**: Stores time series data for line charts
- **`CategoryMetricData`**: Stores category data for bar and pie charts

### 2.2 Core Classes

- **`MonitorManager`**: The central controller of the monitoring system
- **`DataCollector`**: Collects data from environments and agents
- **`MetricProcessor`**: Processes metrics and formats results
- **`MonitorScheduler`**: Schedules periodic metric updates

### 2.3 Utility Classes

- **`MetricAgent`**: An agent that generates monitoring metrics and calculation functions based on scenarios
- **Various Utility Functions**: Safe data access, data transformation, and error handling utilities

## 3. Usage

### 3.1 Basic Usage

```python
from onesim.monitor import MonitorManager, MetricDefinition, VariableSpec

# Create a monitor manager
monitor = MonitorManager()

# Define a metric
economy_metric = MetricDefinition(
    name="economy_growth",
    description="Economic growth rate",
    visualization_type="line",
    variables=[
        VariableSpec(name="gdp", source_type="env", path="gdp"),
        VariableSpec(name="population", source_type="agent", agent_type="Citizen", path="population")
    ],
    update_interval=5  # Update every 5 seconds
)

# Register the metric
monitor.register_metric(economy_metric, calculation_function=calculate_economy_growth)

# Set up the environment
monitor.setup(env)

# Start monitoring
await monitor.start_all_metrics()

# Get metric results
result = monitor.get_result("economy_growth")

# Export charts
monitor.export_metrics_as_images("./output/metrics")

# Stop monitoring
await monitor.stop_all_metrics()
```

### 3.2 Automatic Loading from Scene Files

Define metrics in scene_info.json:

```json
{
  "metrics": [
    {
      "name": "Economy Growth",
      "description": "Economic growth rate over time",
      "visualization_type": "line",
      "update_interval": 5,
      "variables": [
        {
          "name": "gdp",
          "source_type": "env",
          "path": "gdp",
          "required": true
        },
        {
          "name": "population",
          "source_type": "agent",
          "agent_type": "Citizen",
          "path": "population"
        }
      ],
      "function_name": "calculate_economy_growth"
    }
  ]
}
```

Then automatically load the monitoring system in your environment:

```python
# Automatically set up and start the monitor system
monitor_manager = await MonitorManager.setup_metrics(env)
```

### 3.3 Using MetricAgent to Generate Metrics

```python
from onesim.agent import MetricAgent

# Create a MetricAgent
metric_agent = MetricAgent(model_config_name="gpt-4")

# Generate metrics
metrics = metric_agent.generate_metrics(
    scenario_description="Economic growth simulation",
    agent_types=["Citizen", "Company", "Government"],
    system_data_model=data_model
)

# Generate metric calculation module
module_path = metric_agent.generate_metrics_module(
    metrics=metrics, 
    output_dir="./src/envs/economy/code/metrics"
)
```

## 4. Data Structures

### 4.1 Variable Specification

`VariableSpec` defines the data sources required for metric calculations:

```python
@dataclass
class VariableSpec:
    name: str           # Variable name
    source_type: str    # Source type: "env" or "agent"
    path: str           # Variable path (simple variable name)
    required: bool = True                # Whether required
    agent_type: Optional[str] = None     # Agent type if source_type is "agent"
```

### 4.2 Metric Definition

`MetricDefinition` describes all information for a monitoring metric:

```python
@dataclass
class MetricDefinition:
    name: str                     # Metric name
    description: str              # Metric description
    visualization_type: str       # Visualization type: "bar", "pie", "line"
    variables: List[VariableSpec] # Required variables
    calculation_function: str     # Calculation function name or object
    update_interval: int = 60     # Update frequency (seconds)
    visualization_config: Dict = field(default_factory=dict)  # Visualization config
```

### 4.3 Multi-Series Data Format

For line charts, the standard format for multi-series data is:

```python
{
    "xAxis": ["10:00:00", "10:00:05", "10:00:10"],
    "series": [
        {
            "name": "series1",
            "type": "line",
            "data": [10, 20, 30]
        },
        {
            "name": "series2",
            "type": "line", 
            "data": [5, 15, 25]
        }
    ]
}
```

For bar charts, the standard format for multi-series data is:

```python
{
    "xAxis": ["Category1", "Category2", "Category3"],
    "series": [
        {
            "name": "Series1",
            "type": "bar",
            "data": [10, 20, 30]
        },
        {
            "name": "Series2",
            "type": "bar",
            "data": [15, 25, 35]
        }
    ]
}
```

## 5. Metric Calculation

### 5.1 Calculation Function Format

Each metric calculation function should follow this format:

```python
def calculate_metric(data: Dict[str, Any]) -> Any:
    """
    Calculate the metric
    
    Args:
        data: Dictionary containing all variables
        
    Returns:
        Results in different formats based on visualization type:
        - line: Dictionary with series names as keys and values as corresponding numbers
        - bar/pie: Dictionary with categories as keys and values as corresponding numbers
    """
    try:
        # Validate data
        if not data or not isinstance(data, dict):
            return {} if metric_type != "line" else {"default": 0}
            
        # Process environment variables
        env_var = safe_get(data, "variable_name")
        
        # Process agent variables
        agent_var = safe_list(safe_get(data, "agent_type", {}).get("variable_name", []))
        agent_avg = safe_avg(agent_var)
        
        # Calculate results
        # ...
        
        # Return results
        return result
    except Exception as e:
        log_metric_error("metric_name", e)
        return {} if metric_type != "line" else {"default": 0}
```

### 5.2 Calculation Result Format

Calculation functions should return data in a format suitable for the visualization type:

- **Line Chart (line)**: Return a dictionary with series names as keys and values as numbers
  ```python
  return {"Series1": 10, "Series2": 20}
  ```

- **Bar Chart (bar)**: Return a dictionary with categories as keys and values as numbers
  ```python
  return {"Category1": 10, "Category2": 20, "Category3": 30}
  ```

- **Pie Chart (pie)**: Return a dictionary with categories as keys and values as proportions
  ```python
  return {"Category1": 0.2, "Category2": 0.3, "Category3": 0.5}
  ```

## 6. Best Practices

### 6.1 Metric Design Recommendations

- **Naming Convention**: Use descriptive names like "economic_growth_rate" instead of "egr"
- **Complete Description**: Provide detailed metric descriptions including calculation methods and significance
- **Sensible Categorization**: Categorize metrics by function, object, or theme
- **Update Frequency**: Set appropriate update frequencies based on data change rates

### 6.2 Calculation Function Guidelines

- **Robustness**: Handle various edge cases, including None values, empty lists, and type errors
- **Performance Optimization**: Avoid time-consuming operations in calculation functions
- **Utility Functions**: Use utility functions like safe_get, safe_list for safer data access
- **Appropriate Comments**: Add explanatory comments for complex logic

### 6.3 Visualization Selection Guidelines

- **Line Charts**: Suitable for time series data to display trends over time
- **Bar Charts**: Suitable for category comparisons, showing differences between categories
- **Pie Charts**: Suitable for showing proportions, displaying how parts contribute to a whole

## 7. Troubleshooting

### 7.1 Metrics Not Updating

Possible causes:
- Calculation function errors that aren't caught
- Data collection failures
- Scheduler not correctly started

Solutions:
- Check error logs
- Manually trigger an update: `await monitor.update_metric("metric_name")`
- Verify correct data provision in the environment

### 7.2 Multi-Series Data Display Issues

Possible causes:
- Inconsistent data formats
- Calculation function returns formats that don't match expectations

Solutions:
- Ensure calculation functions return the correct multi-series format
- Check if `_normalize_line_data` method is processing correctly
- Use debug mode to output raw data formats for inspection

## 8. Advanced Features

### 8.1 Custom Visualization Configuration

You can customize the visualization style through the `visualization_config` field:

```python
metric_def = MetricDefinition(
    # ...other parameters
    visualization_config={
        "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
        "title": {
            "text": "Custom Title",
            "subtext": "Subtitle"
        },
        "legend": {
            "orient": "vertical",
            "right": 10
        }
    }
)
```

### 8.2 Data Merging and Transformation

The Monitor system supports data merging and transformation:

```python
# Merge two time series datasets
ts_data1.merge(ts_data2)

# Normalize raw data into standard formats
normalized_data = monitor._normalize_line_data(raw_data)
```

### 8.3 Custom Export Formats

You can customize export formats for specific needs:

```python
# Get data in specific format
matplotlib_data = monitor.get_metric_data("metric_name", format="matplotlib")

# Use custom format for export
custom_data = monitor._format_for_api_display(data, "line")
``` 