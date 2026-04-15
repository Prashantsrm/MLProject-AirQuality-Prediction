# Monitoring and Logging System

## Overview

The Air Quality Prediction System includes a comprehensive monitoring and logging infrastructure designed to track system health, performance metrics, and execution statistics. This document describes the monitoring capabilities and how to use them.

## Components

### 1. Structured Logging (`src/utils/logger.py`)

The logging system provides structured, rotated logging with both console and file output.

**Features:**
- Structured logging with timestamps and log levels
- Automatic log rotation (10MB per file, 5 backup files)
- Console and file output
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**Usage:**
```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.warning("High memory usage detected")
logger.error("Failed to process data", exc_info=True)
```

**Configuration:**
- Log file: `logs/system.log`
- Max file size: 10MB
- Backup count: 5 files
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

### 2. System Health Monitoring (`src/utils/monitoring.py`)

Monitors system resources including CPU, memory, and disk usage.

**Features:**
- Real-time CPU, memory, and disk usage tracking
- Historical metrics with configurable window size
- Health status checks with configurable thresholds
- Average, min, and max metrics calculation

**Usage:**
```python
from src.utils.monitoring import SystemHealthMonitor

monitor = SystemHealthMonitor(history_size=100)

# Collect current metrics
metrics = monitor.collect_metrics()
print(f"CPU: {metrics['cpu_percent']:.1f}%")
print(f"Memory: {metrics['memory_percent']:.1f}%")

# Check system health
is_healthy, warnings = monitor.is_healthy(
    cpu_threshold=80.0,
    memory_threshold=85.0,
    disk_threshold=90.0
)

if not is_healthy:
    for warning in warnings:
        print(f"Warning: {warning}")
```

### 3. Execution Time Tracking (`src/utils/monitoring.py`)

Tracks execution times for pipeline stages and operations.

**Features:**
- Start/stop timer for operations
- Statistics calculation (min, max, mean, total)
- Multiple timings per operation
- Performance metrics aggregation

**Usage:**
```python
from src.utils.monitoring import ExecutionTimeTracker

tracker = ExecutionTimeTracker()

# Time an operation
tracker.start_timer('data_ingestion')
# ... do work ...
duration = tracker.end_timer('data_ingestion')

# Get statistics
stats = tracker.get_statistics('data_ingestion')
print(f"Mean time: {stats['mean_seconds']:.3f}s")
print(f"Total time: {stats['total_seconds']:.3f}s")

# Get all statistics
all_stats = tracker.get_all_statistics()
```

### 4. Event Metrics Collection (`src/utils/monitoring.py`)

Collects metrics for streaming event processing.

**Features:**
- Event count and error rate tracking
- Latency statistics (min, max, mean, percentiles)
- Throughput calculation
- Configurable history window

**Usage:**
```python
from src.utils.monitoring import EventMetricsCollector

collector = EventMetricsCollector(window_size=1000)

# Record events
collector.record_event(latency_ms=0.85, success=True)
collector.record_event(latency_ms=1.2, success=False)

# Get metrics
metrics = collector.get_metrics()
print(f"Events: {metrics['event_count']}")
print(f"Error rate: {metrics['error_rate'] * 100:.2f}%")
print(f"Mean latency: {metrics['mean_latency_ms']:.2f}ms")
print(f"P95 latency: {metrics['p95_latency_ms']:.2f}ms")
```

### 5. Performance Monitor (`src/utils/monitoring.py`)

Aggregates all monitoring data into comprehensive reports.

**Features:**
- System health reporting
- Performance metrics aggregation
- Full system monitoring reports
- Report export to JSON

**Usage:**
```python
from src.utils.monitoring import get_performance_monitor

monitor = get_performance_monitor()

# Get reports
system_report = monitor.get_system_report()
performance_report = monitor.get_performance_report()
full_report = monitor.get_full_report()

# Save report
filepath = monitor.save_report('monitoring_report')
print(f"Report saved to {filepath}")

# Log report
monitor.log_report()
```

### 6. Logging Integration (`src/utils/logging_integration.py`)

Provides decorators and utilities for consistent logging across components.

**Features:**
- Operation logging decorator
- Data processing logging
- Model training logging
- Prediction logging
- Alert logging
- Structured logger class

**Usage:**
```python
from src.utils.logging_integration import (
    log_operation,
    log_data_processing,
    log_model_training,
    log_prediction,
    log_alert,
    create_structured_logger
)

# Using decorator
@log_operation("data_ingestion", track_performance=True)
def ingest_data():
    pass

# Using logging functions
log_data_processing(
    stage_name="Silver Layer",
    input_records=1000,
    output_records=950,
    rejected_records=50,
    quality_score=95.0
)

log_model_training(
    model_name="XGBoost",
    training_samples=1000,
    test_samples=200,
    metrics={'rmse': 10.5, 'mae': 8.2, 'r2': 0.85},
    duration_seconds=30.5
)

log_prediction(
    city="Delhi",
    current_aqi=150.5,
    predicted_aqi=160.2,
    latency_ms=0.85,
    alert_triggered=True
)

log_alert(
    alert_type="rule-based",
    city="Delhi",
    aqi_value=250.0,
    alert_level="Heavily Polluted",
    message="AQI exceeds safe threshold"
)

# Using structured logger
logger = create_structured_logger('my_module')
logger.log_event('DATA_PROCESSING', 'ingestion', level='INFO', records=1000)
logger.log_performance('etl_pipeline', duration_seconds=120.5, records_processed=5000)
logger.log_data_quality('silver_layer', 1000, 950, 50, 95.0)
logger.log_error('processing', 'Failed to process data', error_type='ProcessingError')
```

## Monitoring Dashboard

The system includes a Streamlit-based monitoring dashboard for real-time visualization of system metrics.

### Starting the Dashboard

```bash
python scripts/start_monitoring_dashboard.py
```

**Options:**
```bash
python scripts/start_monitoring_dashboard.py --port 8502
python scripts/start_monitoring_dashboard.py --host 0.0.0.0 --port 8502
python scripts/start_monitoring_dashboard.py --logger-level debug
```

### Dashboard Features

The dashboard provides multiple views:

1. **System Health**
   - Current CPU, memory, and disk usage
   - System health status
   - Historical average metrics
   - System warnings

2. **Execution Times**
   - Table of all operations with timing statistics
   - Bar chart of average execution times
   - Min, max, and mean times per operation

3. **Event Metrics**
   - Total events processed
   - Error rate
   - Throughput (events/second)
   - Latency statistics (min, max, mean, P95, P99)
   - Latency distribution visualization

4. **System Uptime**
   - Total uptime in human-readable format
   - System start time

5. **Logs**
   - Recent log entries from system.log
   - Configurable number of lines to display

6. **Export**
   - Save current monitoring report
   - View recent saved reports

### Dashboard URL

Default: `http://localhost:8502`

## Generating Reports

The system provides a script to generate comprehensive monitoring reports.

### Basic Usage

```bash
python scripts/generate_monitoring_report.py
```

### Report Types

```bash
# Summary report (default)
python scripts/generate_monitoring_report.py --report-type summary

# Full report with all details
python scripts/generate_monitoring_report.py --report-type full

# System health only
python scripts/generate_monitoring_report.py --report-type system

# Performance metrics only
python scripts/generate_monitoring_report.py --report-type performance
```

### Output Options

```bash
# Save to custom directory
python scripts/generate_monitoring_report.py --output-dir reports

# Print summary to console
python scripts/generate_monitoring_report.py --print-summary
```

### Report Contents

Reports include:
- Timestamp
- System health metrics (CPU, memory, disk)
- Performance metrics (uptime, operations, execution times)
- Event processing metrics (throughput, latency, error rate)
- Detailed execution time statistics
- Historical metrics

## Integration with Pipeline Components

### ETL Pipeline

The ETL pipeline automatically logs:
- Pipeline start and completion
- Record counts at each stage
- Data quality scores
- Processing times for each layer
- Performance metrics

```python
from src.etl_pipeline.pipeline import ETLPipeline

pipeline = ETLPipeline(bronze_path, silver_path, gold_path)
gold_df, metrics = pipeline.run_pipeline(bronze_df, source='kaggle')

# Metrics include timing and record counts
print(metrics['total_time_seconds'])
print(metrics['records']['valid_records'])
```

### Streaming Pipeline

The streaming pipeline automatically tracks:
- Event processing latency
- Feature computation time
- Model prediction time
- Latency statistics

```python
from src.streaming.streaming_inference_pipeline import StreamingInferencePipeline

pipeline = StreamingInferencePipeline(model)
result = pipeline.process_event(event)

# Get latency statistics
stats = pipeline.get_latency_stats()
print(f"Mean latency: {stats['mean_latency_ms']:.2f}ms")
print(f"P95 latency: {stats['p95_latency_ms']:.2f}ms")
```

### Model Training

Model training automatically logs:
- Training start and completion
- Cross-validation metrics
- Model performance
- Training duration

```python
from src.modeling.model_trainer import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_all_models(X_train, y_train, cv_splits=3)

# Results include metrics and timing
for model_name, metrics in results.items():
    print(f"{model_name}: R²={metrics['r2']:.4f}")
```

## Log File Management

### Log Rotation

Logs are automatically rotated when they reach 10MB:
- Main log file: `logs/system.log`
- Backup files: `logs/system.log.1`, `logs/system.log.2`, etc.
- Maximum backups: 5 files

### Accessing Logs

```bash
# View recent logs
tail -f logs/system.log

# Search for errors
grep ERROR logs/system.log

# Search for specific operation
grep "ETL pipeline" logs/system.log
```

## Performance Thresholds

### System Health Thresholds (Default)

- CPU: 80%
- Memory: 85%
- Disk: 90%

### Streaming Latency Thresholds

- Target: < 1 second per event
- Warning: > 1 second
- Critical: > 5 seconds

### ETL Processing Thresholds

- Target: < 5 minutes per day
- Warning: > 5 minutes

## Best Practices

1. **Regular Monitoring**
   - Check the monitoring dashboard regularly
   - Review logs for errors and warnings
   - Generate reports periodically

2. **Alert Response**
   - Investigate system warnings promptly
   - Check logs for root causes
   - Monitor resource usage trends

3. **Performance Optimization**
   - Use execution time statistics to identify bottlenecks
   - Monitor latency percentiles (P95, P99)
   - Track error rates and investigate failures

4. **Reproducibility**
   - Log all major operations with timestamps
   - Save monitoring reports for comparison
   - Track system configuration changes

5. **Troubleshooting**
   - Check logs for error messages and stack traces
   - Review system health metrics
   - Compare current metrics with historical data
   - Generate detailed reports for analysis

## Troubleshooting

### High CPU Usage

1. Check which operations are consuming CPU
2. Review execution time statistics
3. Check for long-running operations
4. Monitor system processes

### High Memory Usage

1. Check memory metrics in dashboard
2. Review event processing metrics
3. Check for memory leaks in logs
4. Monitor buffer sizes

### High Latency

1. Check event processing latency statistics
2. Review execution times for each stage
3. Check system resource usage
4. Monitor network connectivity

### Missing Logs

1. Verify logs directory exists: `logs/`
2. Check file permissions
3. Verify logging is initialized
4. Check log level configuration

## Examples

### Example 1: Monitor ETL Pipeline

```python
from src.etl_pipeline.pipeline import ETLPipeline
from src.utils.monitoring import get_performance_monitor

pipeline = ETLPipeline(bronze_path, silver_path, gold_path)
monitor = get_performance_monitor()

# Run pipeline
gold_df, metrics = pipeline.run_pipeline(bronze_df, source='kaggle')

# Get performance report
report = monitor.get_performance_report()
print(f"ETL time: {metrics['total_time_seconds']:.2f}s")
print(f"Valid records: {metrics['records']['valid_records']}")
print(f"Quality score: {metrics['quality']['quality_score']:.2f}%")

# Save report
monitor.save_report('etl_execution')
```

### Example 2: Monitor Streaming Pipeline

```python
from src.streaming.streaming_inference_pipeline import StreamingInferencePipeline
from src.utils.monitoring import get_performance_monitor

pipeline = StreamingInferencePipeline(model)
monitor = get_performance_monitor()

# Process events
for event in events:
    result = pipeline.process_event(event)
    monitor.event_collector.record_event(
        result['latency_ms'],
        success=result['features_computed']
    )

# Get metrics
stats = pipeline.get_latency_stats()
metrics = monitor.event_collector.get_metrics()

print(f"Events processed: {metrics['event_count']}")
print(f"Mean latency: {metrics['mean_latency_ms']:.2f}ms")
print(f"P95 latency: {metrics['p95_latency_ms']:.2f}ms")
print(f"Error rate: {metrics['error_rate'] * 100:.2f}%")
```

### Example 3: Generate Monitoring Report

```bash
# Generate summary report and print to console
python scripts/generate_monitoring_report.py --report-type summary --print-summary

# Generate full report
python scripts/generate_monitoring_report.py --report-type full --output-dir reports

# Generate system report
python scripts/generate_monitoring_report.py --report-type system
```

## References

- Python logging: https://docs.python.org/3/library/logging.html
- Streamlit: https://streamlit.io/
- psutil: https://psutil.readthedocs.io/

