# Configuration Guide

## Overview

The Air Quality Prediction System uses a hierarchical configuration system that combines YAML configuration files with environment variable overrides. This guide explains how to configure the system for different deployment scenarios.

## Configuration Hierarchy

Configuration is loaded in the following order (later values override earlier ones):

1. **Default values** in code
2. **config.yaml** file
3. **Environment variables** (highest priority)

This allows for flexible deployment across development, staging, and production environments.

## Configuration File (config.yaml)

### Location

The `config.yaml` file should be placed in the project root directory.

### Basic Structure

```yaml
system:
  environment: development
  log_level: INFO
  random_seed: 42

data_sources:
  kaggle:
    dataset_name: "asaniczka/real-time-air-quality-index-aqi-india-2023-2025"
    api_key_path: ~/.kaggle/kaggle.json
  iqair:
    api_key: ${IQAIR_API_KEY}
    base_url: "https://api.waqi.info"

cities:
  - Delhi
  - Mumbai
  - Bangalore
  - Kolkata
  - Chennai
  - Hyderabad
  - Pune
  - Ahmedabad
  - Jaipur
  - Lucknow

etl:
  batch_size: 1000
  processing_interval_hours: 1
  max_retries: 3
  retry_delay_seconds: 30

modeling:
  xgboost:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
  random_forest:
    n_estimators: 100
    max_depth: 15
    min_samples_split: 5
  cv_folds: 3
  test_size: 0.2

streaming:
  kafka_bootstrap_servers: "localhost:9092"
  topic: "aqi-events"
  consumer_group: "aqi-consumer"
  max_latency_ms: 1000

alerts:
  dedup_window_hours: 1
  prediction_threshold: 150

dashboard:
  refresh_interval_minutes: 5
  port: 8501
  host: "0.0.0.0"

storage:
  bronze_path: "data/bronze"
  silver_path: "data/silver"
  gold_path: "data/gold"
  models_path: "data/models"
  logs_path: "logs"
```

## Configuration Sections

### System Section

Controls overall system behavior and logging.

```yaml
system:
  environment: development  # development, staging, production
  log_level: INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL
  random_seed: 42          # Fixed seed for reproducibility
```

**Parameters**:
- `environment`: Affects resource allocation and logging verbosity
- `log_level`: Controls logging output verbosity
- `random_seed`: Ensures reproducible results across runs

**Recommended Values**:
- Development: `log_level: DEBUG`, `environment: development`
- Production: `log_level: INFO`, `environment: production`

### Data Sources Section

Configures connections to external data sources.

```yaml
data_sources:
  kaggle:
    dataset_name: "asaniczka/real-time-air-quality-index-aqi-india-2023-2025"
    api_key_path: ~/.kaggle/kaggle.json
  iqair:
    api_key: ${IQAIR_API_KEY}
    base_url: "https://api.waqi.info"
```

**Kaggle Configuration**:
- `dataset_name`: Kaggle dataset identifier (do not change)
- `api_key_path`: Path to Kaggle credentials file

**IQAir Configuration**:
- `api_key`: Use environment variable reference `${IQAIR_API_KEY}`
- `base_url`: IQAir API endpoint (do not change)

### Cities Section

Lists cities to monitor.

```yaml
cities:
  - Delhi
  - Mumbai
  - Bangalore
  - Kolkata
  - Chennai
  - Hyderabad
  - Pune
  - Ahmedabad
  - Jaipur
  - Lucknow
```

**Supported Cities**:
- Minimum: 1 city
- Recommended: 10+ cities
- Maximum: Limited by API rate limits

**Adding Cities**:
```yaml
cities:
  - Delhi
  - Mumbai
  - Bangalore
  - Kolkata
  - Chennai
  - Hyderabad
  - Pune
  - Ahmedabad
  - Jaipur
  - Lucknow
  - Surat          # Add new city
  - Chandigarh     # Add new city
```

### ETL Section

Configures data processing pipeline.

```yaml
etl:
  batch_size: 1000
  processing_interval_hours: 1
  max_retries: 3
  retry_delay_seconds: 30
```

**Parameters**:
- `batch_size`: Records per batch (increase for better performance, decrease for lower memory)
- `processing_interval_hours`: Frequency of ETL runs (1 = hourly, 24 = daily)
- `max_retries`: Retry attempts for failed operations (3-5 recommended)
- `retry_delay_seconds`: Initial delay for exponential backoff

**Tuning Guide**:
- **High throughput**: `batch_size: 5000`, `processing_interval_hours: 2`
- **Low latency**: `batch_size: 500`, `processing_interval_hours: 0.5`
- **Balanced**: `batch_size: 1000`, `processing_interval_hours: 1`

### Modeling Section

Configures machine learning models.

```yaml
modeling:
  xgboost:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
  random_forest:
    n_estimators: 100
    max_depth: 15
    min_samples_split: 5
  cv_folds: 3
  test_size: 0.2
```

**XGBoost Parameters**:
- `max_depth`: Tree depth (4-8 typical, higher = more complex)
- `learning_rate`: Step size (0.01-0.3, lower = slower but more stable)
- `n_estimators`: Number of trees (50-200 typical)

**Random Forest Parameters**:
- `n_estimators`: Number of trees (50-200 typical)
- `max_depth`: Tree depth (10-20 typical)
- `min_samples_split`: Minimum samples to split (2-10 typical)

**Cross-Validation**:
- `cv_folds`: Number of folds (3-5 typical)
- `test_size`: Test set fraction (0.1-0.3 typical)

**Tuning Guide**:
- **Fast training**: `n_estimators: 50`, `cv_folds: 2`
- **Accurate**: `n_estimators: 200`, `cv_folds: 5`
- **Balanced**: `n_estimators: 100`, `cv_folds: 3`

### Streaming Section

Configures real-time data streaming.

```yaml
streaming:
  kafka_bootstrap_servers: "localhost:9092"
  topic: "aqi-events"
  consumer_group: "aqi-consumer"
  max_latency_ms: 1000
```

**Parameters**:
- `kafka_bootstrap_servers`: Kafka broker addresses (comma-separated)
- `topic`: Kafka topic name
- `consumer_group`: Consumer group identifier
- `max_latency_ms`: Maximum acceptable latency per event

**For Docker Compose**:
```yaml
streaming:
  kafka_bootstrap_servers: "kafka:9092"
  topic: "aqi-events"
  consumer_group: "aqi-consumer"
  max_latency_ms: 1000
```

### Alerts Section

Configures alert generation.

```yaml
alerts:
  dedup_window_hours: 1
  prediction_threshold: 150
```

**Parameters**:
- `dedup_window_hours`: Time window for deduplication (1-24 hours)
- `prediction_threshold`: AQI threshold for model-based alerts (100-200 typical)

**Alert Levels**:
- Good: AQI ≤ 50
- Satisfactory: 51 ≤ AQI ≤ 100
- Moderately Polluted: 101 ≤ AQI ≤ 200
- Heavily Polluted: 201 ≤ AQI ≤ 300
- Severely Polluted: AQI > 300

### Dashboard Section

Configures Streamlit dashboard.

```yaml
dashboard:
  refresh_interval_minutes: 5
  port: 8501
  host: "0.0.0.0"
```

**Parameters**:
- `refresh_interval_minutes`: Data refresh frequency (1-60 minutes)
- `port`: Port number (1024-65535)
- `host`: Binding address (0.0.0.0 = all interfaces, 127.0.0.1 = localhost)

**For Production**:
```yaml
dashboard:
  refresh_interval_minutes: 5
  port: 8501
  host: "0.0.0.0"  # Behind reverse proxy
```

### Storage Section

Configures data storage paths.

```yaml
storage:
  bronze_path: "data/bronze"
  silver_path: "data/silver"
  gold_path: "data/gold"
  models_path: "data/models"
  logs_path: "logs"
```

**Parameters**:
- `bronze_path`: Raw data storage
- `silver_path`: Cleaned data storage
- `gold_path`: Feature-engineered data storage
- `models_path`: Trained models storage
- `logs_path`: System logs storage

**For Production**:
```yaml
storage:
  bronze_path: "/mnt/data/bronze"
  silver_path: "/mnt/data/silver"
  gold_path: "/mnt/data/gold"
  models_path: "/mnt/models"
  logs_path: "/var/log/aqi"
```

## Environment Variables

Environment variables override configuration file values and are essential for sensitive data.

### Variable Naming Convention

Environment variables follow the pattern: `AQI_SECTION_KEY`

For nested values: `AQI_SECTION_SUBSECTION_KEY`

Examples:
- `AQI_SYSTEM_LOG_LEVEL` → `system.log_level`
- `AQI_STORAGE_BRONZE_PATH` → `storage.bronze_path`
- `AQI_MODELING_XGBOOST_MAX_DEPTH` → `modeling.xgboost.max_depth`

### Setting Environment Variables

#### Linux/macOS

```bash
# Set individual variable
export IQAIR_API_KEY="your_api_key"

# Or create .env file
cat > .env << EOF
IQAIR_API_KEY=your_api_key
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
EOF

# Load .env file
source .env
```

#### Windows (PowerShell)

```powershell
$env:IQAIR_API_KEY = "your_api_key"
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY = "your_key"
```

#### Windows (Command Prompt)

```cmd
set IQAIR_API_KEY=your_api_key
set KAGGLE_USERNAME=your_username
set KAGGLE_KEY=your_key
```

#### Docker

```bash
docker run -e IQAIR_API_KEY=your_key -e KAGGLE_USERNAME=user aqi-app
```

#### Docker Compose

```yaml
services:
  app:
    environment:
      - IQAIR_API_KEY=${IQAIR_API_KEY}
      - KAGGLE_USERNAME=${KAGGLE_USERNAME}
      - KAGGLE_KEY=${KAGGLE_KEY}
```

## Configuration Examples

### Development Configuration

```yaml
system:
  environment: development
  log_level: DEBUG
  random_seed: 42

etl:
  batch_size: 500
  processing_interval_hours: 1
  max_retries: 2
  retry_delay_seconds: 10

modeling:
  xgboost:
    n_estimators: 50
  random_forest:
    n_estimators: 50
  cv_folds: 2

dashboard:
  refresh_interval_minutes: 1
```

### Production Configuration

```yaml
system:
  environment: production
  log_level: INFO
  random_seed: 42

etl:
  batch_size: 5000
  processing_interval_hours: 1
  max_retries: 5
  retry_delay_seconds: 60

modeling:
  xgboost:
    n_estimators: 200
  random_forest:
    n_estimators: 200
  cv_folds: 5

dashboard:
  refresh_interval_minutes: 5
  host: "0.0.0.0"
```

### High-Performance Configuration

```yaml
etl:
  batch_size: 10000
  processing_interval_hours: 2

modeling:
  xgboost:
    max_depth: 8
    n_estimators: 200
  random_forest:
    max_depth: 20
    n_estimators: 200
  cv_folds: 5

streaming:
  max_latency_ms: 2000
```

### Low-Resource Configuration

```yaml
etl:
  batch_size: 100
  processing_interval_hours: 4

modeling:
  xgboost:
    max_depth: 4
    n_estimators: 50
  random_forest:
    max_depth: 10
    n_estimators: 50
  cv_folds: 2

storage:
  bronze_path: "/tmp/bronze"
  silver_path: "/tmp/silver"
  gold_path: "/tmp/gold"
```

## Configuration Validation

The system validates configuration on startup. Common validation errors:

### Error: "Configuration section not found"

**Cause**: Missing required section in config.yaml

**Solution**: Ensure all required sections are present in config.yaml

### Error: "Invalid log_level"

**Cause**: Invalid log level value

**Solution**: Use one of: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Error: "Cities list is empty"

**Cause**: No cities configured

**Solution**: Add at least one city to the cities list

### Error: "Missing storage path"

**Cause**: Storage path not configured

**Solution**: Ensure all storage paths are defined

## Configuration Best Practices

1. **Use environment variables for sensitive data**: API keys, credentials
2. **Keep config.yaml in version control**: But not .env file
3. **Use meaningful values**: Don't use placeholder values in production
4. **Document custom configurations**: Add comments explaining non-standard settings
5. **Test configuration changes**: Validate before deploying to production
6. **Use consistent naming**: Follow the AQI_SECTION_KEY convention
7. **Monitor configuration changes**: Track who changed what and when
8. **Backup configurations**: Keep backups of working configurations

## Troubleshooting Configuration Issues

### Configuration not loading

```bash
# Check if config.yaml exists
ls -la config.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check for environment variable conflicts
env | grep AQI_
```

### Environment variables not working

```bash
# Verify environment variable is set
echo $IQAIR_API_KEY

# Check variable name format
# Should be: AQI_SECTION_KEY

# Reload environment
source .env
```

### Configuration values not applied

```bash
# Check configuration loading in logs
tail -f logs/system.log | grep -i config

# Verify override priority
# Environment variables > config.yaml > defaults
```

## Advanced Configuration

### Custom Storage Paths

```yaml
storage:
  bronze_path: "/mnt/nfs/bronze"
  silver_path: "/mnt/nfs/silver"
  gold_path: "/mnt/nfs/gold"
  models_path: "/mnt/models"
  logs_path: "/var/log/aqi"
```

### Multiple Kafka Brokers

```yaml
streaming:
  kafka_bootstrap_servers: "kafka1:9092,kafka2:9092,kafka3:9092"
```

### Custom Model Hyperparameters

```yaml
modeling:
  xgboost:
    max_depth: 8
    learning_rate: 0.05
    n_estimators: 150
    subsample: 0.8
    colsample_bytree: 0.8
  random_forest:
    n_estimators: 150
    max_depth: 20
    min_samples_split: 3
    min_samples_leaf: 1
```

## Configuration Reload

To reload configuration without restarting:

```python
from src.utils.config_loader import load_config

config = load_config('config.yaml')
config.reload()
```

## Support

For configuration issues:
- Check logs in `logs/system.log`
- Review this guide
- Verify environment variables are set
- Validate YAML syntax
- Contact system administrator
