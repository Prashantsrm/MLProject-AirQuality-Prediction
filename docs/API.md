# API Documentation

Air Quality Prediction System - Comprehensive API Reference

## Table of Contents

1. [Data Ingestion](#data-ingestion)
2. [ETL Pipeline](#etl-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Machine Learning Models](#machine-learning-models)
5. [Streaming Pipeline](#streaming-pipeline)
6. [Alert System](#alert-system)
7. [Dashboard](#dashboard)
8. [Utilities](#utilities)

---

## Data Ingestion

### KaggleDataIngestion

Handles data ingestion from Kaggle dataset with authentication and retry logic.

```python
from src.data_ingestion.kaggle_ingestion import KaggleDataIngestion

# Initialize
ingestion = KaggleDataIngestion(
    api_key_path='~/.kaggle/kaggle.json',
    dataset_name='rohanrao/air-quality-data-in-india'
)

# Fetch data
df = ingestion.fetch_data(output_path='data/raw')
```

#### Methods

**`__init__(api_key_path: str, dataset_name: str)`**
- Initializes Kaggle API client with authentication
- **Parameters:**
  - `api_key_path`: Path to Kaggle API credentials file
  - `dataset_name`: Kaggle dataset identifier
- **Raises:** `KaggleConnectionError` if authentication fails

**`fetch_data(output_path: str) -> pd.DataFrame`**
- Downloads and loads historical AQI data from Kaggle
- **Parameters:**
  - `output_path`: Local directory to store downloaded files
- **Returns:** DataFrame with columns: city, timestamp, aqi, pm25, pm10, no2, o3, so2, co
- **Raises:** `KaggleConnectionError` if download fails after 3 retries

### IQAirDataIngestion

Handles real-time data ingestion from IQAir API with retry logic.

```python
from src.data_ingestion.iqair_ingestion import IQAirDataIngestion

# Initialize
ingestion = IQAirDataIngestion(
    api_key='your_iqair_api_key',
    cities=['Delhi', 'Mumbai', 'Bangalore']
)

# Fetch current AQI for a city
aqi_data = ingestion.fetch_current_aqi('Delhi')

# Fetch all cities
all_data = ingestion.fetch_all_cities_aqi()
```

#### Methods

**`__init__(api_key: str, cities: Optional[List[str]] = None)`**
- Initializes IQAir API client
- **Parameters:**
  - `api_key`: IQAir API authentication key
  - `cities`: List of cities to monitor (default: 10 major Indian cities)
- **Raises:** `IQAirAPIError` if API key is invalid

**`fetch_current_aqi(city: str) -> Dict`**
- Fetches current AQI for a specific city
- **Parameters:**
  - `city`: City name
- **Returns:** Dictionary with keys: city, timestamp, aqi, pm25, pm10, no2, o3, so2, co
- **Raises:** `IQAirAPIError` if API call fails after 3 retries

**`fetch_all_cities_aqi() -> pd.DataFrame`**
- Fetches current AQI for all configured cities
- **Returns:** DataFrame with current AQI data for all cities
- **Raises:** `IQAirAPIError` if any API call fails

---

## ETL Pipeline

### ETLPipeline

Orchestrates complete ETL pipeline: Bronze → Silver → Gold transformations.

```python
from src.etl_pipeline.pipeline import ETLPipeline
from pyspark.sql import SparkSession

# Initialize
spark = SparkSession.builder.appName("AQI-ETL").getOrCreate()
pipeline = ETLPipeline(
    bronze_path='data/bronze',
    silver_path='data/silver',
    gold_path='data/gold',
    spark=spark
)

# Run pipeline
gold_df, metrics = pipeline.run_pipeline(bronze_df, source='kaggle')

# Get performance metrics
metrics = pipeline.get_performance_metrics()
```

#### Methods

**`__init__(bronze_path: str, silver_path: str, gold_path: str, spark: Optional[SparkSession] = None)`**
- Initializes ETL pipeline with layer paths
- **Parameters:**
  - `bronze_path`: Path to Bronze Layer storage
  - `silver_path`: Path to Silver Layer storage
  - `gold_path`: Path to Gold Layer storage
  - `spark`: Optional SparkSession (creates new if None)
- **Raises:** `ETLPipelineError` if initialization fails

**`run_pipeline(bronze_df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, Dict]`**
- Executes complete ETL pipeline
- **Parameters:**
  - `bronze_df`: Raw data from Bronze Layer
  - `source`: Data source identifier ('kaggle' or 'iqair')
- **Returns:** Tuple of (gold_layer_dataframe, performance_metrics)
- **Raises:** `ETLPipelineError` if pipeline execution fails

**`get_performance_metrics() -> Dict`**
- Returns performance metrics from last pipeline run
- **Returns:** Dictionary with timing, record counts, and quality metrics

### BronzeLayer

Stores raw, unmodified data with metadata.

```python
from src.etl_pipeline.bronze_layer import BronzeLayer

bronze = BronzeLayer('data/bronze', spark)
records_stored = bronze.store_data(df, source='kaggle')
```

#### Methods

**`store_data(df: pd.DataFrame, source: str) -> int`**
- Stores raw data in Bronze Layer with metadata
- **Parameters:**
  - `df`: Raw data DataFrame
  - `source`: Data source identifier
- **Returns:** Number of records stored
- **Raises:** `BronzeLayerError` if storage fails

### SilverLayer

Cleans, validates, and deduplicates data.

```python
from src.etl_pipeline.silver_layer import SilverLayer

silver = SilverLayer('data/silver', spark)
cleaned_df, total, valid, rejected = silver.transform_bronze_to_silver(bronze_df)
```

#### Methods

**`transform_bronze_to_silver(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, int]`**
- Transforms Bronze to Silver with validation
- **Parameters:**
  - `df`: Raw data from Bronze Layer
- **Returns:** Tuple of (cleaned_dataframe, total_records, valid_records, rejected_records)
- **Validation checks:**
  - Removes duplicates on (city, timestamp, pollutant_type)
  - Validates AQI ∈ [0, 500]
  - Validates chronological timestamp ordering per city
  - Removes records with missing critical fields

### GoldLayer

Performs feature engineering for modeling.

```python
from src.etl_pipeline.gold_layer import GoldLayer

gold = GoldLayer('data/gold', spark)
features_df = gold.transform_silver_to_gold(silver_df)
```

#### Methods

**`transform_silver_to_gold(df: pd.DataFrame) -> pd.DataFrame`**
- Transforms Silver to Gold with feature engineering
- **Parameters:**
  - `df`: Cleaned data from Silver Layer
- **Returns:** DataFrame with engineered features
- **Features computed:**
  - Lag features: aqi_lag_1h, aqi_lag_3h, aqi_lag_6h, aqi_lag_12h, aqi_lag_24h
  - Rolling statistics: mean, std, min, max over 3h, 6h, 12h, 24h windows
  - Temporal: hour_of_day, day_of_week, month, is_weekend
  - Seasonal: season (Winter, Summer, Monsoon, Post-Monsoon)

### DataQualityValidator

Validates data quality and generates reports.

```python
from src.etl_pipeline.data_validator import DataQualityValidator

validator = DataQualityValidator()
quality_report = validator.validate_data(df)
```

#### Methods

**`validate_data(df: pd.DataFrame) -> Dict`**
- Validates data quality
- **Parameters:**
  - `df`: DataFrame to validate
- **Returns:** Dictionary with validation results and alerts

**`generate_quality_report(df: pd.DataFrame) -> Dict`**
- Generates comprehensive quality report
- **Returns:** Dictionary with quality metrics and statistics

---

## Feature Engineering

### FeatureProcessor

Orchestrates all feature engineering transformations.

```python
from src.feature_engineering.feature_processor import FeatureProcessor

processor = FeatureProcessor(
    lag_offsets=[1, 3, 6, 12, 24],
    rolling_windows=[3, 6, 12, 24]
)
features_df = processor.process(df)
```

#### Methods

**`__init__(lag_offsets: List[int], rolling_windows: List[int])`**
- Initializes feature processor with configuration
- **Parameters:**
  - `lag_offsets`: Lag offsets in hours
  - `rolling_windows`: Rolling window sizes in hours

**`process(df: pd.DataFrame) -> pd.DataFrame`**
- Processes features for pandas DataFrame
- **Parameters:**
  - `df`: Input DataFrame with city, timestamp, aqi columns
- **Returns:** DataFrame with all engineered features

**`process_spark(sdf: DataFrame) -> DataFrame`**
- Processes features for Spark DataFrame (distributed)
- **Parameters:**
  - `sdf`: Input Spark DataFrame
- **Returns:** Spark DataFrame with engineered features

### TimeSeriesSplitter

Splits data for time-series cross-validation without data leakage.

```python
from src.feature_engineering.time_series_splitter import TimeSeriesSplitter

splitter = TimeSeriesSplitter(n_splits=3)
for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

#### Methods

**`split(X: pd.DataFrame, y: pd.Series) -> Iterator`**
- Generates time-series cross-validation splits
- **Parameters:**
  - `X`: Feature DataFrame
  - `y`: Target Series
- **Yields:** Tuples of (train_indices, test_indices)

**`get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple`**
- Creates single train-test split respecting temporal ordering
- **Returns:** Tuple of (X_train, X_test, y_train, y_test)

### FeatureAnalyzer

Analyzes feature statistics and correlations.

```python
from src.feature_engineering.feature_analyzer import FeatureAnalyzer

analyzer = FeatureAnalyzer(df, target_col='aqi')
stats = analyzer.compute_feature_statistics()
corr_matrix = analyzer.compute_correlation_matrix()
```

#### Methods

**`compute_feature_statistics() -> pd.DataFrame`**
- Computes statistics for all features
- **Returns:** DataFrame with mean, std, min, max, median for each feature

**`compute_correlation_matrix() -> pd.DataFrame`**
- Computes correlation between features and target
- **Returns:** Correlation matrix

---

## Machine Learning Models

### ModelTrainer

Trains multiple forecasting models with cross-validation.

```python
from src.modeling.model_trainer import ModelTrainer

trainer = ModelTrainer(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

xgb_model = trainer.train_xgboost()
rf_model = trainer.train_random_forest()
```

#### Methods

**`train_xgboost() -> XGBoostModel`**
- Trains XGBoost model with time-series cross-validation
- **Returns:** Trained XGBoostModel instance
- **Hyperparameters:** max_depth=6, learning_rate=0.1, n_estimators=100

**`train_random_forest() -> RandomForestModel`**
- Trains Random Forest model with time-series cross-validation
- **Returns:** Trained RandomForestModel instance
- **Hyperparameters:** n_estimators=100, max_depth=15, min_samples_split=5

**`train_all_models() -> Dict[str, object]`**
- Trains all available models
- **Returns:** Dictionary mapping model names to trained models

**`evaluate_all_models() -> Dict[str, Dict]`**
- Evaluates all trained models
- **Returns:** Dictionary with evaluation metrics for each model

**`get_best_model() -> Tuple[str, object]`**
- Returns best performing model based on R² score
- **Returns:** Tuple of (model_name, model_instance)

### XGBoostModel

XGBoost forecasting model for AQI prediction.

```python
from src.modeling.xgboost_model import XGBoostModel

model = XGBoostModel(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

#### Methods

**`train(X_train: pd.DataFrame, y_train: pd.Series) -> Dict`**
- Trains model with time-series cross-validation
- **Returns:** Dictionary with CV metrics (RMSE, MAE, R²)

**`predict(X: pd.DataFrame) -> np.ndarray`**
- Generates predictions
- **Returns:** Array of predicted AQI values

### RandomForestModel

Random Forest forecasting model for AQI prediction.

```python
from src.modeling.random_forest_model import RandomForestModel

model = RandomForestModel(
    n_estimators=100,
    max_depth=15
)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

#### Methods

**`train(X_train: pd.DataFrame, y_train: pd.Series) -> Dict`**
- Trains model with time-series cross-validation
- **Returns:** Dictionary with CV metrics

**`predict(X: pd.DataFrame) -> np.ndarray`**
- Generates predictions
- **Returns:** Array of predicted AQI values

### ModelEvaluator

Evaluates model performance with multiple metrics.

```python
from src.modeling.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_true, y_pred)
residuals = evaluator.analyze_residuals()
```

#### Methods

**`evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict`**
- Evaluates model predictions
- **Returns:** Dictionary with RMSE, MAE, R² metrics

**`analyze_residuals() -> Dict[str, float]`**
- Analyzes prediction residuals
- **Returns:** Dictionary with residual statistics

### ModelRegistry

Manages model versioning and persistence.

```python
from src.modeling.model_registry import ModelRegistry

registry = ModelRegistry('data/models')
registry.register_model(model, 'xgboost_v1', metrics)
loaded_model = registry.get_model('xgboost_v1')
```

#### Methods

**`register_model(model: object, model_id: str, metrics: Dict) -> None`**
- Registers and serializes a trained model
- **Parameters:**
  - `model`: Trained model instance
  - `model_id`: Unique model identifier
  - `metrics`: Model performance metrics

**`get_model(model_id: str) -> object`**
- Retrieves a registered model
- **Parameters:**
  - `model_id`: Model identifier
- **Returns:** Deserialized model instance

---

## Streaming Pipeline

### StreamingDataProducer

Produces streaming events to Kafka topics.

```python
from src.streaming.streaming_data_producer import StreamingDataProducer

producer = StreamingDataProducer(
    bootstrap_servers='localhost:9092',
    cities=['Delhi', 'Mumbai']
)
producer.send_event({'city': 'Delhi', 'aqi': 150, 'timestamp': '2024-01-01T12:00:00'})
```

#### Methods

**`send_event(event: Dict, city: Optional[str] = None) -> None`**
- Sends event to appropriate Kafka topic
- **Parameters:**
  - `event`: Event dictionary with city, aqi, timestamp
  - `city`: Optional city override

### StreamingDataConsumer

Consumes streaming events from Kafka topics.

```python
from src.streaming.streaming_data_consumer import StreamingDataConsumer

consumer = StreamingDataConsumer(
    bootstrap_servers='localhost:9092',
    group_id='aqi-consumer-group'
)
consumer.subscribe(['delhi', 'mumbai'])
for event in consumer.consume():
    print(event)
```

#### Methods

**`subscribe(topics: List[str]) -> None`**
- Subscribes to Kafka topics
- **Parameters:**
  - `topics`: List of topic names

**`consume(timeout_ms: int = 1000) -> Iterator[Dict]`**
- Consumes events from subscribed topics
- **Yields:** Event dictionaries

### StreamingFeatureComputer

Computes features in real-time for streaming events.

```python
from src.streaming.streaming_feature_computer import StreamingFeatureComputer

computer = StreamingFeatureComputer(window_size_hours=24)
features = computer.compute_features(event)
```

#### Methods

**`compute_features(event: Dict) -> Dict`**
- Computes features for a streaming event
- **Parameters:**
  - `event`: Event with city, aqi, timestamp
- **Returns:** Dictionary with computed features

### StreamingInferencePipeline

End-to-end streaming inference pipeline.

```python
from src.streaming.streaming_inference_pipeline import StreamingInferencePipeline

pipeline = StreamingInferencePipeline(
    model_path='data/models/xgboost_v1',
    feature_columns=['aqi_lag_1h', 'aqi_lag_3h', ...]
)
result = pipeline.process_event(event)
```

#### Methods

**`process_event(event: Dict) -> Dict`**
- Processes event through complete inference pipeline
- **Parameters:**
  - `event`: Streaming event
- **Returns:** Dictionary with prediction and latency metrics
- **Latency target:** < 1 second per event

---

## Alert System

### RuleBasedAlertSystem

Generates alerts based on AQI thresholds.

```python
from src.streaming.rule_based_alert_system import RuleBasedAlertSystem

alert_system = RuleBasedAlertSystem()
alert = alert_system.evaluate_current_aqi(city='Delhi', aqi=250)
```

#### Methods

**`evaluate_current_aqi(city: str, aqi: float, timestamp: Optional[str] = None) -> Optional[Dict]`**
- Evaluates current AQI and generates alert if needed
- **Parameters:**
  - `city`: City name
  - `aqi`: Current AQI value
  - `timestamp`: Optional timestamp
- **Returns:** Alert dictionary or None if no alert
- **Alert levels:**
  - Good: AQI ≤ 50
  - Satisfactory: 51 ≤ AQI ≤ 100
  - Moderately Polluted: 101 ≤ AQI ≤ 200
  - Heavily Polluted: 201 ≤ AQI ≤ 300
  - Severely Polluted: AQI > 300

### ModelBasedAlertSystem

Generates alerts based on predicted AQI.

```python
from src.streaming.model_based_alert_system import ModelBasedAlertSystem

alert_system = ModelBasedAlertSystem(threshold=150)
alert = alert_system.evaluate_prediction(city='Delhi', predicted_aqi=180)
```

#### Methods

**`evaluate_prediction(city: str, predicted_aqi: float, timestamp: Optional[str] = None) -> Optional[Dict]`**
- Evaluates predicted AQI and generates alert if needed
- **Parameters:**
  - `city`: City name
  - `predicted_aqi`: Predicted AQI value
  - `timestamp`: Optional timestamp
- **Returns:** Alert dictionary or None if no alert
- **Default threshold:** 150 (Moderately Polluted)

### AlertDeduplicator

Prevents duplicate alerts within time window.

```python
from src.streaming.alert_deduplicator import AlertDeduplicator

deduplicator = AlertDeduplicator(dedup_window_hours=1)
should_send = deduplicator.should_send_alert(city='Delhi', alert_level='warning')
```

#### Methods

**`should_send_alert(city: str, alert_level: str) -> bool`**
- Checks if alert should be sent (not duplicate)
- **Parameters:**
  - `city`: City name
  - `alert_level`: Alert level
- **Returns:** True if alert should be sent, False if duplicate

### AlertStore

Persists alerts to SQLite database.

```python
from src.streaming.alert_store import AlertStore

store = AlertStore('data/alerts.db')
alert_id = store.store_alert(alert_dict)
```

#### Methods

**`store_alert(alert: Dict) -> str`**
- Stores alert in database
- **Parameters:**
  - `alert`: Alert dictionary
- **Returns:** Alert ID

**`get_active_alerts(city: Optional[str] = None) -> List[Dict]`**
- Retrieves active alerts
- **Parameters:**
  - `city`: Optional city filter
- **Returns:** List of active alerts

### AlertService

Orchestrates rule-based and model-based alerts.

```python
from src.streaming.alert_service import AlertService

service = AlertService()
service.process_current_aqi(city='Delhi', aqi=250)
service.process_prediction(city='Delhi', predicted_aqi=180)
```

#### Methods

**`process_current_aqi(city: str, aqi: float) -> None`**
- Processes current AQI and generates alerts
- **Parameters:**
  - `city`: City name
  - `aqi`: Current AQI value

**`process_prediction(city: str, predicted_aqi: float) -> None`**
- Processes predicted AQI and generates alerts
- **Parameters:**
  - `city`: City name
  - `predicted_aqi`: Predicted AQI value

---

## Dashboard

### DataStore

Provides data access interface for dashboard.

```python
from src.dashboard.data_store import DataStore

store = DataStore(
    gold_layer_path='data/gold',
    alerts_db_path='data/alerts.db'
)
current_aqi = store.get_latest_aqi('Delhi')
forecast = store.get_forecast('Delhi', hours=24)
```

#### Methods

**`get_latest_aqi(city: str) -> Optional[Dict]`**
- Gets latest AQI for a city
- **Parameters:**
  - `city`: City name
- **Returns:** Dictionary with city, aqi, timestamp, or None

**`get_forecast(city: str, hours: int = 24) -> Optional[pd.DataFrame]`**
- Gets AQI forecast for a city
- **Parameters:**
  - `city`: City name
  - `hours`: Number of hours to forecast
- **Returns:** DataFrame with timestamp and predicted_aqi columns

**`get_cities() -> List[str]`**
- Gets list of monitored cities
- **Returns:** List of city names

**`get_active_alerts() -> List[Dict]`**
- Gets active pollution alerts
- **Returns:** List of alert dictionaries

---

## Utilities

### ConfigLoader

Loads and manages system configuration.

```python
from src.utils.config_loader import ConfigLoader

config = ConfigLoader('config.yaml')
cities = config.get('cities')
api_key = config.get('iqair_api_key')
```

#### Methods

**`get(key: str, default: Any = None) -> Any`**
- Gets configuration value
- **Parameters:**
  - `key`: Configuration key (supports dot notation: 'section.key')
  - `default`: Default value if key not found
- **Returns:** Configuration value

**`get_all() -> Dict`**
- Gets entire configuration
- **Returns:** Configuration dictionary

### LoggerConfig

Configures structured logging for the system.

```python
from src.utils.logger import LoggerConfig, get_logger

LoggerConfig.setup_logging(log_file='logs/system.log')
logger = get_logger('my_module')
logger.info('Processing started')
```

#### Methods

**`setup_logging(log_file: str = 'logs/system.log', level: str = 'INFO') -> None`**
- Configures logging with file rotation
- **Parameters:**
  - `log_file`: Path to log file
  - `level`: Logging level (DEBUG, INFO, WARNING, ERROR)

**`get_logger(module_name: str = 'aqi_system') -> logging.Logger`**
- Gets logger instance for a module
- **Parameters:**
  - `module_name`: Module name for logger
- **Returns:** Configured logger instance

### SystemHealthMonitor

Monitors system health metrics.

```python
from src.utils.monitoring import SystemHealthMonitor

monitor = SystemHealthMonitor()
metrics = monitor.collect_metrics()
avg_metrics = monitor.get_average_metrics()
```

#### Methods

**`collect_metrics() -> Dict[str, float]`**
- Collects current system metrics
- **Returns:** Dictionary with CPU, memory, disk usage

**`get_average_metrics() -> Dict[str, float]`**
- Gets average metrics over history
- **Returns:** Dictionary with average metrics

**`is_healthy(cpu_threshold: float = 80.0, memory_threshold: float = 80.0) -> bool`**
- Checks if system is healthy
- **Parameters:**
  - `cpu_threshold`: CPU usage threshold
  - `memory_threshold`: Memory usage threshold
- **Returns:** True if system is healthy

---

## Usage Examples

### Complete Pipeline Example

```python
from src.data_ingestion.kaggle_ingestion import KaggleDataIngestion
from src.etl_pipeline.pipeline import ETLPipeline
from src.modeling.model_trainer import ModelTrainer
from src.feature_engineering.time_series_splitter import TimeSeriesSplitter
from pyspark.sql import SparkSession

# 1. Ingest data
ingestion = KaggleDataIngestion('~/.kaggle/kaggle.json', 'dataset_name')
raw_df = ingestion.fetch_data('data/raw')

# 2. Run ETL pipeline
spark = SparkSession.builder.appName("AQI").getOrCreate()
pipeline = ETLPipeline('data/bronze', 'data/silver', 'data/gold', spark)
gold_df, metrics = pipeline.run_pipeline(raw_df, 'kaggle')

# 3. Split data
splitter = TimeSeriesSplitter(n_splits=3)
X_train, X_test, y_train, y_test = splitter.get_train_test_split(
    gold_df.drop('aqi', axis=1),
    gold_df['aqi']
)

# 4. Train models
trainer = ModelTrainer(X_train, y_train, X_test, y_test)
xgb_model = trainer.train_xgboost()
rf_model = trainer.train_random_forest()

# 5. Get best model
best_name, best_model = trainer.get_best_model()
print(f"Best model: {best_name}")
```

### Streaming Inference Example

```python
from src.streaming.streaming_inference_pipeline import StreamingInferencePipeline
from src.streaming.alert_service import AlertService

# Initialize pipeline
pipeline = StreamingInferencePipeline(
    model_path='data/models/xgboost_v1',
    feature_columns=[...]
)

# Initialize alert service
alert_service = AlertService()

# Process streaming event
event = {
    'city': 'Delhi',
    'aqi': 150,
    'timestamp': '2024-01-01T12:00:00'
}

result = pipeline.process_event(event)
alert_service.process_current_aqi(event['city'], event['aqi'])
```

---

## Error Handling

All modules raise custom exceptions for error handling:

- `KaggleConnectionError`: Kaggle API connection issues
- `IQAirAPIError`: IQAir API errors
- `ETLPipelineError`: ETL pipeline execution errors
- `BronzeLayerError`: Bronze Layer storage errors
- `SilverLayerError`: Silver Layer transformation errors
- `GoldLayerError`: Gold Layer feature engineering errors
- `ModelTrainerError`: Model training errors
- `ModelEvaluatorError`: Model evaluation errors
- `FeatureProcessorError`: Feature engineering errors

Example error handling:

```python
from src.data_ingestion.kaggle_ingestion import KaggleDataIngestion, KaggleConnectionError

try:
    ingestion = KaggleDataIngestion(api_key_path, dataset_name)
    df = ingestion.fetch_data(output_path)
except KaggleConnectionError as e:
    logger.error(f"Failed to ingest Kaggle data: {e}")
    # Handle error appropriately
```

---

## Configuration

System configuration is managed via `config.yaml`:

```yaml
cities:
  - Delhi
  - Mumbai
  - Bangalore

data_paths:
  bronze: data/bronze
  silver: data/silver
  gold: data/gold
  models: data/models

iqair:
  api_key: ${IQAIR_API_KEY}
  timeout: 10

kaggle:
  api_key_path: ~/.kaggle/kaggle.json
  dataset_name: rohanrao/air-quality-data-in-india

models:
  xgboost:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
  random_forest:
    n_estimators: 100
    max_depth: 15

streaming:
  bootstrap_servers: localhost:9092
  window_hours: 24

alerts:
  dedup_window_hours: 1
  model_threshold: 150
```

---

## Performance Targets

- ETL pipeline: < 5 minutes per day
- Streaming latency: < 1 second per event
- Model inference: < 1 second per city
- Dashboard refresh: Every 5 minutes
- Model R²: > 0.75

---

## Support

For issues or questions, refer to:
- README.md: Setup and usage instructions
- MONITORING_AND_LOGGING.md: Logging and monitoring guide
- REPRODUCIBILITY_VALIDATION.md: Reproducibility details
