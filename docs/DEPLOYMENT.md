# Deployment and Configuration Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Configuration Parameters](#configuration-parameters)
3. [Environment Variable Setup](#environment-variable-setup)
4. [Installation and Setup](#installation-and-setup)
5. [Docker Deployment](#docker-deployment)
6. [Running the System](#running-the-system)
7. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
8. [Production Deployment](#production-deployment)

---

## System Requirements

### Hardware Requirements

- **CPU**: Minimum 4 cores (8+ cores recommended for production)
- **Memory**: Minimum 8 GB RAM (16+ GB recommended for production)
- **Disk Space**: Minimum 50 GB for data storage (100+ GB recommended)
- **Network**: Stable internet connection for API access

### Software Requirements

- **Python**: 3.8 or higher
- **Java**: 11 or higher (required for PySpark)
- **Git**: For version control
- **Docker**: 20.10+ (optional, for containerized deployment)
- **Docker Compose**: 1.29+ (optional, for multi-container setup)

### Operating System Support

- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+
- **macOS**: 10.14+ (Intel or Apple Silicon)
- **Windows**: Windows 10/11 with WSL2 or Docker Desktop

---

## Configuration Parameters

### Configuration File Structure

The system uses `config.yaml` as the primary configuration file. All parameters can be overridden using environment variables.

### System Configuration

```yaml
system:
  environment: development  # development, staging, production
  log_level: INFO           # DEBUG, INFO, WARNING, ERROR, CRITICAL
  random_seed: 42           # For reproducibility
```

**Parameters**:
- `environment`: Deployment environment (affects logging verbosity and resource allocation)
- `log_level`: Logging verbosity level
- `random_seed`: Fixed seed for reproducible results

### Data Sources Configuration

```yaml
data_sources:
  kaggle:
    dataset_name: "asaniczka/real-time-air-quality-index-aqi-india-2023-2025"
    api_key_path: ~/.kaggle/kaggle.json
  iqair:
    api_key: ${IQAIR_API_KEY}
    base_url: "https://api.waqi.info"
```

**Parameters**:
- `kaggle.dataset_name`: Kaggle dataset identifier
- `kaggle.api_key_path`: Path to Kaggle API credentials file
- `iqair.api_key`: IQAir API key (use environment variable)
- `iqair.base_url`: IQAir API endpoint

### Cities Configuration

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

**Parameters**:
- List of Indian cities to monitor (minimum 1, recommended 10+)

### ETL Pipeline Configuration

```yaml
etl:
  batch_size: 1000                    # Records per batch
  processing_interval_hours: 1        # Interval between ETL runs
  max_retries: 3                      # Retry attempts for failed operations
  retry_delay_seconds: 30             # Initial retry delay (exponential backoff)
```

**Parameters**:
- `batch_size`: Number of records to process in each batch
- `processing_interval_hours`: Frequency of ETL pipeline execution
- `max_retries`: Maximum retry attempts for API calls and data operations
- `retry_delay_seconds`: Initial delay for exponential backoff retry logic

### Machine Learning Configuration

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

**Parameters**:
- `xgboost.*`: XGBoost hyperparameters
- `random_forest.*`: Random Forest hyperparameters
- `cv_folds`: Number of cross-validation folds
- `test_size`: Fraction of data for testing (0.0-1.0)

### Streaming Configuration

```yaml
streaming:
  kafka_bootstrap_servers: "localhost:9092"
  topic: "aqi-events"
  consumer_group: "aqi-consumer"
  max_latency_ms: 1000
```

**Parameters**:
- `kafka_bootstrap_servers`: Kafka broker addresses
- `topic`: Kafka topic for AQI events
- `consumer_group`: Consumer group identifier
- `max_latency_ms`: Maximum acceptable latency per event (milliseconds)

### Alert Configuration

```yaml
alerts:
  dedup_window_hours: 1       # Deduplication window
  prediction_threshold: 150   # AQI threshold for model-based alerts
```

**Parameters**:
- `dedup_window_hours`: Time window for alert deduplication
- `prediction_threshold`: AQI threshold for triggering model-based alerts

### Dashboard Configuration

```yaml
dashboard:
  refresh_interval_minutes: 5
  port: 8501
  host: "0.0.0.0"
```

**Parameters**:
- `refresh_interval_minutes`: Dashboard data refresh frequency
- `port`: Port for Streamlit dashboard
- `host`: Host binding address

### Storage Configuration

```yaml
storage:
  bronze_path: "data/bronze"
  silver_path: "data/silver"
  gold_path: "data/gold"
  models_path: "data/models"
  logs_path: "logs"
```

**Parameters**:
- `bronze_path`: Path for raw data storage
- `silver_path`: Path for cleaned data storage
- `gold_path`: Path for feature-engineered data storage
- `models_path`: Path for trained model storage
- `logs_path`: Path for system logs

---

## Environment Variable Setup

### Required Environment Variables

Environment variables override configuration file values and are essential for sensitive data.

#### IQAir API Configuration

```bash
export IQAIR_API_KEY="your_iqair_api_key_here"
```

**Description**: API key for IQAir real-time air quality data access
**Obtain from**: https://waqi.info/api/

#### Kaggle API Configuration

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```

**Description**: Credentials for Kaggle dataset access
**Obtain from**: https://www.kaggle.com/settings/account

**Setup Instructions**:
1. Create account at https://www.kaggle.com
2. Go to Account Settings → API
3. Click "Create New API Token"
4. This downloads `kaggle.json`
5. Place file at `~/.kaggle/kaggle.json`
6. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Optional Environment Variables

#### System Configuration

```bash
export AQI_SYSTEM_ENVIRONMENT="production"
export AQI_SYSTEM_LOG_LEVEL="DEBUG"
export AQI_SYSTEM_RANDOM_SEED="42"
```

#### Storage Paths

```bash
export AQI_STORAGE_BRONZE_PATH="/data/bronze"
export AQI_STORAGE_SILVER_PATH="/data/silver"
export AQI_STORAGE_GOLD_PATH="/data/gold"
export AQI_STORAGE_MODELS_PATH="/data/models"
export AQI_STORAGE_LOGS_PATH="/logs"
```

#### Streaming Configuration

```bash
export AQI_STREAMING_KAFKA_BOOTSTRAP_SERVERS="kafka:9092"
export AQI_STREAMING_TOPIC="aqi-events"
export AQI_STREAMING_CONSUMER_GROUP="aqi-consumer"
```

#### Dashboard Configuration

```bash
export AQI_DASHBOARD_PORT="8501"
export AQI_DASHBOARD_HOST="0.0.0.0"
```

### Setting Environment Variables

#### Linux/macOS

Create `.env` file in project root:

```bash
# .env
IQAIR_API_KEY=your_api_key
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
AQI_SYSTEM_ENVIRONMENT=production
```

Load environment variables:

```bash
source .env
```

Or use `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
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

---

## Installation and Setup

### Prerequisites

1. Python 3.8+ installed
2. Java 11+ installed
3. Git installed
4. Internet connection for API access

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/air-quality-prediction-system.git
cd air-quality-prediction-system
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Step 4: Configure System

```bash
# Copy example configuration
cp config.yaml.example config.yaml

# Edit configuration with your settings
nano config.yaml  # or use your preferred editor
```

### Step 5: Set Up Kaggle API

```bash
# Create Kaggle credentials directory
mkdir -p ~/.kaggle

# Place kaggle.json in the directory
# (Download from https://www.kaggle.com/settings/account)

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Step 6: Set Environment Variables

```bash
# Create .env file
cat > .env << EOF
IQAIR_API_KEY=your_iqair_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
EOF

# Load environment variables
source .env
```

### Step 7: Verify Installation

```bash
# Test Python imports
python -c "import pyspark; import xgboost; import streamlit; print('All imports successful')"

# Test configuration loading
python -c "from src.utils.config_loader import load_config; config = load_config(); print('Configuration loaded successfully')"

# Test data ingestion
python scripts/test_data_ingestion.py
```

---

## Docker Deployment

### Docker Setup

#### Prerequisites

- Docker 20.10+
- Docker Compose 1.29+

#### Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/bronze data/silver data/gold data/models logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SPARK_LOCAL_IP=127.0.0.1

# Expose ports
EXPOSE 8501 9092

# Default command
CMD ["python", "run_pipeline.py"]
```

#### Docker Compose Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: aqi-app
    ports:
      - "8501:8501"  # Streamlit dashboard
    environment:
      - IQAIR_API_KEY=${IQAIR_API_KEY}
      - KAGGLE_USERNAME=${KAGGLE_USERNAME}
      - KAGGLE_KEY=${KAGGLE_KEY}
      - AQI_SYSTEM_ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ~/.kaggle:/root/.kaggle:ro
    depends_on:
      - kafka
    networks:
      - aqi-network

  kafka:
    image: confluentinc/cp-kafka:7.0.0
    container_name: aqi-kafka
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    depends_on:
      - zookeeper
    networks:
      - aqi-network

  zookeeper:
    image: confluentinc/cp-zookeeper:7.0.0
    container_name: aqi-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    networks:
      - aqi-network

networks:
  aqi-network:
    driver: bridge
```

### Building and Running Docker

```bash
# Build Docker image
docker build -t aqi-prediction-system:latest .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

---

## Running the System

### Main Pipeline Execution

```bash
# Run complete pipeline
python run_pipeline.py

# Run with specific configuration
python run_pipeline.py --config config.yaml

# Run with verbose logging
python run_pipeline.py --log-level DEBUG
```

### Component-Specific Execution

```bash
# Data ingestion only
python scripts/ingest_data.py

# ETL pipeline only
python scripts/run_etl.py

# Model training only
python scripts/train_models.py

# Streaming pipeline
python scripts/run_streaming.py

# Dashboard
streamlit run src/dashboard/app.py
```

### Monitoring Dashboard

```bash
# Start Streamlit dashboard
streamlit run src/dashboard/app.py

# Access dashboard
# Open browser to http://localhost:8501
```

### Monitoring and Logging

```bash
# View system logs
tail -f logs/system.log

# View monitoring dashboard
python scripts/start_monitoring_dashboard.py

# Generate monitoring report
python scripts/generate_monitoring_report.py
```

---

## Monitoring and Troubleshooting

### Common Issues and Solutions

#### Issue: Kaggle API Authentication Failed

**Symptoms**: `KaggleApiException: 401 - Unauthorized`

**Solution**:
1. Verify `~/.kaggle/kaggle.json` exists
2. Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. Verify credentials are correct
4. Regenerate API token from Kaggle website

#### Issue: IQAir API Rate Limiting

**Symptoms**: `HTTPError: 429 - Too Many Requests`

**Solution**:
1. Check API rate limits (typically 10,000 calls/month for free tier)
2. Increase `retry_delay_seconds` in config
3. Reduce `processing_interval_hours` frequency
4. Upgrade to paid IQAir plan if needed

#### Issue: PySpark Memory Error

**Symptoms**: `java.lang.OutOfMemoryError: Java heap space`

**Solution**:
1. Increase Spark memory allocation:
   ```bash
   export SPARK_DRIVER_MEMORY=4g
   export SPARK_EXECUTOR_MEMORY=4g
   ```
2. Reduce batch size in config
3. Increase system RAM or use cloud deployment

#### Issue: Kafka Connection Failed

**Symptoms**: `KafkaError: Failed to connect to broker`

**Solution**:
1. Verify Kafka is running: `docker-compose ps`
2. Check Kafka bootstrap servers in config
3. Verify network connectivity: `telnet localhost 9092`
4. Restart Kafka: `docker-compose restart kafka`

#### Issue: Dashboard Not Loading

**Symptoms**: `StreamlitAPIException` or blank page

**Solution**:
1. Check Streamlit logs: `tail -f logs/streamlit.log`
2. Verify port 8501 is not in use: `lsof -i :8501`
3. Clear Streamlit cache: `streamlit cache clear`
4. Restart dashboard: `streamlit run src/dashboard/app.py --logger.level=debug`

### Performance Optimization

#### Optimize ETL Pipeline

```yaml
# config.yaml
etl:
  batch_size: 5000          # Increase batch size
  processing_interval_hours: 2  # Reduce frequency
```

#### Optimize Model Training

```yaml
# config.yaml
modeling:
  xgboost:
    n_estimators: 50        # Reduce estimators
    max_depth: 4            # Reduce depth
  cv_folds: 2               # Reduce folds
```

#### Optimize Streaming

```yaml
# config.yaml
streaming:
  max_latency_ms: 2000      # Increase acceptable latency
```

### Monitoring Metrics

Key metrics to monitor:

- **ETL Pipeline**: Processing time, record count, validation errors
- **Model Training**: Training time, cross-validation scores, R² metric
- **Streaming**: Event latency, throughput, error rate
- **System**: CPU usage, memory usage, disk space
- **API**: Response time, error rate, rate limit status

View monitoring dashboard:

```bash
python scripts/start_monitoring_dashboard.py
```

---

## Production Deployment

### Pre-Production Checklist

- [ ] All tests passing (>80% coverage)
- [ ] Configuration validated for production environment
- [ ] Sensitive credentials stored in environment variables
- [ ] Logging configured for production (INFO level)
- [ ] Monitoring and alerting configured
- [ ] Backup and disaster recovery plan in place
- [ ] Performance tested with full dataset
- [ ] Security review completed

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
  cv_folds: 5
  test_size: 0.2

streaming:
  max_latency_ms: 1000

dashboard:
  refresh_interval_minutes: 5
```

### Deployment Strategies

#### Cloud Deployment (AWS)

```bash
# Push Docker image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag aqi-prediction-system:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/aqi-prediction-system:latest

docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/aqi-prediction-system:latest

# Deploy to ECS
aws ecs create-service --cluster aqi-cluster --service-name aqi-service --task-definition aqi-task:1 --desired-count 1
```

#### Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aqi-prediction-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aqi-prediction-system
  template:
    metadata:
      labels:
        app: aqi-prediction-system
    spec:
      containers:
      - name: aqi-app
        image: aqi-prediction-system:latest
        ports:
        - containerPort: 8501
        env:
        - name: IQAIR_API_KEY
          valueFrom:
            secretKeyRef:
              name: aqi-secrets
              key: iqair-api-key
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
```

Deploy to Kubernetes:

```bash
kubectl apply -f k8s-deployment.yaml
kubectl expose deployment aqi-prediction-system --type=LoadBalancer --port=8501
```

### Backup and Recovery

```bash
# Backup data directories
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/

# Backup models
cp -r data/models data/models-backup-$(date +%Y%m%d)

# Restore from backup
tar -xzf backup-20240101.tar.gz
```

### Monitoring and Alerting

Set up monitoring for:

- System health (CPU, memory, disk)
- API availability and response times
- Model performance degradation
- Data quality issues
- Error rates and exceptions

Use tools like:
- Prometheus for metrics collection
- Grafana for visualization
- AlertManager for alerting

---

## Support and Documentation

For additional help:

- **README.md**: Project overview and quick start
- **API.md**: API documentation
- **MONITORING_AND_LOGGING.md**: Detailed monitoring guide
- **REPRODUCIBILITY_VALIDATION.md**: Reproducibility validation procedures

For issues or questions:
- Check logs in `logs/system.log`
- Review troubleshooting section above
- Contact system administrator
