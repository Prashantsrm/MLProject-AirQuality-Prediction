# Quick Start Guide

Get the Air Quality Prediction System up and running in minutes.

## Prerequisites

- Python 3.8+
- Java 11+
- Git
- Docker and Docker Compose (optional, for containerized setup)

## Option 1: Local Setup (5 minutes)

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-org/air-quality-prediction-system.git
cd air-quality-prediction-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Credentials

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Add your API keys:
```
IQAIR_API_KEY=your_iqair_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### Step 3: Setup Kaggle API

```bash
# Create Kaggle credentials directory
mkdir -p ~/.kaggle

# Place kaggle.json (download from https://www.kaggle.com/settings/account)
# Then set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Step 4: Run the System

```bash
# Load environment variables
source .env

# Run complete pipeline
python run_pipeline.py

# Or run individual components
python scripts/ingest_data.py      # Data ingestion
python scripts/run_etl.py          # ETL pipeline
python scripts/train_models.py     # Model training
streamlit run src/dashboard/app.py # Dashboard (http://localhost:8501)
```

## Option 2: Docker Setup (3 minutes)

### Step 1: Prepare Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

### Step 2: Build and Run

```bash
# Build Docker image
docker build -t aqi-prediction-system:latest .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f app

# Access dashboard
# Open browser to http://localhost:8501
```

### Step 3: Stop Services

```bash
# Stop all services
docker-compose down

# Remove volumes (careful - deletes data)
docker-compose down -v
```

## Verify Installation

```bash
# Test Python imports
python -c "import pyspark; import xgboost; import streamlit; print('✓ All imports successful')"

# Test configuration
python -c "from src.utils.config_loader import load_config; config = load_config(); print('✓ Configuration loaded')"

# Test data ingestion
python scripts/test_data_ingestion.py
```

## Common Issues

### Issue: Kaggle API not found

```bash
# Solution: Ensure kaggle.json is in correct location
ls -la ~/.kaggle/kaggle.json

# If not found, download from https://www.kaggle.com/settings/account
```

### Issue: IQAir API key invalid

```bash
# Solution: Verify API key in .env
echo $IQAIR_API_KEY

# Get new key from https://waqi.info/api/
```

### Issue: PySpark memory error

```bash
# Solution: Increase memory allocation
export SPARK_DRIVER_MEMORY=4g
export SPARK_EXECUTOR_MEMORY=4g
```

### Issue: Port 8501 already in use

```bash
# Solution: Use different port
streamlit run src/dashboard/app.py --server.port 8502
```

## Next Steps

1. **Read the full documentation**:
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Comprehensive deployment guide
   - [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference
   - [API.md](API.md) - API documentation

2. **Explore the system**:
   - Check `src/` directory for source code
   - Review `tests/` for test examples
   - Look at `scripts/` for utility scripts

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

4. **View monitoring**:
   ```bash
   python scripts/start_monitoring_dashboard.py
   ```

5. **Generate reports**:
   ```bash
   python scripts/generate_monitoring_report.py
   ```

## Project Structure

```
air-quality-prediction-system/
├── src/                          # Source code
│   ├── data_ingestion/          # Data ingestion services
│   ├── etl_pipeline/            # ETL pipeline
│   ├── feature_engineering/     # Feature engineering
│   ├── modeling/                # ML models
│   ├── streaming/               # Streaming pipeline
│   ├── dashboard/               # Streamlit dashboard
│   └── utils/                   # Utilities
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── data/                         # Data storage
│   ├── bronze/                  # Raw data
│   ├── silver/                  # Cleaned data
│   ├── gold/                    # Features
│   └── models/                  # Trained models
├── docs/                         # Documentation
├── config.yaml                   # Configuration file
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image
├── docker-compose.yml            # Docker Compose config
└── run_pipeline.py              # Main entry point
```

## Configuration

### Basic Configuration (config.yaml)

```yaml
system:
  environment: development
  log_level: INFO

cities:
  - Delhi
  - Mumbai
  - Bangalore

etl:
  batch_size: 1000
  processing_interval_hours: 1

modeling:
  xgboost:
    n_estimators: 100
  random_forest:
    n_estimators: 100
```

### Environment Variables

```bash
# Required
IQAIR_API_KEY=your_key
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key

# Optional
AQI_SYSTEM_LOG_LEVEL=DEBUG
AQI_SYSTEM_ENVIRONMENT=production
```

See [CONFIGURATION.md](CONFIGURATION.md) for complete reference.

## Dashboard Access

Once running, access the dashboard at:

```
http://localhost:8501
```

Features:
- Current AQI for all cities
- 24-hour forecasts
- Active alerts
- Historical trends
- Multi-city comparison

## Monitoring

View system health and metrics:

```bash
# Start monitoring dashboard
python scripts/start_monitoring_dashboard.py

# Generate monitoring report
python scripts/generate_monitoring_report.py

# View logs
tail -f logs/system.log
```

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Check troubleshooting in [DEPLOYMENT.md](DEPLOYMENT.md)
- **Logs**: Check `logs/system.log` for detailed information

## Next: Production Deployment

For production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md#production-deployment) for:
- Cloud deployment (AWS, GCP, Azure)
- Kubernetes setup
- Monitoring and alerting
- Backup and recovery
