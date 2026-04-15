# Reproducibility Validation Guide

## Overview

This document describes the reproducibility validation framework for the Air Quality Prediction System. The system is designed to produce deterministic results across multiple runs with fixed random seeds, ensuring that all results are reproducible and verifiable.

## Requirements

The reproducibility validation addresses the following requirements:

- **Requirement 10.4**: THE System SHALL use fixed random seeds for all stochastic components (model training, data sampling)
- **Requirement 10.5**: THE System SHALL log all execution steps with timestamps and status messages to a log file

## Design Properties

The reproducibility validation validates the following design properties:

- **Property 37: End-to-End Reproducibility** - For any execution of the main pipeline script, running it multiple times with identical inputs shall produce identical results (deterministic behavior)
- **Property 38: Random Seed Determinism** - For any stochastic component (model training, data sampling), fixed random seeds shall ensure deterministic behavior across runs
- **Property 39: Execution Logging** - For any pipeline execution, all major operations shall be logged with timestamps and status messages to a log file

## Random Seed Configuration

### Fixed Random Seed Value

The system uses a fixed random seed value of **42** throughout all components:

```python
# src/utils/constants.py
RANDOM_SEED = 42
```

This seed is used in:

1. **ETL Pipeline** - Spark session random seed
2. **Model Training** - XGBoost and Random Forest models
3. **Data Sampling** - NumPy and Pandas random operations
4. **Feature Engineering** - Random feature transformations

### Configuration

The random seed is configured in `config.yaml`:

```yaml
system:
  environment: development
  log_level: INFO
  random_seed: 42
```

## Reproducibility Validation Script

### Usage

Run the reproducibility validation script to verify deterministic behavior:

```bash
python scripts/validate_reproducibility.py
```

### What It Does

The script performs the following steps:

1. **Verifies Random Seed Configuration**
   - Checks that RANDOM_SEED constant is 42
   - Checks that config.yaml has random_seed = 42
   - Sets NumPy and Pandas random seeds

2. **Creates Deterministic Sample Data**
   - Generates sample air quality data with fixed random seed
   - Uses first 3 cities for faster testing
   - Creates 100 records with realistic AQI values

3. **Runs Pipeline Multiple Times**
   - Executes complete pipeline 3 times with identical inputs
   - Resets random seeds before each run
   - Tracks outputs and metrics from each run

4. **Validates Reproducibility**
   - Compares Gold DataFrame outputs across runs
   - Compares ETL metrics across runs
   - Compares model training metrics across runs
   - Generates hash values for comparison

5. **Generates Report**
   - Creates comprehensive reproducibility report
   - Saves report to `reports/reproducibility_report.json`
   - Prints summary to console

### Output

The script generates:

1. **Console Output** - Summary of validation results
2. **Log File** - Detailed execution logs in `logs/system.log`
3. **Report File** - JSON report in `reports/reproducibility_report.json`

### Report Structure

The reproducibility report contains:

```json
{
  "validation_timestamp": "2024-01-15T10:30:45.123456",
  "n_runs": 3,
  "random_seed": 42,
  "validation_duration_seconds": 125.45,
  "runs_data": [
    {
      "iteration": 1,
      "timestamp": "2024-01-15T10:30:45.123456",
      "duration_seconds": 42.15,
      "gold_df_shape": [95, 28],
      "gold_df_hash": "abc123...",
      "etl_metrics_hash": "def456...",
      "model_metrics_hash": "ghi789...",
      "etl_metrics": {...},
      "model_metrics": {...}
    },
    ...
  ],
  "reproducibility_checks": [
    {
      "type": "Gold DataFrame Mismatch",
      "run": 2,
      "description": "Run 2 Gold DataFrame differs from Run 1...",
      "severity": "CRITICAL"
    }
  ],
  "is_reproducible": true,
  "summary": {
    "total_issues": 0,
    "critical_issues": 0,
    "warning_issues": 0,
    "is_reproducible": true,
    "status": "PASS",
    "message": "All runs produced identical results. System is reproducible."
  }
}
```

## Testing

### Unit Tests

Run unit tests for random seed configuration:

```bash
python -m pytest tests/integration/test_reproducibility_validation.py::TestRandomSeedConfiguration -v
```

Tests:
- `test_random_seed_constant_value` - Verifies RANDOM_SEED = 42
- `test_numpy_seed_reproducibility` - Verifies NumPy reproducibility
- `test_pandas_seed_reproducibility` - Verifies Pandas reproducibility

### Integration Tests

Run integration tests for reproducibility validation:

```bash
python -m pytest tests/integration/test_reproducibility_validation.py::TestReproducibilityValidation -v
```

Tests:
- `test_random_seed_is_fixed` - Verifies fixed random seed
- `test_etl_pipeline_deterministic` - Verifies ETL determinism
- `test_model_training_deterministic` - Verifies model training determinism
- `test_multiple_runs_identical_outputs` - Verifies 3 runs produce identical outputs
- `test_random_seed_affects_results` - Verifies random seeds affect behavior
- `test_execution_logging_present` - Verifies execution logging
- `test_reproducibility_report_structure` - Verifies report structure
- `test_reproducibility_validation_passes` - Verifies validation passes

### Running All Tests

```bash
python -m pytest tests/integration/test_reproducibility_validation.py -v
```

## Reproducibility Checklist

To ensure reproducibility of your results:

- [ ] Use Python 3.8+ with pinned dependency versions (see `requirements.txt`)
- [ ] Set environment variable `RANDOM_SEED=42` or use default from config.yaml
- [ ] Run pipeline with `python scripts/validate_reproducibility.py`
- [ ] Verify all runs produce identical results (status = PASS)
- [ ] Check that no critical issues are reported
- [ ] Save reproducibility report for documentation

## Troubleshooting

### Issue: Reproducibility validation fails with "Gold DataFrame Mismatch"

**Cause**: Non-deterministic behavior in ETL pipeline

**Solution**:
1. Check that RANDOM_SEED is set to 42 in constants.py
2. Verify config.yaml has `random_seed: 42`
3. Check for any non-deterministic operations (e.g., using system time, random without seed)
4. Review recent changes to ETL pipeline code

### Issue: Model metrics differ across runs

**Cause**: Non-deterministic model training

**Solution**:
1. Verify model hyperparameters include `random_state=42`
2. Check that cross-validation uses fixed random seed
3. Ensure data splitting is deterministic
4. Review model training code for any randomness

### Issue: Execution logging is missing

**Cause**: Logging not configured properly

**Solution**:
1. Check that logger is initialized in each module
2. Verify log level is set to INFO or DEBUG
3. Check that log file path is writable
4. Review logging configuration in `src/utils/logger.py`

## Best Practices

1. **Always use fixed random seeds** - Never rely on system randomness for critical operations
2. **Document random seed usage** - Clearly indicate where random seeds are used
3. **Test reproducibility regularly** - Run validation script before each release
4. **Version dependencies** - Use pinned versions in requirements.txt
5. **Log all operations** - Include timestamps and status messages in logs
6. **Validate results** - Compare outputs across multiple runs

## References

- [Reproducibility in Machine Learning](https://en.wikipedia.org/wiki/Reproducibility)
- [NumPy Random Seed Documentation](https://numpy.org/doc/stable/reference/random/generated.html)
- [Pandas Random State Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html)
- [XGBoost Random State](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [Scikit-learn Random State](https://scikit-learn.org/stable/glossary.html#term-random_state)

## Contact

For questions or issues related to reproducibility validation, please contact the Machine Learning Team.
