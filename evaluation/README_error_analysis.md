# Error Analysis for AirIO Evaluation

This document describes the error analysis functionality added to the AirIO evaluation system to analyze error growth patterns over time.

## Overview

The error analysis system helps understand how position and velocity errors grow over time in inertial odometry systems. It can determine whether errors grow linearly, quadratically, or follow other patterns, which is crucial for understanding system performance and drift characteristics.

## Features

### 1. Individual Sequence Error Analysis
- **Error vs Time Plots**: Shows how position and velocity errors evolve over time
- **Log-Log Analysis**: Determines the growth rate and type of error accumulation
- **Growth Classification**: Automatically classifies errors as Linear, Quadratic, Square Root, or Power Law
- **Statistical Analysis**: Provides R² values and confidence metrics

### 2. Cross-Sequence Summary Analysis
- **Growth Rate Comparison**: Compares error growth rates across different sequences
- **Error Type Distribution**: Shows the distribution of error growth types
- **Correlation Analysis**: Analyzes relationships between position and velocity error growth
- **Summary Statistics**: Provides average growth rates and standard deviations

## Usage

### Running Error Analysis with Evaluation

```bash
# Run evaluation with error analysis
python evaluation/evaluate_motion.py \
    --exp experiments/your_experiment \
    --dataconf configs/datasets/YourDataset/your_config.conf \
    --savedir ./results/error_analysis \
    --seqlen 1000
```

### Testing with Synthetic Data

```bash
# Test the error analysis functionality
python evaluation/test_error_analysis.py
```

## Output Files

### Individual Sequence Plots
- `{sequence_name}_position_error_vs_time.png`: Position error analysis
- `{sequence_name}_velocity_error_vs_time.png`: Velocity error analysis

Each plot contains:
1. **Top**: Error vs time (linear scale)
2. **Bottom**: Log-log plot with fitted growth curve
3. **Analysis Box**: Growth rate, type, and R² value

### Summary Plots
- `error_growth_summary.png`: Comprehensive analysis across all sequences

Contains four subplots:
1. **Position Error Growth Rates**: Bar chart comparing growth rates
2. **Velocity Error Growth Rates**: Bar chart comparing growth rates  
3. **Position vs Velocity Correlation**: Scatter plot
4. **Error Type Distribution**: Distribution of growth types

### Results JSON
- `result.json`: Contains all metrics including growth rates

## Understanding Error Growth Types

### Linear Growth (slope ≈ 1.0)
- Error grows proportionally with time
- Typical for systems with constant bias or drift
- Example: `error(t) = a * t + b`

### Quadratic Growth (slope ≈ 2.0)
- Error grows quadratically with time
- Common in double-integration systems
- Example: `error(t) = a * t² + b`

### Square Root Growth (slope ≈ 0.5)
- Error grows with square root of time
- Typical for random walk processes
- Example: `error(t) = a * √t + b`

### Power Law Growth (other slopes)
- Error follows a power law relationship
- Example: `error(t) = a * t^slope + b`

## Interpretation Guidelines

### Good Performance Indicators
- **Low Growth Rates**: Slopes close to 0.5 or 1.0
- **High R² Values**: R² > 0.8 indicates good fit
- **Consistent Patterns**: Similar growth rates across sequences

### Warning Signs
- **High Growth Rates**: Slopes > 2.0 indicate rapid error accumulation
- **Low R² Values**: R² < 0.5 suggests poor fit or noisy data
- **Inconsistent Patterns**: Large variations in growth rates

### System-Specific Considerations
- **Short Sequences**: May not show clear growth patterns
- **Noisy Data**: Can mask underlying growth trends
- **Different Motion Types**: May have different error characteristics

## Example Output

```
ERROR GROWTH ANALYSIS SUMMARY
==================================================
Average Position Error Growth Rate: 1.234 ± 0.456
Average Velocity Error Growth Rate: 0.987 ± 0.234
Position Error Growth Types: {'Linear': 3, 'Quadratic': 1, 'Other': 2}
Velocity Error Growth Types: {'Linear': 4, 'Square Root': 1, 'Other': 1}
==================================================
```

## Technical Details

### Mathematical Background
The analysis uses log-log regression to fit the relationship:
```
log(error) = slope * log(time) + intercept
```

This corresponds to:
```
error = exp(intercept) * time^slope
```

### Implementation Notes
- **Data Filtering**: Removes zero and negative values for log analysis
- **Minimum Data Points**: Requires at least 10 valid points for analysis
- **Robust Fitting**: Uses scipy.stats.linregress for reliable regression
- **Error Handling**: Gracefully handles insufficient data cases

## Troubleshooting

### Common Issues
1. **"Insufficient data for log-log analysis"**
   - Solution: Check if error values are all zero or negative
   - Ensure sequence has enough time points

2. **"No valid error growth rates found"**
   - Solution: Verify that evaluation completed successfully
   - Check if network output files exist

3. **Poor R² values**
   - Solution: Check for noisy or irregular error patterns
   - Consider filtering or smoothing data

### Performance Tips
- Use longer sequences for better growth pattern analysis
- Ensure consistent time sampling rates
- Consider averaging multiple runs for noisy data

## Future Enhancements

Potential improvements to consider:
- **Confidence Intervals**: Add uncertainty quantification
- **Multiple Regression Models**: Compare different growth models
- **Real-time Analysis**: Monitor error growth during training
- **Custom Growth Models**: Support for domain-specific error models 