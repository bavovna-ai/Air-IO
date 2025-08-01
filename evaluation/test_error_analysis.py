#!/usr/bin/env python3
"""
Test script for error analysis functionality
This script demonstrates how to use the error analysis features with synthetic data
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats
import os

def plot_error_with_time(time, error, error_type, save_path, sequence_name):
    """
    Plot error vs time to analyze error growth pattern (linear, quadratic, etc.)
    
    Args:
        time: Time array
        error: Error array
        error_type: Type of error ('position', 'velocity', 'orientation')
        save_path: Path to save the plot
        sequence_name: Name of the sequence for title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert to numpy if needed
    if torch.is_tensor(time):
        time = time.cpu().numpy()
    if torch.is_tensor(error):
        error = error.cpu().numpy()
    
    # Ensure time and error arrays have the same length
    if len(time) != len(error):
        print(f"Warning: Time array length ({len(time)}) != Error array length ({len(error)})")
        # Truncate the longer array to match the shorter one
        min_length = min(len(time), len(error))
        time = time[:min_length]
        error = error[:min_length]
        print(f"Truncated both arrays to length {min_length}")
    
    # Plot 1: Error vs Time
    ax1.plot(time, error, 'b-', linewidth=1, alpha=0.7, label=f'{error_type} Error')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'{error_type.capitalize()} Error')
    ax1.set_title(f'{error_type.capitalize()} Error vs Time - {sequence_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Log-Log plot to determine growth rate
    # Remove zero values for log plot
    valid_mask = (error > 0) & (time > 0)
    if np.sum(valid_mask) > 10:  # Need enough points for meaningful analysis
        time_valid = time[valid_mask]
        error_valid = error[valid_mask]
        
        ax2.loglog(time_valid, error_valid, 'ro', markersize=3, alpha=0.7, label='Data points')
        
        # Fit linear regression in log space to determine growth rate
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log(time_valid), np.log(error_valid)
        )
        
        # Generate fitted line
        time_fit = np.linspace(time_valid.min(), time_valid.max(), 100)
        error_fit = np.exp(intercept) * (time_fit ** slope)
        
        ax2.loglog(time_fit, error_fit, 'g-', linewidth=2, 
                  label=f'Fit: y = {np.exp(intercept):.3f} * x^{slope:.3f}\nR² = {r_value**2:.3f}')
        
        # Determine growth type
        if abs(slope - 1.0) < 0.1:
            growth_type = "Linear"
        elif abs(slope - 2.0) < 0.1:
            growth_type = "Quadratic"
        elif abs(slope - 0.5) < 0.1:
            growth_type = "Square Root"
        else:
            growth_type = f"Power Law (x^{slope:.2f})"
        
        ax2.set_xlabel('Time (s) - Log Scale')
        ax2.set_ylabel(f'{error_type.capitalize()} Error - Log Scale')
        ax2.set_title(f'Error Growth Analysis - {growth_type} Growth')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for log-log analysis', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_xlabel('Time (s) - Log Scale')
        ax2.set_ylabel(f'{error_type.capitalize()} Error - Log Scale')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return slope if 'slope' in locals() else None
