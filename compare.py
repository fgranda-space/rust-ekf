#!/usr/bin/env python3
"""
Data Comparison Script

This script compares IMU data from FLY479_IMU_data.csv with the filtered EKF data
from ekf_state_batch_filtered.csv. It handles different data sizes by interpolating
and aligning data based on timestamps and sample indices.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy import interpolate
import argparse

# Set the style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_imu_data(imu_path, assumed_sample_rate=150.0):
    """
    Load IMU data and create a timestamp vector.
    
    Args:
        imu_path: Path to FLY479_IMU_data.csv
        assumed_sample_rate: Assumed sample rate in Hz
    
    Returns:
        DataFrame with timestamp added
    """
    try:
        df = pd.read_csv(imu_path)
        print(f"📁 Loaded IMU data: {len(df)} samples")
        print(f"   Columns: {list(df.columns)}")
        
        # Create timestamp vector based on assumed sample rate
        duration = len(df) / assumed_sample_rate
        timestamps = np.linspace(0, duration, len(df))
        df['timestamp'] = timestamps
        
        print(f"   Generated timestamps: 0 to {duration:.3f}s at {assumed_sample_rate} Hz")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading IMU data: {e}")
        return None

def load_ekf_data(ekf_path):
    """Load EKF filtered data."""
    try:
        df = pd.read_csv(ekf_path)
        print(f"📁 Loaded EKF data: {len(df)} samples")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Time range: {df['timestamp'].min():.3f}s to {df['timestamp'].max():.3f}s")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading EKF data: {e}")
        return None

def align_data_by_interpolation(imu_df, ekf_df):
    """
    Align data by interpolating IMU data to match EKF timestamps.
    
    Args:
        imu_df: IMU DataFrame with generated timestamps
        ekf_df: EKF DataFrame with original timestamps
    
    Returns:
        Tuple of (aligned_imu_df, ekf_df) with matching timestamps
    """
    print("\n🔄 Aligning data using interpolation...")
    
    # Use EKF timestamps as the reference
    target_timestamps = ekf_df['timestamp'].values
    
    # Interpolate IMU data to match EKF timestamps
    aligned_imu = pd.DataFrame({'timestamp': target_timestamps})
    
    # Interpolate each IMU column
    for col in ['a_x', 'a_y', 'a_z', 'roll', 'pitch']:
        if col in imu_df.columns:
            # Create interpolation function
            f = interpolate.interp1d(
                imu_df['timestamp'], 
                imu_df[col], 
                kind='linear', 
                bounds_error=False, 
                fill_value=np.nan
            )
            aligned_imu[col] = f(target_timestamps)
            
            # Count valid interpolated points
            valid_count = (~np.isnan(aligned_imu[col])).sum()
            print(f"   {col}: {valid_count}/{len(target_timestamps)} valid interpolated points")
    
    print(f"✓ Data aligned to {len(target_timestamps)} time points")
    
    return aligned_imu, ekf_df

def plot_accelerometer_comparison(imu_df, ekf_df):
    """Compare accelerometer data between IMU and EKF."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Accelerometer Data Comparison: IMU vs EKF', fontsize=16, fontweight='bold')
    
    accel_axes = ['a_x', 'a_y', 'a_z']
    colors_imu = ['darkred', 'darkgreen', 'darkblue']
    colors_ekf = ['lightcoral', 'lightgreen', 'lightblue']
    labels = ['X-axis', 'Y-axis', 'Z-axis']
    
    for i, (axis, color_imu, color_ekf, label) in enumerate(zip(accel_axes, colors_imu, colors_ekf, labels)):
        ax = axes[i]
        
        # Plot IMU data
        if axis in imu_df.columns:
            mask = ~np.isnan(imu_df[axis])
            ax.plot(imu_df['timestamp'][mask], imu_df[axis][mask], 
                   color=color_imu, linewidth=1, alpha=0.8, label=f'IMU {axis}')
        
        # Plot EKF data
        if axis in ekf_df.columns:
            ax.plot(ekf_df['timestamp'], ekf_df[axis], 
                   color=color_ekf, linewidth=1, alpha=0.8, label=f'EKF {axis}')
        
        ax.set_title(f'Accelerometer {label} Comparison', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add gravity reference for z-axis
        if axis == 'a_z':
            ax.axhline(y=9.81, color='black', linestyle=':', alpha=0.5, label='Gravity')
    
    plt.tight_layout()
    return fig

def plot_attitude_comparison(imu_df, ekf_df):
    """Compare attitude data between IMU and EKF."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Attitude Data Comparison: IMU vs EKF', fontsize=16, fontweight='bold')
    
    attitude_axes = ['roll', 'pitch']
    colors_imu = ['darkred', 'darkgreen']
    colors_ekf = ['lightcoral', 'lightgreen']
    labels = ['Roll', 'Pitch']
    
    for i, (axis, color_imu, color_ekf, label) in enumerate(zip(attitude_axes, colors_imu, colors_ekf, labels)):
        ax = axes[i]
        
        # Plot IMU data (convert to degrees)
        if axis in imu_df.columns:
            mask = ~np.isnan(imu_df[axis])
            imu_data_deg = imu_df[axis][mask]
            ax.plot(imu_df['timestamp'][mask], imu_data_deg, 
                   color=color_imu, linewidth=1, alpha=0.8, label=f'IMU {axis}')
        
        # Plot EKF data (convert to degrees)
        if axis in ekf_df.columns:
            ekf_data_deg = np.degrees(ekf_df[axis])
            ax.plot(ekf_df['timestamp'], ekf_data_deg, 
                   color=color_ekf, linewidth=1, alpha=0.8, label=f'EKF {axis}')
        
        ax.set_title(f'{label} Angle Comparison', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_difference_analysis(imu_df, ekf_df):
    """Plot the differences between IMU and EKF data."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Difference Analysis: IMU - EKF', fontsize=16, fontweight='bold')
    
    # Accelerometer differences
    accel_axes = ['a_x', 'a_y', 'a_z']
    for i, axis in enumerate(accel_axes):
        ax = axes[0, i]
        
        if axis in imu_df.columns and axis in ekf_df.columns:
            mask = ~np.isnan(imu_df[axis])
            diff = imu_df[axis][mask] - ekf_df[axis][mask]
            
            ax.plot(imu_df['timestamp'][mask], diff, color='red', linewidth=1, alpha=0.7)
            ax.set_title(f'Accel {axis.upper()} Difference', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Difference (m/s²)')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add statistics
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            ax.text(0.02, 0.98, f'Mean: {mean_diff:.3f}\nStd: {std_diff:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Attitude differences
    attitude_axes = ['roll', 'pitch']
    for i, axis in enumerate(attitude_axes):
        ax = axes[1, i]
        
        if axis in imu_df.columns and axis in ekf_df.columns:
            mask = ~np.isnan(imu_df[axis])
            # Convert both to degrees for comparison
            imu_deg = np.degrees(imu_df[axis][mask])
            ekf_deg = np.degrees(ekf_df[axis][mask])
            diff = imu_deg - ekf_deg
            
            ax.plot(imu_df['timestamp'][mask], diff, color='blue', linewidth=1, alpha=0.7)
            ax.set_title(f'{axis.capitalize()} Difference', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Difference (degrees)')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add statistics
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            ax.text(0.02, 0.98, f'Mean: {mean_diff:.3f}°\nStd: {std_diff:.3f}°', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Correlation analysis
    ax = axes[1, 2]
    
    # Calculate correlations for available data
    correlations = {}
    for axis in ['a_x', 'a_y', 'a_z', 'roll', 'pitch']:
        if axis in imu_df.columns and axis in ekf_df.columns:
            mask = ~np.isnan(imu_df[axis])
            if mask.sum() > 1:  # Need at least 2 points for correlation
                corr = np.corrcoef(imu_df[axis][mask], ekf_df[axis][mask])[0, 1]
                correlations[axis] = corr
    
    if correlations:
        axes_names = list(correlations.keys())
        corr_values = list(correlations.values())
        
        bars = ax.bar(axes_names, corr_values, 
                     color=['red', 'green', 'blue', 'orange', 'purple'][:len(axes_names)])
        ax.set_title('Correlation: IMU vs EKF', fontweight='bold')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for bar, corr in zip(bars, corr_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{corr:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_summary_statistics(imu_df, ekf_df):
    """Plot summary statistics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Summary Statistics Comparison', fontsize=16, fontweight='bold')
    
    # Data availability
    ax1 = axes[0, 0]
    imu_count = len(imu_df)
    ekf_count = len(ekf_df)
    
    ax1.bar(['IMU Data', 'EKF Data'], [imu_count, ekf_count], 
           color=['darkblue', 'lightblue'])
    ax1.set_title('Data Sample Count', fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    
    for i, v in enumerate([imu_count, ekf_count]):
        ax1.text(i, v + max(imu_count, ekf_count) * 0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    # Time coverage
    ax2 = axes[0, 1]
    imu_duration = imu_df['timestamp'].max() - imu_df['timestamp'].min()
    ekf_duration = ekf_df['timestamp'].max() - ekf_df['timestamp'].min()
    
    ax2.bar(['IMU Data', 'EKF Data'], [imu_duration, ekf_duration], 
           color=['darkgreen', 'lightgreen'])
    ax2.set_title('Time Coverage', fontweight='bold')
    ax2.set_ylabel('Duration (seconds)')
    
    for i, v in enumerate([imu_duration, ekf_duration]):
        ax2.text(i, v + max(imu_duration, ekf_duration) * 0.01, f'{v:.1f}s', 
                ha='center', va='bottom', fontweight='bold')
    
    # Mean absolute differences
    ax3 = axes[1, 0]
    mad_values = []
    axis_labels = []
    
    for axis in ['a_x', 'a_y', 'a_z', 'roll', 'pitch']:
        if axis in imu_df.columns and axis in ekf_df.columns:
            mask = ~np.isnan(imu_df[axis])
            if mask.sum() > 0:
                if axis in ['roll', 'pitch']:
                    # Convert to degrees for attitude
                    diff = np.degrees(imu_df[axis][mask]) - np.degrees(ekf_df[axis][mask])
                    mad = np.mean(np.abs(diff))
                    axis_labels.append(f'{axis} (°)')
                else:
                    diff = imu_df[axis][mask] - ekf_df[axis][mask]
                    mad = np.mean(np.abs(diff))
                    axis_labels.append(f'{axis} (m/s²)')
                
                mad_values.append(mad)
    
    if mad_values:
        bars = ax3.bar(axis_labels, mad_values, 
                      color=['red', 'green', 'blue', 'orange', 'purple'][:len(mad_values)])
        ax3.set_title('Mean Absolute Differences', fontweight='bold')
        ax3.set_ylabel('MAD')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, mad in zip(bars, mad_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(mad_values) * 0.01,
                   f'{mad:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Data quality metrics
    ax4 = axes[1, 1]
    
    # Calculate missing data percentages
    imu_missing = {}
    ekf_missing = {}
    
    for axis in ['a_x', 'a_y', 'a_z', 'roll', 'pitch']:
        if axis in imu_df.columns:
            imu_missing[axis] = (np.isnan(imu_df[axis]).sum() / len(imu_df)) * 100
        if axis in ekf_df.columns:
            ekf_missing[axis] = (np.isnan(ekf_df[axis]).sum() / len(ekf_df)) * 100
    
    if imu_missing and ekf_missing:
        common_axes = set(imu_missing.keys()) & set(ekf_missing.keys())
        if common_axes:
            x = np.arange(len(common_axes))
            width = 0.35
            
            imu_vals = [imu_missing[axis] for axis in common_axes]
            ekf_vals = [ekf_missing[axis] for axis in common_axes]
            
            ax4.bar(x - width/2, imu_vals, width, label='IMU', color='darkred', alpha=0.7)
            ax4.bar(x + width/2, ekf_vals, width, label='EKF', color='darkblue', alpha=0.7)
            
            ax4.set_title('Missing Data Percentage', fontweight='bold')
            ax4.set_ylabel('Missing (%)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(list(common_axes))
            ax4.legend()
    
    plt.tight_layout()
    return fig

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare IMU and EKF data')
    parser.add_argument('--imu-sample-rate', type=float, default=150.0,
                       help='Assumed sample rate for IMU data in Hz (default: 150.0)')
    parser.add_argument('--no-show', action='store_true', 
                       help='Don\'t display plots, only save them')
    
    args = parser.parse_args()
    
    # File paths
    base_dir = Path(__file__).parent
    imu_path = base_dir / "data" / "FLY479_IMU_data.csv"
    ekf_path = base_dir / "data" / "ekf_state_batch_filtered.csv"
    
    print("🔍 EKF vs IMU Data Comparison")
    print("=" * 50)
    
    # Check if files exist
    if not imu_path.exists():
        print(f"❌ IMU data file not found: {imu_path}")
        return
    
    if not ekf_path.exists():
        print(f"❌ EKF data file not found: {ekf_path}")
        return
    
    # Load data
    print("\n📂 Loading data...")
    imu_df = load_imu_data(imu_path, args.imu_sample_rate)
    ekf_df = load_ekf_data(ekf_path)
    
    if imu_df is None or ekf_df is None:
        print("❌ Failed to load data files")
        return
    
    # Align data
    aligned_imu, ekf_data = align_data_by_interpolation(imu_df, ekf_df)
    
    # Generate comparison plots
    figures = []
    plot_names = []
    
    print(f"\n📊 Generating comparison plots...")
    
    # Accelerometer comparison
    fig1 = plot_accelerometer_comparison(aligned_imu, ekf_data)
    figures.append(fig1)
    plot_names.append("comparison_accelerometer.png")
    print("✓ Accelerometer comparison plot generated")
    
    # Attitude comparison  
    fig2 = plot_attitude_comparison(aligned_imu, ekf_data)
    figures.append(fig2)
    plot_names.append("comparison_attitude.png")
    print("✓ Attitude comparison plot generated")
    
    # Difference analysis
    fig3 = plot_difference_analysis(aligned_imu, ekf_data)
    figures.append(fig3)
    plot_names.append("comparison_differences.png")
    print("✓ Difference analysis plot generated")
    
    # Summary statistics
    fig4 = plot_summary_statistics(aligned_imu, ekf_data)
    figures.append(fig4)
    plot_names.append("comparison_statistics.png")
    print("✓ Summary statistics plot generated")
    
    # Save plots
    output_dir = base_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving plots to {output_dir}...")
    for fig, name in zip(figures, plot_names):
        fig.savefig(output_dir / name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {name}")
    
    # Print summary
    print(f"\n📈 Comparison Summary:")
    print(f"   IMU samples: {len(imu_df):,}")
    print(f"   EKF samples: {len(ekf_df):,}")
    print(f"   Aligned samples: {len(aligned_imu):,}")
    print(f"   Time overlap: {ekf_data['timestamp'].min():.3f}s to {ekf_data['timestamp'].max():.3f}s")
    
    print(f"\n🎉 Comparison complete!")
    print(f"📁 Check the '{output_dir}' directory for saved plots")
    
    # Show plots unless --no-show is specified
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
