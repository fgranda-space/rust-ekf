#!/usr/bin/env python3
"""
EKF Data Visualization Script

This script reads the EKF state batch filtered data and creates comprehensive plots
for quaternions, Euler angles, gyroscope data, accelerometer data, and more.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set the style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(csv_path):
    """Load the EKF data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} data points from {csv_path}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_quaternions(df):
    """Plot quaternion components over time."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Quaternion Components Over Time', fontsize=16, fontweight='bold')
    
    quaternions = ['q_0', 'q_1', 'q_2', 'q_3']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (q, color) in enumerate(zip(quaternions, colors)):
        row, col = i // 2, i % 2
        axes[row, col].plot(df['timestamp'], df[q], color=color, linewidth=1.5, alpha=0.8)
        axes[row, col].set_title(f'{q} ({"w" if q == "q_0" else q[-1]} component)', fontweight='bold')
        axes[row, col].set_xlabel('Time (s)')
        axes[row, col].set_ylabel('Value')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    return fig

def plot_euler_angles(df):
    """Plot Euler angles (roll, pitch, yaw) over time."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Euler Angles Over Time', fontsize=16, fontweight='bold')
    
    euler_angles = ['roll', 'pitch', 'yaw']
    colors = ['red', 'green', 'blue']
    labels = ['Roll', 'Pitch', 'Yaw']
    
    for i, (angle, color, label) in enumerate(zip(euler_angles, colors, labels)):
        # Convert to degrees for better understanding
        angle_deg = np.degrees(df[angle])
        
        axes[i].plot(df['timestamp'], angle_deg, color=color, linewidth=1.5, alpha=0.8)
        axes[i].set_title(f'{label} Angle', fontweight='bold')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Angle (degrees)')
        axes[i].grid(True, alpha=0.3)
        
        axes[i].legend()
    
    plt.tight_layout()
    return fig

def plot_gyroscope_data(df):
    """Plot gyroscope data over time."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Gyroscope Data Over Time', fontsize=16, fontweight='bold')
    
    gyro_axes = ['g_x', 'g_y', 'g_z']
    colors = ['red', 'green', 'blue']
    labels = ['X-axis', 'Y-axis', 'Z-axis']
    
    for i, (axis, color, label) in enumerate(zip(gyro_axes, colors, labels)):
        # Convert to degrees per second
        gyro_dps = np.degrees(df[axis])
        
        axes[i].plot(df['timestamp'], gyro_dps, color=color, linewidth=1, alpha=0.8)
        axes[i].set_title(f'Gyroscope {label}', fontweight='bold')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Angular Velocity (deg/s)')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(gyro_dps)
        std_val = np.std(gyro_dps)
        axes[i].axhline(y=mean_val, color=color, linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}°/s')
        axes[i].legend()
    
    plt.tight_layout()
    return fig

def plot_accelerometer_data(df):
    """Plot accelerometer data over time."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Accelerometer Data Over Time', fontsize=16, fontweight='bold')
    
    accel_axes = ['a_x', 'a_y', 'a_z']
    colors = ['red', 'green', 'blue']
    labels = ['X-axis', 'Y-axis', 'Z-axis']
    
    for i, (axis, color, label) in enumerate(zip(accel_axes, colors, labels)):
        axes[i].plot(df['timestamp'], df[axis], color=color, linewidth=1, alpha=0.8)
        axes[i].set_title(f'Accelerometer {label}', fontweight='bold')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Acceleration (m/s²)')
        axes[i].grid(True, alpha=0.3)
        
        # Add gravity reference line for z-axis
        if axis == 'a_z':
            axes[i].axhline(y=9.81, color='black', linestyle=':', alpha=0.7, label='Gravity (9.81 m/s²)')
        
        # Add statistics
        mean_val = np.mean(df[axis])
        std_val = np.std(df[axis])
        axes[i].axhline(y=mean_val, color=color, linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f} m/s²')
        axes[i].legend()
    
    plt.tight_layout()
    return fig

def plot_3d_orientation(df):
    """Plot 3D orientation visualization using quaternions."""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot quaternion magnitude over time
    ax1 = plt.subplot(131)
    q_magnitude = np.sqrt(df['q_0']**2 + df['q_1']**2 + df['q_2']**2 + df['q_3']**2)
    ax1.plot(df['timestamp'], q_magnitude, color='purple', linewidth=1.5)
    ax1.set_title('Quaternion Magnitude', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Unit quaternion')
    ax1.legend()
    
    # Plot 3D trajectory in Euler space
    ax2 = plt.subplot(132, projection='3d')
    roll_deg = np.degrees(df['roll'])
    pitch_deg = np.degrees(df['pitch'])
    yaw_deg = np.degrees(df['yaw'])
    
    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    ax2.scatter(roll_deg, pitch_deg, yaw_deg, c=colors, s=1, alpha=0.6)
    ax2.set_xlabel('Roll (degrees)')
    ax2.set_ylabel('Pitch (degrees)')
    ax2.set_zlabel('Yaw (degrees)')
    ax2.set_title('3D Orientation Trajectory', fontweight='bold')
    
    # Plot combined Euler angles
    ax3 = plt.subplot(133)
    ax3.plot(df['timestamp'], np.degrees(df['roll']), label='Roll', alpha=0.7)
    ax3.plot(df['timestamp'], np.degrees(df['pitch']), label='Pitch', alpha=0.7)
    ax3.plot(df['timestamp'], np.degrees(df['yaw']), label='Yaw', alpha=0.7)
    ax3.set_title('All Euler Angles', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (degrees)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_sensor_correlation(df):
    """Plot correlation matrix and sensor relationships."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Correlation matrix
    sensor_columns = ['g_x', 'g_y', 'g_z', 'a_x', 'a_y', 'a_z', 'roll', 'pitch', 'yaw']
    corr_matrix = df[sensor_columns].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[0], fmt='.2f')
    axes[0].set_title('Sensor Data Correlation Matrix', fontweight='bold')
    
    # Gyro vs Accelerometer magnitude
    gyro_mag = np.sqrt(df['g_x']**2 + df['g_y']**2 + df['g_z']**2)
    accel_mag = np.sqrt(df['a_x']**2 + df['a_y']**2 + df['a_z']**2)
    
    axes[1].scatter(gyro_mag, accel_mag, alpha=0.5, s=1)
    axes[1].set_xlabel('Gyroscope Magnitude (rad/s)')
    axes[1].set_ylabel('Accelerometer Magnitude (m/s²)')
    axes[1].set_title('Gyro vs Accelerometer Magnitude', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_data_statistics(df):
    """Plot statistical summary of the data."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Statistics and Quality', fontsize=16, fontweight='bold')
    
    # Sampling rate analysis
    ax1 = axes[0, 0]
    dt = np.diff(df['timestamp'])
    ax1.hist(dt, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Sampling Rate Distribution', fontweight='bold')
    ax1.set_xlabel('Time Difference (s)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(x=np.mean(dt), color='red', linestyle='--', label=f'Mean: {np.mean(dt):.6f}s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Quaternion components distribution
    ax2 = axes[0, 1]
    for q in ['q_0', 'q_1', 'q_2', 'q_3']:
        ax2.hist(df[q], bins=30, alpha=0.5, label=q)
    ax2.set_title('Quaternion Components Distribution', fontweight='bold')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Data timeline
    ax3 = axes[1, 0]
    ax3.plot(df['timestamp'], range(len(df)), color='green')
    ax3.set_title('Data Timeline', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Sample Index')
    ax3.grid(True, alpha=0.3)
    
    # Missing data check
    ax4 = axes[1, 1]
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_data.plot(kind='bar', ax=ax4, color='red')
        ax4.set_title('Missing Data by Column', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=16, fontweight='bold', color='green')
        ax4.set_title('Data Quality Check', fontweight='bold')
    ax4.set_ylabel('Missing Count')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run all plotting functions."""
    # Path to the CSV file
    csv_path = Path(__file__).parent / "data" / "ekf_state_batch_filtered.csv"
    
    # Load data
    df = load_data(csv_path)
    if df is None:
        return
    
    print(f"\nData summary:")
    print(f"Time range: {df['timestamp'].min():.3f}s to {df['timestamp'].max():.3f}s")
    print(f"Duration: {df['timestamp'].max() - df['timestamp'].min():.3f}s")
    print(f"Average sampling rate: {len(df) / (df['timestamp'].max() - df['timestamp'].min()):.1f} Hz")
    
    # Create all plots
    figures = []
    
    print("\nGenerating plots...")
    
    figures.append(plot_quaternions(df))
    print("✓ Quaternion plots generated")
    
    figures.append(plot_euler_angles(df))
    print("✓ Euler angle plots generated")
    
    figures.append(plot_gyroscope_data(df))
    print("✓ Gyroscope plots generated")
    
    figures.append(plot_accelerometer_data(df))
    print("✓ Accelerometer plots generated")
    
    figures.append(plot_3d_orientation(df))
    print("✓ 3D orientation plots generated")
    
    figures.append(plot_sensor_correlation(df))
    print("✓ Sensor correlation plots generated")
    
    figures.append(plot_data_statistics(df))
    print("✓ Data statistics plots generated")
    
    # Save all plots
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    plot_names = [
        "quaternions.png",
        "euler_angles.png", 
        "gyroscope_data.png",
        "accelerometer_data.png",
        "3d_orientation.png",
        "sensor_correlation.png",
        "data_statistics.png"
    ]
    
    print(f"\nSaving plots to {output_dir}...")
    for fig, name in zip(figures, plot_names):
        fig.savefig(output_dir / name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {name}")
    
    print(f"\n🎉 All plots generated successfully!")
    print(f"📁 Check the '{output_dir}' directory for saved plots")
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
