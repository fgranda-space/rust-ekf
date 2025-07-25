use polars::prelude::*;
use rust_ekf::data_utils::{read_space_separated_file, save_csv};
use rust_ekf::filters::batch;
use rust_ekf::{EKF, GRAVITY};
use std::f64;

/// Convert quaternion to Euler angles (roll, pitch, yaw) in radians
/// q = [w, x, y, z] where w is the scalar part
fn quaternion_to_euler(q: &[f64; 4]) -> [f64; 3] {
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];

    // Roll (x-axis rotation)
    let sin_r_cp = 2.0 * (w * x + y * z);
    let cos_r_cp = 1.0 - 2.0 * (x * x + y * y);
    let roll = sin_r_cp.atan2(cos_r_cp);

    // Pitch (y-axis rotation)
    let sin_p = 2.0 * (w * y - z * x);
    let pitch = if sin_p.abs() >= 1.0 {
        f64::consts::PI / 2.0 * sin_p.signum()
    } else {
        sin_p.asin()
    };

    // Yaw (z-axis rotation)
    let sin_y_cp = 2.0 * (w * z + x * y);
    let cos_y_cp = 1.0 - 2.0 * (y * y + z * z);
    let yaw = sin_y_cp.atan2(cos_y_cp);

    [roll, pitch, yaw]
}

fn main() {
    // Fields to read from the CSV file
    // Read bin files for individual sensors
    // Gyroscope data
    let gyro_data_path = "data/Flight01/20250428_150955/sensors_data/Inertial/gyroscope.bin";
    let gyroscope_fields = ["timestamp", "Gyr_X", "Gyr_Y", "Gyr_Z"];

    let num_rows = 53952; // Total number of rows in the dataset
    let data_size = Some(num_rows);

    let df_gyro_complete = read_space_separated_file(gyro_data_path, &gyroscope_fields, data_size)
        .expect("Failed to read gyroscope data");

    // Print the size of the gyroscope data
    println!("Gyroscope Data Size: {}", df_gyro_complete.height());

    // Accelerometer data
    let accel_data_path = "data/Flight01/20250428_150955/sensors_data/Inertial/accelerometer.bin";
    let accelerometer_fields = ["timestamp", "Acc_X", "Acc_Y", "Acc_Z"];

    let df_accel_complete =
        read_space_separated_file(accel_data_path, &accelerometer_fields, data_size)
            .expect("Failed to read accelerometer data");

    // Print the size of the accelerometer data
    println!("Accelerometer Data Size: {}", df_accel_complete.height());

    // Extract all data into vectors for batch processing
    let timestamps_gyro_raw: Vec<i64> = df_gyro_complete
        .column("timestamp")
        .unwrap()
        .i64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap_or(0))
        .collect();

    let first_timestamp_gyro = timestamps_gyro_raw[0];
    let timestamps_gyro: Vec<f64> = timestamps_gyro_raw
        .into_iter()
        .map(|ts| (ts - first_timestamp_gyro) as f64 / 1_000_000.0)
        .collect();

    // Extract all data into vectors for batch processing
    let timestamps_accel_raw: Vec<i64> = df_accel_complete
        .column("timestamp")
        .unwrap()
        .i64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap_or(0))
        .collect();

    let first_timestamp_accel = timestamps_accel_raw[0];
    let _timestamps_accel: Vec<f64> = timestamps_accel_raw
        .into_iter()
        .map(|ts| (ts - first_timestamp_accel) as f64 / 1_000_000.0)
        .collect();

    let timestamps = timestamps_gyro;

    let accel_x_raw: Vec<f64> = df_accel_complete
        .column("Acc_X")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap_or(0.0) * GRAVITY)
        .collect();

    let accel_y_raw: Vec<f64> = df_accel_complete
        .column("Acc_Y")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap_or(0.0) * GRAVITY)
        .collect();

    let accel_z_raw: Vec<f64> = df_accel_complete
        .column("Acc_Z")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap_or(0.0) * GRAVITY)
        .collect();

    let gyro_x_raw: Vec<f64> = df_gyro_complete
        .column("Gyr_X")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap_or(0.0).to_radians())
        .collect();

    let gyro_y_raw: Vec<f64> = df_gyro_complete
        .column("Gyr_Y")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap_or(0.0).to_radians())
        .collect();

    let gyro_z_raw: Vec<f64> = df_gyro_complete
        .column("Gyr_Z")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap_or(0.0).to_radians())
        .collect();

    // Calculate sampling rate from timestamps
    let dt_avg = if timestamps.len() > 1 {
        (timestamps[timestamps.len() - 1] - timestamps[0]) / (timestamps.len() - 1) as f64
    } else {
        1.0 / 150.0 // Default to 150 Hz
    };
    let sample_rate = 1.0 / dt_avg;
    println!("Detected sampling rate: {:.1} Hz", sample_rate);

    // Apply batch filtering to all IMU data
    println!("Applying batch filtering...");

    let accel_x_filtered = batch::imu_batch_filter_window_size(
        &accel_x_raw,
        batch::IMUFilterType::Accelerometer,
        sample_rate,
        251,
    );
    let accel_y_filtered = batch::imu_batch_filter_window_size(
        &accel_y_raw,
        batch::IMUFilterType::Accelerometer,
        sample_rate,
        126,
    );
    let accel_z_filtered = batch::imu_batch_filter_window_size(
        &accel_z_raw,
        batch::IMUFilterType::Accelerometer,
        sample_rate,
        252,
    );

    let gyro_x_filtered = batch::imu_batch_filter_window_size(
        &gyro_x_raw,
        batch::IMUFilterType::Gyroscope,
        sample_rate,
        38,
    );
    let gyro_y_filtered = batch::imu_batch_filter_window_size(
        &gyro_y_raw,
        batch::IMUFilterType::Gyroscope,
        sample_rate,
        35,
    );
    let gyro_z_filtered = batch::imu_batch_filter_window_size(
        &gyro_z_raw,
        batch::IMUFilterType::Gyroscope,
        sample_rate,
        25,
    );

    // let accel_x_filtered = accel_x_raw.clone();
    // let accel_y_filtered = accel_y_raw.clone();
    // let accel_z_filtered = accel_z_raw.clone();

    // let gyro_x_filtered = gyro_x_raw.clone();
    // let gyro_y_filtered = gyro_y_raw.clone();
    // let gyro_z_filtered = gyro_z_raw.clone();

    println!("Batch filtering completed!");

    // Show comparison of raw vs filtered data for first few samples
    println!("\nComparison of raw vs filtered data (first 3 samples):");
    for i in 0..3.min(timestamps.len()) {
        println!(
            "Sample {}: Raw Accel: [{:.3}, {:.3}, {:.3}] -> Filtered: [{:.3}, {:.3}, {:.3}]",
            i,
            accel_x_raw[i],
            accel_y_raw[i],
            accel_z_raw[i],
            accel_x_filtered[i],
            accel_y_filtered[i],
            accel_z_filtered[i]
        );
        println!(
            "Sample {}: Raw Gyro: [{:.3}, {:.3}, {:.3}] -> Filtered: [{:.3}, {:.3}, {:.3}]",
            i,
            gyro_x_raw[i],
            gyro_y_raw[i],
            gyro_z_raw[i],
            gyro_x_filtered[i],
            gyro_y_filtered[i],
            gyro_z_filtered[i]
        );
    }

    // Now run EKF with filtered data
    println!("\nRunning EKF with filtered data...");
    let mut time_stamp_prev = 0.0;
    let mut ekf = EKF::new(None);

    // Prepare to collect the quaternion, euler angles, gyro, accel, and timestamp
    let mut state_matrix: Vec<[f64; 14]> = Vec::with_capacity(num_rows);

    // Skip first sample to have proper dt calculation
    for i in 1..timestamps.len() {
        let timestamp = timestamps[i];
        let dt = timestamp - time_stamp_prev;

        // Prepare filtered gyro and accel data
        let gyro_data = [gyro_x_filtered[i], gyro_y_filtered[i], gyro_z_filtered[i]];
        let accel_data = [
            accel_x_filtered[i],
            accel_y_filtered[i],
            accel_z_filtered[i],
        ];

        // Update the EKF
        ekf.predict(gyro_data, dt);
        ekf.update(accel_data);

        // Get the updated state vector
        let state = ekf.get_state();

        // Convert quaternion to Euler angles
        let quat = [state[0], state[1], state[2], state[3]];
        let euler = quaternion_to_euler(&quat);

        // Collect quaternion, euler angles, gyro data, accel data, and timestamp
        let mut row = [0.0; 14];
        // Quaternion components
        for j in 0..4 {
            row[j] = state[j];
        }
        // Euler angles (roll, pitch, yaw)
        row[4] = euler[0]; // roll
        row[5] = euler[1]; // pitch
        row[6] = euler[2]; // yaw

        // Gyro data (filtered)
        row[7] = gyro_data[0];
        row[8] = gyro_data[1];
        row[9] = gyro_data[2];

        // Accel data (filtered)
        row[10] = accel_data[0];
        row[11] = accel_data[1];
        row[12] = accel_data[2];

        // Timestamp
        row[13] = timestamp;
        state_matrix.push(row);

        // Update the previous timestamp
        time_stamp_prev = timestamp;
    }

    // Print the final state matrix element
    println!("Final state matrix: {:?}", state_matrix.last());

    // After the loop, create a DataFrame from the collected state_matrix
    let col_names = [
        "q_0",
        "q_1",
        "q_2",
        "q_3",
        "roll",
        "pitch",
        "yaw",
        "g_x",
        "g_y",
        "g_z",
        "a_x",
        "a_y",
        "a_z",
        "timestamp",
    ];
    let state_cols: Vec<Column> = (0..14)
        .map(|j| {
            let col: Vec<f64> = state_matrix.iter().map(|row| row[j]).collect();
            let series = Series::new(PlSmallStr::from(col_names[j]), col);
            Column::from(series)
        })
        .collect();
    let state_df = DataFrame::new(state_cols).expect("Failed to create state DataFrame");

    // Save to CSV
    save_csv(&mut state_df.clone(), "data/ekf_state_batch_filtered.csv")
        .expect("Failed to save EKF state CSV");
    println!("Results saved to 'data/ekf_state_batch_filtered.csv'");
}
