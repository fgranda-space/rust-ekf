use rust_ekf::data_utils::read_space_separated_file;
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

    let data_size = Some(53952);

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
    let timestamps_accel: Vec<f64> = timestamps_accel_raw
        .into_iter()
        .map(|ts| (ts - first_timestamp_accel) as f64 / 1_000_000.0)
        .collect();

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
}
