//! Digital Signal Processing Filters for IMU Data
//!
//! This module provides various digital filters commonly used for IMU data preprocessing:
//! - Low-pass filters for noise reduction
//! - High-pass filters for bias removal
//! - Band-pass filters for frequency range selection
//! - Batch filtering for processing entire datasets at once
//!
//! Common filtering practices for IMU data:
//! - Accelerometer: Low-pass filter at 5-50Hz, High-pass at 0.1-1Hz
//! - Gyroscope: Low-pass filter at 10-100Hz, Band-pass at 0.1-50Hz

use std::collections::VecDeque;

/// Batch filtering functions for processing entire data arrays
pub mod batch {
    /// Apply Savitzky-Golay smoothing filter
    /// Excellent for preserving signal features while removing noise
    pub fn savitzky_golay_filter(
        data: &[f64],
        window_size: usize,
        polynomial_order: usize,
    ) -> Vec<f64> {
        if data.len() < window_size || window_size % 2 == 0 {
            return data.to_vec();
        }

        let half_window = window_size / 2;
        let mut filtered = data.to_vec();

        // Simple implementation for polynomial order 2 (quadratic)
        if polynomial_order == 2 && window_size >= 5 {
            for i in half_window..data.len() - half_window {
                let start = i - half_window;
                let end = i + half_window + 1;
                let window: Vec<f64> = data[start..end].to_vec();

                // Coefficients for 5-point quadratic Savitzky-Golay
                let coeffs = match window_size {
                    5 => vec![-3.0, 12.0, 17.0, 12.0, -3.0],
                    7 => vec![-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0],
                    9 => vec![-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0],
                    _ => (0..window_size).map(|_| 1.0).collect(), // Fallback to moving average
                };

                let sum_coeffs: f64 = coeffs.iter().sum();
                let weighted_sum: f64 = window
                    .iter()
                    .zip(coeffs.iter())
                    .map(|(val, coeff)| val * coeff)
                    .sum();

                filtered[i] = weighted_sum / sum_coeffs;
            }
        } else {
            // Fallback to moving average
            for i in half_window..data.len() - half_window {
                let start = i - half_window;
                let end = i + half_window + 1;
                let sum: f64 = data[start..end].iter().sum();
                filtered[i] = sum / window_size as f64;
            }
        }

        filtered
    }

    /// Apply median filter to remove outliers and spikes
    pub fn median_filter(data: &[f64], window_size: usize) -> Vec<f64> {
        let window_size = if window_size % 2 == 0 {
            window_size + 1
        } else {
            window_size
        };
        let half_window = window_size / 2;
        let mut filtered = data.to_vec();

        for i in half_window..data.len() - half_window {
            let start = i - half_window;
            let end = i + half_window + 1;
            let mut window: Vec<f64> = data[start..end].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            filtered[i] = window[half_window];
        }

        filtered
    }

    /// Apply moving average filter
    pub fn moving_average_filter(data: &[f64], window_size: usize) -> Vec<f64> {
        let mut filtered = Vec::with_capacity(data.len());
        let mut sum = 0.0;
        let mut count = 0;

        for (i, &value) in data.iter().enumerate() {
            sum += value;
            count += 1;

            if count > window_size {
                sum -= data[i - window_size];
                count -= 1;
            }

            filtered.push(sum / count as f64);
        }

        filtered
    }

    /// Apply zero-phase low-pass filter using forward and backward passes
    /// This eliminates phase distortion by applying the filter twice
    pub fn zero_phase_low_pass_filter(
        data: &[f64],
        cutoff_freq: f64,
        sample_rate: f64,
    ) -> Vec<f64> {
        // Calculate filter coefficient
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_freq);
        let dt = 1.0 / sample_rate;
        let alpha = dt / (rc + dt);

        // Forward pass
        let mut forward_filtered = Vec::with_capacity(data.len());
        let mut prev_output = data[0];

        for &input in data {
            let output = alpha * input + (1.0 - alpha) * prev_output;
            forward_filtered.push(output);
            prev_output = output;
        }

        // Backward pass
        let mut backward_filtered = Vec::with_capacity(data.len());
        prev_output = forward_filtered[forward_filtered.len() - 1];

        for &input in forward_filtered.iter().rev() {
            let output = alpha * input + (1.0 - alpha) * prev_output;
            backward_filtered.push(output);
            prev_output = output;
        }

        // Reverse to get correct order
        backward_filtered.reverse();
        backward_filtered
    }

    /// Apply zero-phase high-pass filter
    pub fn zero_phase_high_pass_filter(
        data: &[f64],
        cutoff_freq: f64,
        sample_rate: f64,
    ) -> Vec<f64> {
        // High-pass = Original - Low-pass
        let low_passed = zero_phase_low_pass_filter(data, cutoff_freq, sample_rate);
        data.iter()
            .zip(low_passed.iter())
            .map(|(orig, low)| orig - low)
            .collect()
    }

    /// Apply zero-phase band-pass filter
    pub fn zero_phase_band_pass_filter(
        data: &[f64],
        low_cutoff: f64,
        high_cutoff: f64,
        sample_rate: f64,
    ) -> Vec<f64> {
        let high_passed = zero_phase_high_pass_filter(data, low_cutoff, sample_rate);
        zero_phase_low_pass_filter(&high_passed, high_cutoff, sample_rate)
    }

    /// Comprehensive IMU batch filter with multiple stages
    pub fn imu_batch_filter(
        data: &[f64],
        filter_type: IMUFilterType,
        sample_rate: f64,
    ) -> Vec<f64> {
        match filter_type {
            IMUFilterType::Accelerometer => {
                // Stage 1: Remove outliers with median filter
                let median_filtered = median_filter(data, 200);

                // Stage 2: Smooth with Savitzky-Golay filter
                let smooth_filtered = savitzky_golay_filter(&median_filtered, 7, 2);

                // // Stage 3: Apply zero-phase low-pass filter only (preserve gravity for EKF)
                // EKF handles gravity compensation, so we only remove high-frequency noise
                zero_phase_low_pass_filter(&smooth_filtered, 20.0, sample_rate)
            }
            IMUFilterType::Gyroscope => {
                // Stage 1: Light outlier removal
                let median_filtered = median_filter(data, 20);

                // Stage 2: Smooth with Savitzky-Golay filter
                let smooth_filtered = savitzky_golay_filter(&median_filtered, 5, 2);

                // Stage 3: Apply zero-phase band-pass filter (remove bias drift but preserve motion)
                zero_phase_band_pass_filter(&smooth_filtered, 0.1, 50.0, sample_rate)
            }
        }
    }

    /// Comprehensive IMU batch filter with multiple stages
    pub fn imu_batch_filter_window_size(
        data: &[f64],
        filter_type: IMUFilterType,
        sample_rate: f64,
        window_size: usize,
    ) -> Vec<f64> {
        match filter_type {
            IMUFilterType::Accelerometer => {
                // Stage 1: Remove outliers with median filter
                let median_filtered = median_filter(data, window_size);

                // Stage 2: Smooth with Savitzky-Golay filter
                let smooth_filtered = savitzky_golay_filter(&median_filtered, 7, 2);

                // // Stage 3: Apply zero-phase low-pass filter only (preserve gravity for EKF)
                // EKF handles gravity compensation, so we only remove high-frequency noise
                zero_phase_low_pass_filter(&smooth_filtered, 20.0, sample_rate)
            }
            IMUFilterType::Gyroscope => {
                // Stage 1: Light outlier removal
                let median_filtered = median_filter(data, window_size);

                // Stage 2: Smooth with Savitzky-Golay filter
                let smooth_filtered = savitzky_golay_filter(&median_filtered, 5, 2);

                // Stage 3: Apply zero-phase band-pass filter (remove bias drift but preserve motion)
                zero_phase_band_pass_filter(&smooth_filtered, 0.1, 50.0, sample_rate)
            }
        }
    }

    #[derive(Clone, Copy)]
    pub enum IMUFilterType {
        Accelerometer,
        Gyroscope,
    }
}

/// First-order Butterworth low-pass filter
/// Provides smooth attenuation of high-frequency components
pub struct LowPassFilter {
    alpha: f64,
    previous_output: f64,
    initialized: bool,
}

impl LowPassFilter {
    /// Create a new low-pass filter
    ///
    /// # Arguments
    /// * `cutoff_freq` - Cutoff frequency in Hz
    /// * `sample_rate` - Sampling rate in Hz
    pub fn new(cutoff_freq: f64, sample_rate: f64) -> Self {
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_freq);
        let dt = 1.0 / sample_rate;
        let alpha = dt / (rc + dt);

        Self {
            alpha,
            previous_output: 0.0,
            initialized: false,
        }
    }

    /// Apply the filter to input data
    pub fn filter(&mut self, input: f64) -> f64 {
        if !self.initialized {
            self.previous_output = input;
            self.initialized = true;
            return input;
        }

        let output = self.alpha * input + (1.0 - self.alpha) * self.previous_output;
        self.previous_output = output;
        output
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.previous_output = 0.0;
        self.initialized = false;
    }
}

/// First-order high-pass filter
/// Removes DC bias and low-frequency drift
pub struct HighPassFilter {
    alpha: f64,
    previous_input: f64,
    previous_output: f64,
    initialized: bool,
}

impl HighPassFilter {
    /// Create a new high-pass filter
    ///
    /// # Arguments
    /// * `cutoff_freq` - Cutoff frequency in Hz
    /// * `sample_rate` - Sampling rate in Hz
    pub fn new(cutoff_freq: f64, sample_rate: f64) -> Self {
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_freq);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        Self {
            alpha,
            previous_input: 0.0,
            previous_output: 0.0,
            initialized: false,
        }
    }

    /// Apply the filter to input data
    pub fn filter(&mut self, input: f64) -> f64 {
        if !self.initialized {
            self.previous_input = input;
            self.previous_output = 0.0;
            self.initialized = true;
            return 0.0;
        }

        let output = self.alpha * (self.previous_output + input - self.previous_input);
        self.previous_input = input;
        self.previous_output = output;
        output
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.previous_input = 0.0;
        self.previous_output = 0.0;
        self.initialized = false;
    }
}

/// Band-pass filter (combination of high-pass and low-pass)
/// Allows only frequencies within a specific range to pass through
pub struct BandPassFilter {
    high_pass: HighPassFilter,
    low_pass: LowPassFilter,
}

impl BandPassFilter {
    /// Create a new band-pass filter
    ///
    /// # Arguments
    /// * `low_cutoff` - Low cutoff frequency in Hz
    /// * `high_cutoff` - High cutoff frequency in Hz
    /// * `sample_rate` - Sampling rate in Hz
    pub fn new(low_cutoff: f64, high_cutoff: f64, sample_rate: f64) -> Self {
        Self {
            high_pass: HighPassFilter::new(low_cutoff, sample_rate),
            low_pass: LowPassFilter::new(high_cutoff, sample_rate),
        }
    }

    /// Apply the filter to input data
    pub fn filter(&mut self, input: f64) -> f64 {
        let high_passed = self.high_pass.filter(input);
        self.low_pass.filter(high_passed)
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.high_pass.reset();
        self.low_pass.reset();
    }
}

/// Moving average filter for simple smoothing
/// Good for reducing noise while preserving general signal shape
pub struct MovingAverageFilter {
    window_size: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

impl MovingAverageFilter {
    /// Create a new moving average filter
    ///
    /// # Arguments
    /// * `window_size` - Number of samples to average
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }

    /// Apply the filter to input data
    pub fn filter(&mut self, input: f64) -> f64 {
        if self.buffer.len() == self.window_size {
            let old_value = self.buffer.pop_front().unwrap();
            self.sum -= old_value;
        }

        self.buffer.push_back(input);
        self.sum += input;

        self.sum / self.buffer.len() as f64
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
    }
}

/// Median filter for outlier rejection
/// Excellent for removing spikes and impulse noise
pub struct MedianFilter {
    window_size: usize,
    buffer: VecDeque<f64>,
}

impl MedianFilter {
    /// Create a new median filter
    ///
    /// # Arguments
    /// * `window_size` - Number of samples in the median window (should be odd)
    pub fn new(window_size: usize) -> Self {
        let window_size = if window_size % 2 == 0 {
            window_size + 1
        } else {
            window_size
        };
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
        }
    }

    /// Apply the filter to input data
    pub fn filter(&mut self, input: f64) -> f64 {
        if self.buffer.len() == self.window_size {
            self.buffer.pop_front();
        }

        self.buffer.push_back(input);

        if self.buffer.len() < 3 {
            return input;
        }

        let mut sorted: Vec<f64> = self.buffer.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

/// Composite IMU filter for comprehensive signal conditioning
/// Combines multiple filtering stages optimized for IMU data
pub struct IMUFilter {
    median_filter: MedianFilter,
    low_pass_filter: LowPassFilter,
    // high_pass_filter: Option<HighPassFilter>,
}

impl IMUFilter {
    /// Create a new IMU filter optimized for accelerometer data
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling rate in Hz
    pub fn new_accelerometer(sample_rate: f64) -> Self {
        Self {
            median_filter: MedianFilter::new(11), // Remove spikes
            low_pass_filter: LowPassFilter::new(20.0, sample_rate), // Remove high-freq noise
                                                  // high_pass_filter: Some(HighPassFilter::new(0.5, sample_rate)), // Remove DC drift
        }
    }

    /// Create a new IMU filter optimized for gyroscope data
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling rate in Hz
    pub fn new_gyroscope(sample_rate: f64) -> Self {
        Self {
            median_filter: MedianFilter::new(11), // Light spike removal
            low_pass_filter: LowPassFilter::new(50.0, sample_rate), // Remove high-freq noise
                                                  // high_pass_filter: Some(HighPassFilter::new(0.1, sample_rate)), // Remove bias drift
        }
    }

    /// Create a new IMU filter with custom parameters
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling rate in Hz
    /// * `low_pass_cutoff` - Low-pass filter cutoff frequency in Hz
    /// * `high_pass_cutoff` - Optional high-pass filter cutoff frequency in Hz
    /// * `median_window` - Median filter window size
    pub fn new_custom(
        sample_rate: f64,
        low_pass_cutoff: f64,
        high_pass_cutoff: Option<f64>,
        median_window: usize,
    ) -> Self {
        Self {
            median_filter: MedianFilter::new(median_window),
            low_pass_filter: LowPassFilter::new(low_pass_cutoff, sample_rate),
            // high_pass_filter: high_pass_cutoff.map(|freq| HighPassFilter::new(freq, sample_rate)),
        }
    }

    /// Apply the complete filter chain to input data
    pub fn filter(&mut self, input: f64) -> f64 {
        // Stage 1: Remove spikes
        let median_filtered = self.median_filter.filter(input);

        // Stage 2: Remove high-frequency noise
        let low_pass_filtered = self.low_pass_filter.filter(median_filtered);

        // // Stage 3: Remove DC bias/drift (optional)
        low_pass_filtered
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        self.median_filter.reset();
        self.low_pass_filter.reset();
        // if let Some(ref mut hpf) = self.high_pass_filter {
        //     hpf.reset();
        // }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_pass_filter() {
        let mut filter = LowPassFilter::new(10.0, 100.0);

        // Test with step input - note that first output equals input
        let result1 = filter.filter(1.0); // First call returns input directly
        let result2 = filter.filter(1.0);
        let result3 = filter.filter(1.0);

        // Output should eventually approach input
        assert_eq!(result1, 1.0); // First output equals input due to initialization
        assert!(result3 >= result2); // Output should increase or stay same towards input
        assert!(result3 <= 1.0); // Output should not exceed input
    }

    #[test]
    fn test_high_pass_filter() {
        let mut filter = HighPassFilter::new(1.0, 100.0);

        // Test with DC input
        for _ in 0..10 {
            filter.filter(1.0);
        }
        let dc_result = filter.filter(1.0);

        // DC component should be heavily attenuated
        assert!(dc_result.abs() < 0.1);
    }

    #[test]
    fn test_median_filter() {
        let mut filter = MedianFilter::new(5);

        // Test with spike
        filter.filter(1.0);
        filter.filter(1.0);
        let _result = filter.filter(100.0); // Spike
        filter.filter(1.0);
        let filtered = filter.filter(1.0);

        // Spike should be suppressed
        assert!(filtered < 10.0);
    }

    #[test]
    fn test_imu_filter() {
        let mut filter = IMUFilter::new_accelerometer(100.0);

        // Test basic functionality
        let result = filter.filter(9.81); // Gravity
        assert!(result.is_finite());
    }
}

#[cfg(test)]
mod batch_tests {
    use super::batch::*;

    #[test]
    fn test_median_filter() {
        let data = vec![1.0, 1.0, 10.0, 1.0, 1.0]; // Data with spike
        let filtered = median_filter(&data, 3);
        // The spike should be suppressed
        assert!(filtered[2] < 5.0);
    }

    #[test]
    fn test_moving_average_filter() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let filtered = moving_average_filter(&data, 3);
        assert_eq!(filtered.len(), data.len());
        // Should smooth the data
        assert!(filtered.last().unwrap() > &1.0);
    }

    #[test]
    fn test_savitzky_golay_filter() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let filtered = savitzky_golay_filter(&data, 5, 2);
        assert_eq!(filtered.len(), data.len());
        // Should preserve general shape while smoothing
        assert!(filtered[4] > 3.0); // Peak should still be around the middle
    }

    #[test]
    fn test_zero_phase_filters() {
        let data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let sample_rate = 10.0;

        let low_passed = zero_phase_low_pass_filter(&data, 1.0, sample_rate);
        let high_passed = zero_phase_high_pass_filter(&data, 1.0, sample_rate);
        let band_passed = zero_phase_band_pass_filter(&data, 0.5, 2.0, sample_rate);

        assert_eq!(low_passed.len(), data.len());
        assert_eq!(high_passed.len(), data.len());
        assert_eq!(band_passed.len(), data.len());
    }

    #[test]
    fn test_imu_batch_filter() {
        let data = vec![1.0, 2.0, 10.0, 2.0, 1.0, 2.0, 1.0]; // Data with spike
        let sample_rate = 100.0;

        let accel_filtered = imu_batch_filter(&data, IMUFilterType::Accelerometer, sample_rate);
        let gyro_filtered = imu_batch_filter(&data, IMUFilterType::Gyroscope, sample_rate);

        assert_eq!(accel_filtered.len(), data.len());
        assert_eq!(gyro_filtered.len(), data.len());

        // Spike should be reduced
        assert!(accel_filtered[2] < data[2]);
        assert!(gyro_filtered[2] < data[2]);
    }
}
