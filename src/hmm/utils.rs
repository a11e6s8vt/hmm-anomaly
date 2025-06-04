use ndarray::Array1;
use rand::prelude::*;
use rand::rng;
use rand_distr::{Distribution, Gamma, LogNormal, Normal};
use statrs::function::erf::erfc;

pub fn log_sum_exp(log_probs: &ndarray::Array1<f64>) -> f64 {
    let max = log_probs.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp = log_probs.mapv(|x| (x - max).exp()).sum();
    max + sum_exp.ln()
}

pub fn normalize_log_probs(log_probs: &Array1<f64>) -> Array1<f64> {
    // Step 1: Subtract max log-value to avoid overflow
    let max_log = log_probs.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let shifted_logs = log_probs.mapv(|x| x - max_log);

    // Step 2: Exponentiate and normalize
    let exps = shifted_logs.mapv(|x| x.exp());
    let sum = exps.sum();
    exps / sum
}

pub fn censored_gamma_samples(k: f64, theta: f64, n: usize) -> Vec<f64> {
    let mut rng = rng();
    let gamma = Gamma::new(k, theta).unwrap();
    (0..n)
        .map(|_| {
            let mut sample = gamma.sample(&mut rng);
            if sample <= 1.0 {
                sample = 1.0 + f64::EPSILON;
            }
            sample
        })
        .collect()
}

pub fn normal_samples_above_1(mean: f64, std_dev: f64, n: usize) -> Vec<f64> {
    let mut rng = rng();
    let normal = Normal::new(mean, std_dev).unwrap();
    let mut samples = Vec::with_capacity(n);

    while samples.len() < n {
        let sample = normal.sample(&mut rng);
        if sample > 1.0 {
            samples.push(sample);
        }
    }

    samples
}

pub fn truncated_normal_above_1(mean: f64, std_dev: f64, n: usize) -> Vec<f64> {
    let mut rng = rng();
    let normal = Normal::new(mean, std_dev).unwrap();
    // Compute the CDF at 1.0 using statrs
    let z = (1.0 - mean) / std_dev;
    let cdf_1 = 0.5 * erfc(-z / std::f64::consts::SQRT_2);

    (0..n)
        .map(|_| {
            // Generate uniform in [cdf(1), 1.0]
            let u = rng.gen_range(cdf_1..1.0);
            // Inverse CDF transform
            mean + std_dev * (-2.0f64).sqrt() * erf_inv(2.0 * u - 1.0)
        })
        .collect()
}

// Approximate inverse error function
fn erf_inv(x: f64) -> f64 {
    let a = 8.0 * (std::f64::consts::PI - 3.0)
        / (3.0 * std::f64::consts::PI * (4.0 - std::f64::consts::PI));
    let y = x.abs();
    let z = (-(y * y) + a * y * y / (1.0 - y * y)).ln().sqrt();
    z.copysign(x)
}

pub fn shifted_lognormal(mean: f64, std_dev: f64, n: usize) -> Vec<f64> {
    let mut rng = rng();
    // Adjust parameters to approximate desired mean/stddev
    let lognormal = LogNormal::new(mean.ln(), std_dev.ln()).unwrap();
    (0..n).map(|_| lognormal.sample(&mut rng)).collect()
}
