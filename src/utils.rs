use ndarray::Array1;

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
