use anyhow::Result;
use hmm_anomaly::{plot_anomalies, CliInputs, CpuUtilizationEntry, CsvReader, Hmm, TrainingData};
use ndarray::Array2;

// No of iterations in the gibbs_sampling
const NUM_ITER: usize = 500;
const BURN_IN: usize = 50;

fn main() -> Result<()> {
    // This specific example tries to spot the anomalies in a cpu utilization metric data.
    // We categorise the cpu utilization data as below:
    // Three State: 0 = Normal (0% - 30%), 1 = Warn (30% - 60%), 2 = Critical (>= 60%)
    let n_states = 3usize;
    // No of features in the dataset - only one field in the dataset, the cpu utilization percent
    let n_features: usize = 1;
    let (training_data_file, input_data_file) = CliInputs::read_cli()?;

    // INPUT FILE
    let mut input_data: TrainingData<CpuUtilizationEntry> = TrainingData::default();
    let _ = input_data.read_csv(input_data_file);
    let test_input = input_data
        .records
        .iter()
        .map(|x| x.utilization)
        .collect::<Vec<_>>();
    let test_input = Array2::from_shape_vec((test_input.len(), 1), test_input)?;

    // TRAINING FILE
    let mut train_data: TrainingData<CpuUtilizationEntry> = TrainingData::default();
    let _ = train_data.read_csv(training_data_file);
    let observations = train_data
        .records
        .iter()
        .map(|x| x.utilization)
        .collect::<Vec<_>>();
    let observations = Array2::from_shape_vec((observations.len(), 1), observations)?;
    let n_obs = observations.nrows();
    let mut hmm = Hmm::new(n_features, n_states, n_obs);
    let (transition_samples, emission_means_samples, emission_variances_samples) =
        hmm.learn_gibbs_sampling(&observations, NUM_ITER, BURN_IN)?;

    // check anomalies
    let scores = hmm.anomalies(&input_data)?;
    // let score = compute_anomaly_score(
    //     &test_input,
    //     n_states,
    //     transition_samples,
    //     emission_means_samples,
    //     emission_variances_samples,
    //     BURN_IN,
    // )?;
    // println!("Anomaly score: {}", score);
    plot_anomalies(scores, 8.0)?;

    Ok(())
}
