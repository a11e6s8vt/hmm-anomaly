use crate::CpuUtilizationEntry;
use crate::TrainingData;
use ndarray::Array2;

pub trait GibbsSampler {
    fn sample_latent_states(&mut self, observations: &Array2<f64>) -> anyhow::Result<Vec<usize>>;
    fn sample_transition_probs(&mut self, states: &[usize]);
    fn sample_emission_params(
        &mut self,
        observations: &Array2<f64>,
        states: &[usize],
    ) -> anyhow::Result<()>;
}

pub trait Bayesian: GibbsSampler {
    /// Gibbs Sampling
    /// i/p: PDF for X= {x_1,···,x_L} (We will use Gaussian)
    /// o/p: HMM, M= {A,B,C, π}
    ///     where:
    ///         A = transition_probs_samples,
    ///         B = emission_means_samples,
    ///         C = emission_variance_samples,
    ///         π = initial probabilities
    fn learn_gibbs_sampling(
        &mut self,
        observations: &Array2<f64>,
        num_iter: usize,
        burn_in: usize,
    ) -> anyhow::Result<()>;
}

pub trait AnalyticsEngine {
    fn anomalies(
        &self,
        test_data: &TrainingData<CpuUtilizationEntry>,
    ) -> anyhow::Result<Vec<(String, f64, f64)>>;
}
