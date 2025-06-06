//
// TODO: ZIG (Zero-Inflated Gamma) for handling mostly zero observations
// with positve continuous values
//
use crate::traits::{AnalyticsEngine, Bayesian, GibbsSampler};
use crate::utils::{log_sum_exp, normalize_log_probs};
use crate::{censored_gamma_samples, shifted_lognormal};
use ndarray::{Array1, Array2, Array3, Axis};
use polars::frame::DataFrame;

use rand_distr::{Dirichlet, Distribution, Gamma, Normal};
use statrs::distribution::{Continuous, Normal as StatNormal};

#[derive(Clone, Debug)]
pub enum Means {
    Univariate(Array1<f64>),
    Multivariate(Array2<f64>),
}

#[derive(Clone, Debug)]
pub enum Covariance {
    Univariate(Array1<f64>),
    Multivariate(Array3<f64>),
}

#[derive(Debug)]
pub struct Hmm {
    // Number of states
    n_states: usize,

    // Priors for the states - π
    init_probs: Array1<f64>,

    // A:
    transition_probs: Array2<f64>,

    // B: (mu_k)
    emission_means: Means,

    // C: (sigma_k)
    // Diagonal Covariance (Uncorrelated emissions)
    emission_variance: Covariance,
}

impl Hmm {
    pub fn new(
        n_features: usize,
        n_states: usize,
        s_mean: f64,
        s_variance: f64,
        shape: f64,
        scale: f64,
    ) -> Self {
        let init_probs: Array1<f64> = Array1::ones(n_states) / n_states as f64;
        let init_probs = init_probs.mapv(|p| p.ln());

        let transition_probs: Array2<f64> = Array2::ones((n_states, n_states)) / n_states as f64;
        let transition_probs = transition_probs.mapv(|p| p.ln());

        let emission_means: Means = if n_features == 1 {
            let samples: Vec<f64> = shifted_lognormal(s_mean, s_variance.sqrt(), n_states);
            let emission_means = Array1::from(samples);
            Means::Univariate(emission_means.iter().map(|p| p.ln()).collect::<Array1<_>>())
        } else {
            unimplemented!()
        };

        let emission_variance: Covariance = if n_features == 1 {
            let samples: Vec<f64> = censored_gamma_samples(shape, scale, n_states);
            let emission_variance = Array1::from(samples);
            Covariance::Univariate(
                emission_variance
                    .iter()
                    .map(|p| p.ln())
                    .collect::<Array1<_>>(),
            )
        } else {
            unimplemented!()
        };

        Self {
            n_states,
            init_probs,
            transition_probs,
            emission_means,
            emission_variance,
            // hidden_states,
        }
    }

    pub fn print_model_params(&self) {
        println!("Number of hidden states: {:?}", self.n_states);
        println!("Initial probabilities for states: {:?}", self.init_probs);
        println!(
            "Initial transition probabilities: {:?}",
            self.transition_probs
        );
        println!("Initial emission means: {:?}", self.emission_means);
        println!("Initial emission variances: {:?}", self.emission_variance);
    }
}

impl Bayesian for Hmm {
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
    ) -> anyhow::Result<()> {
        let mut transition_probs_samples: Vec<Array2<f64>> = Vec::new();
        let mut emission_means_samples: Vec<crate::Means> = Vec::new();
        let mut emission_variances_samples: Vec<crate::Covariance> = Vec::new();
        let mut latent_states = Vec::new();

        println!("num_iter = {}, burn_in = {}", num_iter, burn_in);
        // burn_in: we discard the samples until the Markov chain reaches its
        // stationary distribution. It's the initial phase of the Gibbs Sampling
        for index in 0..num_iter {
            let states = self.sample_latent_states(observations)?;
            self.sample_transition_probs(&states);
            self.sample_emission_params(observations, &states)?;
            if index >= burn_in {
                transition_probs_samples.push(self.transition_probs.to_owned());
                emission_means_samples.push(self.emission_means.clone());
                emission_variances_samples.push(self.emission_variance.clone());
                latent_states.push(states);
            }
        }

        Ok(())
    }
}

impl GibbsSampler for Hmm {
    fn sample_latent_states(&mut self, observations: &Array2<f64>) -> anyhow::Result<Vec<usize>> {
        /*
         * Sample hidden states
         */
        // Forward-Backward Algorithm
        //  - used for inferencing P(z_k|x_1:num_obs)
        let n_obs = observations.nrows();
        let mut states = vec![0; n_obs];
        let mut rng = rand::rng();

        // Forward Step
        //
        // alpha_t(i) = probability of seeing observations x_1,...,x_t and ending at state q_t = S_i
        // [[0, 0, 0], [0, 0, 0], ...] - for three states
        let mut alpha: Array2<f64> = Array2::zeros((n_obs, self.n_states));

        //
        // Base Case, t = 0
        //
        let row = observations.row(0);
        for k in 0..self.n_states {
            // assuming random variable X has only a single value at time step t
            let mu = match &self.emission_means {
                Means::Univariate(arr) => arr[k],
                Means::Multivariate(_) => {
                    unimplemented!()
                }
            };
            let sigma = match &self.emission_variance {
                Covariance::Univariate(arr) => arr[k],
                Covariance::Multivariate(_) => {
                    unimplemented!()
                }
            };
            let normal = StatNormal::new(mu, sigma.sqrt())?;
            let log_likelihood = normal.ln_pdf(row[0]);
            // log(a.b) = log(a) + log(b)
            alpha[[0, k]] = self.init_probs[k] + log_likelihood;
        }

        // Normalise alpha
        // let row_sum = alpha.row(0).sum();
        // let mut row = alpha.row_mut(0);
        // row.mapv_inplace(|x| x / row_sum);
        let row = alpha.row(0);
        let row = normalize_log_probs(&row.to_owned());
        alpha.index_axis_mut(Axis(0), 0).assign(&row);

        //
        // Inductive Step (t = 1..n_obs)
        //
        for t in 1..n_obs {
            for k in 0..self.n_states {
                let row = observations.row(t);
                let mu = match &self.emission_means {
                    Means::Univariate(arr) => arr[k],
                    Means::Multivariate(_arr) => {
                        unimplemented!()
                    }
                };
                let sigma = match &self.emission_variance {
                    Covariance::Univariate(arr) => arr[k],
                    Covariance::Multivariate(_arr) => {
                        unimplemented!()
                    }
                };
                let normal = StatNormal::new(mu, sigma.sqrt())?;
                let log_likelihood = normal.ln_pdf(row[0]); // assuming random variable X has only a
                                                            // single value at time step t
                let mut temp = Array1::<f64>::zeros(self.n_states);
                for prev_state in 0..self.n_states {
                    temp[prev_state] =
                        alpha[[t - 1, prev_state]] + self.transition_probs[[prev_state, k]];
                }
                alpha[[t, k]] = log_sum_exp(&temp) + log_likelihood;
            }

            // Normalise alpha
            let row = alpha.row(t);
            let row = normalize_log_probs(&row.to_owned());
            alpha.index_axis_mut(Axis(0), t).assign(&row);
        }

        let dist = rand::distr::weighted::WeightedIndex::new(&alpha.row(n_obs - 1))?;
        states[n_obs - 1] = dist.sample(&mut rng);

        // Backward Step
        for t in (0..n_obs - 1).rev() {
            let mut probs: Array1<f64> = Array1::zeros(self.n_states);
            for k in 0..self.n_states {
                probs[k] = alpha[[t, k]] + self.transition_probs[[k, states[t + 1]]];
            }
            probs = normalize_log_probs(&probs);
            let dist = rand::distr::weighted::WeightedIndex::new(&probs)?;
            states[t] = dist.sample(&mut rng);
        }

        Ok(states)
    }

    fn sample_transition_probs(&mut self, states: &[usize]) {
        /*
         * Sample Transition Probabilities
         */

        let mut rng = rand::rng();
        let mut total_transitions: Array2<f64> = Array2::zeros((self.n_states, self.n_states));
        for window in states.windows(2) {
            total_transitions[[window[0], window[1]]] += 1.0;
        }

        for k in 0..self.n_states {
            let v = (total_transitions.row(k).to_owned() + &self.init_probs).to_vec();
            // TODO: hardcoded array size. Need to find a new way
            let arr: [f64; 3] = v.try_into().expect("Wrong length");
            let dirichlet = Dirichlet::new(arr).unwrap();
            let sample = dirichlet.sample(&mut rng);
            let row = Array1::from_vec(sample.to_vec());
            self.transition_probs.row_mut(k).assign(&row);
        }
    }

    fn sample_emission_params(
        &mut self,
        observations: &Array2<f64>,
        states: &[usize],
    ) -> anyhow::Result<()> {
        /*
         * Sample emissions - variance and means samples
         */

        let mut rng = rand::rng();
        let prior_mean = 0.0;
        let prior_variance_scale = 20.0;
        let prior_variance_shape = 2.0;

        // Group observations by state {0: Normal, 1: Warn, 2: Critical}
        for state in 0..self.n_states {
            let obs_group_by_state = observations
                .iter()
                .zip(states.iter())
                .filter_map(|(obs, s)| if *s == state { Some(*obs) } else { None })
                .filter_map(Option::Some)
                .map(|x| x.ln())
                .collect::<Vec<f64>>();

            if obs_group_by_state.is_empty() {
                continue;
            }

            // update emission means
            let count_obs = obs_group_by_state.len() as f64;
            if count_obs == 0.0 {
                return Ok(());
            }
            let obs_mean: f64 = obs_group_by_state.iter().sum::<f64>() / count_obs;
            let posterior_mean = (prior_mean / prior_variance_scale + obs_mean * count_obs)
                / (1.0 / prior_variance_scale + count_obs);
            let posterior_var = 1.0 / (1.0 / prior_variance_scale + count_obs);
            let normal = Normal::new(posterior_mean, posterior_var.sqrt())?;
            match self.emission_means {
                Means::Univariate(ref mut arr) => arr[state] = normal.sample(&mut rng),
                Means::Multivariate(ref mut arr) => {
                    unimplemented!()
                }
            }
            //
            // update emission variances
            //
            //  squared deviation
            let emission_means_state = match &self.emission_means {
                Means::Univariate(arr) => arr[state],
                Means::Multivariate(arr) => {
                    unimplemented!()
                }
            };
            let sq_dev_sum: f64 = obs_group_by_state
                .iter()
                .map(|x| (*x - emission_means_state).powi(2))
                .sum();
            // Prior hyperparameters
            assert!(prior_variance_scale > 0.0);
            assert!(prior_variance_shape > 0.0);
            // Inverse-Gamma distribution  - division by 2.0 because arises as a conjugate prior in Bayesian inference for the variance
            let posterior_shape = (prior_variance_shape + count_obs / 2.0).max(1e-6);
            let posterior_scale = (prior_variance_scale + 0.5 * sq_dev_sum).max(1e-6);
            assert!(posterior_scale > 0.0, "posterior_scale must be positive");
            let inv_gamma = Gamma::new(posterior_shape, 1.0 / posterior_scale)?;
            match self.emission_variance {
                Covariance::Univariate(ref mut arr) => {
                    arr[state] = 1.0 / inv_gamma.sample(&mut rng).clamp(1e-6, 1e2)
                }
                Covariance::Multivariate(ref mut arr) => {
                    unimplemented!()
                }
            };
        }

        Ok(())
    }
}

impl AnalyticsEngine for Hmm {
    fn anomaly_scores(
        &self,
        test_data: &DataFrame,
        threshold: f64,
    ) -> anyhow::Result<Vec<(String, f64, f64)>> {
        const MIN_PROB: f64 = 1e-300;
        let mut scores: Vec<(String, f64, f64)> = Vec::new();
        let columns = test_data.get_columns();
        let timestamps = columns[0].clone();
        let timestamps = timestamps.str().unwrap();
        let observations = columns[1].clone();
        let observations = observations.f64().unwrap();

        for (i, (time, data)) in timestamps.iter().zip(observations.iter()).enumerate() {
            let time = time.unwrap();
            let data = data.unwrap();
            let data_log = data.max(MIN_PROB).ln();
            match (self.emission_means.clone(), self.emission_variance.clone()) {
                (Means::Univariate(means_arr), Covariance::Univariate(cov_arr)) => {
                    let log_likelihoods: Vec<f64> = means_arr
                        .iter()
                        .zip(cov_arr.iter())
                        .map(|(&mu, &var)| {
                            let std = var.sqrt().max(1e-6);
                            let dist = StatNormal::new(mu, std).unwrap();
                            dist.ln_pdf(data_log)
                        })
                        .collect();

                    let score = -log_likelihoods
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    // - A low score (e.g. -0.64) → normal
                    // - A higher score (e.g. greater than 5) → possibly anomalous
                    // - a score like 7 or higher = a rare point = more anomalous
                    // println!(
                    //     "{:?} {:.2}, log = {:.2}, log_likes = {:?}, score = {:.2}",
                    //     time, data, data_log, log_likelihoods, score
                    // );
                    scores.push((time.to_string(), data, score));
                }
                (Means::Multivariate(_means_arr), Covariance::Multivariate(_cov_arr)) => {
                    unimplemented!()
                }
                (_, _) => {}
            };
        }

        for (i, (time, data, score)) in scores.iter().enumerate() {
            if *score > threshold {
                println!(
                    "Anomaly at {}: data = {:.2}, score = {:.2}",
                    time, data, score
                );
            }
        }
        //
        // for (i, (time, data, score)) in scores.iter().enumerate() {
        //     if *score > threshold_crit {
        //         println!(
        //             "Anomaly at {}: data = {:.2}, score = {:.2}",
        //             time, data, score
        //         );
        //     }
        // }
        Ok(scores)
    }
}
