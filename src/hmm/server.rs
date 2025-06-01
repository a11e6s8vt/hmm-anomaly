use crate::plot_anomalies;
use crate::{AnalyticsEngine, Hmm};
use ndarray::Array2;
use polars::frame::DataFrame;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::Bayesian;

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerCommand {
    UploadTraining {
        name: String,
        data: DataFrame,
    },
    ListTraining,
    DescTraining {
        name: String,
    },
    TrainModel {
        model_name: String,
        n_states: usize,
        training_data: String,
        field: String,
    },
    FindAnomalies {
        model_name: String,
        data: DataFrame,
        threshold: f64,
    },
    // CorrelateAnomalies {
    //     model_names: Vec<String>,
    //     data: Vec<Vec<f64>>,
    // },
    // GetModelInfo {
    //     name: String,
    // },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerResponse {
    Success(String),
    Error(String),
    TrainingList(Vec<String>),
    DescTraining(Vec<(String, String)>),
    AnomalyResults(Vec<(String, f64, f64)>), // (index, anomaly_score)
    CorrelationResults(Vec<(String, Vec<usize>)>), // (model_name, anomaly_indices)
    ModelInfo(String),
}

pub struct HmmServer {
    models: Arc<Mutex<HashMap<String, Hmm>>>,
    training_data: Arc<Mutex<HashMap<String, DataFrame>>>,
}

impl HmmServer {
    fn new() -> Self {
        HmmServer {
            models: Arc::new(Mutex::new(HashMap::new())),
            training_data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn handle_command(&self, command: ServerCommand) -> anyhow::Result<ServerResponse> {
        match command {
            ServerCommand::UploadTraining { name, data } => {
                let mut training_data = self.training_data.lock().await;
                training_data.insert(name.clone(), data);
                Ok(ServerResponse::Success(format!(
                    "Training data '{}' uploaded",
                    name
                )))
            }
            ServerCommand::ListTraining => {
                let training_data = self.training_data.lock().await;
                let l = training_data
                    .keys()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>();
                Ok(ServerResponse::TrainingList(l))
            }
            ServerCommand::DescTraining { name } => {
                let training_data = self.training_data.lock().await;
                if let Some(data) = training_data.get(&name) {
                    let s = data
                        .schema()
                        .iter()
                        .map(|(n, t)| (n.to_string(), t.to_string()))
                        .collect::<Vec<(String, String)>>();
                    Ok(ServerResponse::DescTraining(s))
                } else {
                    Ok(ServerResponse::Error(format!(
                        "Training data, {} not found!",
                        name
                    )))
                }
            }
            ServerCommand::TrainModel {
                model_name,
                n_states,
                training_data,
                field,
            } => {
                let training_data_lock = self.training_data.lock().await;
                if let Some(data) = training_data_lock.get(&training_data) {
                    // Convert to ndarray format
                    // TODO: Change unwrap()
                    let observations = data.column(&field).unwrap().f64().unwrap();
                    let observations: Vec<f64> = observations.into_no_null_iter().collect();
                    let observations =
                        Array2::from_shape_vec((observations.len(), 1), observations)?;

                    // TODO: compute it from number of fields in training_data
                    let n_features: usize = 1;

                    // TODO: Make it a command line input parameter
                    // No of iterations in the gibbs_sampling
                    const NUM_ITER: usize = 1000;
                    const BURN_IN: usize = 50;

                    // Initialise HMM
                    let mut hmm = Hmm::new(n_features, n_states);

                    // Train HMM - Gibbs Sampling
                    hmm.learn_gibbs_sampling(&observations, NUM_ITER, BURN_IN)
                        .unwrap(); // Add proper error handling

                    // Store model
                    let mut models = self.models.lock().await;
                    models.insert(model_name.clone(), hmm);

                    Ok(ServerResponse::Success(format!(
                        "Model '{}' trained with {} states",
                        model_name, n_states
                    )))
                } else {
                    Ok(ServerResponse::Error(format!(
                        "No training data found for '{}'",
                        model_name
                    )))
                }
            }
            ServerCommand::FindAnomalies {
                model_name,
                data,
                threshold,
            } => {
                let models = self.models.lock().await;
                if let Some(model) = models.get(&model_name) {
                    // Find anomalies
                    let scores = model.anomaly_scores(&data)?;
                    plot_anomalies(&scores, threshold)?;
                    let anomalies: Vec<_> = scores
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, _, score))| score > &threshold)
                        .map(|(_, (time, obs, score))| (time.to_string(), *obs, *score))
                        .collect();

                    Ok(ServerResponse::AnomalyResults(anomalies))
                } else {
                    Ok(ServerResponse::Error(format!(
                        "Anomaly scores calculations failed using model '{}'",
                        model_name
                    )))
                }
            } // ServerCommand::CorrelateAnomalies { model_names, data } => {
              //     let models = self.models.lock().await;
              //     let mut results = Vec::new();
              //
              //     for model_name in model_names {
              //         if let Some(model) = models.get(&model_name) {
              //             // Convert data to ndarray
              //             let obs = data.len();
              //             let features = data[0].len();
              //             let mut observations = Array2::zeros((obs, features));
              //
              //             for (i, row) in data.iter().enumerate() {
              //                 for (j, &val) in row.iter().enumerate() {
              //                     observations[[i, j]] = val;
              //                 }
              //             }
              //
              //             // Find anomalies (using default threshold for correlation)
              //             let scores = model.anomaly_scores(&observations);
              //             let threshold =
              //                 scores.iter().fold(0.0, |acc, &x| acc + x) / scores.len() as f64 + 2.0;
              //             let anomalies: Vec<_> = scores
              //                 .iter()
              //                 .enumerate()
              //                 .filter(|(_, &score)| score > threshold)
              //                 .map(|(i, _)| i)
              //                 .collect();
              //
              //             results.push((model_name, anomalies));
              //         }
              //     }
              //
              //     ServerResponse::CorrelationResults(results)
              // }
              // ServerCommand::GetModelInfo { name } => {
              //     let models = self.models.lock().await;
              //     if let Some(model) = models.get(&name) {
              //         ServerResponse::ModelInfo(format!("{:?}", model))
              //     } else {
              //         ServerResponse::Error(format!("Model '{}' not found", name))
              //     }
              // }
        }
    }
}

impl Default for HmmServer {
    fn default() -> Self {
        Self::new()
    }
}
