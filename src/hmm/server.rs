use crate::Hmm;
use polars::frame::DataFrame;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerCommand {
    UploadTraining { name: String, data: DataFrame },
    // TrainModel {
    //     name: String,
    //     states: usize,
    // },
    // FindAnomalies {
    //     model_name: String,
    //     data: Vec<Vec<f64>>,
    //     threshold: f64,
    // },
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
    AnomalyResults(Vec<(usize, f64)>), // (index, anomaly_score)
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

    pub async fn handle_command(&self, command: ServerCommand) -> ServerResponse {
        match command {
            ServerCommand::UploadTraining { name, data } => {
                let mut training_data = self.training_data.lock().await;
                training_data.insert(name.clone(), data);
                ServerResponse::Success(format!("Training data '{}' uploaded", name))
            } // ServerCommand::TrainModel { name, states } => {
              //     let training_data = self.training_data.lock().await;
              //     if let Some(data) = training_data.get(&name) {
              //         // Convert to ndarray format
              //         let obs = data.len();
              //         let features = data[0].len();
              //         let mut observations = Array2::zeros((obs, features));
              //
              //         for (i, row) in data.iter().enumerate() {
              //             for (j, &val) in row.iter().enumerate() {
              //                 observations[[i, j]] = val;
              //             }
              //         }
              //
              //         // Train HMM
              //         let mut hmm = GaussianHMM::new(states, features);
              //         hmm.fit(&observations, 100).unwrap(); // Add proper error handling
              //
              //         // Store model
              //         let mut models = self.models.lock().await;
              //         models.insert(name, hmm);
              //
              //         ServerResponse::Success(format!(
              //             "Model '{}' trained with {} states",
              //             name, states
              //         ))
              //     } else {
              //         ServerResponse::Error(format!("No training data found for '{}'", name))
              //     }
              // }
              // ServerCommand::FindAnomalies {
              //     model_name,
              //     data,
              //     threshold,
              // } => {
              //     let models = self.models.lock().await;
              //     if let Some(model) = models.get(&model_name) {
              //         // Convert data to ndarray
              //         let obs = data.len();
              //         let features = data[0].len();
              //         let mut observations = Array2::zeros((obs, features));
              //
              //         for (i, row) in data.iter().enumerate() {
              //             for (j, &val) in row.iter().enumerate() {
              //                 observations[[i, j]] = val;
              //             }
              //         }
              //
              //         // Find anomalies
              //         let scores = model.anomaly_scores(&observations);
              //         let anomalies: Vec<_> = scores
              //             .iter()
              //             .enumerate()
              //             .filter(|(_, &score)| score > threshold)
              //             .map(|(i, &score)| (i, score))
              //             .collect();
              //
              //         ServerResponse::AnomalyResults(anomalies)
              //     } else {
              //         ServerResponse::Error(format!("Model '{}' not found", model_name))
              //     }
              // }
              // ServerCommand::CorrelateAnomalies { model_names, data } => {
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
