use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, NaiveDateTime};
use chrono::{TimeZone, Utc};
use clap::{command, CommandFactory, Parser};
use csv::Reader;
use plotters::element::CoordMapper;
use plotters::prelude::*;
use serde::Deserialize;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[derive(Debug, Deserialize)]
pub struct CliInputs {
    #[arg(short, long)]
    training_data: Option<PathBuf>,

    #[arg(short, long)]
    input_data: Option<PathBuf>,
}

const DATA: [f64; 14] = [
    137.24, 136.37, 138.43, 137.41, 139.69, 140.41, 141.58, 139.55, 139.68, 139.10, 138.24, 135.67,
    137.12, 138.12,
];

impl CliInputs {
    pub fn read_cli() -> Result<(PathBuf, PathBuf)> {
        let cli_inputs = CliInputs::parse();
        let training_data_file = cli_inputs.training_data.as_deref().unwrap_or_else(|| {
            let mut cmd = CliInputs::command();
            cmd.error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "--training-data is required!",
            )
            .exit()
        });

        let input_data_file = cli_inputs.input_data.as_deref().unwrap_or_else(|| {
            let mut cmd = CliInputs::command();
            cmd.error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "--input-data is required!",
            )
            .exit()
        });
        Ok((
            training_data_file.to_path_buf(),
            input_data_file.to_path_buf(),
        ))
    }
}

pub trait CsvReader<T> {
    fn read_csv(&mut self, path: PathBuf) -> Result<()>;
}

// cpu utilization is a time series data with a single f64 field
#[derive(Debug, Deserialize)]
pub struct CpuUtilizationEntry {
    pub time: String,
    pub utilization: f64,
}

#[derive(Default)]
pub struct TrainingData<T> {
    pub records: Vec<T>,
}

impl Default for TrainingData<CpuUtilizationEntry> {
    fn default() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

impl CsvReader<CpuUtilizationEntry> for TrainingData<CpuUtilizationEntry> {
    fn read_csv(&mut self, path: PathBuf) -> Result<()> {
        let mut csv_reader = Reader::from_path(path)?;
        for result in csv_reader.deserialize() {
            let record = result?;
            self.records.push(record);
        }
        Ok(())
    }
}

pub fn plot_anomalies(scores: Vec<(String, f64, f64)>, score_threshold: f64) -> anyhow::Result<()> {
    let home = match std::env::var("HOME") {
        Ok(path) => path,
        Err(e) => return Err(anyhow!("Getting path information failed!")),
    };

    let exe_path = std::env::current_exe().expect("Failed to get executable path");
    let program_name = exe_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown");
    let output_path = std::path::Path::new(&home).join(program_name);
    if !output_path.exists() {
        std::fs::create_dir_all(&output_path)?;
    }
    let output_path = output_path.join("anomaly.png");

    let root = BitMapBackend::new(&output_path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let format = "%Y/%m/%d %H:%M:%S";

    let t = scores.first().unwrap().clone().0.to_string();
    let start_time: DateTime<Utc> =
        match DateTime::parse_from_str(&format!("{} +0000", t), &format!("{} %z", format)) {
            Ok(t) => t.with_timezone(&Utc),
            Err(e) => {
                eprint!("{:?}", e);
                return Err(anyhow!("Failed to parse time: {}", e));
            }
        };

    let end_time: DateTime<Utc> = DateTime::parse_from_str(
        &format!("{} +0000", scores.last().unwrap().clone().0),
        &format!("{} %z", format),
    )?
    .with_timezone(&Utc);

    let mut ctx = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("MSFT daily close price", ("sans-serif", 40))
        .build_cartesian_2d(start_time..end_time, 0.0..100.0)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    let mut graph_points: Vec<_> = Vec::new();
    for (t, val, score) in scores.iter() {
        let time: DateTime<Utc> =
            DateTime::parse_from_str(&format!("{} +0000", t), &format!("{} %z", format))?
                .with_timezone(&Utc);
        println!("{:?} {:?}", time, val);
        graph_points.push((time, *val));
    }
    ctx.draw_series(LineSeries::new(graph_points, &BLUE))
        .unwrap();

    Ok(())
}
