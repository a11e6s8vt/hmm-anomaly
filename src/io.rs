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
        .caption("CPU Utilization with Anomalies", ("sans-serif", 40))
        .build_cartesian_2d(start_time..end_time, 0.0..100.0)
        .unwrap();

    ctx.configure_mesh()
        .x_labels(40)
        .x_label_formatter(&|dt| format!("{}", dt.format("%m-%d %H")))
        .x_label_style(
            ("sans-serif", 15)
                .into_font()
                .transform(FontTransform::Rotate270),
        )
        .y_desc("CPU Usage")
        .x_desc("Time")
        .draw()
        .unwrap();

    let mut graph_points: Vec<_> = Vec::new();
    for (t, val, score) in scores.iter() {
        let time: DateTime<Utc> =
            DateTime::parse_from_str(&format!("{} +0000", t), &format!("{} %z", format))?
                .with_timezone(&Utc);
        graph_points.push((time, *val));
    }
    ctx.draw_series(LineSeries::new(graph_points, &BLUE))
        .unwrap();

    let mut anomaly_points: Vec<(DateTime<Utc>, f64)> = Vec::new();

    for (t, val, score) in scores.iter() {
        if score >= &score_threshold {
            let time: DateTime<Utc> =
                DateTime::parse_from_str(&format!("{} +0000", t), &format!("{} %z", format))?
                    .with_timezone(&Utc);
            anomaly_points.push((time, *val));
        } else {
            continue;
        }
    }

    ctx.draw_series(
        anomaly_points
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 1.2, RED.filled())),
    )?;

    Ok(())
}
