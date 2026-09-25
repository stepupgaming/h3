use clap::Parser;

mod host;
mod video;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let queue = match host::queue::QueueGuard::acquire_if_needed(&args) {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("h3: {error:#}");
            return ExitCode::from(1);
        }
    };
    let config = match host::config::H3Config::load() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("h3: {error:#}");
            return ExitCode::from(1);
        }
    };
    let parsed = video::h3::H3Args::parse();
    let result = video::h3::run_h3(parsed, &config);
    drop(queue);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("h3: {error:#}");
            ExitCode::from(1)
        }
    }
}
