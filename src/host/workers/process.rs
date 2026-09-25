use crate::host::workers::validation::{OutputExpectation, validate_outputs};
use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub struct WorkerSpec {
    cmd: Command,
    label: String,
    verbose: bool,
    expected_outputs: Vec<OutputExpectation>,
    forward_stdout_to_stderr: bool,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct WorkerOutcome {
    pub label: String,
    pub status: std::process::ExitStatus,
    pub duration: Duration,
}

impl WorkerSpec {
    pub fn new(cmd: Command, label: impl Into<String>) -> Self {
        Self {
            cmd,
            label: label.into(),
            verbose: false,
            expected_outputs: Vec::new(),
            forward_stdout_to_stderr: false,
        }
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Route worker stdout to the parent's stderr. Robot-envelope commands use
    /// this so the envelope stays the only JSON on stdout while worker chatter
    /// (e.g. the parakeet ASR metadata line) remains visible for debugging.
    pub fn forward_stdout_to_stderr(mut self) -> Self {
        self.forward_stdout_to_stderr = true;
        self
    }

    #[allow(dead_code)]
    pub fn expect_output(mut self, output: OutputExpectation) -> Self {
        self.expected_outputs.push(output);
        self
    }

    pub fn expect_outputs(mut self, outputs: impl IntoIterator<Item = OutputExpectation>) -> Self {
        self.expected_outputs.extend(outputs);
        self
    }

    pub fn run_inherited(mut self) -> Result<WorkerOutcome> {
        crate::host::workers::env::apply_isolated_runtime_env(&mut self.cmd);
        // Cooperative cancel before launch (and periodically is handled by the
        // child gemmy process when the worker is a Gemmy child).
        crate::host::workers::env::check_supervisor_cancel_signal()?;
        if self.verbose {
            eprintln!("[{}] launching: {:?}", self.label, self.cmd);
        }
        let started = Instant::now();
        // Own the process tree so Ctrl+C / parent death / force-cancel does not
        // leave orphan GPU workers. Kill-on-job-close covers Drop without wait.
        let process_tree = crate::host::windows_job::OwnedProcessTree::create()
            .with_context(|| format!("create process ownership for {}", self.label))?;
        let mut child = self
            .cmd
            .stdin(Stdio::inherit())
            .stdout(if self.forward_stdout_to_stderr {
                Stdio::piped()
            } else {
                Stdio::inherit()
            })
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("Failed to launch {}", self.label))?;
        let stdout_forwarder = if self.forward_stdout_to_stderr {
            child.stdout.take().map(|mut stream| {
                std::thread::spawn(move || {
                    let mut buffer = [0_u8; 8192];
                    loop {
                        match stream.read(&mut buffer) {
                            Ok(0) | Err(_) => break,
                            Ok(read) => {
                                let _ = std::io::stderr().write_all(&buffer[..read]);
                                let _ = std::io::stderr().flush();
                            }
                        }
                    }
                })
            })
        } else {
            None
        };
        let process_tree = match process_tree.assign(&child) {
            Ok(()) => Some(process_tree),
            Err(error) => {
                // The setup exe already runs inside a job. A second job is
                // refused there. The child keeps running; parent-death kill is
                // the only thing we lose.
                eprintln!(
                    "[{}] continuing without a job object ({error:#})",
                    self.label
                );
                None
            }
        };
        let status = loop {
            crate::host::workers::env::check_supervisor_cancel_signal().map_err(|error| {
                if let Some(tree) = process_tree.as_ref() {
                    let _ = tree.terminate(0xC000_013A);
                } else {
                    let _ = child.kill();
                }
                let _ = child.wait();
                error
            })?;
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(error) => {
                    if let Some(tree) = process_tree.as_ref() {
                        let _ = tree.terminate(0xC000_013A);
                    } else {
                        let _ = child.kill();
                    }
                    let _ = child.wait();
                    return Err(error).with_context(|| format!("poll {}", self.label));
                }
            }
        };
        // Drop the job after wait so kill-on-close does not race a clean exit.
        drop(process_tree);
        if let Some(forwarder) = stdout_forwarder {
            let _ = forwarder.join();
        }
        if !status.success() {
            bail!("{} exited with {}", self.label, status);
        }
        validate_outputs(&self.expected_outputs)
            .with_context(|| format!("{} output validation failed", self.label))?;
        Ok(WorkerOutcome {
            label: self.label,
            status,
            duration: started.elapsed(),
        })
    }
}

#[allow(dead_code)]
pub fn run_inherited(cmd: Command, label: &str, verbose: bool) -> Result<()> {
    WorkerSpec::new(cmd, label)
        .verbose(verbose)
        .run_inherited()
        .map(|_| ())
}
