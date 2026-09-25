//! The error type `h3 edit` returns when a mask gate fails.
//! Exit codes match Gemmy's taxonomy so a delegated workflow can read them.

use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CliErrorCode {
    Ok,
    NotFound,
    InvalidArgs,
    ValidationFailed,
    RuntimeMissing,
    Conflict,
    Canceled,
    Unavailable,
    Internal,
}

impl CliErrorCode {
    pub fn exit_code(self) -> u8 {
        match self {
            Self::Ok => 0,
            Self::NotFound => 1,
            Self::InvalidArgs | Self::ValidationFailed => 2,
            Self::RuntimeMissing => 3,
            Self::Conflict => 4,
            Self::Canceled => 5,
            Self::Unavailable => 6,
            Self::Internal => 10,
        }
    }

    pub fn retryable(self) -> bool {
        matches!(self, Self::Unavailable | Self::Conflict)
    }
}

#[derive(Clone, Debug)]
pub struct CliError {
    pub code: CliErrorCode,
    pub message: String,
    pub suggestions: Vec<String>,
    pub details: BTreeMap<String, Value>,
    pub retryable: bool,
    pub command: Option<String>,
}

impl CliError {
    pub fn new(code: CliErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            suggestions: Vec::new(),
            details: BTreeMap::new(),
            retryable: code.retryable(),
            command: None,
        }
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(CliErrorCode::Conflict, message)
    }

    pub fn with_command(mut self, command: impl Into<String>) -> Self {
        self.command = Some(command.into());
        self
    }

    pub fn suggest(mut self, suggestion: impl Into<String>) -> Self {
        let text = suggestion.into();
        if !text.trim().is_empty() && self.suggestions.len() < 3 {
            self.suggestions.push(text);
        }
        self
    }

    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CliError {}
