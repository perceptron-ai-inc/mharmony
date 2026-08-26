// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
use std::fmt;

#[cfg(feature = "python-binding")]
use pyo3::{basic::CompareOp, prelude::*, types::PyModule};

use crate::chat::ChannelConfig;

/// A canonical Harmony assistant channel.
///
/// Conversations may still carry custom or legacy channel strings via
/// [`crate::chat::Message::channel`] and [`crate::chat::ChannelConfig`]. This
/// typed enum only owns the canonical channels that Harmony itself treats as
/// built-in semantics.
#[cfg_attr(feature = "python-binding", pyclass(name = "CanonicalChannel"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CanonicalChannel {
    Analysis,
    Commentary,
    Final,
}

impl CanonicalChannel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Analysis => "analysis",
            Self::Commentary => "commentary",
            Self::Final => "final",
        }
    }

    pub fn parse(channel: &str) -> Option<Self> {
        canonical_assistant_channels()
            .find(|canonical_channel| canonical_channel.as_str() == channel)
    }
}

impl AsRef<str> for CanonicalChannel {
    fn as_ref(&self) -> &str {
        (*self).as_str()
    }
}

impl fmt::Display for CanonicalChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str((*self).as_str())
    }
}

impl From<CanonicalChannel> for String {
    fn from(channel: CanonicalChannel) -> Self {
        channel.as_str().to_string()
    }
}

impl TryFrom<&str> for CanonicalChannel {
    type Error = ();

    fn try_from(channel: &str) -> Result<Self, Self::Error> {
        Self::parse(channel).ok_or(())
    }
}

#[cfg(feature = "python-binding")]
#[pymethods]
impl CanonicalChannel {
    #[staticmethod]
    fn canonical() -> Vec<Self> {
        canonical_assistant_channels().collect()
    }

    #[getter]
    fn value(&self) -> &'static str {
        self.as_str()
    }

    fn __str__(&self) -> &'static str {
        self.as_str()
    }

    fn __repr__(&self) -> String {
        format!("CanonicalChannel.{}", python_channel_name(self.as_str()))
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        let is_equal = other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other_channel| *other_channel == *self);
        Ok(match op {
            CompareOp::Eq => is_equal,
            CompareOp::Ne => !is_equal,
            _ => false,
        })
    }

    fn __hash__(&self) -> isize {
        *self as isize
    }

    fn __name__(&self) -> String {
        python_channel_name(self.as_str())
    }
}

#[cfg(feature = "python-binding")]
fn python_channel_name(channel: &str) -> String {
    channel
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

pub fn canonical_assistant_channels() -> impl Iterator<Item = CanonicalChannel> {
    [
        CanonicalChannel::Analysis,
        CanonicalChannel::Commentary,
        CanonicalChannel::Final,
    ]
    .into_iter()
}

#[cfg(feature = "python-binding")]
pub(crate) fn register_python_bindings(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
) -> PyResult<()> {
    module.add_class::<CanonicalChannel>()?;
    let canonical_channel_type = module.getattr("CanonicalChannel")?;
    for channel in canonical_assistant_channels() {
        canonical_channel_type
            .setattr(python_channel_name(channel.as_str()), Py::new(py, channel)?)?;
    }
    Ok(())
}

pub fn canonical_assistant_channel_names() -> impl Iterator<Item = &'static str> {
    canonical_assistant_channels().map(|channel| channel.as_str())
}

pub fn canonical_assistant_channels_display() -> String {
    canonical_assistant_channel_names()
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn is_canonical_assistant_channel(channel: &str) -> bool {
    CanonicalChannel::parse(channel).is_some()
}

pub fn is_channel(channel: Option<&str>, expected: CanonicalChannel) -> bool {
    channel == Some(expected.as_str())
}

pub(crate) fn render_valid_channels_header(
    channel_config: &ChannelConfig,
    conversation_has_function_tools: bool,
) -> Option<String> {
    if channel_config.valid_channels.is_empty() {
        return None;
    }

    let channels_str = channel_config.valid_channels.join(", ");
    let mut channels_header = format!("# Valid channels: {channels_str}.");
    if channel_config.channel_required {
        channels_header.push_str(" Channel must be included for every message.");
    }
    if conversation_has_function_tools {
        channels_header.push('\n');
        channels_header.push_str(&format!(
            "Calls to these tools must go to the {} channel: 'functions'.",
            CanonicalChannel::Commentary
        ));
    }

    Some(channels_header)
}
