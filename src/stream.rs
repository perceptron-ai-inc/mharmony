// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
use crate::preprocess::PreprocessConfig;
use serde::{de::Error as _, ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModalityType {
    Text,
    SpecialToken,
    /// A registered media modality's wire name (for example `image_frame`).
    Media(String),
    Vector,
    RoleMarker,
    ModalityTag,
}

impl ModalityType {
    fn as_str(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::SpecialToken => "special_token",
            Self::Media(modality) => modality,
            Self::Vector => "vector",
            Self::RoleMarker => "role_marker",
            Self::ModalityTag => "modality_tag",
        }
    }
}

impl Serialize for ModalityType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ModalityType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "text" => Self::Text,
            "special_token" => Self::SpecialToken,
            "vector" => Self::Vector,
            "role_marker" => Self::RoleMarker,
            "modality_tag" => Self::ModalityTag,
            _ => Self::Media(value),
        })
    }
}

use crate::preprocess::ProcessedVector;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediaEventData {
    pub kind: String,
    pub payload: Value,
}

#[derive(Clone, Debug)]
pub enum EventData {
    Tokens(Vec<i64>),
    Media(MediaEventData),
    Vector(ProcessedVector),
}

impl Serialize for EventData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(1))?;
        match self {
            Self::Tokens(tokens) => map.serialize_entry("Tokens", tokens)?,
            Self::Vector(vector) => map.serialize_entry("Vector", vector)?,
            Self::Media(media) => match media.kind.as_str() {
                // Keep the established wire discriminants for shipped kinds while
                // representing both through one public Rust variant.
                "image" => map.serialize_entry("Image", &media.payload)?,
                "video" => map.serialize_entry("Video", &media.payload)?,
                _ => map.serialize_entry("Media", media)?,
            },
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for EventData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let object = value
            .as_object()
            .cloned()
            .ok_or_else(|| D::Error::custom("event data must be a single-key object"))?;
        if object.len() != 1 {
            return Err(D::Error::custom("event data must be a single-key object"));
        }
        let (kind, payload) = object
            .into_iter()
            .next()
            .ok_or_else(|| D::Error::custom("event data must not be empty"))?;
        match kind.as_str() {
            "Tokens" => serde_json::from_value(payload)
                .map(Self::Tokens)
                .map_err(D::Error::custom),
            "Vector" => serde_json::from_value(payload)
                .map(Self::Vector)
                .map_err(D::Error::custom),
            "Image" => Ok(Self::Media(MediaEventData {
                kind: "image".to_string(),
                payload,
            })),
            "Video" => Ok(Self::Media(MediaEventData {
                kind: "video".to_string(),
                payload,
            })),
            "Media" => serde_json::from_value(payload)
                .map(Self::Media)
                .map_err(D::Error::custom),
            other => Err(D::Error::custom(format!(
                "unknown event data kind: {other}"
            ))),
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub time: (f64, f64),
    pub modality: ModalityType,
    pub role: Option<crate::chat::Role>,
    pub channel: Option<String>,
    pub tags: HashMap<String, serde_json::Value>,
    pub data: Option<EventData>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stream {
    pub events: Vec<Event>,
    pub priority: Vec<ModalityType>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderedStream {
    pub stream: Stream,
    pub config: PreprocessConfig,
}

impl Stream {
    pub fn schedule(&mut self) {
        let priority_map: HashMap<ModalityType, usize> = self
            .priority
            .iter()
            .enumerate()
            .map(|(i, m)| (m.clone(), i))
            .collect();

        self.events.sort_by(|a, b| {
            let time_cmp = a
                .time
                .0
                .partial_cmp(&b.time.0)
                .unwrap_or(std::cmp::Ordering::Equal);
            if time_cmp != std::cmp::Ordering::Equal {
                return time_cmp;
            }

            let end_cmp = a
                .time
                .1
                .partial_cmp(&b.time.1)
                .unwrap_or(std::cmp::Ordering::Equal);
            if end_cmp != std::cmp::Ordering::Equal {
                return end_cmp;
            }

            let a_pri = priority_map.get(&a.modality).copied().unwrap_or(usize::MAX);
            let b_pri = priority_map.get(&b.modality).copied().unwrap_or(usize::MAX);
            a_pri.cmp(&b_pri)
        });
    }

    pub fn merge(streams: Vec<Stream>, priority: Vec<ModalityType>) -> Stream {
        let events = streams
            .into_iter()
            .flat_map(|s| s.events.into_iter())
            .collect();
        let mut merged = Stream { events, priority };
        merged.schedule();
        merged
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            priority: vec![
                ModalityType::RoleMarker,
                ModalityType::ModalityTag,
                ModalityType::Text,
                ModalityType::SpecialToken,
                ModalityType::Vector,
                ModalityType::Media("image_frame".to_string()),
                ModalityType::Media("video_i_frame".to_string()),
                ModalityType::Media("video_p_frame".to_string()),
            ],
        }
    }
}
