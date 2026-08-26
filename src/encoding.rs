// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
use crate::{
    channels::{self, CanonicalChannel},
    chat::{
        Author, Content, Message, PointCollectionContent, PointingContent, ReasoningEffort, Role,
        SystemContent, TextContent, TokensContent,
    },
    point_stream::{
        split_text_with_pointing_view, stable_text_from_point_spans, IncrementalPointParser,
    },
    tiktoken::{CoreBPE, Rank},
};
use anyhow::Context as _;
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
    vec,
};
use unicode_normalization_alignments::UnicodeNormalization;

pub use crate::stream::RenderedStream;

const REPLACEMENT: &str = "\u{FFFD}";
const DEFAULT_QWEN_TIMESTAMP_PRECISION: usize = 1;

pub(crate) fn normalize_text_for_tokenizer(text: &str) -> String {
    text.nfc().map(|(ch, _)| ch).collect()
}

fn validate_token_ids(tokens: &[i64]) -> anyhow::Result<()> {
    for (idx, tok) in tokens.iter().enumerate() {
        if *tok < 0 {
            anyhow::bail!("token id at index {idx} is negative: {tok}");
        }
        if *tok > u32::MAX as i64 {
            anyhow::bail!("token id at index {idx} exceeds u32: {tok}");
        }
    }
    Ok(())
}

fn extract_tag_str<'a>(tags: &'a HashMap<String, Value>, key: &str) -> Option<&'a str> {
    tags.get(key).and_then(|val| val.as_str())
}

fn extract_tag_i64(tags: &HashMap<String, Value>, key: &str) -> anyhow::Result<Option<i64>> {
    tags.get(key)
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| anyhow::anyhow!("{key} must be an integer in the i64 range"))
        })
        .transpose()
}

fn active_channel_config<'a>(messages: &[&'a Message]) -> Option<&'a crate::chat::ChannelConfig> {
    let mut active = None;
    for msg in messages {
        if msg.author.role != Role::System {
            continue;
        }
        for content in &msg.content {
            if let Content::SystemContent(sys) = content {
                if let Some(channel_config) = sys.channel_config.as_ref() {
                    active = Some(channel_config);
                }
            }
        }
    }
    active
}

fn render_options_for_conversation(messages: &[&Message]) -> RenderOptions {
    let conversation_has_function_tools = messages.iter().any(|message| {
        message.content.iter().any(|content| {
            let Content::DeveloperContent(developer) = content else {
                return false;
            };
            developer
                .tools
                .as_ref()
                .and_then(|tools| tools.get("functions"))
                .is_some_and(|namespace| !namespace.tools.is_empty())
        })
    });

    RenderOptions {
        conversation_has_function_tools,
    }
}

fn validate_content_role(message: &Message, content: &Content) -> anyhow::Result<()> {
    match content {
        Content::SystemContent(_) => anyhow::ensure!(
            message.author.role == Role::System,
            "SystemContent may only appear in system messages, found in {:?}",
            message.author.role
        ),
        Content::DeveloperContent(_) => anyhow::ensure!(
            message.author.role == Role::Developer,
            "DeveloperContent may only appear in developer messages, found in {:?}",
            message.author.role
        ),
        _ => {}
    }
    Ok(())
}

fn validate_conversation(messages: &[&Message]) -> anyhow::Result<()> {
    validate_conversation_channels(messages)?;
    for message in messages {
        for content in &message.content {
            validate_content_role(message, content)?;
        }
    }
    Ok(())
}

pub(crate) fn conversation_messages_to_render<'a>(
    messages: &[&'a Message],
    config: &RenderConversationConfig,
) -> Vec<&'a Message> {
    let last_assistant_is_final = messages
        .iter()
        .rev()
        .find_map(|message| {
            (message.author.role == Role::Assistant)
                .then(|| channels::is_channel(message.channel.as_deref(), CanonicalChannel::Final))
        })
        .unwrap_or(false);
    let should_drop_analysis = config.auto_drop_analysis && last_assistant_is_final;
    let first_final_idx = messages.iter().position(|message| {
        channels::is_channel(message.channel.as_deref(), CanonicalChannel::Final)
    });

    messages
        .iter()
        .enumerate()
        .filter_map(|(idx, message)| {
            (!(should_drop_analysis
                && first_final_idx.is_some_and(|first| idx < first)
                && channels::is_channel(message.channel.as_deref(), CanonicalChannel::Analysis)))
            .then_some(*message)
        })
        .collect()
}

pub(crate) fn validate_conversation_channels(messages: &[&Message]) -> anyhow::Result<()> {
    let Some(channel_config) = active_channel_config(messages) else {
        return Ok(());
    };

    let valid_channels: HashSet<&str> = channel_config
        .valid_channels
        .iter()
        .map(String::as_str)
        .collect();

    // SystemContent.channel_config constrains assistant response channels. Other
    // roles can legitimately omit channels or carry legacy/internal routing.
    for (idx, msg) in messages.iter().enumerate() {
        if msg.author.role != Role::Assistant {
            continue;
        }

        if channel_config.channel_required && msg.channel.is_none() {
            anyhow::bail!(
                "assistant message at index {idx} is missing a channel while the active system channel config requires one"
            );
        }

        if let Some(channel) = msg.channel.as_deref() {
            if !valid_channels.is_empty()
                && !valid_channels.contains(channel)
                && !channels::is_canonical_assistant_channel(channel)
            {
                anyhow::bail!(
                    "assistant message at index {idx} uses channel '{channel}', which is not in the accepted assistant channel set [{}] or the active valid channel set [{}]",
                    channels::canonical_assistant_channels_display(),
                    channel_config.valid_channels.join(", ")
                );
            }
        }
    }

    Ok(())
}

// Parsed representation of a message header.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParsedHeader {
    author: Author,
    recipient: Option<String>,
    channel: Option<String>,
    content_type: Option<String>,
}

/// Structured completion parse plus per-message append-safe display text.
///
/// `stable_message_texts[i]` is index-aligned with `messages[i]` and is safe
/// to use for progressive display of that message. Stable text never contains
/// mHarmony point sentinels such as `<|point_start|>` or `<|point_end|>`, but
/// it may include structurally complete point children from an incomplete point
/// span before the span itself closes. Channel and role visibility policy is not
/// applied here; callers should filter `messages` and use the aligned sidecar
/// for each visible message.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompletionParseView {
    pub messages: Vec<Message>,
    pub stable_message_texts: Vec<String>,
}

/// Options for [`HarmonyEncoding::completion_stream_parser`].
///
/// The default is assistant completion-suffix mode: `role` is
/// `Some(Role::Assistant)` and `strict` is `false`. In that mode the parser
/// expects generated assistant tokens that begin after the already-rendered
/// `<|start|>assistant` prefix, for example
/// `<|channel|>final<|message|>...<|return|>`.
///
/// Set `role` to `None` explicitly when parsing a full mHarmony message stream
/// that includes `<|start|>` and a role in each message header.
///
/// The stream parser reports `stop` only after consuming implemented terminal
/// stop tokens (final `<|return|>`/`<|end|>`, or tool-call `<|call|>` when
/// `stop_on_tool_call` is enabled). If the provider reports completion without
/// emitting a stop token, call [`CompletionStreamParser::finish`] to flush and
/// finalize the open partial message.
#[derive(Clone, Debug)]
pub struct CompletionStreamOptions {
    pub role: Option<Role>,
    pub strict: bool,
    pub stop_on_tool_call: bool,
    pub include_thinking_channels: bool,
}

impl Default for CompletionStreamOptions {
    fn default() -> Self {
        Self {
            role: Some(Role::Assistant),
            strict: false,
            stop_on_tool_call: false,
            include_thinking_channels: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamStopKind {
    Final,
    ToolCall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StreamStop {
    pub kind: StreamStopKind,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StreamTextUpdate {
    pub visible_delta: String,
    pub visible_full: String,
    pub final_delta: String,
    pub final_full: String,
    pub reasoning_delta: String,
    pub reasoning_full: String,
    pub visible_delta_parts: Vec<String>,
    pub final_delta_parts: Vec<String>,
    pub reasoning_delta_parts: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompletionStreamUpdate {
    pub token_count: usize,
    pub completed_messages: Vec<Message>,
    pub stop: Option<StreamStop>,
    pub text: StreamTextUpdate,
}

pub struct CompletionStreamParser {
    inner: StreamableParser,
    options: CompletionStreamOptions,
    completed_count: usize,
    visible_full: String,
    final_full: String,
    reasoning_full: String,
    finished_stop: Option<StreamStop>,
    finished_without_stop: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ReservedTokenGroup {
    pub(crate) name: Option<String>,
    pub(crate) tokenizer: Option<String>,
    pub(crate) size: usize,
    pub(crate) offset: i64,
}

impl ReservedTokenGroup {
    fn matches(&self, tags: &HashMap<String, Value>) -> bool {
        let group = extract_tag_str(tags, "token_group");
        let tokenizer = extract_tag_str(tags, "tokenizer");
        let name_match = self.name.as_deref().is_some_and(|name| group == Some(name));
        let tokenizer_match = self
            .tokenizer
            .as_deref()
            .is_some_and(|tok| tokenizer == Some(tok));
        match (self.name.is_some(), self.tokenizer.is_some()) {
            (true, true) => name_match && tokenizer_match,
            (true, false) => name_match,
            (false, true) => tokenizer_match,
            (false, false) => false,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum RenderFormattingTokenError {
    #[error("tried to render unmapped formatting token {0}")]
    UnmappedToken(FormattingToken),

    #[error("tokenizer panicked while encoding formatting token {token}")]
    TokenizerFailure { token: FormattingToken },

    #[error(
        "Expected encoding of formatting token {token} to be a single token, but got {encoding:?}"
    )]
    InvalidEncoding {
        token: FormattingToken,
        encoding: Vec<Rank>,
    },
}

/// These are formatting tokens that the renderer can use to generically
/// format the output of the model, but at formatting time, they are replaced
/// by actual tokens from the tokenizers vocabulary.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FormattingToken {
    Start,
    Message,
    EndMessage,
    EndMessageDoneSampling,
    EndMessageAssistantToTool,
    Refusal,
    ConstrainedFormat,
    Channel,
    BeginUntrusted,
    EndUntrusted,
    MetaSep,
    MetaEnd,
    Image,
    Video,
    Vector,
    PointStart,
    PointEnd,
}

impl FormattingToken {
    pub(crate) fn symbol(self) -> &'static str {
        match self {
            FormattingToken::Start => "start",
            FormattingToken::Message => "message",
            FormattingToken::EndMessage => "end_message",
            FormattingToken::EndMessageDoneSampling => "end_message_done_sampling",
            FormattingToken::EndMessageAssistantToTool => "end_message_assistant_to_tool",
            FormattingToken::Refusal => "refusal",
            FormattingToken::ConstrainedFormat => "constrained_format",
            FormattingToken::Channel => "channel",
            FormattingToken::BeginUntrusted => "begin_untrusted",
            FormattingToken::EndUntrusted => "end_untrusted",
            FormattingToken::MetaSep => "meta_sep",
            FormattingToken::MetaEnd => "meta_end",
            FormattingToken::Image => "image",
            FormattingToken::Video => "video",
            FormattingToken::Vector => "vector",
            FormattingToken::PointStart => "point_start",
            FormattingToken::PointEnd => "point_end",
        }
    }

    pub(crate) fn from_symbol(value: &str) -> Option<Self> {
        Some(match value {
            "start" => Self::Start,
            "message" => Self::Message,
            "end_message" => Self::EndMessage,
            "end_message_done_sampling" => Self::EndMessageDoneSampling,
            "end_message_assistant_to_tool" => Self::EndMessageAssistantToTool,
            "refusal" => Self::Refusal,
            "constrained_format" => Self::ConstrainedFormat,
            "channel" => Self::Channel,
            "begin_untrusted" => Self::BeginUntrusted,
            "end_untrusted" => Self::EndUntrusted,
            "meta_sep" => Self::MetaSep,
            "meta_end" => Self::MetaEnd,
            "image" => Self::Image,
            "video" => Self::Video,
            "vector" => Self::Vector,
            "point_start" => Self::PointStart,
            "point_end" => Self::PointEnd,
            _ => return None,
        })
    }
}

/// Core-side part of a media-kind registration. Processor callbacks and
/// validators are supplied by the host; the encoding only needs to know how
/// to represent the kind in its token and event streams.
#[derive(Clone, Debug)]
pub(crate) struct RegisteredMediaKind {
    pub(crate) token_id: Rank,
    pub(crate) modality: String,
    pub(crate) empty_payload: serde_json::Value,
}

impl RegisteredMediaKind {
    pub(crate) fn empty_payload(&self) -> serde_json::Value {
        self.empty_payload.clone()
    }
}

impl std::fmt::Display for FormattingToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.symbol())
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct HarmonyEncoding {
    pub(crate) name: String,
    pub(crate) n_ctx: usize,
    pub(crate) max_message_tokens: usize,
    pub(crate) max_action_length: usize,
    pub(crate) tokenizer_name: String,
    pub(crate) tokenizer: Arc<CoreBPE>,
    pub(crate) format_token_mapping: HashMap<FormattingToken, String>,
    pub(crate) format_token_ids: HashMap<FormattingToken, Rank>,
    pub(crate) media_kinds: HashMap<String, RegisteredMediaKind>,
    pub(crate) stop_formatting_tokens: HashSet<FormattingToken>,
    pub(crate) stop_formatting_tokens_for_assistant_actions: HashSet<FormattingToken>,
    /// Whether to apply NFC normalization before tokenizing (required for Qwen3)
    pub(crate) use_nfc_normalization: bool,
    pub(crate) reserved_token_groups: Vec<ReservedTokenGroup>,
}

impl std::fmt::Debug for HarmonyEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarmonyEncoding")
            .field("name", &self.name)
            .field("tokenizer_name", &self.tokenizer_name)
            .field("n_ctx", &self.n_ctx)
            .field("max_message_tokens", &self.max_message_tokens)
            .field("max_action_length", &self.max_action_length)
            .finish()
    }
}

impl std::fmt::Display for HarmonyEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Renderer({})", self.name)
    }
}

// General methods
impl HarmonyEncoding {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tokenizer_name(&self) -> &str {
        &self.tokenizer_name
    }

    pub fn max_message_tokens(&self) -> usize {
        self.max_message_tokens
    }

    pub fn tokenizer(&self) -> &CoreBPE {
        &self.tokenizer
    }

    fn find_reserved_token_group(
        &self,
        tags: &HashMap<String, Value>,
    ) -> Option<&ReservedTokenGroup> {
        if self.reserved_token_groups.is_empty() {
            return None;
        }
        self.reserved_token_groups
            .iter()
            .find(|group| group.matches(tags))
    }

    pub(crate) fn offset_tokens_for_tags(
        &self,
        tokens: &[i64],
        tags: &HashMap<String, Value>,
    ) -> anyhow::Result<Vec<i64>> {
        let explicit_offset = extract_tag_i64(tags, "token_offset")?;
        let group = self.find_reserved_token_group(tags);

        let (offset, size) = match (explicit_offset, group) {
            (Some(offset), _) => (offset, None),
            (None, Some(group)) => (group.offset, Some(group.size)),
            (None, None) => return Ok(tokens.to_vec()),
        };
        if offset < 0 {
            anyhow::bail!("token_offset must be non-negative (got {offset})");
        }

        let any_ge = tokens.iter().any(|tok| *tok >= offset);
        let any_lt = tokens.iter().any(|tok| *tok < offset);
        if any_ge && !any_lt {
            if let Some(size) = size {
                let max_token = tokens.iter().copied().max().unwrap_or(-1);
                let max_allowed = offset
                    .checked_add(i64::try_from(size)? - 1)
                    .ok_or_else(|| anyhow::anyhow!("token offset range overflows i64"))?;
                if max_token > max_allowed {
                    anyhow::bail!(
                        "token id {max_token} exceeds reserved group size {size} (offset {offset})"
                    );
                }
            }
            validate_token_ids(tokens)?;
            return Ok(tokens.to_vec());
        }
        if any_ge && any_lt {
            anyhow::bail!("token group contains mixed offset tokens (offset={offset})");
        }

        if let Some(size) = size {
            let max_token = tokens.iter().copied().max().unwrap_or(-1);
            if max_token >= size as i64 {
                anyhow::bail!(
                    "token id {max_token} exceeds reserved group size {size} (offset {offset})"
                );
            }
        }

        tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let token = token.checked_add(offset).ok_or_else(|| {
                    anyhow::anyhow!("token id at index {index} overflows after offset {offset}")
                })?;
                if token > i64::from(Rank::MAX) {
                    anyhow::bail!(
                        "token id at index {index} exceeds u32 after offset {offset}: {token}"
                    );
                }
                Ok(token)
            })
            .collect()
    }

    fn guard_tokenizer_panic<T>(
        &self,
        context: &'static str,
        f: impl FnOnce() -> T,
    ) -> anyhow::Result<T> {
        catch_unwind(AssertUnwindSafe(f))
            .map_err(|_| anyhow::anyhow!("tokenizer panicked while {context}"))
    }

    #[allow(dead_code)]
    pub(crate) fn encode_text(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
    ) -> anyhow::Result<Vec<Rank>> {
        if self.use_nfc_normalization {
            let normalized = normalize_text_for_tokenizer(text);
            self.guard_tokenizer_panic("encoding text", || {
                self.tokenizer.encode(&normalized, allowed_special).0
            })
        } else {
            self.guard_tokenizer_panic("encoding text", || {
                self.tokenizer.encode(text, allowed_special).0
            })
        }
    }

    pub(crate) fn encode_ordinary_text(&self, text: &str) -> anyhow::Result<Vec<Rank>> {
        if self.use_nfc_normalization {
            let normalized = normalize_text_for_tokenizer(text);
            self.guard_tokenizer_panic("encoding ordinary text", || {
                self.tokenizer.encode_ordinary(&normalized)
            })
        } else {
            self.guard_tokenizer_panic("encoding ordinary text", || {
                self.tokenizer.encode_ordinary(text)
            })
        }
    }

    pub fn encode_timestamps_qwen(
        &self,
        timestamps: &[f64],
        precision: Option<usize>,
    ) -> anyhow::Result<Vec<Vec<i64>>> {
        let precision = precision.unwrap_or(DEFAULT_QWEN_TIMESTAMP_PRECISION);
        let mut out: Vec<Vec<i64>> = Vec::with_capacity(timestamps.len());
        for &timestamp in timestamps {
            // Match Python's float-based rounding behavior for halfway cases like
            // 2.675 -> 2.67 and 4.335 -> 4.33 at precision=2.
            let text = format!("<{timestamp:.precision$} seconds>");
            let tokens = self.encode_ordinary_text(&text)?;
            out.push(tokens.into_iter().map(|t| t as i64).collect());
        }
        Ok(out)
    }

    /// Returns whether this encoding requires NFC normalization before tokenizing.
    pub fn use_nfc_normalization(&self) -> bool {
        self.use_nfc_normalization
    }

    pub fn stop_tokens(&self) -> anyhow::Result<HashSet<Rank>> {
        self.render_stop_tokens(&self.stop_formatting_tokens)
    }

    pub fn stop_tokens_for_assistant_actions(&self) -> anyhow::Result<HashSet<Rank>> {
        self.render_stop_tokens(&self.stop_formatting_tokens_for_assistant_actions)
    }

    fn render_stop_tokens(
        &self,
        formatting_tokens: &HashSet<FormattingToken>,
    ) -> anyhow::Result<HashSet<Rank>> {
        formatting_tokens
            .iter()
            .copied()
            .map(|t| match self.render_formatting_token(t) {
                Ok(t) => Ok(t),
                Err(RenderFormattingTokenError::UnmappedToken(_)) => Err(anyhow::anyhow!(
                    "token {t} was specified as a stop token, but is not mapped"
                )),
                Err(e) => Err(anyhow::anyhow!(e).context("could not render stop token")),
            })
            .collect()
    }

    fn token_event(
        message: &Message,
        tokens: Vec<Rank>,
        modality: crate::stream::ModalityType,
        tags: HashMap<String, Value>,
    ) -> crate::stream::Event {
        crate::stream::Event {
            data: Some(crate::stream::EventData::Tokens(
                tokens.into_iter().map(i64::from).collect(),
            )),
            time: (0.0, 0.0),
            modality,
            role: Some(message.author.role.clone()),
            channel: message.channel.clone(),
            tags,
        }
    }

    /// Render a conversation into a multimodal RenderedStream.
    ///
    /// This produces a stream of events that is a strict superset of render_conversation():
    /// - The token sequence is identical when flattened
    /// - Events include framing tokens (header/footer) using the same render_message_header/footer helpers
    /// - Registered media content is emitted as separate events with its data
    pub fn render_conversation_multimodal<'a, I>(
        &self,
        conversation: I,
        preprocess: Option<&crate::preprocess::PreprocessConfig>,
    ) -> anyhow::Result<RenderedStream>
    where
        I: IntoIterator<Item = &'a Message>,
    {
        let cfg = preprocess
            .cloned()
            .unwrap_or_else(crate::preprocess::PreprocessConfig::default);
        cfg.validate()?;

        let messages: Vec<_> = conversation.into_iter().collect();
        validate_conversation(&messages)?;
        let render_options = render_options_for_conversation(&messages);
        let render_config = RenderConversationConfig::default();

        let mut events = Vec::new();
        for msg in conversation_messages_to_render(&messages, &render_config) {
            // Emit header as token event: <|start|>{role}{channel}<|message|>
            let mut header_tokens: Vec<Rank> = vec![];
            self.render_message_header_into(msg, &mut header_tokens)?;
            events.push(Self::token_event(
                msg,
                header_tokens,
                crate::stream::ModalityType::RoleMarker,
                Default::default(),
            ));

            // Emit content events
            for content in &msg.content {
                match content {
                    Content::Media(media) => {
                        let registration = self.registered_media_kind_for(media)?;
                        let mut tags = HashMap::new();
                        tags.insert(
                            "media_ref_id".to_string(),
                            Value::String(media.media_ref.id.clone()),
                        );
                        tags.insert(
                            "media_ref_type".to_string(),
                            Value::String(media.kind.clone()),
                        );
                        events.push(Self::token_event(
                            msg,
                            vec![self.render_registered_media_token(registration)?],
                            crate::stream::ModalityType::SpecialToken,
                            tags.clone(),
                        ));
                        events.push(crate::stream::Event {
                            data: Some(crate::stream::EventData::Media(
                                crate::stream::MediaEventData {
                                    kind: media.kind.clone(),
                                    payload: registration.empty_payload(),
                                },
                            )),
                            time: (0.0, 0.0),
                            modality: crate::stream::ModalityType::Media(
                                registration.modality.clone(),
                            ),
                            role: Some(msg.author.role.clone()),
                            channel: msg.channel.clone(),
                            tags,
                        });
                    }
                    Content::Vector(vector) => {
                        events.push(Self::token_event(
                            msg,
                            vec![self.render_formatting_token(FormattingToken::Vector)?],
                            crate::stream::ModalityType::SpecialToken,
                            Default::default(),
                        ));
                        let mut tags: HashMap<String, serde_json::Value> = vector
                            .metadata
                            .as_ref()
                            .and_then(|v| v.as_object())
                            .map(|obj| obj.clone().into_iter().collect())
                            .unwrap_or_default();
                        if let Some(name) = &vector.name {
                            tags.insert(
                                "name".to_string(),
                                serde_json::Value::String(name.clone()),
                            );
                        }
                        if let Some(dtype) = &vector.dtype {
                            tags.insert(
                                "dtype".to_string(),
                                serde_json::Value::String(dtype.clone()),
                            );
                        }
                        events.push(crate::stream::Event {
                            data: Some(crate::stream::EventData::Vector(
                                crate::preprocess::ProcessedVector {
                                    data: vector.values.clone(),
                                    shape: vector.shape.clone(),
                                },
                            )),
                            time: (0.0, 0.0),
                            modality: crate::stream::ModalityType::Vector,
                            role: Some(msg.author.role.clone()),
                            channel: msg.channel.clone(),
                            tags,
                        });
                    }
                    _ => {
                        let mut tokens = Vec::new();
                        Render::<Content>::render(
                            self,
                            content,
                            &mut tokens,
                            Some(&render_options),
                        )?;
                        let preserve_empty_event =
                            matches!(content, Content::Text(_) | Content::Tokens(_));
                        if preserve_empty_event || !tokens.is_empty() {
                            let tags = match content {
                                Content::Tokens(tokens) => tokens.tags.clone(),
                                _ => Default::default(),
                            };
                            events.push(Self::token_event(
                                msg,
                                tokens,
                                crate::stream::ModalityType::Text,
                                tags,
                            ));
                        }
                    }
                }
            }

            // Emit footer as token event: <|end|> or <|call|>
            let mut footer_tokens: Vec<Rank> = vec![];
            self.render_message_footer_into(msg, &mut footer_tokens)?;
            events.push(Self::token_event(
                msg,
                footer_tokens,
                crate::stream::ModalityType::SpecialToken,
                Default::default(),
            ));
        }

        let stream = crate::stream::Stream {
            events,
            priority: crate::stream::Stream::default().priority,
        };
        Ok(RenderedStream {
            stream,
            config: cfg,
        })
    }
}

// Methods for rendering conversations
impl HarmonyEncoding {
    /// Renders a conversation into a collection of tokens.
    pub fn render_conversation_into<'a, I, B>(
        &self,
        conversation: I,
        into: &mut B,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = &'a Message>,
        B: Extend<Rank>,
    {
        let messages: Vec<_> = conversation.into_iter().collect();
        validate_conversation(&messages)?;
        let render_options = render_options_for_conversation(&messages);
        let default_config = RenderConversationConfig::default();
        let config = config.unwrap_or(&default_config);

        let result = conversation_messages_to_render(&messages, config)
            .into_iter()
            .try_for_each(|message| self.render_into(message, into, Some(&render_options)));
        result?;
        Ok(())
    }

    /// Renders a conversation into a collection of tokens, adding the next turn role.
    ///
    /// This method is used to prepare a conversation for inference.
    pub fn render_conversation_for_completion_into<'a, I, B>(
        &self,
        conversation: I,
        next_turn_role: Role,
        into: &mut B,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = &'a Message>,
        B: Extend<Rank>,
    {
        self.render_conversation_into(conversation, into, config)?;
        self.render_formatting_token_into(FormattingToken::Start, into)?;
        self.render_text_into(next_turn_role.as_str(), into)?;
        Ok(())
    }

    pub fn render_conversation_for_completion<'a, I>(
        &self,
        conversation: I,
        next_turn_role: Role,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<Vec<Rank>>
    where
        I: IntoIterator<Item = &'a Message>,
    {
        let mut into = vec![];
        self.render_conversation_for_completion_into(
            conversation,
            next_turn_role,
            &mut into,
            config,
        )?;
        Ok(into)
    }

    /// Render a conversation for training.
    ///
    /// If the last message in the conversation is an assistant message to the
    /// `final` channel, replace the trailing `<|end|>` token with `<|return|>`.
    pub fn render_conversation_for_training<'a, I>(
        &self,
        conversation: I,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<Vec<Rank>>
    where
        I: IntoIterator<Item = &'a Message>,
    {
        let messages: Vec<&Message> = conversation.into_iter().collect();
        let mut out = vec![];
        self.render_conversation_into(messages.iter().copied(), &mut out, config)?;
        if let Some(last) = messages.last() {
            if last.author.role == Role::Assistant
                && channels::is_channel(last.channel.as_deref(), CanonicalChannel::Final)
            {
                if let Some(last_token) = out.last_mut() {
                    *last_token =
                        self.render_formatting_token(FormattingToken::EndMessageDoneSampling)?;
                }
            }
        }
        Ok(out)
    }

    /// Render a conversation without appending a new role.
    pub fn render_conversation<'a, I>(
        &self,
        conversation: I,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<Vec<Rank>>
    where
        I: IntoIterator<Item = &'a Message>,
    {
        let mut out = vec![];
        self.render_conversation_into(conversation, &mut out, config)?;
        Ok(out)
    }

    /// Render a single message into tokens.
    pub fn render(
        &self,
        message: &Message,
        render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<Vec<Rank>> {
        let mut out = vec![];
        Render::<Message>::render(self, message, &mut out, render_options)?;
        Ok(out)
    }

    /// Render a single message into the provided buffer.
    pub fn render_into<B>(
        &self,
        message: &Message,
        into: &mut B,
        render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        Render::<Message>::render(self, message, into, render_options)
    }
}

// Rendering helper methods
impl HarmonyEncoding {
    pub(crate) fn validate_media_content(media: &crate::chat::MediaContent) -> anyhow::Result<()> {
        anyhow::ensure!(
            !media.kind.is_empty() && media.kind.trim() == media.kind,
            "media kind must be non-empty and contain no surrounding whitespace"
        );
        anyhow::ensure!(
            !media.media_ref.id.is_empty() && media.media_ref.id.trim() == media.media_ref.id,
            "media ref id must be non-empty and contain no surrounding whitespace"
        );
        Ok(())
    }

    pub(crate) fn registered_media_kind(&self, kind: &str) -> anyhow::Result<&RegisteredMediaKind> {
        self.media_kinds
            .get(kind)
            .ok_or_else(|| anyhow::anyhow!("unregistered media kind: {kind}"))
    }

    pub(crate) fn registered_media_kind_for<'a>(
        &'a self,
        media: &crate::chat::MediaContent,
    ) -> anyhow::Result<&'a RegisteredMediaKind> {
        Self::validate_media_content(media)?;
        self.registered_media_kind(&media.kind)
    }

    fn mapped_format_token(&self, t: FormattingToken) -> Option<&str> {
        self.format_token_mapping.get(&t).map(|s| s.as_str())
    }

    pub(crate) fn render_formatting_token(
        &self,
        t: FormattingToken,
    ) -> Result<Rank, RenderFormattingTokenError> {
        self.format_token_ids
            .get(&t)
            .copied()
            .ok_or(RenderFormattingTokenError::UnmappedToken(t))
    }

    fn render_registered_media_token(
        &self,
        registration: &RegisteredMediaKind,
    ) -> anyhow::Result<Rank> {
        Ok(registration.token_id)
    }

    fn render_formatting_token_into<B>(
        &self,
        t: FormattingToken,
        into: &mut B,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        let r = self.render_formatting_token(t)?;
        into.extend(std::iter::once(r));
        Ok(())
    }

    fn render_text_into<T, B>(&self, text: T, into: &mut B) -> anyhow::Result<()>
    where
        T: AsRef<str>,
        B: Extend<Rank>,
    {
        into.extend(self.encode_ordinary_text(text.as_ref())?);
        Ok(())
    }

    /// Create a high-level completion stream parser.
    ///
    /// Use `CompletionStreamOptions::default()` for assistant completion-suffix
    /// mode (`role: Some(Role::Assistant)`, `strict: false`). Pass
    /// `CompletionStreamOptions { role: None, ..CompletionStreamOptions::default() }`
    /// to parse a full mHarmony message stream instead.
    pub fn completion_stream_parser(
        &self,
        options: CompletionStreamOptions,
    ) -> anyhow::Result<CompletionStreamParser> {
        let parser_options = ParseOptions {
            strict: options.strict,
        };
        Ok(CompletionStreamParser {
            inner: StreamableParser::new_with_options(
                self.clone(),
                options.role.clone(),
                parser_options,
            )?,
            options,
            completed_count: 0,
            visible_full: String::new(),
            final_full: String::new(),
            reasoning_full: String::new(),
            finished_stop: None,
            finished_without_stop: false,
        })
    }

    pub fn parse_messages_from_completion_tokens_with_options<I>(
        &self,
        tokens: I,
        role: Option<Role>,
        options: ParseOptions,
    ) -> anyhow::Result<Vec<Message>>
    where
        I: IntoIterator<Item = Rank>,
    {
        let mut parser = StreamableParser::new_with_options(self.clone(), role, options)?;
        for token in tokens {
            parser.process(token)?;
        }
        parser.process_eos()?;
        Ok(parser.into_messages())
    }

    pub fn parse_messages_from_completion_tokens<I>(
        &self,
        tokens: I,
        role: Option<Role>,
    ) -> anyhow::Result<Vec<Message>>
    where
        I: IntoIterator<Item = Rank>,
    {
        self.parse_messages_from_completion_tokens_with_options(
            tokens,
            role,
            ParseOptions::default(),
        )
    }

    /// Parse completion tokens once and return structured messages with aligned
    /// stable display text for each message.
    ///
    /// Stable message text is append-safe for streaming display and never emits
    /// mHarmony point sentinels. It can expose completed point children from an
    /// incomplete point span while suppressing incomplete or malformed point
    /// fragments. The returned text is per-message; channel, role, and recipient
    /// visibility filtering remains the caller's responsibility.
    pub fn parse_completion_view_from_tokens<I>(
        &self,
        tokens: I,
        role: Option<Role>,
        options: ParseOptions,
    ) -> anyhow::Result<CompletionParseView>
    where
        I: IntoIterator<Item = Rank>,
    {
        let mut parser = StreamableParser::new_with_options(self.clone(), role, options)?;
        for token in tokens {
            parser.process(token)?;
        }
        parser.process_eos()?;
        parser.into_completion_parse_view()
    }

    pub fn stable_visible_text_from_point_spans(&self, text: &str) -> String {
        stable_text_from_point_spans(
            text,
            self.mapped_format_token(FormattingToken::PointStart),
            self.mapped_format_token(FormattingToken::PointEnd),
        )
    }

    /// Helper to convert a JSON schema (OpenAPI style) to a TypeScript type definition.
    fn json_schema_to_typescript(schema: &serde_json::Value, indent: &str) -> String {
        // Helper to check if this schema is an enum
        fn is_enum(schema: &serde_json::Value) -> bool {
            schema
                .get("enum")
                .and_then(|e| e.as_array())
                .is_some_and(|arr| !arr.is_empty())
        }

        // Handle oneOf at the top level
        if let Some(one_of) = schema.get("oneOf") {
            if let Some(arr) = one_of.as_array() {
                let mut out = String::new();
                let mut first = true;
                for variant in arr {
                    if !first {
                        out.push('\n');
                        out.push_str(&format!("{indent} | "));
                    } else {
                        out.push_str(&format!("\n{indent} | "));
                        first = false;
                    }
                    let type_str =
                        Self::json_schema_to_typescript(variant, &format!("{indent}   "));
                    let mut type_str = type_str;
                    if variant
                        .get("nullable")
                        .and_then(|n| n.as_bool())
                        .unwrap_or(false)
                        && !type_str.contains("null")
                    {
                        type_str = format!("{type_str} | null");
                    }
                    out.push_str(&type_str);
                    // Add trailing comments (description, default)
                    let mut trailing_comments = Vec::new();
                    if let Some(desc) = variant.get("description") {
                        if let Some(desc_str) = desc.as_str() {
                            trailing_comments.push(desc_str.to_string());
                        }
                    }
                    if let Some(default) = variant.get("default") {
                        if default.is_string() && !is_enum(variant) {
                            trailing_comments
                                .push(format!("default: \"{}\"", default.as_str().unwrap()));
                        } else {
                            trailing_comments.push(format!("default: {default}"));
                        }
                    }
                    if !trailing_comments.is_empty() {
                        out.push_str(&format!(" // {}", trailing_comments.join(" ")));
                    }
                }
                return out;
            }
        }
        // Handle type as array (e.g., ["number", "string"])
        if let Some(types) = schema.get("type").and_then(|v| v.as_array()) {
            let mut type_strings = Vec::new();
            for ty in types {
                if let Some(ty_str) = ty.as_str() {
                    let mapped = match ty_str {
                        "integer" => "number",
                        other => other,
                    };
                    type_strings.push(mapped.to_string());
                }
            }
            if !type_strings.is_empty() {
                return type_strings.join(" | ");
            }
        }
        // Handle type
        if let Some(ty) = schema.get("type").and_then(|v| v.as_str()) {
            match ty {
                "object" => {
                    let mut out = String::new();
                    // Render object-level description as comment
                    if let Some(desc) = schema.get("description") {
                        if let Some(desc_str) = desc.as_str() {
                            out.push_str(&format!("{indent}// {desc_str}\n"));
                        }
                    }
                    out.push_str("{\n");

                    if let Some(props) = schema.get("properties") {
                        if let Some(props_map) = props.as_object() {
                            // Determine required fields
                            let mut required = std::collections::HashSet::new();
                            if let Some(req) = schema.get("required") {
                                if let Some(req_arr) = req.as_array() {
                                    for r in req_arr {
                                        if let Some(s) = r.as_str() {
                                            required.insert(s);
                                        }
                                    }
                                }
                            }
                            for (key, val) in props_map {
                                // Render title, description, and examples as comments
                                if let Some(title) = val.get("title") {
                                    if let Some(title_str) = title.as_str() {
                                        out.push_str(&format!(
                                            "{indent}// {title_str}\n{indent}//\n"
                                        ));
                                    }
                                }
                                // Only render description here if not a oneOf property
                                if val.get("oneOf").is_none() {
                                    if let Some(desc) = val.get("description") {
                                        if let Some(desc_str) = desc.as_str() {
                                            out.push_str(&format!("{indent}// {desc_str}\n"));
                                        }
                                    }
                                }
                                if let Some(examples) = val.get("examples") {
                                    if let Some(arr) = examples.as_array() {
                                        if !arr.is_empty() {
                                            out.push_str(&format!("{indent}// Examples:\n"));
                                            for ex in arr {
                                                if let Some(ex_str) = ex.as_str() {
                                                    out.push_str(&format!(
                                                        "{indent}// - \"{ex_str}\"\n"
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                                // Handle oneOf at the property level
                                if let Some(one_of) = val.get("oneOf") {
                                    if let Some(arr) = one_of.as_array() {
                                        // Deduplicate property-level description if it matches the first variant's description
                                        let mut property_desc: Option<&str> = None;
                                        if let Some(desc) = val.get("description") {
                                            if let Some(desc_str) = desc.as_str() {
                                                property_desc = Some(desc_str);
                                            }
                                        }
                                        let mut skip_property_desc = false;
                                        if let Some(desc_str) = property_desc {
                                            if let Some(first_variant) = arr.first() {
                                                if let Some(variant_desc) =
                                                    first_variant.get("description")
                                                {
                                                    if let Some(variant_desc_str) =
                                                        variant_desc.as_str()
                                                    {
                                                        if desc_str == variant_desc_str {
                                                            skip_property_desc = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        // Add property-level comments above the property name if not skipped
                                        let mut rendered_property_desc_above = false;
                                        if !skip_property_desc {
                                            if let Some(desc_str) = property_desc {
                                                out.push_str(&format!("{indent}// {desc_str}\n"));
                                                rendered_property_desc_above = true;
                                            }
                                        }
                                        if let Some(default) = val.get("default") {
                                            if default.is_string() && !is_enum(val) {
                                                out.push_str(&format!(
                                                    "{}// default: \"{}\"\n",
                                                    indent,
                                                    default.as_str().unwrap()
                                                ));
                                            } else if default.is_string() {
                                                out.push_str(&format!(
                                                    "{}// default: {}\n",
                                                    indent,
                                                    default.as_str().unwrap()
                                                ));
                                            } else {
                                                out.push_str(&format!(
                                                    "{indent}// default: {default}\n"
                                                ));
                                            }
                                        }
                                        // Add property name and optional marker
                                        out.push_str(&format!(
                                            "{}{}{}:\n",
                                            indent,
                                            key,
                                            if required.contains(key.as_str()) {
                                                ""
                                            } else {
                                                "?"
                                            }
                                        ));
                                        // Render each variant
                                        for (i, variant) in arr.iter().enumerate() {
                                            out.push_str(&format!("{indent} | "));
                                            let type_str = Self::json_schema_to_typescript(
                                                variant,
                                                &format!("{indent}   "),
                                            );
                                            // Handle nullable in variant
                                            let mut type_str = type_str;
                                            if variant
                                                .get("nullable")
                                                .and_then(|n| n.as_bool())
                                                .unwrap_or(false)
                                                && !type_str.contains("null")
                                            {
                                                type_str = format!("{type_str} | null");
                                            }
                                            out.push_str(&type_str);
                                            // Add variant-level comments after the type
                                            let mut trailing_comments = Vec::new();
                                            if i == 0 && rendered_property_desc_above {
                                                // Do not add any description for the first variant if property-level description was rendered above
                                            } else if let Some(desc) = variant.get("description") {
                                                if let Some(desc_str) = desc.as_str() {
                                                    // Only render if not equal to property-level description
                                                    if Some(desc_str) != property_desc {
                                                        trailing_comments
                                                            .push(desc_str.to_string());
                                                    }
                                                }
                                            }
                                            if let Some(default) = variant.get("default") {
                                                if default.is_string() && !is_enum(variant) {
                                                    trailing_comments.push(format!(
                                                        "default: \"{}\"",
                                                        default.as_str().unwrap()
                                                    ));
                                                } else if default.is_string() {
                                                    trailing_comments.push(format!(
                                                        "default: {}",
                                                        default.as_str().unwrap()
                                                    ));
                                                } else {
                                                    trailing_comments
                                                        .push(format!("default: {default}"));
                                                }
                                            }
                                            if !trailing_comments.is_empty() {
                                                out.push_str(&format!(
                                                    " // {}",
                                                    trailing_comments.join(" ")
                                                ));
                                            }
                                            out.push('\n');
                                        }
                                        out.push_str(&format!("{indent},\n"));
                                        continue;
                                    }
                                }
                                // Normal property rendering
                                out.push_str(&format!(
                                    "{}{}{}: ",
                                    indent,
                                    key,
                                    if required.contains(key.as_str()) {
                                        ""
                                    } else {
                                        "?"
                                    }
                                ));
                                // Handle nullable
                                let mut type_str =
                                    Self::json_schema_to_typescript(val, &format!("{indent}    "));
                                if val
                                    .get("nullable")
                                    .and_then(|n| n.as_bool())
                                    .unwrap_or(false)
                                    && !type_str.contains("null")
                                {
                                    type_str = format!("{type_str} | null");
                                }
                                out.push_str(&type_str);
                                out.push(',');
                                // Add default as comment if present (and not already handled)
                                if val.get("oneOf").is_none() {
                                    if let Some(default) = val.get("default") {
                                        if default.is_string() && !is_enum(val) {
                                            out.push_str(&format!(
                                                " // default: \"{}\"",
                                                default.as_str().unwrap()
                                            ));
                                        } else if default.is_string() {
                                            out.push_str(&format!(
                                                " // default: {}",
                                                default.as_str().unwrap()
                                            ));
                                        } else {
                                            out.push_str(&format!(" // default: {default}"));
                                        }
                                    }
                                }
                                out.push('\n');
                            }
                        }
                    }
                    out.push_str(&format!("{indent}}}"));
                    out
                }
                "string" => {
                    if let Some(enum_vals) = schema.get("enum") {
                        if let Some(arr) = enum_vals.as_array() {
                            let enums: Vec<String> = arr
                                .iter()
                                .filter_map(|v| v.as_str().map(|s| format!("\"{s}\"")))
                                .collect();
                            if !enums.is_empty() {
                                return enums.join(" | ");
                            }
                        }
                    }
                    "string".to_string()
                }
                "number" => "number".to_string(),
                "integer" => "number".to_string(),
                "boolean" => "boolean".to_string(),
                "array" => {
                    if let Some(items) = schema.get("items") {
                        format!("{}[]", Self::json_schema_to_typescript(items, indent))
                    } else {
                        "Array<any>".to_string()
                    }
                }
                _ => "any".to_string(),
            }
        } else if let Some(one_of) = schema.get("oneOf") {
            // Defensive: already handled above, but just in case
            if let Some(arr) = one_of.as_array() {
                let mut out = String::new();
                let mut first = true;
                for variant in arr {
                    if !first {
                        out.push_str("\n | ");
                    } else {
                        first = false;
                    }
                    out.push_str(&Self::json_schema_to_typescript(variant, indent));
                }
                return out;
            }
            "any".to_string()
        } else {
            "any".to_string()
        }
    }

    /// Helper to template the tools section for system content rendering.
    pub(crate) fn template_tools_section(
        tools: &std::collections::BTreeMap<String, crate::chat::ToolNamespaceConfig>,
    ) -> String {
        let mut tool_sections = Vec::<String>::new();
        tool_sections.push("# Tools".to_string());
        for ns_config in tools.values() {
            let mut tool_section_content = Vec::<String>::new();
            tool_section_content.push(format!("## {}\n", ns_config.name));
            if let Some(desc) = &ns_config.description {
                for line in desc.lines() {
                    if !ns_config.tools.is_empty() {
                        tool_section_content.push(format!("// {line}"));
                    } else {
                        tool_section_content.push(line.to_string());
                    }
                }
            }
            if !ns_config.tools.is_empty() {
                tool_section_content.push(format!("namespace {} {{\n", ns_config.name));
                for tool in &ns_config.tools {
                    for line in tool.description.lines() {
                        tool_section_content.push(format!("// {line}"));
                    }
                    if let Some(params) = &tool.parameters {
                        let param_type = Self::json_schema_to_typescript(params, "");
                        tool_section_content.push(format!(
                            "type {} = (_: {}) => any;\n",
                            tool.name, param_type
                        ));
                    } else {
                        tool_section_content.push(format!("type {} = () => any;\n", tool.name));
                    }
                }
                tool_section_content.push(format!("}} // namespace {}", ns_config.name));
            }
            tool_sections.push(tool_section_content.join("\n"));
        }
        tool_sections.join("\n\n")
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderOptions {
    pub conversation_has_function_tools: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MessageHeader {
    pub author: Author,
    pub recipient: Option<String>,
    pub channel: Option<String>,
    pub content_type: Option<String>,
}

impl MessageHeader {
    pub fn from_author(author: impl Into<Author>) -> Self {
        Self {
            author: author.into(),
            recipient: None,
            channel: None,
            content_type: None,
        }
    }

    pub fn from_role(role: Role) -> Self {
        Self::from_author(role)
    }

    pub fn assistant() -> Self {
        Self::from_role(Role::Assistant)
    }

    pub fn from_message(message: &Message) -> Self {
        Self {
            author: message.author.clone(),
            recipient: message.recipient.clone(),
            channel: message.channel.clone(),
            content_type: message.content_type.clone(),
        }
    }

    pub fn with_recipient<S>(mut self, recipient: S) -> Self
    where
        S: Into<String>,
    {
        self.recipient = Some(recipient.into());
        self
    }

    pub fn with_channel<S>(mut self, channel: S) -> Self
    where
        S: Into<String>,
    {
        self.channel = Some(channel.into());
        self
    }

    pub fn with_content_type<S>(mut self, content_type: S) -> Self
    where
        S: Into<String>,
    {
        self.content_type = Some(content_type.into());
        self
    }
}

impl From<&Message> for MessageHeader {
    fn from(message: &Message) -> Self {
        Self::from_message(message)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HeaderRenderOptions {
    pub include_start_token: bool,
    pub include_author: bool,
    pub include_recipient: bool,
    pub include_channel_marker: bool,
    pub include_channel_value: bool,
    pub include_content_type: bool,
    pub include_message_token: bool,
}

impl Default for HeaderRenderOptions {
    fn default() -> Self {
        Self {
            include_start_token: true,
            include_author: true,
            include_recipient: true,
            include_channel_marker: true,
            include_channel_value: true,
            include_content_type: true,
            include_message_token: true,
        }
    }
}

impl HeaderRenderOptions {
    pub const fn none() -> Self {
        Self {
            include_start_token: false,
            include_author: false,
            include_recipient: false,
            include_channel_marker: false,
            include_channel_value: false,
            include_content_type: false,
            include_message_token: false,
        }
    }

    pub const fn with_start_token(mut self, include: bool) -> Self {
        self.include_start_token = include;
        self
    }

    pub const fn with_author(mut self, include: bool) -> Self {
        self.include_author = include;
        self
    }

    pub const fn with_recipient(mut self, include: bool) -> Self {
        self.include_recipient = include;
        self
    }

    pub const fn with_channel_marker(mut self, include: bool) -> Self {
        self.include_channel_marker = include;
        self
    }

    pub const fn with_channel_value(mut self, include: bool) -> Self {
        self.include_channel_value = include;
        self
    }

    pub const fn with_content_type(mut self, include: bool) -> Self {
        self.include_content_type = include;
        self
    }

    pub const fn with_message_token(mut self, include: bool) -> Self {
        self.include_message_token = include;
        self
    }
}

trait Render<T: ?Sized> {
    fn render<B>(
        &self,
        item: &T,
        into: &mut B,
        render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>;
}

enum HeaderSegment {
    FormattingToken(FormattingToken),
    Text(String),
}

// Helper methods for message header/footer rendering (used by both render() and render_conversation_multimodal())
impl HarmonyEncoding {
    fn formatting_token_text(&self, token: FormattingToken) -> anyhow::Result<&str> {
        self.mapped_format_token(token)
            .ok_or_else(|| RenderFormattingTokenError::UnmappedToken(token).into())
    }

    fn message_header_segments(
        &self,
        header: &MessageHeader,
        options: HeaderRenderOptions,
    ) -> anyhow::Result<Vec<HeaderSegment>> {
        let mut segments = Vec::new();

        if options.include_start_token {
            segments.push(HeaderSegment::FormattingToken(FormattingToken::Start));
        }

        if options.include_author {
            if matches!(header.author.role, Role::Tool) {
                if let Some(name) = &header.author.name {
                    segments.push(HeaderSegment::Text(name.clone()));
                } else {
                    anyhow::bail!("Tools should have a name!");
                }
            } else {
                segments.push(HeaderSegment::Text(header.author.role.as_str().to_string()));
                if let Some(name) = &header.author.name {
                    segments.push(HeaderSegment::Text(format!(":{name}")));
                }
            };
        }

        if options.include_recipient {
            if let Some(recipient) = &header.recipient {
                if recipient != "all" {
                    segments.push(HeaderSegment::Text(format!(" to={recipient}")));
                }
            }
        }

        if let Some(channel) = &header.channel {
            if options.include_channel_marker {
                segments.push(HeaderSegment::FormattingToken(FormattingToken::Channel));
            }
            if options.include_channel_value {
                segments.push(HeaderSegment::Text(channel.clone()));
            }
        }

        if options.include_content_type {
            if let Some(content_type) = &header.content_type {
                // <|constrain|> is a unique case which needs to be tokenized as a special token
                if let Ok(constrain_marker) =
                    self.formatting_token_text(FormattingToken::ConstrainedFormat)
                {
                    if let Some(rest) = content_type.strip_prefix(constrain_marker) {
                        // Render the space, then the constrain marker as a special token, then the rest as text (if any)
                        segments.push(HeaderSegment::Text(" ".to_string()));
                        segments.push(HeaderSegment::FormattingToken(
                            FormattingToken::ConstrainedFormat,
                        ));
                        if !rest.is_empty() {
                            segments.push(HeaderSegment::Text(rest.to_string()));
                        }
                    } else {
                        segments.push(HeaderSegment::Text(format!(" {content_type}")));
                    }
                } else {
                    segments.push(HeaderSegment::Text(format!(" {content_type}")));
                }
            }
        }

        if options.include_message_token {
            segments.push(HeaderSegment::FormattingToken(FormattingToken::Message));
        }

        Ok(segments)
    }

    fn render_message_header_segments_into<B>(
        &self,
        segments: &[HeaderSegment],
        into: &mut B,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        for segment in segments {
            match segment {
                HeaderSegment::FormattingToken(token) => {
                    self.render_formatting_token_into(*token, into)?
                }
                HeaderSegment::Text(text) => self.render_text_into(text, into)?,
            }
        }
        Ok(())
    }

    pub fn render_message_header_tokens(
        &self,
        header: &MessageHeader,
        options: HeaderRenderOptions,
    ) -> anyhow::Result<Vec<Rank>> {
        let segments = self.message_header_segments(header, options)?;
        let mut tokens = Vec::new();
        self.render_message_header_segments_into(&segments, &mut tokens)?;
        Ok(tokens)
    }

    pub fn render_message_header_text(
        &self,
        header: &MessageHeader,
        options: HeaderRenderOptions,
    ) -> anyhow::Result<String> {
        let segments = self.message_header_segments(header, options)?;
        let mut text = String::new();
        for segment in segments {
            match segment {
                HeaderSegment::FormattingToken(token) => {
                    text.push_str(self.formatting_token_text(token)?)
                }
                HeaderSegment::Text(segment_text) => text.push_str(&segment_text),
            }
        }
        Ok(text)
    }

    /// Render message header: <|start|>{role}{recipient}{channel}{content_type}<|message|>
    pub(crate) fn render_message_header_into<B>(
        &self,
        message: &Message,
        into: &mut B,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        let header = MessageHeader::from(message);
        let segments = self.message_header_segments(&header, HeaderRenderOptions::default())?;
        self.render_message_header_segments_into(&segments, into)
    }

    /// Render message footer: <|end|> or <|call|> for tool calls
    pub(crate) fn render_message_footer_into<B>(
        &self,
        message: &Message,
        into: &mut B,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        // If there is a tool call we should render a tool call token
        if message.author.role == crate::chat::Role::Assistant && message.recipient.is_some() {
            self.render_formatting_token_into(FormattingToken::EndMessageAssistantToTool, into)?;
        } else {
            self.render_formatting_token_into(FormattingToken::EndMessage, into)?;
        }
        Ok(())
    }
}

impl Render<Message> for HarmonyEncoding {
    fn render<B>(
        &self,
        message: &Message,
        into: &mut B,
        render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        // Render header: <|start|>{role}{recipient}{channel}{content_type}<|message|>
        self.render_message_header_into(message, into)?;

        // Render content with validation
        for content in message.content.iter() {
            validate_content_role(message, content)?;
            Render::<Content>::render(self, content, into, render_options)?;
        }

        // Render footer: <|end|> or <|call|>
        self.render_message_footer_into(message, into)?;
        Ok(())
    }
}

// Dispatch Content variants to their specific Render implementations
impl Render<Content> for HarmonyEncoding {
    fn render<B>(
        &self,
        content: &Content,
        into: &mut B,
        render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        match content {
            Content::Text(text) => Render::<TextContent>::render(self, text, into, render_options),
            Content::Tokens(tokens) => {
                Render::<TokensContent>::render(self, tokens, into, render_options)
            }
            Content::Vector(vector) => {
                Render::<crate::chat::VectorContent>::render(self, vector, into, render_options)
            }
            Content::Media(media) => {
                Render::<crate::chat::MediaContent>::render(self, media, into, render_options)
            }
            Content::Pointing(pointing) => {
                Render::<PointingContent>::render(self, pointing, into, render_options)
            }
            Content::PointCollection(point_collection) => Render::<PointCollectionContent>::render(
                self,
                point_collection,
                into,
                render_options,
            ),
            Content::SystemContent(sys) => {
                Render::<SystemContent>::render(self, sys, into, render_options)
            }
            Content::DeveloperContent(dev) => {
                Render::<crate::chat::DeveloperContent>::render(self, dev, into, render_options)
            }
        }
    }
}

// Render plain text content
impl Render<TextContent> for HarmonyEncoding {
    fn render<B>(
        &self,
        text: &TextContent,
        into: &mut B,
        _render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        self.render_text_into(&text.text, into)
    }
}

impl Render<TokensContent> for HarmonyEncoding {
    fn render<B>(
        &self,
        tokens: &TokensContent,
        into: &mut B,
        _render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        validate_token_ids(&tokens.tokens)?;
        let offset_tokens = self.offset_tokens_for_tags(&tokens.tokens, &tokens.tags)?;
        into.extend(offset_tokens.iter().map(|t| *t as Rank));
        Ok(())
    }
}

impl Render<PointingContent> for HarmonyEncoding {
    fn render<B>(
        &self,
        pointing: &PointingContent,
        into: &mut B,
        _render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        self.render_formatting_token_into(FormattingToken::PointStart, into)?;
        self.render_text_into(pointing.to_html(), into)?;
        self.render_formatting_token_into(FormattingToken::PointEnd, into)?;
        Ok(())
    }
}

impl Render<PointCollectionContent> for HarmonyEncoding {
    fn render<B>(
        &self,
        point_collection: &PointCollectionContent,
        into: &mut B,
        _render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        self.render_formatting_token_into(FormattingToken::PointStart, into)?;
        self.render_text_into(point_collection.to_html(), into)?;
        self.render_formatting_token_into(FormattingToken::PointEnd, into)?;
        Ok(())
    }
}

// Placeholder renderers for non-text content. Host callbacks attach payloads
// when rendering a multimodal stream.
impl Render<crate::chat::VectorContent> for HarmonyEncoding {
    fn render<B>(
        &self,
        _vector: &crate::chat::VectorContent,
        into: &mut B,
        _render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        self.render_formatting_token_into(FormattingToken::Vector, into)
    }
}

impl Render<crate::chat::MediaContent> for HarmonyEncoding {
    fn render<B>(
        &self,
        media: &crate::chat::MediaContent,
        into: &mut B,
        _render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        let registration = self.registered_media_kind_for(media)?;
        into.extend(std::iter::once(
            self.render_registered_media_token(registration)?,
        ));
        Ok(())
    }
}

// Render system-specific content (model identity, instructions, effort)
impl Render<SystemContent> for HarmonyEncoding {
    fn render<B>(
        &self,
        sys: &SystemContent,
        into: &mut B,
        render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        let mut sections = Vec::<String>::new();

        let mut top_section = Vec::<String>::new();
        if let Some(model_id) = &sys.model_identity {
            top_section.push(model_id.clone());
        }
        if let Some(knowledge_cutoff) = &sys.knowledge_cutoff {
            top_section.push(format!("Knowledge cutoff: {knowledge_cutoff}"));
        }
        if let Some(conversation_start_date) = &sys.conversation_start_date {
            top_section.push(format!("Current date: {conversation_start_date}"));
        }
        if !top_section.is_empty() {
            sections.push(top_section.join("\n"));
        }

        let mut instructions_and_reasoning = Vec::<String>::new();
        if let Some(effort) = sys.reasoning_effort {
            let effort_str = match effort {
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High => "high",
            };
            instructions_and_reasoning.push(format!("Reasoning: {effort_str}"));
        }
        if let Some(hints) = &sys.hints {
            if !hints.is_empty() {
                let hint_strs: Vec<&str> = hints.iter().map(|hint| hint.as_str()).collect();
                instructions_and_reasoning.push(format!("Hints: {}", hint_strs.join(" ")));
            }
        }
        if !instructions_and_reasoning.is_empty() {
            sections.push(instructions_and_reasoning.join("\n"));
        }

        if let Some(tools) = &sys.tools {
            if !tools.is_empty() {
                sections.push(Self::template_tools_section(tools));
            }
        }

        if let Some(channel_config) = &sys.channel_config {
            if let Some(channels_header) = channels::render_valid_channels_header(
                channel_config,
                render_options.is_some_and(|o| o.conversation_has_function_tools),
            ) {
                sections.push(channels_header);
            }
        }
        let formatted = sections.join("\n\n");
        self.render_text_into(&formatted, into)?;
        Ok(())
    }
}

// Render developer-specific content (instructions, tools)
impl Render<crate::chat::DeveloperContent> for HarmonyEncoding {
    fn render<B>(
        &self,
        dev: &crate::chat::DeveloperContent,
        into: &mut B,
        _render_options: Option<&RenderOptions>,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        let mut sections = Vec::<String>::new();

        if let Some(instr) = &dev.instructions {
            sections.push("# Instructions".to_string());
            sections.push(instr.clone());
        }

        if let Some(tools) = &dev.tools {
            if !tools.is_empty() {
                sections.push(Self::template_tools_section(tools));
            }
        }
        let formatted = sections.join("\n\n");
        self.render_text_into(&formatted, into)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParseOptions {
    pub strict: bool,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self { strict: true }
    }
}

#[derive(Clone, Debug)]
pub struct MessageStreamAccumulator {
    header: ParsedHeader,
    content_tokens: Vec<Rank>,
    raw_text: String,
    point_parser: IncrementalPointParser,
    stable_text_full: String,
}

#[derive(Clone, Debug)]
struct FinishedStreamMessage {
    message: Message,
    stable_text: String,
    final_delta_parts: Vec<String>,
}

impl MessageStreamAccumulator {
    fn new(encoding: &HarmonyEncoding, header: ParsedHeader) -> Self {
        Self {
            header,
            content_tokens: Vec::new(),
            raw_text: String::new(),
            point_parser: IncrementalPointParser::new(
                encoding.mapped_format_token(FormattingToken::PointStart),
                encoding.mapped_format_token(FormattingToken::PointEnd),
            ),
            stable_text_full: String::new(),
        }
    }

    fn push_decoded_text(&mut self, decoded: &str) -> Vec<String> {
        self.raw_text.push_str(decoded);
        let parts = self.point_parser.push_text(decoded);
        for part in &parts {
            self.stable_text_full.push_str(part);
        }
        parts
    }

    fn finish(
        mut self,
        encoding: &HarmonyEncoding,
        strict: bool,
    ) -> anyhow::Result<FinishedStreamMessage> {
        let final_delta_parts = self.point_parser.finish();
        for part in &final_delta_parts {
            self.stable_text_full.push_str(part);
        }
        let split = split_text_with_pointing_view(
            &self.raw_text,
            encoding.mapped_format_token(FormattingToken::PointStart),
            encoding.mapped_format_token(FormattingToken::PointEnd),
            !strict,
        );
        Ok(FinishedStreamMessage {
            message: Message {
                author: self.header.author,
                recipient: self.header.recipient,
                channel: self.header.channel,
                content_type: self.header.content_type,
                content: split.content,
            },
            stable_text: self.stable_text_full,
            final_delta_parts,
        })
    }
}

/// Incremental parser that can consume tokens one by one.
///
/// It keeps track of all tokens seen so far, exposes all fully parsed messages
/// and retains the partially parsed state of the current message.
pub struct StreamableParser {
    encoding: HarmonyEncoding,
    next_role: Option<Role>,
    tokens: Vec<Rank>,
    messages: Vec<Message>,
    stable_message_texts: Vec<String>,
    state: StreamState,
    stop_tokens: HashSet<Rank>,
    last_content_delta: Option<String>,
    last_stable_delta_parts: Vec<String>,
    last_stop: Option<StreamStop>,
    undecoded_tokens: Vec<Rank>,
    undecoded_bytes: Vec<u8>,
    options: ParseOptions,
}

#[derive(Clone, Debug)]
// Boxing the accumulator would change the parser's public state representation and
// add indirection on every content token. Keep the established layout deliberately.
#[allow(clippy::large_enum_variant)]
pub enum StreamState {
    ExpectStart,
    Header {
        header_tokens: Vec<Rank>,
    },
    Content {
        accumulator: MessageStreamAccumulator,
    },
}

impl StreamableParser {
    /// Create a new streaming parser starting with the given role.
    pub fn new(encoding: HarmonyEncoding, role: Option<Role>) -> anyhow::Result<Self> {
        Self::new_with_options(encoding, role, ParseOptions::default())
    }

    /// Create a new streaming parser with explicit options.
    pub fn new_with_options(
        encoding: HarmonyEncoding,
        role: Option<Role>,
        options: ParseOptions,
    ) -> anyhow::Result<Self> {
        let stop_tokens = encoding.stop_tokens()?;
        let (state, next_role) = match role {
            Some(role) => (
                StreamState::Header {
                    header_tokens: Vec::new(),
                },
                Some(role),
            ),
            None => (StreamState::ExpectStart, None),
        };
        Ok(Self {
            encoding,
            next_role,
            tokens: Vec::new(),
            messages: Vec::new(),
            stable_message_texts: Vec::new(),
            state,
            stop_tokens,
            last_content_delta: None,
            last_stable_delta_parts: Vec::new(),
            last_stop: None,
            undecoded_tokens: Vec::new(),
            undecoded_bytes: Vec::new(),
            options,
        })
    }

    /// Consume a single token and update the internal state.
    fn process_next(&mut self, token: Option<Rank>) -> anyhow::Result<&mut Self> {
        self.last_content_delta = None;
        self.last_stable_delta_parts.clear();
        self.last_stop = None;
        if let Some(token) = token {
            self.tokens.push(token);
        }
        // Clone next_role up front to avoid borrow checker issues
        let next_role_clone = self.next_role.clone();
        match &mut self.state {
            StreamState::ExpectStart => {
                let start = self
                    .encoding
                    .render_formatting_token(FormattingToken::Start)?;
                match token {
                    Some(token) if token == start => {
                        self.state = StreamState::Header {
                            header_tokens: Vec::new(),
                        };
                    }
                    Some(token) => {
                        anyhow::bail!(
                            "Unexpected token {} while expecting start token {}",
                            token,
                            start
                        );
                    }
                    None => {
                        // receiving EOS while waiting for start token is actually fine
                        // as we may have just parsed a stop token. in this case we can
                        // simple keep state as is
                    }
                }
            }
            StreamState::Header { header_tokens } => {
                let msg_tok = self
                    .encoding
                    .render_formatting_token(FormattingToken::Message)?;
                match token {
                    Some(token) if token == msg_tok => {
                        // Clone the tokens and next_role, then clear the state before parsing
                        let header_tokens_cloned = header_tokens.clone();
                        let next_role_cloned = next_role_clone;
                        // Set state to dummy to drop mutable borrow
                        self.state = StreamState::ExpectStart;
                        let header =
                            self.parse_header_from_tokens(&header_tokens_cloned, next_role_cloned)?;
                        self.next_role = None;
                        self.state = StreamState::Content {
                            accumulator: MessageStreamAccumulator::new(&self.encoding, header),
                        };
                    }
                    Some(token) if self.options.strict && self.stop_tokens.contains(&token) => {
                        anyhow::bail!(
                            "unexpected stop token {} while parsing header (missing message token)",
                            token
                        );
                    }
                    Some(token) if !self.options.strict && self.stop_tokens.contains(&token) => {
                        // Encountered a stop token while in Header state. This means we have
                        // accumulated header tokens but never saw a <|message|> token, so the
                        // message is malformed. If we have a role, parse header metadata and
                        // treat remaining tokens as content.
                        if let Some(role) = next_role_clone {
                            if !header_tokens.is_empty() {
                                let decoded =
                                    self.encoding.tokenizer().decode_utf8(header_tokens)?;
                                let (header, remaining_content) =
                                    self.parse_header_from_string(decoded, Some(role), false)?;

                                // Use remaining content if present, otherwise empty string
                                let text = remaining_content.unwrap_or_default();
                                let split = split_text_with_pointing_view(
                                    &text,
                                    self.encoding
                                        .mapped_format_token(FormattingToken::PointStart),
                                    self.encoding.mapped_format_token(FormattingToken::PointEnd),
                                    true,
                                );
                                let message = Message {
                                    author: header.author.clone(),
                                    recipient: header.recipient.clone(),
                                    channel: header.channel.clone(),
                                    content_type: header.content_type.clone(),
                                    content: split.content,
                                };
                                self.messages.push(message);
                                self.stable_message_texts.push(split.stable_text);
                            }
                        }
                        // Transition to ExpectStart to wait for the next message
                        self.state = StreamState::ExpectStart;
                        self.next_role = None;
                    }
                    Some(token) => {
                        header_tokens.push(token);
                    }
                    None => {
                        // Allow an incomplete header at EOS if it parses cleanly. This happens
                        // for generation prompts that end with a bare "<|start|>{role}" header.
                        let header_tokens_cloned = header_tokens.clone();
                        let next_role_cloned = next_role_clone;
                        // Drop the mutable borrow before parsing.
                        self.state = StreamState::ExpectStart;
                        if self
                            .parse_header_from_tokens(&header_tokens_cloned, next_role_cloned)
                            .is_ok()
                        {
                            self.next_role = None;
                            return Ok(self);
                        }
                        // Restore state for consistency before erroring.
                        self.state = StreamState::Header {
                            header_tokens: header_tokens_cloned,
                        };
                        anyhow::bail!(
                            "Failed to parse incomplete header tokens while handling EOS"
                        );
                    }
                }
            }
            StreamState::Content { accumulator } => {
                let stop_token = token.filter(|token| self.stop_tokens.contains(token));
                let is_eos = if let Some(token) = token {
                    if stop_token.is_some() {
                        true
                    } else {
                        accumulator.content_tokens.push(token);
                        self.undecoded_tokens.push(token);
                        if let Ok(decoded_bytes) = self
                            .encoding
                            .tokenizer()
                            .decode_bytes(&self.undecoded_tokens)
                        {
                            self.undecoded_bytes.extend(decoded_bytes.iter().copied());
                            match String::from_utf8(self.undecoded_bytes.clone()) {
                                Ok(decoded_str) => {
                                    self.last_stable_delta_parts =
                                        accumulator.push_decoded_text(&decoded_str);
                                    self.last_content_delta = Some(decoded_str);
                                    self.undecoded_bytes.clear();
                                }
                                Err(e) => {
                                    let utf8_error = e.utf8_error();
                                    let decoded_bytes = e.into_bytes();
                                    let valid_len = utf8_error.valid_up_to();
                                    let mut content_delta = String::new();
                                    if valid_len > 0 {
                                        let valid_str =
                                            String::from_utf8(decoded_bytes[..valid_len].to_vec())
                                                .unwrap();
                                        self.last_stable_delta_parts
                                            .extend(accumulator.push_decoded_text(&valid_str));
                                        content_delta.push_str(&valid_str);
                                        self.undecoded_bytes.drain(..valid_len);
                                    }
                                    if let Some(error_len) = utf8_error.error_len() {
                                        self.last_stable_delta_parts
                                            .extend(accumulator.push_decoded_text(REPLACEMENT));
                                        content_delta.push_str(REPLACEMENT);
                                        self.undecoded_bytes.drain(..error_len);
                                    }
                                    if !content_delta.is_empty() {
                                        self.last_content_delta = Some(content_delta);
                                    }
                                }
                            }
                            self.undecoded_tokens.clear();
                        }
                        false
                    }
                } else {
                    true
                };
                if is_eos {
                    let tokens_text = match self
                        .encoding
                        .tokenizer()
                        .decode_utf8(self.undecoded_tokens.clone())
                    {
                        Ok(text) => text,
                        Err(_) => REPLACEMENT.to_string(),
                    };
                    let bytes_text = String::from_utf8_lossy(&self.undecoded_bytes).to_string();
                    if !tokens_text.is_empty() {
                        self.last_stable_delta_parts
                            .extend(accumulator.push_decoded_text(&tokens_text));
                    }
                    if !bytes_text.is_empty() {
                        self.last_stable_delta_parts
                            .extend(accumulator.push_decoded_text(&bytes_text));
                    }
                    let accumulator =
                        match std::mem::replace(&mut self.state, StreamState::ExpectStart) {
                            StreamState::Content { accumulator } => accumulator,
                            _ => unreachable!(),
                        };
                    let finished = accumulator.finish(&self.encoding, self.options.strict)?;
                    self.last_stable_delta_parts
                        .extend(finished.final_delta_parts.clone());
                    self.last_stop = stop_token
                        .and_then(|token| self.classify_stream_stop(&finished.message, token));
                    self.stable_message_texts.push(finished.stable_text);
                    self.messages.push(finished.message);
                    self.last_content_delta = None;
                    self.undecoded_tokens.clear();
                    self.undecoded_bytes.clear();
                }
            }
        }
        Ok(self)
    }

    pub fn process(&mut self, token: Rank) -> anyhow::Result<&mut Self> {
        self.process_next(Some(token))
    }

    pub fn process_eos(&mut self) -> anyhow::Result<&mut Self> {
        self.process_next(None)?;
        Ok(self)
    }

    /// Helper to parse header metadata from a decoded string.
    /// Returns the parsed header and any remaining content after extracting header parts.
    ///
    /// If `parse_recipient_and_type` is true, tries to parse recipient and content_type from
    /// whitespace-separated tokens (normal header parsing). If false, treats all remaining
    /// text after extracting channel as content (for malformed messages).
    fn parse_header_from_string(
        &self,
        mut header_string: String,
        role: Option<Role>,
        parse_recipient_and_type: bool,
    ) -> anyhow::Result<(ParsedHeader, Option<String>)> {
        let mut channel: Option<String> = None;
        if let Some(channel_marker) = self.encoding.mapped_format_token(FormattingToken::Channel) {
            if let Some(idx) = header_string.find(channel_marker) {
                let after_marker = &header_string[idx + channel_marker.len()..];
                let channel_end = after_marker
                    .find(|c: char| c.is_whitespace() || c == '<')
                    .unwrap_or(after_marker.len());
                let channel_value = &after_marker[..channel_end];
                if channel_value.is_empty() {
                    anyhow::bail!("channel marker present but no channel value found in header");
                }
                channel = Some(channel_value.to_string());

                let mut new_header = String::new();
                new_header.push_str(&header_string[..idx]);
                new_header.push_str(&after_marker[channel_end..]);
                header_string = new_header;
            }
        }

        // Trim extraneous whitespace that may have been introduced when we
        // removed the channel section.
        header_string = header_string.trim().to_string();

        // If the constrained format marker is present but not preceded by
        // whitespace (e.g. "to=foo<|constrain|>json"), insert a space before
        // the marker so that splitting on whitespace treats the content type
        // as a separate token.
        if let Some(constrain_marker) = self
            .encoding
            .mapped_format_token(FormattingToken::ConstrainedFormat)
        {
            if header_string.contains(constrain_marker) {
                header_string = header_string
                    .replace(constrain_marker, &format!(" {constrain_marker}"))
                    .trim()
                    .to_string();
            }
        }

        let mut parts: Vec<&str> = header_string.split_ascii_whitespace().collect();

        let mut role_str_opt: Option<String> = None;
        let role = match role {
            Some(r) => r,
            None => {
                let role_str = parts
                    .first()
                    .context("message header did not contain a role")?;
                role_str_opt = Some((*role_str).to_string());
                let parsed_role = Role::try_from(*role_str);
                let out = match parsed_role {
                    Ok(r) => r,
                    Err(_) => {
                        // If recipient is present, treat as tool call
                        if parts.len() > 1 || (parts.len() == 1 && parts[0].starts_with("to=")) {
                            parts.remove(0); // Remove the unknown role string
                            Role::Tool
                        } else {
                            return Err(anyhow::anyhow!("Unknown role: {}", role_str));
                        }
                    }
                };
                out
            }
        };

        if let Some(&first) = parts.first() {
            if first == role.as_str() {
                parts.remove(0);
            }
        }

        let mut recipient: Option<String> = None;
        let mut content_type: Option<String> = None;
        let remaining_content: Option<String>;

        if parse_recipient_and_type && !parts.is_empty() {
            let num_parts = parts.len();
            // SAFETY: we know that there is at least one part remaining, because of is_empty check above
            let last_part = parts.pop().unwrap();

            if let Some(stripped) = last_part.strip_prefix("to=") {
                // The header contains a recipient but *no* content-type.
                recipient = Some(stripped.to_string());
            } else if num_parts == 1 {
                // Only one part total (after potential role removal) and it doesn't start
                // with "to=" => interpret it as a standalone recipient.
                recipient = Some(last_part.to_string());
            } else {
                // More than one token and the last one is not a recipient -> treat as content-type.
                content_type = Some(last_part.to_string());

                // After removing the content-type there may be exactly one token describing the recipient.
                if let Some(raw_recipient) = parts.pop() {
                    recipient = if let Some(stripped) = raw_recipient.strip_prefix("to=") {
                        Some(stripped.to_string())
                    } else {
                        Some(raw_recipient.to_string())
                    };
                }
            }

            // Any remaining parts are content (not header metadata)
            remaining_content = if !parts.is_empty() {
                Some(parts.join(" "))
            } else {
                None
            };
        } else {
            // Treat all remaining parts as content when not parsing recipient and content type
            remaining_content = if !parts.is_empty() {
                Some(parts.join(" "))
            } else {
                None
            };
        }

        let author = if role == Role::Tool {
            let name = role_str_opt;
            Author { role, name }
        } else {
            Author { role, name: None }
        };
        Ok((
            ParsedHeader {
                author,
                recipient,
                channel,
                content_type,
            },
            remaining_content,
        ))
    }

    fn parse_header_from_tokens(
        &self,
        header_tokens: &[Rank],
        role: Option<Role>,
    ) -> anyhow::Result<ParsedHeader> {
        let header_string = self
            .encoding
            .tokenizer()
            .decode_utf8(header_tokens)
            .context("could not decode header")?;

        let (header, remaining_content) =
            self.parse_header_from_string(header_string, role, true)?;

        if remaining_content.is_some() {
            anyhow::bail!(
                "unexpected tokens remaining in message header: {:?}",
                remaining_content
            );
        }

        Ok(header)
    }

    /// Return the textual content of the current message so far.
    pub fn current_content(&self) -> anyhow::Result<String> {
        match &self.state {
            StreamState::Content { accumulator } => Ok(accumulator.raw_text.clone()),
            _ => Ok(String::new()),
        }
    }

    /// Role of the current message if it has been parsed.
    pub fn current_role(&self) -> Option<Role> {
        match &self.state {
            StreamState::Content { accumulator } => Some(accumulator.header.author.role.clone()),
            _ => self.next_role.clone(),
        }
    }

    /// Current content type if known.
    pub fn current_content_type(&self) -> Option<String> {
        match &self.state {
            StreamState::Content { accumulator } => accumulator.header.content_type.clone(),
            _ => None,
        }
    }

    /// Decode the last content delta if available.
    pub fn last_content_delta(&self) -> anyhow::Result<Option<String>> {
        Ok(self.last_content_delta.clone())
    }

    /// Consume the parser and return all parsed messages.
    pub fn into_messages(self) -> Vec<Message> {
        self.messages
    }

    /// Consume the parser and return parsed messages with aligned stable text.
    pub fn into_completion_parse_view(self) -> anyhow::Result<CompletionParseView> {
        if self.messages.len() != self.stable_message_texts.len() {
            anyhow::bail!(
                "internal parser invariant violated: messages/stable text length mismatch ({} != {})",
                self.messages.len(),
                self.stable_message_texts.len()
            );
        }
        Ok(CompletionParseView {
            messages: self.messages,
            stable_message_texts: self.stable_message_texts,
        })
    }

    /// All fully parsed messages so far.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// All tokens that were fed into the parser.
    pub fn tokens(&self) -> &[Rank] {
        &self.tokens
    }

    /// Expose the current state as a JSON string for Python interop.
    pub fn state_json(&self) -> anyhow::Result<String> {
        #[derive(serde::Serialize)]
        #[serde(tag = "state")]
        enum SerializableStreamState<'a> {
            ExpectStart,
            Header {
                header_tokens: &'a [Rank],
            },
            Content {
                header: &'a ParsedHeader,
                content_tokens: &'a [Rank],
            },
        }
        let serializable = match &self.state {
            StreamState::ExpectStart => SerializableStreamState::ExpectStart,
            StreamState::Header { header_tokens } => {
                SerializableStreamState::Header { header_tokens }
            }
            StreamState::Content { accumulator } => SerializableStreamState::Content {
                header: &accumulator.header,
                content_tokens: &accumulator.content_tokens,
            },
        };
        Ok(serde_json::to_string(&serializable)?)
    }

    /// Return the current recipient if known.
    pub fn current_recipient(&self) -> Option<String> {
        match &self.state {
            StreamState::Content { accumulator } => accumulator.header.recipient.clone(),
            _ => None,
        }
    }

    /// Return the current channel if known.
    pub fn current_channel(&self) -> Option<String> {
        match &self.state {
            StreamState::Content { accumulator } => accumulator.header.channel.clone(),
            _ => None,
        }
    }

    fn take_last_stable_delta_parts(&mut self) -> Vec<String> {
        std::mem::take(&mut self.last_stable_delta_parts)
    }

    fn last_stop(&self) -> Option<StreamStop> {
        self.last_stop
    }

    fn classify_stream_stop(&self, message: &Message, token_id: Rank) -> Option<StreamStop> {
        if message.author.role != Role::Assistant {
            return None;
        }
        let is_final_stop_token = self
            .encoding
            .render_formatting_token(FormattingToken::EndMessageDoneSampling)
            .is_ok_and(|stop_token| token_id == stop_token)
            || self
                .encoding
                .render_formatting_token(FormattingToken::EndMessage)
                .is_ok_and(|stop_token| token_id == stop_token);
        if is_final_stop_token
            && channels::is_channel(message.channel.as_deref(), CanonicalChannel::Final)
        {
            return Some(StreamStop {
                kind: StreamStopKind::Final,
            });
        }
        if token_id
            == self
                .encoding
                .render_formatting_token(FormattingToken::EndMessageAssistantToTool)
                .ok()?
            && channels::is_channel(message.channel.as_deref(), CanonicalChannel::Commentary)
            && normalize_recipient(message.recipient.as_deref()).is_some()
        {
            return Some(StreamStop {
                kind: StreamStopKind::ToolCall,
            });
        }
        None
    }
}

fn normalize_recipient(recipient: Option<&str>) -> Option<&str> {
    let recipient = recipient?.trim();
    if recipient.is_empty() || recipient == "assistant" || recipient.starts_with("<|") {
        None
    } else {
        Some(recipient)
    }
}

#[derive(Clone, Copy)]
enum StablePartRoute {
    Hidden,
    VisibleOnly,
    Reasoning { visible: bool },
    Final,
}

fn stable_part_route(
    role: &Role,
    channel: Option<&str>,
    recipient: Option<&str>,
    include_thinking_channels: bool,
) -> StablePartRoute {
    if *role != Role::Assistant {
        return StablePartRoute::Hidden;
    }
    let normalized_channel =
        channel.map(|channel| CanonicalChannel::parse(&channel.to_ascii_lowercase()));
    match normalized_channel {
        Some(Some(CanonicalChannel::Analysis)) if normalize_recipient(recipient).is_some() => {
            if include_thinking_channels {
                StablePartRoute::VisibleOnly
            } else {
                StablePartRoute::Hidden
            }
        }
        Some(Some(CanonicalChannel::Analysis)) => StablePartRoute::Reasoning {
            visible: include_thinking_channels,
        },
        Some(Some(CanonicalChannel::Commentary)) if normalize_recipient(recipient).is_some() => {
            StablePartRoute::Hidden
        }
        None if normalize_recipient(recipient).is_some() => StablePartRoute::Hidden,
        Some(Some(CanonicalChannel::Final)) | Some(Some(CanonicalChannel::Commentary)) | None => {
            StablePartRoute::Final
        }
        Some(None) => StablePartRoute::Hidden,
    }
}

fn append_text_delta(full: &mut String, delta_parts: &mut Vec<String>, text: String) {
    full.push_str(&text);
    delta_parts.push(text);
}

impl CompletionStreamParser {
    pub fn current_recipient(&self) -> Option<String> {
        self.inner.current_recipient()
    }

    pub fn current_channel(&self) -> Option<String> {
        self.inner.current_channel()
    }

    pub fn push_tokens(
        &mut self,
        token_ids_delta: &[Rank],
        absolute_start: usize,
    ) -> anyhow::Result<CompletionStreamUpdate> {
        let expected_start = self.inner.tokens().len();
        if self.finished_without_stop {
            if token_ids_delta.is_empty() && absolute_start == expected_start {
                return Ok(self.finished_noop_update(None));
            }
            anyhow::bail!(
                "completion stream parser is already finished at {expected_start} tokens"
            );
        }
        if let Some(stop) = self.finished_stop {
            if token_ids_delta.is_empty() && absolute_start == expected_start {
                return Ok(self.finished_noop_update(Some(stop)));
            }
            anyhow::bail!(
                "completion stream parser is already finished at {expected_start} tokens"
            );
        }
        if absolute_start != expected_start {
            anyhow::bail!(
                "completion stream parser offset mismatch: expected absolute_start {}, got {}",
                expected_start,
                absolute_start
            );
        }

        let mut visible_delta_parts = Vec::new();
        let mut final_delta_parts = Vec::new();
        let mut reasoning_delta_parts = Vec::new();
        let starting_completed_count = self.completed_count;
        let mut stop = None;

        for token in token_ids_delta {
            self.inner.process(*token)?;

            self.append_last_stable_delta_parts(
                &mut visible_delta_parts,
                &mut final_delta_parts,
                &mut reasoning_delta_parts,
            );

            if let Some(candidate_stop) = self.inner.last_stop() {
                match candidate_stop.kind {
                    StreamStopKind::Final => {
                        stop = Some(candidate_stop);
                        break;
                    }
                    StreamStopKind::ToolCall if self.options.stop_on_tool_call => {
                        stop = Some(candidate_stop);
                        break;
                    }
                    StreamStopKind::ToolCall => {}
                }
            }
        }

        let completed_messages = self.inner.messages()[starting_completed_count..].to_vec();
        self.completed_count = self.inner.messages().len();
        if stop.is_some() {
            self.finished_stop = stop;
        }
        Ok(CompletionStreamUpdate {
            token_count: self.inner.tokens().len(),
            completed_messages,
            stop,
            text: self.text_update(
                visible_delta_parts,
                final_delta_parts,
                reasoning_delta_parts,
            ),
        })
    }

    pub fn finish(&mut self, absolute_start: usize) -> anyhow::Result<CompletionStreamUpdate> {
        let expected_start = self.inner.tokens().len();
        if absolute_start != expected_start {
            anyhow::bail!(
                "completion stream parser finish offset mismatch: expected absolute_start {}, got {}",
                expected_start,
                absolute_start
            );
        }
        if self.finished_without_stop {
            return Ok(self.finished_noop_update(None));
        }
        if let Some(stop) = self.finished_stop {
            return Ok(self.finished_noop_update(Some(stop)));
        }

        let mut visible_delta_parts = Vec::new();
        let mut final_delta_parts = Vec::new();
        let mut reasoning_delta_parts = Vec::new();
        let starting_completed_count = self.completed_count;

        self.inner.process_eos()?;
        self.append_last_stable_delta_parts(
            &mut visible_delta_parts,
            &mut final_delta_parts,
            &mut reasoning_delta_parts,
        );

        let completed_messages = self.inner.messages()[starting_completed_count..].to_vec();
        self.completed_count = self.inner.messages().len();
        self.finished_without_stop = true;
        Ok(CompletionStreamUpdate {
            token_count: self.inner.tokens().len(),
            completed_messages,
            stop: None,
            text: self.text_update(
                visible_delta_parts,
                final_delta_parts,
                reasoning_delta_parts,
            ),
        })
    }

    fn finished_noop_update(&self, stop: Option<StreamStop>) -> CompletionStreamUpdate {
        CompletionStreamUpdate {
            token_count: self.inner.tokens().len(),
            completed_messages: Vec::new(),
            stop,
            text: self.text_update(Vec::new(), Vec::new(), Vec::new()),
        }
    }

    fn append_last_stable_delta_parts(
        &mut self,
        visible_delta_parts: &mut Vec<String>,
        final_delta_parts: &mut Vec<String>,
        reasoning_delta_parts: &mut Vec<String>,
    ) {
        let (role, channel, recipient) = self.delta_route_metadata();
        let parts = self.inner.take_last_stable_delta_parts();
        if parts.is_empty() {
            return;
        }
        let route = stable_part_route(
            &role,
            channel.as_deref(),
            recipient.as_deref(),
            self.options.include_thinking_channels,
        );
        for part in parts {
            if part.is_empty() {
                continue;
            }
            match route {
                StablePartRoute::Hidden => {}
                StablePartRoute::VisibleOnly => {
                    append_text_delta(&mut self.visible_full, visible_delta_parts, part);
                }
                StablePartRoute::Reasoning { visible } => {
                    if visible {
                        append_text_delta(
                            &mut self.visible_full,
                            visible_delta_parts,
                            part.clone(),
                        );
                    }
                    append_text_delta(&mut self.reasoning_full, reasoning_delta_parts, part);
                }
                StablePartRoute::Final => {
                    append_text_delta(&mut self.visible_full, visible_delta_parts, part.clone());
                    append_text_delta(&mut self.final_full, final_delta_parts, part);
                }
            }
        }
    }

    fn text_update(
        &self,
        visible_delta_parts: Vec<String>,
        final_delta_parts: Vec<String>,
        reasoning_delta_parts: Vec<String>,
    ) -> StreamTextUpdate {
        StreamTextUpdate {
            visible_delta: visible_delta_parts.concat(),
            visible_full: self.visible_full.clone(),
            final_delta: final_delta_parts.concat(),
            final_full: self.final_full.clone(),
            reasoning_delta: reasoning_delta_parts.concat(),
            reasoning_full: self.reasoning_full.clone(),
            visible_delta_parts,
            final_delta_parts,
            reasoning_delta_parts,
        }
    }

    fn delta_route_metadata(&self) -> (Role, Option<String>, Option<String>) {
        if let StreamState::Content { accumulator } = &self.inner.state {
            return (
                accumulator.header.author.role.clone(),
                accumulator.header.channel.clone(),
                accumulator.header.recipient.clone(),
            );
        }
        if let Some(message) = self.inner.messages.last() {
            return (
                message.author.role.clone(),
                message.channel.clone(),
                message.recipient.clone(),
            );
        }
        (Role::Assistant, None, None)
    }
}

// Add config struct for rendering
#[derive(Clone, Debug)]
pub struct RenderConversationConfig {
    pub auto_drop_analysis: bool,
}

impl Default for RenderConversationConfig {
    fn default() -> Self {
        Self {
            auto_drop_analysis: true,
        }
    }
}
