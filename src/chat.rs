// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
use core::fmt;

use crate::channels;
use regex::Regex;
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::{fmt::Display, marker::PhantomData, sync::LazyLock};

static RAW_MARKUP_ATTR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(\w+)\s*=\s*"([^"]*)""#).expect("valid attribute regex"));
static POINT_COORD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*\)").expect("valid coordinate regex")
});
static CLIP_TIME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(\d+(?:\.\d+)?)(?:\s*seconds|s)?").expect("valid clip time regex")
});

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Author {
    pub role: Role,
    pub name: Option<String>,
}

impl Author {
    pub fn new(role: Role, name: impl Into<String>) -> Self {
        Self {
            role,
            name: Some(name.into()),
        }
    }
}

impl From<Role> for Author {
    fn from(role: Role) -> Self {
        Self { role, name: None }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
    Developer,
    Tool,
}

impl TryFrom<&str> for Role {
    type Error = &'static str;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "system" => Ok(Role::System),
            "developer" => Ok(Role::Developer),
            "tool" => Ok(Role::Tool),
            _ => Err("Unknown role"),
        }
    }
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Developer => "developer",
            Role::Tool => "tool",
        }
    }
}

impl Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Content {
    Text(TextContent),
    Tokens(TokensContent),
    Vector(VectorContent),
    /// Modality-agnostic media envelope. Concrete kinds are registered by the
    /// host rather than represented in the public type system.
    Media(MediaContent),
    Pointing(PointingContent),
    PointCollection(PointCollectionContent),
    /// Special content for system-level instructions
    SystemContent(SystemContent),
    /// Special content for developer-level instructions
    DeveloperContent(DeveloperContent),
}

impl<T> From<T> for Content
where
    T: Into<String>,
{
    fn from(text: T) -> Self {
        Self::Text(TextContent { text: text.into() })
    }
}

impl From<SystemContent> for Content {
    fn from(sys: SystemContent) -> Self {
        Self::SystemContent(sys)
    }
}

impl From<DeveloperContent> for Content {
    fn from(dev: DeveloperContent) -> Self {
        Self::DeveloperContent(dev)
    }
}

impl From<PointingContent> for Content {
    fn from(pointing: PointingContent) -> Self {
        Self::Pointing(pointing)
    }
}

impl From<PointCollectionContent> for Content {
    fn from(point_collection: PointCollectionContent) -> Self {
        Self::PointCollection(point_collection)
    }
}

impl From<TokensContent> for Content {
    fn from(tokens: TokensContent) -> Self {
        Self::Tokens(tokens)
    }
}

impl From<MediaContent> for Content {
    fn from(media: MediaContent) -> Self {
        Self::Media(media)
    }
}

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Message {
    /// An object representing the author of the message, including
    /// their role (e.g., user, assistant) and any additional metadata.
    #[serde(flatten)]
    pub author: Author,

    /// The intended recipient of the message. If not set, the message is
    /// is intend for all (this is the default). Can also be set to specific
    /// identifiers (e.g., 'user', 'assistant', etc.) In the case of a tool call,
    /// the recipient is the name of the tool.
    pub recipient: Option<String>,

    /// The main content of the message. This can be of various types
    /// (e.g., text, code) and structures, depending on the type of `MessageContent` used.
    #[serde(
        deserialize_with = "de_string_or_content_vec",
        serialize_with = "se_string_or_content_vec"
    )]
    pub content: Vec<Content>,

    /// Specifies the target channel (context) for the message, allowing
    /// models to annotate their responses and e.g. control message visibility.
    /// By default, messages do not have channel set (None) and it's not rendered.
    /// When set and render_channel=True, the channel is rendered in message header as
    /// <|channel|>CHANNEL_VALUE (usually, "<|meta_sep|>CHANNEL_VALUE"). To not render
    /// channels, use `formatter.render_channel = False`. (note: parsing will raise an error
    /// if render_channel=False, but a channel was sampled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,

    /// Content type of the message. This is typically only set by the model, you probably don't need to set this.
    pub content_type: Option<String>,
}

impl Message {
    pub fn from_author_and_content<C>(author: Author, content: C) -> Self
    where
        C: Into<Content>,
    {
        Message {
            author,
            content: vec![content.into()],
            channel: None,
            recipient: None,
            content_type: None,
        }
    }

    pub fn from_role_and_content<C>(role: Role, content: C) -> Self
    where
        C: Into<Content>,
    {
        Self::from_author_and_content(Author { role, name: None }, content)
    }

    pub fn from_role_and_contents<I>(role: Role, content: I) -> Self
    where
        I: IntoIterator<Item = Content>,
    {
        Message {
            author: Author { role, name: None },
            content: content.into_iter().collect(),
            channel: None,
            recipient: None,
            content_type: None,
        }
    }
    pub fn adding_content<C>(mut self, content: C) -> Self
    where
        C: Into<Content>,
    {
        self.content.push(content.into());
        self
    }
    pub fn with_channel<S>(mut self, channel: S) -> Self
    where
        S: Into<String>,
    {
        self.channel = Some(channel.into());
        self
    }
    pub fn with_recipient<S>(mut self, recipient: S) -> Self
    where
        S: Into<String>,
    {
        self.recipient = Some(recipient.into());
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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TextContent {
    pub text: String,
}

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct TokensContent {
    pub tokens: Vec<i64>,
    #[serde(default)]
    pub tags: HashMap<String, Value>,
}

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct VectorContent {
    pub values: Vec<f32>,
    pub shape: Vec<i64>,
    pub dtype: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub name: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MediaRef {
    pub id: String,
    pub mime: Option<String>,
    pub bytes_b64: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MediaContent {
    /// Registered media kind, for example `image` or `video`.
    pub kind: String,
    pub media_ref: MediaRef,
    pub metadata: Option<serde_json::Value>,
    /// Optional host-defined source descriptor. The core keeps this value
    /// opaque and forwards it to the registered processor callback.
    pub source: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PointingKind {
    Point,
    Box,
    Polygon,
    Clip,
}

impl PointingKind {
    fn html_tag(self) -> &'static str {
        match self {
            PointingKind::Point => "point",
            PointingKind::Box => "point_box",
            PointingKind::Polygon => "polygon",
            PointingKind::Clip => "clip",
        }
    }

    fn from_html_tag(tag: &str) -> Option<Self> {
        match tag {
            "point" => Some(PointingKind::Point),
            "point_box" => Some(PointingKind::Box),
            "polygon" => Some(PointingKind::Polygon),
            "clip" => Some(PointingKind::Clip),
            _ => None,
        }
    }

    fn is_self_closing(self) -> bool {
        matches!(self, PointingKind::Clip)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum PointCollectionKind {
    #[default]
    Collection,
    Track,
}

impl PointCollectionKind {
    fn html_tag(self) -> &'static str {
        match self {
            PointCollectionKind::Collection => "collection",
            PointCollectionKind::Track => "track",
        }
    }

    fn from_html_tag(tag: &str) -> Option<Self> {
        match tag {
            "collection" => Some(PointCollectionKind::Collection),
            "track" => Some(PointCollectionKind::Track),
            _ => None,
        }
    }

    fn is_collection(kind: &Self) -> bool {
        matches!(kind, PointCollectionKind::Collection)
    }
}

/// Either a leaf pointing element or a nested point collection, used to share
/// child-tag dispatch between `PointCollectionContent::from_html` and the
/// streaming-stable boundary finder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChildKind {
    Pointing(PointingKind),
    Collection(PointCollectionKind),
}

impl ChildKind {
    fn from_html_tag(tag: &str) -> Option<Self> {
        PointingKind::from_html_tag(tag)
            .map(Self::Pointing)
            .or_else(|| PointCollectionKind::from_html_tag(tag).map(Self::Collection))
    }

    fn is_self_closing(self) -> bool {
        matches!(self, ChildKind::Pointing(kind) if kind.is_self_closing())
    }
}

fn close_tag(tag: &str) -> String {
    format!("</{tag}>")
}

fn escape_attr(value: &str) -> String {
    html_escape::encode_double_quoted_attribute(value).into_owned()
}

fn unescape_attr(value: &str) -> String {
    html_escape::decode_html_entities(value).into_owned()
}

fn format_t(t: f64) -> String {
    if t.fract() == 0.0 {
        format!("{:.1}", t)
    } else {
        let mut s = format!("{:.15}", t);
        if let Some(dot) = s.find('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.len() == dot + 1 {
                s.push('0');
            }
        }
        s
    }
}

fn parse_raw_markup_attrs(attrs_raw: &str) -> anyhow::Result<HashMap<String, String>> {
    // Spatial point attrs use `t` as a scalar float (`t="1.0"`), but clip attrs use human-readable time
    // spans (`t="1 seconds 2.5 seconds"` or legacy `t="1s 2.5s"`). Parse attrs as raw strings first so
    // clip parsing can interpret the `t` value without the scalar float parser rejecting it.
    let mut attrs = HashMap::new();
    for cap in RAW_MARKUP_ATTR_RE.captures_iter(attrs_raw) {
        let key = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let raw_val = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        attrs.insert(key.to_string(), unescape_attr(raw_val));
    }
    Ok(attrs)
}

fn parse_scalar_time_attr(val: &str) -> anyhow::Result<f64> {
    let trimmed = val.trim();
    let lower = trimmed.to_ascii_lowercase();
    let numeric = if lower.ends_with("seconds") {
        &trimmed[..trimmed.len() - "seconds".len()]
    } else if lower.ends_with("second") {
        &trimmed[..trimmed.len() - "second".len()]
    } else if lower.ends_with('s') {
        &trimmed[..trimmed.len() - 's'.len_utf8()]
    } else {
        trimmed
    }
    .trim();

    if numeric.is_empty() {
        anyhow::bail!("Invalid t value '{val}' in point markup");
    }
    numeric
        .parse::<f64>()
        .map_err(|_| anyhow::anyhow!("Invalid t value '{val}' in point markup"))
}

type ParsedMarkupAttrs = (Option<String>, Option<String>, Option<String>, Option<f64>);

fn parse_markup_attrs(attrs_raw: &str) -> anyhow::Result<ParsedMarkupAttrs> {
    let attrs = parse_raw_markup_attrs(attrs_raw)?;
    let mut mention: Option<String> = None;
    let mut label: Option<String> = None;
    let mut index: Option<String> = None;
    let mut t: Option<f64> = None;
    for (key, val) in attrs {
        match key.as_str() {
            "mention" => {
                if !val.is_empty() {
                    mention = Some(val);
                }
            }
            "label" => {
                if !val.is_empty() {
                    label = Some(val);
                }
            }
            "index" => {
                if !val.is_empty() {
                    index = Some(val);
                }
            }
            "t" => {
                t = Some(parse_scalar_time_attr(&val)?);
            }
            _ => {}
        }
    }
    Ok((mention, label, index, t))
}

fn format_clip_time(seconds: f64) -> String {
    let mut formatted = format!("{seconds:.6}");
    if let Some(dot) = formatted.find('.') {
        while formatted.ends_with('0') {
            formatted.pop();
        }
        if formatted.len() == dot + 1 {
            formatted.pop();
        }
    }
    format!("{formatted} seconds")
}

fn format_clip_attrs(
    mention: Option<&str>,
    index: Option<&str>,
    start_time: Option<f64>,
    end_time: Option<f64>,
) -> String {
    let mut attrs: Vec<String> = Vec::new();
    if let Some(mention) = mention {
        if !mention.is_empty() {
            attrs.push(format!("mention=\"{}\"", escape_attr(mention)));
        }
    }
    if let Some(index) = index {
        if !index.is_empty() {
            attrs.push(format!("index=\"{}\"", escape_attr(index)));
        }
    }
    if let Some(start_time) = start_time {
        let mut time_values = vec![format_clip_time(start_time)];
        if let Some(end_time) = end_time {
            time_values.push(format_clip_time(end_time));
        }
        attrs.push(format!("t=\"{}\"", escape_attr(&time_values.join(" "))));
    }
    if attrs.is_empty() {
        String::new()
    } else {
        format!(" {}", attrs.join(" "))
    }
}

/// Single assembly point for the mention -> t -> index attribute order:
/// `t` arrives PRE-FORMATTED (numeric callers go through `format_t`, display
/// callers pass the final string), so the two public open-tag builders and
/// the `Render` impls cannot drift on ordering or escaping.
fn format_markup_attrs_display_t(
    mention: Option<&str>,
    index: Option<&str>,
    t: Option<&str>,
) -> String {
    let mut attrs: Vec<String> = Vec::new();
    if let Some(mention) = mention {
        if !mention.is_empty() {
            attrs.push(format!("mention=\"{}\"", escape_attr(mention)));
        }
    }
    if let Some(t) = t {
        attrs.push(format!("t=\"{}\"", escape_attr(t)));
    }
    if let Some(index) = index {
        if !index.is_empty() {
            attrs.push(format!("index=\"{}\"", escape_attr(index)));
        }
    }
    if attrs.is_empty() {
        String::new()
    } else {
        format!(" {}", attrs.join(" "))
    }
}

fn format_markup_attrs(mention: Option<&str>, index: Option<&str>, t: Option<f64>) -> String {
    let t_formatted = t.map(format_t);
    format_markup_attrs_display_t(mention, index, t_formatted.as_deref())
}

/// Build the opening markup tag (`<{tag}{ATTRS}>`) for a pointing primitive or
/// collection. Uses the same `mention` / `t` / `index` formatting that
/// `Render<PointingContent>` and `Render<PointCollectionContent>` use, so
/// Python callers that need to emit the opening tag themselves (e.g. when
/// splitting a primitive's render across multiple `HarmonyContent` blocks)
/// stay byte-identical with the legacy text emitter.
///
/// `tag` is one of `"point"`, `"point_box"`, `"polygon"`, `"clip"`,
/// `"collection"` (i.e. any markup tag that uses the standard
/// mention/index/t attribute set).
pub fn format_markup_open_tag(
    tag: &str,
    mention: Option<&str>,
    index: Option<&str>,
    t: Option<f64>,
) -> String {
    let attr_str = format_markup_attrs(mention, index, t);
    format!("<{tag}{attr_str}>")
}

/// Like [`format_markup_open_tag`], but `t` is a pre-formatted display string
/// emitted verbatim (attr-escaped) in the same attribute position — e.g.
/// `t="1.2 seconds"`. Used by the Python coord-token render for temporally
/// pre-aligned primitives, whose `t` carries the final `"X seconds"`
/// presentation form (the in-band per-primitive "aligned" marker) instead of
/// a raw numeric frame index.
pub fn format_markup_open_tag_display_t(
    tag: &str,
    mention: Option<&str>,
    index: Option<&str>,
    t: Option<&str>,
) -> String {
    let attr_str = format_markup_attrs_display_t(mention, index, t);
    format!("<{tag}{attr_str}>")
}

struct MarkupOpenTag<'a> {
    tag: &'a str,
    attrs_raw: &'a str,
    open_end: usize,
}

fn parse_markup_open_tag_at(input: &str, start: usize) -> Option<MarkupOpenTag<'_>> {
    let tail = input.get(start..)?;
    let after_open = tail.strip_prefix('<')?;
    let open_end = start + 1 + after_open.find('>')?;
    let open_tag = &input[(start + 1)..open_end];
    let tag = open_tag.split_whitespace().next()?;
    let attrs_raw = &open_tag[tag.len()..];
    Some(MarkupOpenTag {
        tag,
        attrs_raw,
        open_end,
    })
}

fn parse_markup_tag_and_body(input: &str) -> anyhow::Result<(&str, &str, &str)> {
    let s = input.trim();
    let open_tag = parse_markup_open_tag_at(s, 0)
        .ok_or_else(|| anyhow::anyhow!("No opening tag found in '{s}'"))?;
    if open_tag.attrs_raw.trim_end().ends_with('/') {
        return Ok((open_tag.tag, open_tag.attrs_raw, ""));
    }
    let close_tag = close_tag(open_tag.tag);
    let close_start = s
        .rfind(&close_tag)
        .ok_or_else(|| anyhow::anyhow!("Missing closing tag for '{}'", open_tag.tag))?;
    if close_start <= open_tag.open_end {
        anyhow::bail!("Invalid point markup '{s}'");
    }
    let body = s[(open_tag.open_end + 1)..close_start].trim();
    Ok((open_tag.tag, open_tag.attrs_raw, body))
}

fn pointing_min_coordinate(pointing: &PointingContent) -> (i64, i64) {
    match pointing.kind {
        PointingKind::Point | PointingKind::Box => {
            pointing.coords.first().copied().unwrap_or((0, 0))
        }
        PointingKind::Polygon => pointing
            .coords
            .iter()
            .min_by_key(|(x, y)| (*y, *x))
            .copied()
            .unwrap_or((0, 0)),
        PointingKind::Clip => (0, 0),
    }
}

fn pointing_start_time(pointing: &PointingContent) -> f64 {
    match pointing.kind {
        PointingKind::Clip => pointing.start_time.unwrap_or(0.0),
        _ => pointing.t.unwrap_or(0.0),
    }
}

fn pointing_end_time(pointing: &PointingContent) -> f64 {
    match pointing.kind {
        PointingKind::Clip => {
            let start_time = pointing.start_time.unwrap_or(0.0);
            pointing.end_time.unwrap_or(start_time)
        }
        _ => pointing.t.unwrap_or(0.0),
    }
}

fn compare_pointings(a: &PointingContent, b: &PointingContent) -> Ordering {
    let time_cmp = pointing_start_time(a).total_cmp(&pointing_start_time(b));
    if time_cmp != Ordering::Equal {
        return time_cmp;
    }
    let end_time_cmp = pointing_end_time(a).total_cmp(&pointing_end_time(b));
    if end_time_cmp != Ordering::Equal {
        return end_time_cmp;
    }
    let (ax, ay) = pointing_min_coordinate(a);
    let (bx, by) = pointing_min_coordinate(b);
    ay.cmp(&by).then_with(|| ax.cmp(&bx))
}

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PointingContent {
    pub kind: PointingKind,
    #[serde(default)]
    pub coords: Vec<(i64, i64)>,
    pub target: Option<u32>,
    pub t: Option<f64>,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub mention: Option<String>,
    pub label: Option<String>,
    pub index: Option<String>,
}

impl PointingContent {
    pub fn to_html(&self) -> String {
        let tag = self.kind.html_tag();

        if self.kind.is_self_closing() {
            let attr_str = format_clip_attrs(
                self.mention.as_deref(),
                self.index.as_deref(),
                self.start_time,
                self.end_time,
            );
            return format!("<{tag}{attr_str} />");
        }

        let attr_str = format_markup_attrs(self.mention.as_deref(), self.index.as_deref(), self.t);
        let coords = self
            .coords
            .iter()
            .map(|(x, y)| format!("({x},{y})"))
            .collect::<Vec<_>>()
            .join(" ");

        format!("<{tag}{attr_str}> {coords} </{tag}>")
    }

    pub fn from_html(input: &str) -> anyhow::Result<Self> {
        let (tag, attrs_raw, body) = parse_markup_tag_and_body(input)?;

        let kind = PointingKind::from_html_tag(tag)
            .ok_or_else(|| anyhow::anyhow!("Unsupported point tag '{tag}'"))?;

        if kind == PointingKind::Clip {
            let raw_attrs = parse_raw_markup_attrs(attrs_raw)?;
            let mention = raw_attrs
                .get("mention")
                .filter(|val| !val.is_empty())
                .cloned();
            let label = raw_attrs
                .get("label")
                .filter(|val| !val.is_empty())
                .cloned();
            let index = raw_attrs
                .get("index")
                .filter(|val| !val.is_empty())
                .cloned();
            let t_value = raw_attrs.get("t").ok_or_else(|| {
                anyhow::anyhow!("Clip requires t attribute for start and optional end time")
            })?;
            let (start_time, end_time) = parse_clip_times(t_value)?;
            return Ok(PointingContent {
                kind,
                coords: vec![],
                target: None,
                t: None,
                start_time: Some(start_time),
                end_time,
                mention,
                label,
                index,
            });
        }

        let (mention, label, index, t) = parse_markup_attrs(attrs_raw)?;

        let mut coords: Vec<(i64, i64)> = POINT_COORD_RE
            .captures_iter(body)
            .map(|cap| Ok((cap["x"].parse()?, cap["y"].parse()?)))
            .collect::<anyhow::Result<_>>()?;

        // Coord-token decoder overrides surface consecutive coord ids as
        // whitespace-separated integers, e.g. `<point> 10  20 </point>`,
        // rather than the legacy `(10,20)` representation. Only use this
        // strict fallback when no parenthesized coordinates were found so the
        // established legacy parser behaviour remains unchanged.
        if coords.is_empty() && !body.trim().is_empty() {
            let values = body
                .split_whitespace()
                .map(|value| {
                    if !value.chars().all(|ch| ch.is_ascii_digit()) {
                        anyhow::bail!(
                            "Expected whitespace-separated integer coordinates, got '{value}'"
                        );
                    }
                    value.parse::<i64>().map_err(Into::into)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            if values.len() % 2 != 0 {
                anyhow::bail!(
                    "Expected an even number of coordinate values, got {}",
                    values.len()
                );
            }
            coords = values
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect();
        }

        match kind {
            PointingKind::Point => {
                if coords.len() != 1 {
                    anyhow::bail!("Expected exactly one point, got {}", coords.len());
                }
            }
            PointingKind::Box => {
                if coords.len() != 2 {
                    anyhow::bail!("Expected two points for bounding box, got {}", coords.len());
                }
            }
            PointingKind::Polygon => {
                if coords.is_empty() {
                    anyhow::bail!("Expected at least one point for polygon");
                }
            }
            PointingKind::Clip => {}
        }

        Ok(PointingContent {
            kind,
            coords,
            target: None,
            t,
            start_time: None,
            end_time: None,
            mention,
            label,
            index,
        })
    }
}

fn parse_clip_times(t_value: &str) -> anyhow::Result<(f64, Option<f64>)> {
    let tokens: Vec<&str> = CLIP_TIME_RE
        .captures_iter(t_value)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str()))
        .collect();

    if tokens.is_empty() {
        anyhow::bail!("Clip time attribute is empty");
    }
    if tokens.len() > 2 {
        anyhow::bail!("Invalid clip time value '{t_value}'");
    }

    let start_time = tokens[0].parse::<f64>()?;
    let end_time = if tokens.len() == 2 {
        Some(tokens[1].parse::<f64>()?)
    } else {
        None
    };
    Ok((start_time, end_time))
}

#[serde_with::skip_serializing_none]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PointCollectionContent {
    pub points: Vec<PointCollectionItem>,
    #[serde(default, skip_serializing_if = "PointCollectionKind::is_collection")]
    pub kind: PointCollectionKind,
    pub t: Option<f64>,
    pub mention: Option<String>,
    pub label: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum PointCollectionItem {
    Pointing(PointingContent),
    PointCollection(PointCollectionContent),
}

impl From<PointingContent> for PointCollectionItem {
    fn from(pointing: PointingContent) -> Self {
        Self::Pointing(pointing)
    }
}

impl From<PointCollectionContent> for PointCollectionItem {
    fn from(point_collection: PointCollectionContent) -> Self {
        Self::PointCollection(point_collection)
    }
}

impl PointCollectionItem {
    fn canonicalize(&mut self) {
        if let PointCollectionItem::PointCollection(collection) = self {
            collection.canonicalize();
        }
    }

    fn to_html(&self) -> String {
        match self {
            PointCollectionItem::Pointing(pointing) => pointing.to_html(),
            PointCollectionItem::PointCollection(collection) => collection.to_html(),
        }
    }

    fn min_pointing(&self) -> Option<&PointingContent> {
        match self {
            PointCollectionItem::Pointing(pointing) => Some(pointing),
            PointCollectionItem::PointCollection(collection) => collection
                .points
                .iter()
                .filter_map(PointCollectionItem::min_pointing)
                .min_by(|a, b| compare_pointings(a, b)),
        }
    }
}

fn compare_point_collection_items(a: &PointCollectionItem, b: &PointCollectionItem) -> Ordering {
    match (a.min_pointing(), b.min_pointing()) {
        (Some(a_pointing), Some(b_pointing)) => compare_pointings(a_pointing, b_pointing),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

impl PointCollectionContent {
    pub fn canonicalize(&mut self) {
        // Track containers carry the mention on the container itself, so child
        // mentions are redundant. Strip them here so render and parse paths
        // share a single source of truth for the invariant.
        if self.kind == PointCollectionKind::Track {
            for item in self.points.iter_mut() {
                if let PointCollectionItem::Pointing(point) = item {
                    point.mention = None;
                }
            }
        }
        for point in self.points.iter_mut() {
            point.canonicalize();
        }
        self.points.sort_by(compare_point_collection_items);
    }

    pub fn canonicalized(mut self) -> Self {
        self.canonicalize();
        self
    }

    pub fn to_html(&self) -> String {
        let canonical = self.clone().canonicalized();
        let points = canonical
            .points
            .into_iter()
            .map(|point| point.to_html().trim_end_matches('\n').to_string())
            .collect::<Vec<_>>();
        let attr_str = format_markup_attrs(self.mention.as_deref(), None, self.t);
        let col_sep = "\n";
        let body = points.join("\n ");
        let tag = self.kind.html_tag();
        format!("<{tag}{attr_str}>{col_sep} {body}{col_sep}</{tag}>{col_sep}")
    }

    pub fn from_html(input: &str) -> anyhow::Result<Self> {
        let (tag, attrs_raw, body) = parse_markup_tag_and_body(input)?;
        let kind = PointCollectionKind::from_html_tag(tag)
            .ok_or_else(|| anyhow::anyhow!("Unsupported collection tag '{tag}'"))?;

        let (mention, label, _index, t) = parse_markup_attrs(attrs_raw)?;
        let mut points: Vec<PointCollectionItem> = Vec::new();
        let mut cursor = 0usize;
        while cursor < body.len() {
            let remaining = &body[cursor..];
            if remaining.trim().is_empty() {
                break;
            }

            let rel_start = remaining
                .find('<')
                .ok_or_else(|| anyhow::anyhow!("Unsupported content inside collection"))?;
            if !remaining[..rel_start].trim().is_empty() {
                anyhow::bail!("Unsupported content inside collection");
            }

            let child_start = cursor + rel_start;
            let child_open_end = body[child_start..]
                .find('>')
                .map(|idx| child_start + idx)
                .ok_or_else(|| anyhow::anyhow!("Missing child tag terminator in collection"))?;
            let child_open_tag = &body[(child_start + 1)..child_open_end];
            let child_tag = child_open_tag
                .split_whitespace()
                .next()
                .ok_or_else(|| anyhow::anyhow!("Missing child tag name in collection"))?;
            let child_kind = ChildKind::from_html_tag(child_tag)
                .ok_or_else(|| anyhow::anyhow!("Unsupported content inside collection"))?;

            let child_end = if child_kind.is_self_closing()
                && child_open_tag.trim_end().ends_with('/')
            {
                child_open_end + 1
            } else {
                let close_tag = close_tag(child_tag);
                let rel_close = body[(child_open_end + 1)..]
                    .find(&close_tag)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Missing closing tag for child '{child_tag}'")
                    })?;
                child_open_end + 1 + rel_close + close_tag.len()
            };
            let child_markup = &body[child_start..child_end];
            points.push(match child_kind {
                ChildKind::Pointing(_) => {
                    PointCollectionItem::Pointing(PointingContent::from_html(child_markup)?)
                }
                ChildKind::Collection(_) => PointCollectionItem::PointCollection(
                    PointCollectionContent::from_html(child_markup)?,
                ),
            });
            cursor = child_end;
        }
        if points.is_empty() {
            anyhow::bail!("No valid child points found in collection");
        }
        if kind == PointCollectionKind::Track {
            let first_kind = match &points[0] {
                PointCollectionItem::Pointing(point) => point.kind,
                PointCollectionItem::PointCollection(_) => {
                    anyhow::bail!("Track point collection can only contain pointing children")
                }
            };
            if first_kind == PointingKind::Clip {
                anyhow::bail!("Track points must be point, point_box, or polygon");
            }
            for item in points.iter() {
                let PointCollectionItem::Pointing(point) = item else {
                    anyhow::bail!("Track point collection can only contain pointing children");
                };
                if point.kind != first_kind {
                    anyhow::bail!("Track points must all be the same type");
                }
            }
        } else {
            for item in points.iter() {
                if let PointCollectionItem::PointCollection(collection) = item {
                    if collection.kind != PointCollectionKind::Track {
                        anyhow::bail!(
                            "Collection point collection can contain track children, but not nested collections"
                        );
                    }
                }
            }
        }

        Ok(PointCollectionContent {
            points,
            kind,
            t,
            mention,
            label,
        }
        .canonicalized())
    }
}

fn stable_pointing_child_end(payload: &str, start: usize) -> Option<usize> {
    let open_tag = parse_markup_open_tag_at(payload, start)?;
    let kind = PointingKind::from_html_tag(open_tag.tag)?;
    let child_end = if kind.is_self_closing() {
        if !open_tag.attrs_raw.trim_end().ends_with('/') {
            return None;
        }
        open_tag.open_end + 1
    } else {
        let close_tag = close_tag(kind.html_tag());
        open_tag.open_end
            + 1
            + payload.get((open_tag.open_end + 1)..)?.find(&close_tag)?
            + close_tag.len()
    };
    Some(child_end)
}

/// Streaming-stable boundary finder for a child of a `PointCollectionContent`.
///
/// Pointing children defer to `stable_pointing_child_end`. Collection children
/// are only accepted in the one nesting we support — `Collection > Track`.
/// Anything else (`Track > *`, `Collection > Collection`) returns `None`, which
/// causes the streaming parser to leave the markup as raw text rather than
/// emit an incomplete structured span.
fn stable_point_collection_child_end(
    payload: &str,
    start: usize,
    parent_kind: PointCollectionKind,
) -> Option<usize> {
    let open_tag = parse_markup_open_tag_at(payload, start)?;
    match ChildKind::from_html_tag(open_tag.tag)? {
        ChildKind::Pointing(kind) => stable_pointing_child_end(payload, start)
            .filter(|_| !kind.is_self_closing() || open_tag.attrs_raw.trim_end().ends_with('/')),
        ChildKind::Collection(kind) => {
            if parent_kind != PointCollectionKind::Collection || kind != PointCollectionKind::Track
            {
                return None;
            }
            let close_tag = close_tag(kind.html_tag());
            Some(
                open_tag.open_end
                    + 1
                    + payload.get((open_tag.open_end + 1)..)?.find(&close_tag)?
                    + close_tag.len(),
            )
        }
    }
}

fn stable_top_level_point_prefix(payload: &str) -> &str {
    let leading_ws_len = payload.len() - payload.trim_start().len();
    let Some(child_end) = stable_pointing_child_end(payload, leading_ws_len) else {
        return "";
    };

    let mut stable_end = child_end;
    while stable_end < payload.len() {
        let Some(ch) = payload[stable_end..].chars().next() else {
            break;
        };
        if !ch.is_whitespace() {
            break;
        }
        stable_end += ch.len_utf8();
    }
    &payload[..stable_end]
}

fn stable_point_collection_prefix(payload: &str) -> &str {
    let leading_ws_len = payload.len() - payload.trim_start().len();
    let Some(open_tag) = parse_markup_open_tag_at(payload, leading_ws_len) else {
        return "";
    };
    let Some(kind) = PointCollectionKind::from_html_tag(open_tag.tag) else {
        return "";
    };
    if parse_markup_attrs(open_tag.attrs_raw).is_err() {
        return "";
    }

    let mut cursor = open_tag.open_end + 1;
    let mut stable_end = cursor;
    let collection_close_tag = close_tag(kind.html_tag());

    while cursor < payload.len() {
        let mut whitespace_end = cursor;
        while whitespace_end < payload.len() {
            let Some(ch) = payload[whitespace_end..].chars().next() else {
                break;
            };
            if !ch.is_whitespace() {
                break;
            }
            whitespace_end += ch.len_utf8();
        }

        stable_end = whitespace_end;

        if payload[whitespace_end..].starts_with(&collection_close_tag) {
            stable_end = whitespace_end + collection_close_tag.len();
            while stable_end < payload.len() {
                let Some(ch) = payload[stable_end..].chars().next() else {
                    break;
                };
                if !ch.is_whitespace() {
                    break;
                }
                stable_end += ch.len_utf8();
            }
            break;
        }

        let Some(child_end) = stable_point_collection_child_end(payload, whitespace_end, kind)
        else {
            break;
        };
        cursor = child_end;
        stable_end = child_end;
    }

    &payload[..stable_end]
}

pub(crate) fn stable_point_payload_prefix(payload: &str) -> &str {
    let leading_ws_len = payload.len() - payload.trim_start().len();
    match parse_markup_open_tag_at(payload, leading_ws_len).map(|tag| tag.tag) {
        Some(tag) if PointCollectionKind::from_html_tag(tag).is_some() => {
            stable_point_collection_prefix(payload)
        }
        _ => stable_top_level_point_prefix(payload),
    }
}

pub fn parse_point_payload(input: &str) -> anyhow::Result<Content> {
    let trimmed = input.trim_start();
    if parse_markup_open_tag_at(trimmed, 0)
        .is_some_and(|tag| PointCollectionKind::from_html_tag(tag.tag).is_some())
    {
        Ok(Content::PointCollection(PointCollectionContent::from_html(
            input,
        )?))
    } else {
        Ok(Content::Pointing(PointingContent::from_html(input)?))
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SystemHint {
    Point,
    Box,
    Polygon,
    Clip,
    Focus,
    Think,
    Tools,
}

impl SystemHint {
    pub fn as_str(&self) -> &'static str {
        match self {
            SystemHint::Point => "POINT",
            SystemHint::Box => "BOX",
            SystemHint::Polygon => "POLYGON",
            SystemHint::Clip => "CLIP",
            SystemHint::Focus => "FOCUS",
            SystemHint::Think => "THINK",
            SystemHint::Tools => "TOOLS",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct ChannelConfig {
    /// List of valid channels to instruct the model it can generate.
    ///
    /// If empty, this part of the system message will not be rendered.
    pub valid_channels: Vec<String>,

    /// If True, every assistant's message must have channel value set.
    pub channel_required: bool,
}

impl ChannelConfig {
    pub fn require_channels<I, T>(channels: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            valid_channels: channels.into_iter().map(|c| c.into()).collect(),
            channel_required: true,
        }
    }

    pub fn require_canonical_channels() -> Self {
        Self::require_channels(channels::canonical_assistant_channels())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolNamespaceConfig {
    pub name: String,
    pub description: Option<String>,
    pub tools: Vec<ToolDescription>,
}

impl ToolNamespaceConfig {
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        tools: Vec<ToolDescription>,
    ) -> Self {
        Self {
            name: name.into(),
            description,
            tools,
        }
    }

    pub fn browser() -> Self {
        ToolNamespaceConfig::new(
            "browser",
            Some("Tool for browsing.\nThe `cursor` appears in brackets before each browsing display: `[{cursor}]`.\nCite information from the tool using the following format:\n`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\nDo not quote more than 10 words directly from the tool output.\nsources=web (default: web)".to_string()),
            vec![
                ToolDescription::new(
                    "search",
                    "Searches for information related to `query` and displays `topn` results.",
                    Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "topn": {"type": "number", "default": 10},
                            "source": {"type": "string"}
                        },
                        "required": ["query"]
                    })),
                ),
                ToolDescription::new(
                    "open",
                    "Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\nValid link ids are displayed with the formatting: `【{id}†.*】`.\nIf `cursor` is not provided, the most recent page is implied.\nIf `id` is a string, it is treated as a fully qualified URL associated with `source`.\nIf `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\nUse this function without `id` to scroll to a new location of an opened page.",
                    Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": ["number", "string"],
                                "default": -1
                            },
                            "cursor": {"type": "number", "default": -1},
                            "loc": {"type": "number", "default": -1},
                            "num_lines": {"type": "number", "default": -1},
                            "view_source": {"type": "boolean", "default": false},
                            "source": {"type": "string"}
                        }
                    })),
                ),
                ToolDescription::new(
                    "find",
                    "Finds exact matches of `pattern` in the current page, or the page given by `cursor`.",
                    Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "cursor": {"type": "number", "default": -1}
                        },
                        "required": ["pattern"]
                    })),
                ),
            ],
        )
    }

    pub fn python() -> Self {
        ToolNamespaceConfig::new(
            "python",
            Some("Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n\nWhen you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.".to_string()),
            vec![],
        )
    }
}

/// Content specific to system messages, includes model identity and its instructions
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SystemContent {
    pub model_identity: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub hints: Option<Vec<SystemHint>>,
    pub tools: Option<BTreeMap<String, ToolNamespaceConfig>>,
    /// Date/Time at which the conversation is taking place.
    /// Must be an isoformat date for portability to javascript.
    pub conversation_start_date: Option<String>,

    /// The date at which the model's training data ends.
    pub knowledge_cutoff: Option<String>,

    /// Channel configuration for the system message.
    pub channel_config: Option<ChannelConfig>,
}

impl Default for SystemContent {
    fn default() -> Self {
        Self {
            model_identity: Some("You are ChatGPT, a large language model.".to_string()),
            reasoning_effort: Some(ReasoningEffort::Medium),
            hints: None,
            tools: None,
            conversation_start_date: None,
            knowledge_cutoff: Some("2024-06".to_string()),
            channel_config: Some(ChannelConfig::require_canonical_channels()),
        }
    }
}

impl SystemContent {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn empty() -> Self {
        Self {
            model_identity: None,
            reasoning_effort: None,
            hints: None,
            tools: None,
            conversation_start_date: None,
            knowledge_cutoff: None,
            channel_config: None,
        }
    }
    pub fn with_model_identity(mut self, model_identity: impl Into<String>) -> Self {
        self.model_identity = Some(model_identity.into());
        self
    }
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }
    pub fn with_hints(mut self, hints: Vec<SystemHint>) -> Self {
        self.hints = Some(hints);
        self
    }
    pub fn with_hint(mut self, hint: SystemHint) -> Self {
        let mut hints = self.hints.take().unwrap_or_default();
        hints.push(hint);
        self.hints = Some(hints);
        self
    }
    pub fn with_tools(mut self, ns_config: ToolNamespaceConfig) -> Self {
        let ns = ns_config.name.clone();
        if let Some(ref mut map) = self.tools {
            map.insert(ns, ns_config);
        } else {
            let mut map = BTreeMap::new();
            map.insert(ns, ns_config);
            self.tools = Some(map);
        }
        self
    }
    pub fn with_conversation_start_date(
        mut self,
        conversation_start_date: impl Into<String>,
    ) -> Self {
        self.conversation_start_date = Some(conversation_start_date.into());
        self
    }
    pub fn with_knowledge_cutoff(mut self, knowledge_cutoff: impl Into<String>) -> Self {
        self.knowledge_cutoff = Some(knowledge_cutoff.into());
        self
    }
    pub fn with_channel_config(mut self, channel_config: ChannelConfig) -> Self {
        self.channel_config = Some(channel_config);
        self
    }
    pub fn with_required_channels<I, T>(mut self, channels: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.channel_config = Some(ChannelConfig::require_channels(channels));
        self
    }

    pub fn with_browser_tool(mut self) -> Self {
        self = self.with_tools(ToolNamespaceConfig::browser());
        self
    }

    pub fn with_python_tool(mut self) -> Self {
        self = self.with_tools(ToolNamespaceConfig::python());
        self
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolDescription {
    pub name: String,
    pub description: String,
    pub parameters: Option<serde_json::Value>,
}

impl ToolDescription {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Option<serde_json::Value>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Conversation {
    pub messages: Vec<Message>,
}

impl Conversation {
    pub fn from_messages<I>(messages: I) -> Self
    where
        I: IntoIterator<Item = Message>,
    {
        Self {
            messages: messages.into_iter().collect(),
        }
    }
}

impl<'a> IntoIterator for &'a Conversation {
    type Item = &'a Message;
    type IntoIter = std::slice::Iter<'a, Message>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter()
    }
}

fn de_string_or_content_vec<'de, D>(deserializer: D) -> Result<Vec<Content>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrContentVec(PhantomData<fn() -> Vec<Content>>);

    impl<'de> Visitor<'de> for StringOrContentVec {
        type Value = Vec<Content>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or list of content")
        }

        fn visit_str<E>(self, value: &str) -> Result<Vec<Content>, E>
        where
            E: de::Error,
        {
            Ok(vec![Content::Text(TextContent {
                text: value.to_owned(),
            })])
        }

        fn visit_seq<A>(self, seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(StringOrContentVec(PhantomData))
}

fn se_string_or_content_vec<S>(value: &Vec<Content>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if value.len() == 1 {
        if let Content::Text(TextContent { text }) = &value[0] {
            return serializer.serialize_str(text);
        }
    }
    value.serialize(serializer)
}

/// Content specific to developer messages, includes developer identity and its instructions
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct DeveloperContent {
    pub instructions: Option<String>,
    pub tools: Option<BTreeMap<String, ToolNamespaceConfig>>,
}

impl DeveloperContent {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
    pub fn with_tools(mut self, ns_config: ToolNamespaceConfig) -> Self {
        let ns = ns_config.name.clone();
        if let Some(ref mut map) = self.tools {
            map.insert(ns, ns_config);
        } else {
            let mut map = BTreeMap::new();
            map.insert(ns, ns_config);
            self.tools = Some(map);
        }
        self
    }
    pub fn with_function_tools(mut self, tools: Vec<ToolDescription>) -> Self {
        self = self.with_tools(ToolNamespaceConfig::new("functions", None, tools));
        self
    }
}
