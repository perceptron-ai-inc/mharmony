// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
use crate::chat::{parse_point_payload, stable_point_payload_prefix, Content, TextContent};

const POINT_START_TOKEN: &str = "<|point_start|>";
const POINT_END_TOKEN: &str = "<|point_end|>";

#[derive(Clone, Debug)]
pub(crate) struct PointSplit {
    pub(crate) content: Vec<Content>,
    pub(crate) stable_text: String,
}

#[derive(Clone, Debug)]
pub(crate) struct IncrementalPointParser {
    start_token: String,
    end_token: String,
    state: PointParserState,
}

#[derive(Clone, Debug)]
enum PointParserState {
    Outside(String),
    Inside(String),
}

impl Default for PointParserState {
    fn default() -> Self {
        Self::Outside(String::new())
    }
}

impl IncrementalPointParser {
    pub(crate) fn new(start_token: Option<&str>, end_token: Option<&str>) -> Self {
        Self {
            start_token: start_token.unwrap_or(POINT_START_TOKEN).to_string(),
            end_token: end_token.unwrap_or(POINT_END_TOKEN).to_string(),
            state: PointParserState::default(),
        }
    }

    pub(crate) fn push_text(&mut self, delta: &str) -> Vec<String> {
        match &mut self.state {
            PointParserState::Outside(text) | PointParserState::Inside(text) => {
                text.push_str(delta)
            }
        }
        self.drain()
    }

    pub(crate) fn finish(&mut self) -> Vec<String> {
        let mut parts = Vec::new();
        match std::mem::take(&mut self.state) {
            PointParserState::Outside(pending_text) => {
                self.push_stripped_part(&mut parts, &pending_text)
            }
            PointParserState::Inside(mut payload_pending) => {
                self.emit_stable_incomplete_point_payload(&mut payload_pending, &mut parts);
            }
        }
        parts
    }

    fn drain(&mut self) -> Vec<String> {
        let mut parts = Vec::new();
        loop {
            match std::mem::take(&mut self.state) {
                PointParserState::Outside(mut pending_text) => {
                    if let Some(start_idx) = pending_text.find(&self.start_token) {
                        let before_start = &pending_text[..start_idx];
                        self.push_stripped_part(&mut parts, before_start);
                        let after_start =
                            pending_text[(start_idx + self.start_token.len())..].to_string();
                        self.state = PointParserState::Inside(after_start);
                        continue;
                    }

                    let retain_len = longest_suffix_that_prefixes_any(
                        &pending_text,
                        &[self.start_token.as_str(), self.end_token.as_str()],
                    );
                    let emit_len = pending_text.len().saturating_sub(retain_len);
                    if emit_len > 0 {
                        let emit = pending_text[..emit_len].to_string();
                        self.push_stripped_part(&mut parts, &emit);
                        pending_text.drain(..emit_len);
                    }
                    self.state = PointParserState::Outside(pending_text);
                    return parts;
                }
                PointParserState::Inside(mut payload_pending) => {
                    if let Some(end_idx) = payload_pending.find(&self.end_token) {
                        let before_end = payload_pending[..end_idx].to_string();
                        let after_end =
                            payload_pending[(end_idx + self.end_token.len())..].to_string();
                        // Once the end sentinel is observed, the span is complete and its remaining payload is safe
                        // to display verbatim (minus sentinels). Incomplete spans stay on the conservative path below.
                        self.push_stripped_part(&mut parts, &before_end);
                        self.state = PointParserState::Outside(after_end);
                        continue;
                    }

                    self.drain_stable_point_payload_prefixes(&mut payload_pending, &mut parts);
                    self.state = PointParserState::Inside(payload_pending);
                    return parts;
                }
            }
        }
    }

    fn emit_stable_incomplete_point_payload(
        &self,
        payload_pending: &mut String,
        parts: &mut Vec<String>,
    ) {
        self.drain_stable_point_payload_prefixes(payload_pending, parts);
        self.emit_safe_point_payload_remainder_at_end(payload_pending, parts);
    }

    fn push_stripped_part(&self, parts: &mut Vec<String>, text: &str) {
        push_part(
            parts,
            strip_point_sentinels(text, &self.start_token, &self.end_token),
        );
    }

    fn drain_stable_point_payload_prefixes(
        &self,
        payload_pending: &mut String,
        parts: &mut Vec<String>,
    ) {
        loop {
            let stable_prefix = stable_point_payload_prefix(payload_pending);
            if stable_prefix.is_empty() {
                break;
            }
            self.push_stripped_part(parts, stable_prefix);
            payload_pending.drain(..stable_prefix.len());
        }
    }

    fn emit_safe_point_payload_remainder_at_end(
        &self,
        payload_pending: &mut String,
        parts: &mut Vec<String>,
    ) {
        let whitespace_len = payload_pending
            .char_indices()
            .find_map(|(idx, ch)| (!ch.is_whitespace()).then_some(idx))
            .unwrap_or(payload_pending.len());
        if whitespace_len > 0 {
            push_part(parts, payload_pending[..whitespace_len].to_string());
            payload_pending.drain(..whitespace_len);
        }
        let trimmed = payload_pending.trim_start();
        if trimmed.is_empty() || trimmed.starts_with("</collection>") {
            self.push_stripped_part(parts, payload_pending);
            payload_pending.clear();
        }
    }
}

pub(crate) fn stable_text_from_point_spans(
    text: &str,
    start_token: Option<&str>,
    end_token: Option<&str>,
) -> String {
    let mut parser = IncrementalPointParser::new(start_token, end_token);
    parser
        .push_text(text)
        .into_iter()
        .chain(parser.finish())
        .collect()
}

pub(crate) fn split_text_with_pointing_view(
    text: &str,
    start_token: Option<&str>,
    end_token: Option<&str>,
    suppress_incomplete_pointing_span: bool,
) -> PointSplit {
    let start_token = start_token.unwrap_or(POINT_START_TOKEN);
    let end_token = end_token.unwrap_or(POINT_END_TOKEN);

    if !text.contains(start_token) {
        return PointSplit {
            content: vec![text_content(text)],
            stable_text: strip_point_sentinels(text, start_token, end_token),
        };
    }

    let stable_text = stable_text_from_point_spans(text, Some(start_token), Some(end_token));

    let mut parts: Vec<Content> = Vec::new();
    let mut cursor = 0;
    while let Some(rel_start) = text[cursor..].find(start_token) {
        let start = cursor + rel_start;
        push_text_content(&mut parts, &text[cursor..start]);
        let after_start = start + start_token.len();
        if let Some(rel_end) = text[after_start..].find(end_token) {
            let end = after_start + rel_end;
            let payload = &text[after_start..end];
            match parse_point_payload(payload.trim()) {
                Ok(content) => parts.push(content),
                Err(_) => push_text_content(&mut parts, &text[start..(end + end_token.len())]),
            }
            cursor = end + end_token.len();
        } else {
            if !suppress_incomplete_pointing_span {
                push_text_content(&mut parts, &text[start..]);
            }
            cursor = text.len();
            break;
        }
    }

    push_text_content(&mut parts, &text[cursor..]);

    PointSplit {
        content: parts,
        stable_text,
    }
}

fn push_text_content(parts: &mut Vec<Content>, text: &str) {
    if !text.is_empty() {
        parts.push(text_content(text));
    }
}

fn text_content(text: &str) -> Content {
    Content::Text(TextContent {
        text: text.to_string(),
    })
}

fn push_part(parts: &mut Vec<String>, text: String) {
    if !text.is_empty() {
        parts.push(text);
    }
}

fn strip_point_sentinels(text: &str, start_token: &str, end_token: &str) -> String {
    if !text.contains(start_token) && !text.contains(end_token) {
        return text.to_string();
    }
    text.replace(start_token, "").replace(end_token, "")
}

fn longest_suffix_that_prefixes_any(text: &str, needles: &[&str]) -> usize {
    let mut best = 0;
    let max_len = needles
        .iter()
        .map(|needle| needle.len())
        .max()
        .unwrap_or(0)
        .min(text.len());
    for needle in needles {
        for len in 1..=max_len {
            if text.is_char_boundary(text.len() - len)
                && needle.starts_with(&text[(text.len() - len)..])
            {
                best = best.max(len);
            }
        }
    }
    best
}
