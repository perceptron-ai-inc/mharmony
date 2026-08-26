// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
//! Generic-media parity baseline.
//!
//! Golden `RenderedStream` snapshots for the shipped multimodal surfaces — image,
//! video, and pointing (incl. temporal clip pointing). These fixtures are captured
//! before the media-hook genericization. The same established wire format must
//! reproduce byte-identically through the generic core representation.
//!
//! The core never interprets `Processed*` semantically (it emits placeholder tokens and
//! delegates to a host processor callback), so rendering with the default preprocess
//! config is deterministic and needs no external media data.
//!
//! Regenerate goldens after an intentional, reviewed change:
//!   UPDATE_PARITY_FIXTURES=1 cargo test -p mharmony parity

use std::path::{Path, PathBuf};

use crate::chat::{
    Content, Conversation, MediaContent, MediaRef, Message, PointingContent, PointingKind, Role,
};
use crate::encoding::RenderedStream;
use crate::{load_harmony_encoding, HarmonyEncodingName};

/// Encoding under which the baseline is captured. Matches the multimodal tests in `tests.rs`.
const PARITY_ENCODING: HarmonyEncodingName = HarmonyEncodingName::HarmonyGptOss;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-data")
        .join("parity")
}

/// Render a conversation under the parity encoding via the multimodal renderer.
fn render(convo: &Conversation) -> RenderedStream {
    let encoding =
        load_harmony_encoding(PARITY_ENCODING).expect("parity encoding must load offline");
    encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .expect("render_conversation_multimodal failed")
}

/// Recursively sort object keys so the snapshot is stable across runs.
///
/// `Event.tags` is a `HashMap`, and with serde_json's `preserve_order` feature the
/// serializer emits map keys in (random) `HashMap` iteration order rather than sorting
/// them. Canonicalizing here removes that source of false drift.
fn canonicalize(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (_, v) in map.iter_mut() {
                canonicalize(v);
            }
            map.sort_keys();
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(canonicalize),
        _ => {}
    }
}

/// Canonical, key-sorted pretty JSON for a rendered stream.
fn canonical_json(rendered: &RenderedStream) -> String {
    let mut value = serde_json::to_value(rendered).expect("RenderedStream must serialize");
    canonicalize(&mut value);
    serde_json::to_string_pretty(&value).expect("canonical JSON must serialize")
}

/// Assert a rendered stream matches its committed golden snapshot byte-for-byte.
///
/// With `UPDATE_PARITY_FIXTURES` set, (re)writes the golden instead of asserting.
fn assert_golden(name: &str, rendered: &RenderedStream) {
    let actual = canonical_json(rendered);
    let path = golden_dir().join(format!("{name}.json"));

    if std::env::var_os("UPDATE_PARITY_FIXTURES").is_some() {
        std::fs::create_dir_all(golden_dir()).expect("create parity golden dir");
        std::fs::write(&path, format!("{actual}\n")).expect("write parity golden");
        return;
    }

    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing parity golden {path:?}.\n\
             Generate the baseline with:  UPDATE_PARITY_FIXTURES=1 cargo test -p mharmony parity"
        )
    });

    assert_eq!(
        expected.trim_end(),
        actual,
        "RenderedStream parity drift for fixture '{name}'. If this change is intentional and reviewed, \
         regenerate with UPDATE_PARITY_FIXTURES=1 cargo test -p mharmony parity"
    );
}

// --- Fixture builders -------------------------------------------------------
// Kept tiny and explicit; these define the frozen baseline surface, so prefer
// adding a new fixture over mutating an existing one.

fn media_content(kind: &str, id: &str, mime: &str) -> MediaContent {
    MediaContent {
        kind: kind.to_string(),
        media_ref: MediaRef {
            id: id.to_string(),
            mime: Some(mime.to_string()),
            bytes_b64: None,
            metadata: None,
        },
        metadata: None,
        source: None,
    }
}

fn point(mention: &str, index: &str) -> PointingContent {
    PointingContent {
        kind: PointingKind::Point,
        coords: vec![(10, 20)],
        target: Some(0),
        t: Some(1.0),
        start_time: None,
        end_time: None,
        mention: Some(mention.to_string()),
        label: None,
        index: Some(index.to_string()),
    }
}

fn clip(mention: &str, index: &str) -> PointingContent {
    PointingContent {
        kind: PointingKind::Clip,
        coords: vec![],
        target: Some(0),
        t: None,
        start_time: Some(1.25),
        end_time: Some(3.75),
        mention: Some(mention.to_string()),
        label: None,
        index: Some(index.to_string()),
    }
}

// --- Baseline fixtures ------------------------------------------------------

#[test]
fn parity_image() {
    let convo = Conversation::from_messages([Message::from_role_and_contents(
        Role::User,
        [
            "Describe this image.".into(),
            Content::Media(media_content("image", "image_0", "image/png")),
        ],
    )]);
    assert_golden("image", &render(&convo));
}

#[test]
fn parity_video() {
    let convo = Conversation::from_messages([Message::from_role_and_contents(
        Role::User,
        [
            "Summarize this video.".into(),
            Content::Media(media_content("video", "video_0", "video/mp4")),
        ],
    )]);
    assert_golden("video", &render(&convo));
}

#[test]
fn parity_pointing_point() {
    let convo = Conversation::from_messages([Message::from_role_and_contents(
        Role::Assistant,
        [Content::Pointing(point("the cat", "image_0"))],
    )]);
    assert_golden("pointing_point", &render(&convo));
}

#[test]
fn parity_pointing_clip() {
    let convo = Conversation::from_messages([Message::from_role_and_contents(
        Role::Assistant,
        [Content::Pointing(clip("the scene", "video_0"))],
    )]);
    assert_golden("pointing_clip", &render(&convo));
}

/// Mixed, multi-turn conversation exercising image + pointing together with text framing —
/// the closest fixture to a real request, and the one most sensitive to event ordering.
#[test]
fn parity_mixed_conversation() {
    let convo = Conversation::from_messages([
        Message::from_role_and_contents(
            Role::User,
            [
                "What is in this image?".into(),
                Content::Media(media_content("image", "image_0", "image/png")),
            ],
        ),
        Message::from_role_and_contents(
            Role::Assistant,
            [
                "A cat, here:".into(),
                Content::Pointing(point("the cat", "image_0")),
            ],
        ),
    ]);
    assert_golden("mixed_conversation", &render(&convo));
}
