// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
#![doc = include_str!("../README.md")]

pub mod channels;
pub mod chat;
pub mod encoding;
mod point_stream;
pub mod preprocess;
pub mod registry;
pub mod stream;
mod tiktoken;
pub mod tiktoken_ext;
pub mod token_tables;

pub use encoding::{
    CompletionParseView, CompletionStreamOptions, CompletionStreamParser, CompletionStreamUpdate,
    HarmonyEncoding, HeaderRenderOptions, MessageHeader, ParseOptions, RenderConversationConfig,
    RenderOptions, RenderedStream, StreamStop, StreamStopKind, StreamTextUpdate, StreamableParser,
};
pub use registry::{
    load_harmony_encoding, load_harmony_encoding_with_options, reserved_token_group_pool,
    HarmonyEncodingName, HarmonyEncodingOptions, MediaKindRegistrationConfig,
    ReservedTokenGroupConfig,
};

#[cfg(test)]
pub mod tests;

#[cfg(test)]
mod parity;

#[cfg(feature = "python-binding")]
mod py_module;

#[cfg(feature = "wasm-binding")]
mod wasm_module;
