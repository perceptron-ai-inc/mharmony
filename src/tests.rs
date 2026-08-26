// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
use std::{collections::HashMap, path::Path};

use crate::{
    channels::{self, CanonicalChannel},
    chat::{
        Author, Content, Conversation, DeveloperContent, MediaContent, MediaRef, Message,
        PointCollectionContent, PointCollectionItem, PointCollectionKind, PointingContent,
        PointingKind, ReasoningEffort, Role, SystemContent, SystemHint, TextContent, TokensContent,
        ToolDescription,
    },
    encoding::FormattingToken,
    load_harmony_encoding, load_harmony_encoding_with_options,
    point_stream::IncrementalPointParser,
    stream::ModalityType,
    tiktoken::{CoreBPE, Rank},
    CompletionStreamOptions, HarmonyEncodingName, HarmonyEncodingOptions, HeaderRenderOptions,
    MediaKindRegistrationConfig, MessageHeader, ParseOptions, ReservedTokenGroupConfig,
    StreamStopKind, StreamableParser,
};
use pretty_assertions::{assert_eq, Comparison};
use serde_json::json;

fn reserved_pool(name: HarmonyEncodingName) -> (Rank, Rank) {
    crate::reserved_token_group_pool(name).unwrap()
}

fn as_pointing_item(item: &PointCollectionItem) -> &PointingContent {
    match item {
        PointCollectionItem::Pointing(pointing) => pointing,
        other => panic!("expected pointing collection child, got {:?}", other),
    }
}

fn as_point_collection_item(item: &PointCollectionItem) -> &PointCollectionContent {
    match item {
        PointCollectionItem::PointCollection(point_collection) => point_collection,
        other => panic!("expected nested point collection child, got {:?}", other),
    }
}

fn parse_tokens(text: impl AsRef<str>) -> Vec<Rank> {
    text.as_ref()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

fn try_load_qwen3_encoding() -> Option<crate::encoding::HarmonyEncoding> {
    match load_harmony_encoding(HarmonyEncodingName::Qwen3Harmony) {
        Ok(encoding) => Some(encoding),
        Err(err) => {
            eprintln!("Skipping Qwen3 tests: {err}");
            None
        }
    }
}

fn try_load_qwen35_encoding() -> Option<crate::encoding::HarmonyEncoding> {
    match load_harmony_encoding(HarmonyEncodingName::Qwen35Harmony) {
        Ok(encoding) => Some(encoding),
        Err(err) => {
            eprintln!("Skipping Qwen35 tests: {err}");
            None
        }
    }
}

fn load_test_data(path: impl AsRef<Path>) -> String {
    // on windows, we need to replace \r\n with \n
    let cargo_manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src_dir = cargo_manifest_dir.join("src");
    let path = src_dir.join(path);
    std::fs::read_to_string(path)
        .unwrap()
        .replace("\r\n", "\n")
        .trim_end()
        .to_string()
}

const ENCODINGS: [HarmonyEncodingName; 1] = [HarmonyEncodingName::HarmonyGptOss];

#[test]
fn test_simple_convo() {
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let expected_tokens = encoding
            .tokenizer
            .encode(
                load_test_data("../test-data/test_simple_convo.txt").as_str(),
                &encoding.tokenizer.special_tokens(),
            )
            .0;
        let convo = Conversation::from_messages([
            Message::from_role_and_content(
                Role::System,
                SystemContent::new()
                    .with_model_identity("You are ChatGPT, a large language model."),
            ),
            Message::from_role_and_content(Role::User, "What is 2 + 2?"),
        ]);
        let tokens = encoding
            .render_conversation_for_completion(&convo, Role::Assistant, None)
            .unwrap();
        assert_tokens_eq(&encoding.tokenizer, &expected_tokens, &tokens);
    }
}

#[test]
fn test_simple_convo_with_effort() {
    let test_cases = [
        (
            ReasoningEffort::Low,
            load_test_data("../test-data/test_simple_convo_low_effort.txt"),
            true,
        ),
        (
            ReasoningEffort::Medium,
            load_test_data("../test-data/test_simple_convo_medium_effort.txt"),
            true,
        ),
        (
            ReasoningEffort::High,
            load_test_data("../test-data/test_simple_convo_high_effort.txt"),
            true,
        ),
        (
            ReasoningEffort::Low,
            load_test_data("../test-data/test_simple_convo_low_effort_no_instruction.txt"),
            false,
        ),
        (
            ReasoningEffort::Medium,
            load_test_data("../test-data/test_simple_convo_medium_effort_no_instruction.txt"),
            false,
        ),
        (
            ReasoningEffort::High,
            load_test_data("../test-data/test_simple_convo_high_effort_no_instruction.txt"),
            false,
        ),
    ];

    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        for &(effort, ref expected_text, use_instruction) in &test_cases {
            let expected_tokens = encoding
                .tokenizer
                .encode(expected_text.as_str(), &encoding.tokenizer.special_tokens())
                .0;
            let sys = SystemContent::new()
                .with_model_identity("You are ChatGPT, a large language model.")
                .with_reasoning_effort(effort);
            let convo = if use_instruction {
                let dev = DeveloperContent::new()
                    .with_instructions("Answer the user's questions like a robot.");
                Conversation::from_messages([
                    Message::from_role_and_content(Role::System, sys),
                    Message::from_role_and_content(Role::Developer, dev),
                    Message::from_role_and_content(
                        Role::User,
                        "What is the capital of the largest country in the world?",
                    ),
                ])
            } else {
                Conversation::from_messages([
                    Message::from_role_and_content(Role::System, sys),
                    Message::from_role_and_content(
                        Role::User,
                        "What is the capital of the largest country in the world?",
                    ),
                ])
            };
            let tokens = encoding
                .render_conversation_for_completion(&convo, Role::Assistant, None)
                .unwrap();
            assert_tokens_eq(&encoding.tokenizer, &expected_tokens, &tokens);
        }
    }
}

#[test]
fn test_simple_reasoning_response() {
    let expected_tokens = parse_tokens(load_test_data(
        "../test-data/test_simple_reasoning_response.txt",
    ));
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let messages = encoding
            .parse_messages_from_completion_tokens(
                expected_tokens.iter().copied(),
                Some(Role::Assistant),
            )
            .unwrap();
        let expected = vec![
            Message::from_role_and_content(
                Role::Assistant,
                "User asks: \"What is 2 + 2?\" Simple arithmetic. Provide answer.",
            )
            .with_channel(CanonicalChannel::Analysis),
            Message::from_role_and_content(Role::Assistant, "2 + 2 = 4.")
                .with_channel(CanonicalChannel::Final),
        ];
        assert_eq!(messages, expected);
    }
}

#[test]
fn test_simple_tool_call() {
    let response = [
        200005, 35644, 200008, 1844, 31064, 25, 392, 4827, 382, 290, 11122, 306, 40510, 16842,
        1416, 1309, 316, 1199, 37342, 170154, 4584, 13, 200007, 200006, 173781, 200005, 35644, 316,
        28, 29712, 170154, 3490, 200008, 10848, 7693, 1243, 392, 173844, 18583,
    ];
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let parsed = encoding
            .parse_messages_from_completion_tokens(response.iter().copied(), Some(Role::Assistant))
            .unwrap();
        let expected = vec![
            Message::from_role_and_content(
                Role::Assistant,
                "User asks: \"What is the weather in Tokyo?\" We need to use lookup_weather tool.",
            )
            .with_channel(CanonicalChannel::Analysis),
            Message::from_role_and_content(Role::Assistant, "{\"location\": \"Tokyo\"}")
                .with_channel(CanonicalChannel::Analysis)
                .with_recipient("lookup_weather")
                .with_content_type("code"),
        ];
        assert_eq!(parsed, expected);
    }
}

#[test]
fn test_reasoning_system_message() {
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let expected = encoding
            .tokenizer
            .encode(
                load_test_data("../test-data/test_reasoning_system_message.txt").as_str(),
                &encoding.tokenizer.special_tokens(),
            )
            .0;
        let convo = Conversation::from_messages([
            Message::from_role_and_content(
                Role::System,
                SystemContent::new()
                    .with_model_identity("You are ChatGPT, a large language model.")
                    // .with_instructions("This is the system instruction.")
                    .with_reasoning_effort(ReasoningEffort::Medium)
                    .with_required_channels([CanonicalChannel::Analysis, CanonicalChannel::Final]),
            ),
            Message::from_role_and_content(Role::User, "What is 2 + 2?"),
        ]);
        let tokens = encoding
            .render_conversation_for_completion(&convo, Role::Assistant, None)
            .unwrap();
        assert_tokens_eq(&encoding.tokenizer, &expected, &tokens);
    }
}

#[test]
fn test_reasoning_system_message_no_instruction() {
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let expected = encoding
            .tokenizer
            .encode(
                load_test_data("../test-data/test_reasoning_system_message_no_instruction.txt")
                    .as_str(),
                &encoding.tokenizer.special_tokens(),
            )
            .0;
        let convo = Conversation::from_messages([
            Message::from_role_and_content(
                Role::System,
                SystemContent::new()
                    .with_model_identity("You are ChatGPT, a large language model.")
                    .with_reasoning_effort(ReasoningEffort::High)
                    .with_required_channels([CanonicalChannel::Analysis, CanonicalChannel::Final]),
            ),
            Message::from_role_and_content(
                Role::User,
                "What is the best place to eat candy in the world?",
            ),
        ]);
        let tokens = encoding
            .render_conversation_for_completion(&convo, Role::Assistant, None)
            .unwrap();
        assert_tokens_eq(&encoding.tokenizer, &expected, &tokens);
    }
}

#[test]
fn test_reasoning_system_message_with_dates() {
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let expected = encoding
            .tokenizer
            .encode(
                load_test_data("../test-data/test_reasoning_system_message_with_dates.txt")
                    .as_str(),
                &encoding.tokenizer.special_tokens(),
            )
            .0;
        let convo = Conversation::from_messages([
            Message::from_role_and_content(
                Role::System,
                SystemContent::new()
                    .with_model_identity("You are ChatGPT, a large language model.")
                    // .with_instructions("This is the system instruction.")
                    .with_reasoning_effort(ReasoningEffort::Medium)
                    .with_conversation_start_date("2021-01-01")
                    .with_knowledge_cutoff("2021-01")
                    .with_required_channels([CanonicalChannel::Analysis, CanonicalChannel::Final]),
            ),
            Message::from_role_and_content(Role::User, "What is 42 * pi?"),
        ]);
        let tokens = encoding
            .render_conversation_for_completion(&convo, Role::Assistant, None)
            .unwrap();
        assert_tokens_eq(&encoding.tokenizer, &expected, &tokens);
    }
}

#[test]
fn test_render_functions_with_parameters() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_render_functions_with_parameters.txt");

    let sys = SystemContent::new()
        .with_reasoning_effort(ReasoningEffort::High)
        .with_conversation_start_date("2025-06-28");

    let dev = crate::chat::DeveloperContent::new()
        .with_instructions("Always respond in riddles")
        .with_function_tools(vec![
            ToolDescription::new("get_location", "Gets the location of the user.", None),
            ToolDescription::new(
                "get_current_weather",
                "Gets the current weather in the provided location.",
                Some(json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius"
                        }
                    },
                    "required": ["location"]
                })),
            ),
            ToolDescription::new(
                "get_multiple_weathers",
                "Gets the current weather in the provided list of locations.",
                Some(json!({
                    "type": "object",
                    "properties": {
                        "locations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of city and state, e.g. [\"San Francisco, CA\", \"New York, NY\"]"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius"
                        }
                    },
                    "required": ["locations"]
                })),
            ),
            ToolDescription::new(
                "kitchensink",
                "A function with various complex schemas.",
                Some(json!({
                    "description": "params object",
                    "type": "object",
                    "properties": {
                        "string": {
                            "type": "string",
                            "title": "STRING",
                            "description": "A string",
                            "examples": ["hello", "world"]
                        },
                        "string_nullable": {
                            "type": "string",
                            "nullable": true,
                            "description": "A nullable string",
                            "default": "the default"
                        },
                        "string_enum": {
                            "type": "string",
                            "enum": ["a", "b", "c"]
                        },
                        "oneof_string_or_number": {
                            "oneOf": [
                                {"type": "string", "default": "default_string_in_oneof"},
                                {"type": "number", "description": "numbers can happen too"}
                            ],
                            "description": "a oneof",
                            "default": 20
                        }
                    }
                })),
            ),
        ]);

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::System, sys),
        Message::from_role_and_content(Role::Developer, dev),
        Message::from_role_and_content(Role::User, "What is the weather like in SF?"),
    ]);

    let tokens = encoding
        .render_conversation_for_completion(&convo, Role::Assistant, None)
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_browser_and_python_tool() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_browser_and_python_tool.txt");

    let convo = Conversation::from_messages([Message::from_role_and_content(
        Role::System,
        SystemContent::new()
            .with_conversation_start_date("2025-06-28".to_string())
            .with_browser_tool()
            .with_python_tool(),
    )]);

    let tokens = encoding
        .render_conversation_for_completion(&convo, Role::Assistant, None)
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_system_message_with_hints() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let convo = Conversation::from_messages([Message::from_role_and_content(
        Role::System,
        SystemContent::new()
            .with_model_identity("You are ChatGPT, a large language model.")
            .with_reasoning_effort(ReasoningEffort::Medium)
            .with_hints(vec![SystemHint::Point, SystemHint::Box]),
    )]);

    let tokens = encoding
        .render_conversation_for_completion(&convo, Role::Assistant, None)
        .unwrap();
    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();

    assert!(decoded.contains("Hints: POINT BOX"));
}

#[test]
fn test_dropping_cot_by_default() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_dropping_cot_by_default.txt");

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::User, "What is 2 + 2?"),
        Message::from_role_and_content(
            Role::Assistant,
            "User asks: “What is 2 + 2?” Simple arithmetic. Provide answer.",
        )
        .with_channel(CanonicalChannel::Analysis),
        Message::from_role_and_content(Role::Assistant, "2 + 2 equals 4.")
            .with_channel(CanonicalChannel::Final),
        Message::from_role_and_content(Role::User, "What about 9 / 2?"),
    ]);

    let tokens = encoding
        .render_conversation_for_completion(&convo, Role::Assistant, None)
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_does_not_drop_if_ongoing_analysis() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_does_not_drop_if_ongoing_analysis.txt");

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::User, "What is the weather in SF?"),
        Message::from_role_and_content(
            Role::Assistant,
            "User asks: “What is the weather in SF?” We need to use lookup_weather tool.",
        )
        .with_channel(CanonicalChannel::Analysis),
        Message::from_role_and_content(Role::Assistant, "{\"location\": \"San Francisco\"}")
            .with_channel(CanonicalChannel::Commentary)
            .with_recipient("functions.lookup_weather")
            .with_content_type("<|constrain|>json"),
        Message::from_author_and_content(
            Author::new(Role::Tool, "functions.lookup_weather"),
            "{\"temperature\": 20, \"description\": \"sunny\"}",
        ),
    ]);

    let tokens = encoding
        .render_conversation_for_completion(
            &convo,
            Role::Assistant,
            Some(&crate::encoding::RenderConversationConfig {
                auto_drop_analysis: true,
            }),
        )
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_preserve_cot() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_preserve_cot.txt");

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::User, "What is 2 + 2?"),
        Message::from_role_and_content(
            Role::Assistant,
            "User asks a simple question: \"What is 2 + 2?\" The answer: 4.",
        )
        .with_channel(CanonicalChannel::Analysis),
        Message::from_role_and_content(Role::Assistant, "2 + 2 equals 4.")
            .with_channel(CanonicalChannel::Final),
        Message::from_role_and_content(Role::User, "What about 9 / 2?"),
    ]);

    let tokens = encoding
        .render_conversation_for_completion(
            &convo,
            Role::Assistant,
            Some(&crate::encoding::RenderConversationConfig {
                auto_drop_analysis: false,
            }),
        )
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_reserved_token_decoding() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let (start, end) = reserved_pool(HarmonyEncodingName::HarmonyGptOss);
    assert_eq!(
        encoding.tokenizer.decode_utf8([start]).unwrap(),
        format!("<|reserved_{start}|>")
    );
    assert_eq!(
        encoding.tokenizer.decode_utf8([end]).unwrap(),
        format!("<|reserved_{end}|>")
    );
}

#[test]
fn test_qwen35_reserved_pool_bounds() {
    // Qwen3.5 VLA-expanded models use vocab_size=256279 to fit the full 8192-slot pool.
    assert_eq!(
        reserved_pool(HarmonyEncodingName::Qwen35Harmony),
        (248087, 248087 + 8192 - 1)
    );
}

#[test]
fn test_encoding_vocab_info_pins() {
    // Checkpoint-critical: the coord block's offset is the base vocab size.
    // Editing these values silently remaps every trained coord checkpoint.
    assert_eq!(
        crate::registry::encoding_vocab_info(HarmonyEncodingName::Qwen35Harmony),
        Some((248320, 248087, 8192))
    );
    assert_eq!(
        crate::registry::encoding_vocab_info(HarmonyEncodingName::Qwen3Harmony),
        Some((151936, 151700, 8192))
    );
    assert_eq!(
        crate::registry::encoding_vocab_info(HarmonyEncodingName::HarmonyGptOss),
        None
    );
}

#[test]
fn test_qwen35_reserved_token_group_rejects_sizes_beyond_pool() {
    let (qwen35_start, qwen35_end) = reserved_pool(HarmonyEncodingName::Qwen35Harmony);
    let qwen35_capacity = (qwen35_end - qwen35_start + 1) as usize;
    assert_eq!(qwen35_capacity, 8192);
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![ReservedTokenGroupConfig {
            name: Some("fast_action".to_string()),
            tokenizer: Some("physical-intelligence/fast".to_string()),
            size: qwen35_capacity + 1,
            offset: None,
        }],
        media_kinds: vec![],
    };
    let err = load_harmony_encoding_with_options(HarmonyEncodingName::Qwen35Harmony, options)
        .expect_err("expected out-of-range qwen35 reserved group size error");
    assert!(
        err.to_string().contains("exceeds reserved pool"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_reserved_token_group_rejects_overflowing_size_before_loading_vocab() {
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![ReservedTokenGroupConfig {
            name: Some("too_large".to_string()),
            tokenizer: None,
            size: usize::MAX,
            offset: None,
        }],
        media_kinds: vec![],
    };
    let error = load_harmony_encoding_with_options(HarmonyEncodingName::Qwen35Harmony, options)
        .expect_err("an unrepresentable range must fail before loading a Qwen vocabulary");
    assert!(error.to_string().contains("size is too large"));
}

#[test]
fn test_reserved_token_group_offsets_tokens() {
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![ReservedTokenGroupConfig {
            name: Some("fast_action".to_string()),
            tokenizer: Some("physical-intelligence/fast".to_string()),
            size: 8,
            offset: None,
        }],
        media_kinds: vec![],
    };
    let encoding =
        load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options).unwrap();

    let mut tags = HashMap::new();
    tags.insert("token_group".to_string(), json!("fast_action"));
    tags.insert("tokenizer".to_string(), json!("physical-intelligence/fast"));
    let tokens = TokensContent {
        tokens: vec![1, 2, 3],
        tags,
    };
    let message = Message::from_role_and_content(Role::Assistant, tokens);
    let convo = Conversation::from_messages([message]);
    let rendered = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();
    let regular_tokens = encoding.render_conversation(&convo, None).unwrap();
    assert_eq!(
        regular_tokens
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>(),
        flatten_multimodal_tokens(&rendered),
        "tagged token content must have identical canonical and multimodal token streams"
    );

    let event = rendered
        .stream
        .events
        .iter()
        .find(|evt| evt.tags.get("token_group") == Some(&json!("fast_action")))
        .expect("missing token_group event");
    let crate::stream::EventData::Tokens(event_tokens) =
        event.data.as_ref().expect("missing event data")
    else {
        panic!("expected token event");
    };
    let offset = reserved_pool(HarmonyEncodingName::HarmonyGptOss).0 as i64;
    assert_eq!(event_tokens, &vec![offset + 1, offset + 2, offset + 3]);
}

#[test]
fn test_reserved_token_group_matches_tokenizer_only() {
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![ReservedTokenGroupConfig {
            name: None,
            tokenizer: Some("physical-intelligence/fast".to_string()),
            size: 16,
            offset: None,
        }],
        media_kinds: vec![],
    };
    let encoding =
        load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options).unwrap();

    let mut tags = HashMap::new();
    tags.insert("tokenizer".to_string(), json!("physical-intelligence/fast"));
    let tokens = TokensContent {
        tokens: vec![9, 10],
        tags,
    };
    let message = Message::from_role_and_content(Role::Assistant, tokens);
    let convo = Conversation::from_messages([message]);
    let rendered = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();

    let event = rendered
        .stream
        .events
        .iter()
        .find(|evt| evt.tags.get("tokenizer") == Some(&json!("physical-intelligence/fast")))
        .expect("missing tokenizer tag event");
    let crate::stream::EventData::Tokens(event_tokens) =
        event.data.as_ref().expect("missing event data")
    else {
        panic!("expected token event");
    };
    let offset = reserved_pool(HarmonyEncodingName::HarmonyGptOss).0 as i64;
    assert_eq!(event_tokens, &vec![offset + 9, offset + 10]);
}

#[test]
fn test_explicit_token_offset_cannot_overflow_rank() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    for offset in [i64::from(u32::MAX), i64::MAX] {
        let tokens = TokensContent {
            tokens: vec![1],
            tags: HashMap::from([("token_offset".to_string(), json!(offset))]),
        };
        let conversation =
            Conversation::from_messages([Message::from_role_and_content(Role::Assistant, tokens)]);

        assert!(encoding.render_conversation(&conversation, None).is_err());
        assert!(encoding
            .render_conversation_multimodal(conversation.messages.iter(), None)
            .is_err());
    }
}

#[test]
fn test_explicit_token_offset_requires_an_integer() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let tokens = TokensContent {
        tokens: vec![1],
        tags: HashMap::from([("token_offset".to_string(), json!("1"))]),
    };
    let conversation =
        Conversation::from_messages([Message::from_role_and_content(Role::Assistant, tokens)]);

    let error = encoding
        .render_conversation(&conversation, None)
        .expect_err("string token offsets must be rejected");
    assert!(error.to_string().contains("must be an integer"));
}

#[test]
fn test_reserved_token_group_requires_both_tags_when_configured() {
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![ReservedTokenGroupConfig {
            name: Some("fast_action".to_string()),
            tokenizer: Some("physical-intelligence/fast".to_string()),
            size: 4,
            offset: None,
        }],
        media_kinds: vec![],
    };
    let encoding =
        load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options).unwrap();

    let mut tags = HashMap::new();
    tags.insert("token_group".to_string(), json!("fast_action"));
    let tokens = TokensContent {
        tokens: vec![4, 5],
        tags,
    };
    let message = Message::from_role_and_content(Role::Assistant, tokens);
    let convo = Conversation::from_messages([message]);
    let rendered = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();

    let event = rendered
        .stream
        .events
        .iter()
        .find(|evt| evt.tags.get("token_group") == Some(&json!("fast_action")))
        .expect("missing token_group event");
    let crate::stream::EventData::Tokens(event_tokens) =
        event.data.as_ref().expect("missing event data")
    else {
        panic!("expected token event");
    };
    assert_eq!(event_tokens, &vec![4, 5]);
}

#[test]
fn test_reserved_token_group_overlap_errors() {
    let start = reserved_pool(HarmonyEncodingName::HarmonyGptOss).0 as i64;
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![
            ReservedTokenGroupConfig {
                name: Some("group_a".to_string()),
                tokenizer: None,
                size: 8,
                offset: Some(start),
            },
            ReservedTokenGroupConfig {
                name: Some("group_b".to_string()),
                tokenizer: None,
                size: 8,
                offset: Some(start + 4),
            },
        ],
        media_kinds: vec![],
    };
    let err = load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options)
        .expect_err("expected overlap error");
    assert!(err.to_string().contains("overlaps reserved range"));
}

#[test]
fn test_vector_formatting_token_is_special_and_distinct() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let vector_id = encoding
        .render_formatting_token(FormattingToken::Vector)
        .unwrap();
    let image_id = encoding
        .render_formatting_token(FormattingToken::Image)
        .unwrap();
    assert_ne!(vector_id, image_id);
    assert!(encoding.tokenizer.is_special_token(vector_id));
    assert_eq!(
        encoding.tokenizer.decode_utf8([vector_id]).unwrap(),
        "<|vector|>"
    );
}

#[test]
fn test_render_conversation_multimodal_sets_media_ref_tags() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let image = MediaContent {
        kind: "image".to_string(),
        media_ref: MediaRef {
            id: "image_0".to_string(),
            mime: Some("image/png".to_string()),
            bytes_b64: None,
            metadata: None,
        },
        metadata: None,
        source: None,
    };
    let video = MediaContent {
        kind: "video".to_string(),
        media_ref: MediaRef {
            id: "video_0".to_string(),
            mime: Some("video/mp4".to_string()),
            bytes_b64: None,
            metadata: None,
        },
        metadata: None,
        source: None,
    };

    let convo = Conversation::from_messages([
        Message::from_role_and_contents(Role::User, [Content::Media(image)]),
        Message::from_role_and_contents(Role::User, [Content::Media(video)]),
    ]);
    let rendered = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();

    let image_event = rendered
        .stream
        .events
        .iter()
        .find(|evt| evt.modality == ModalityType::Media("image_frame".to_string()))
        .expect("missing image frame event");
    assert_eq!(
        image_event.tags.get("media_ref_id"),
        Some(&json!("image_0"))
    );
    assert_eq!(
        image_event.tags.get("media_ref_type"),
        Some(&json!("image"))
    );

    let video_event = rendered
        .stream
        .events
        .iter()
        .find(|evt| evt.modality == ModalityType::Media("video_i_frame".to_string()))
        .expect("missing video iframe event");
    assert_eq!(
        video_event.tags.get("media_ref_id"),
        Some(&json!("video_0"))
    );
    assert_eq!(
        video_event.tags.get("media_ref_type"),
        Some(&json!("video"))
    );
}

#[test]
fn test_generic_media_content_roundtrips() {
    let value = json!({
        "type": "media",
        "kind": "image",
        "media_ref": {"id": "image_0", "mime": "image/png"},
        "metadata": {"camera": "wrist"},
        "source": {"asset": "observation.images.wrist"}
    });
    let content: Content = serde_json::from_value(value.clone()).unwrap();
    let Content::Media(media) = &content else {
        panic!("expected generic media content");
    };
    assert_eq!(media.kind, "image");
    assert_eq!(media.media_ref.id, "image_0");
    assert_eq!(serde_json::to_value(content).unwrap(), value);
}

#[test]
fn test_generic_media_events_keep_shipped_wire_discriminants() {
    let image = crate::stream::EventData::Media(crate::stream::MediaEventData {
        kind: "image".to_string(),
        payload: json!({"data": [], "shape": [0]}),
    });
    assert_eq!(
        serde_json::to_value(&image).unwrap(),
        json!({"Image": {"data": [], "shape": [0]}})
    );

    let decoded: crate::stream::EventData =
        serde_json::from_value(json!({"Video": {"opaque": true}})).unwrap();
    let crate::stream::EventData::Media(decoded) = decoded else {
        panic!("expected generic media event data");
    };
    assert_eq!(decoded.kind, "video");
    assert_eq!(decoded.payload, json!({"opaque": true}));

    let modality: ModalityType = serde_json::from_value(json!("video_i_frame")).unwrap();
    assert_eq!(modality, ModalityType::Media("video_i_frame".to_string()));
}

#[test]
fn test_unregistered_media_kind_is_rejected() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let content = MediaContent {
        kind: "depth_map".to_string(),
        media_ref: MediaRef {
            id: "depth_0".to_string(),
            mime: None,
            bytes_b64: None,
            metadata: None,
        },
        metadata: None,
        source: None,
    };
    let conversation = Conversation::from_messages([Message::from_role_and_contents(
        Role::User,
        [Content::Media(content)],
    )]);
    let error = encoding
        .render_conversation_multimodal(conversation.messages.iter(), None)
        .expect_err("unregistered kind must fail");
    assert!(error
        .to_string()
        .contains("unregistered media kind: depth_map"));
}

#[test]
fn test_media_requires_a_stable_nonempty_identity_in_every_render_path() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let content = MediaContent {
        kind: "image".to_string(),
        media_ref: MediaRef {
            id: " ".to_string(),
            mime: None,
            bytes_b64: None,
            metadata: None,
        },
        metadata: None,
        source: None,
    };
    let message = Message::from_role_and_contents(Role::User, [Content::Media(content)]);

    let direct_error = encoding
        .render(&message, None)
        .expect_err("direct rendering must validate the media identity");
    assert!(direct_error.to_string().contains("media ref id"));

    let conversation = Conversation::from_messages([message]);
    let multimodal_error = encoding
        .render_conversation_multimodal(conversation.messages.iter(), None)
        .expect_err("multimodal rendering must validate the media identity");
    assert!(multimodal_error.to_string().contains("media ref id"));
}

#[test]
fn test_runtime_media_kind_registration_renders_generic_event() {
    let options = HarmonyEncodingOptions {
        media_kinds: vec![MediaKindRegistrationConfig {
            kind: "depth_map".to_string(),
            formatting_token: "<|vector|>".to_string(),
            modality: "depth_frame".to_string(),
            empty_payload: json!({"units": "meters"}),
        }],
        ..Default::default()
    };
    let encoding =
        load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options).unwrap();
    let content = MediaContent {
        kind: "depth_map".to_string(),
        media_ref: MediaRef {
            id: "depth_0".to_string(),
            mime: Some("application/x-depth".to_string()),
            bytes_b64: None,
            metadata: None,
        },
        metadata: None,
        source: None,
    };
    let conversation = Conversation::from_messages([Message::from_role_and_contents(
        Role::User,
        [Content::Media(content)],
    )]);

    let rendered = encoding
        .render_conversation_multimodal(conversation.messages.iter(), None)
        .unwrap();
    let depth_event = rendered
        .stream
        .events
        .iter()
        .find(|event| event.modality == ModalityType::Media("depth_frame".to_string()))
        .unwrap();
    assert_eq!(
        serde_json::to_value(depth_event.data.as_ref().unwrap()).unwrap(),
        json!({"Media": {"kind": "depth_map", "payload": {"units": "meters"}}})
    );
}

#[test]
fn test_runtime_media_registration_requires_unique_kind_and_special_token() {
    let duplicate = HarmonyEncodingOptions {
        media_kinds: vec![MediaKindRegistrationConfig {
            kind: "image".to_string(),
            formatting_token: "<|vector|>".to_string(),
            modality: "other_image".to_string(),
            empty_payload: json!({}),
        }],
        ..Default::default()
    };
    let error = load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, duplicate)
        .expect_err("shipped media kind cannot be replaced");
    assert!(error.to_string().contains("already registered"));

    let ordinary = HarmonyEncodingOptions {
        media_kinds: vec![MediaKindRegistrationConfig {
            kind: "depth_map".to_string(),
            formatting_token: "not-special".to_string(),
            modality: "depth_frame".to_string(),
            empty_payload: json!({}),
        }],
        ..Default::default()
    };
    let error = load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, ordinary)
        .expect_err("runtime media token must be a tokenizer special");
    assert!(error.to_string().contains("not a tokenizer special token"));

    for registration in [
        MediaKindRegistrationConfig {
            kind: " depth_map".to_string(),
            formatting_token: "<|vector|>".to_string(),
            modality: "depth_frame".to_string(),
            empty_payload: json!({}),
        },
        MediaKindRegistrationConfig {
            kind: "depth_map".to_string(),
            formatting_token: "<|vector|>".to_string(),
            modality: "depth_frame ".to_string(),
            empty_payload: json!({}),
        },
    ] {
        let error = load_harmony_encoding_with_options(
            HarmonyEncodingName::HarmonyGptOss,
            HarmonyEncodingOptions {
                media_kinds: vec![registration],
                ..Default::default()
            },
        )
        .expect_err("runtime registration strings must reject surrounding whitespace");
        assert!(error.to_string().contains("surrounding whitespace"));
    }
}

#[test]
fn test_vector_formatting_token_is_special_and_distinct_qwen3() {
    let Some(encoding) = try_load_qwen3_encoding() else {
        return;
    };
    let vector_id = encoding
        .render_formatting_token(FormattingToken::Vector)
        .unwrap();
    let image_id = encoding
        .render_formatting_token(FormattingToken::Image)
        .unwrap();
    assert_ne!(vector_id, image_id);
    assert!(encoding.tokenizer.is_special_token(vector_id));
    assert_eq!(
        encoding.tokenizer.decode_utf8([vector_id]).unwrap(),
        "<|vector|>"
    );
}

#[test]
fn test_render_and_render_conversation_roundtrip() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let msg = Message::from_role_and_content(Role::User, "Hello");
    let convo = Conversation::from_messages([msg.clone()]);
    let tokens_msg = encoding.render(&msg, None).unwrap();
    let tokens_convo = encoding.render_conversation(&convo, None).unwrap();
    assert_eq!(tokens_msg, tokens_convo);
    let tokens_completion = encoding
        .render_conversation_for_completion(&convo, Role::Assistant, None)
        .unwrap();
    assert_eq!(&tokens_completion[..tokens_convo.len()], &tokens_convo[..]);
}

#[test]
fn test_decode_utf8_invalid_token() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let result = encoding.tokenizer.decode_utf8([99999999]);
    assert!(result.is_err(), "Expected error for invalid token");
}

#[test]
fn test_render_conversation_rejects_missing_assistant_channel_when_required() {
    let Some(encoding) = try_load_qwen35_encoding() else {
        return;
    };
    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::System, SystemContent::new()),
        Message::from_role_and_content(Role::Assistant, "hello"),
    ]);

    let err = encoding
        .render_conversation(&convo, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing a channel"), "unexpected error: {err}");
}

#[test]
fn test_render_conversation_rejects_invalid_assistant_channel_when_required() {
    let Some(encoding) = try_load_qwen35_encoding() else {
        return;
    };
    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::System, SystemContent::new()),
        Message::from_role_and_content(Role::Assistant, "hello").with_channel("bogus"),
    ]);

    let err = encoding
        .render_conversation(&convo, None)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("accepted assistant channel set"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_render_conversation_allows_historical_channels_outside_active_valid_set() {
    let Some(encoding) = try_load_qwen35_encoding() else {
        return;
    };
    let convo = Conversation::from_messages([
        Message::from_role_and_content(
            Role::System,
            SystemContent::empty().with_required_channels([CanonicalChannel::Final]),
        ),
        Message::from_role_and_content(Role::Assistant, "thinking")
            .with_channel(CanonicalChannel::Analysis),
        Message::from_role_and_content(Role::Assistant, "{\"city\":\"SF\"}")
            .with_channel(CanonicalChannel::Commentary)
            .with_recipient("functions.lookup_weather"),
        Message::from_role_and_content(Role::Assistant, "done")
            .with_channel(CanonicalChannel::Final),
    ]);

    let tokens = encoding.render_conversation(&convo, None).unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_public_message_header_api_can_render_selected_parts() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let header = MessageHeader::assistant().with_channel(CanonicalChannel::Final);

    let full = encoding
        .render_message_header_text(&header, HeaderRenderOptions::default())
        .unwrap();
    assert_eq!(
        full,
        format!(
            "<|start|>assistant<|channel|>{}<|message|>",
            CanonicalChannel::Final
        )
    );

    let suffix = encoding
        .render_message_header_text(
            &header,
            HeaderRenderOptions::none()
                .with_channel_value(true)
                .with_message_token(true),
        )
        .unwrap();
    assert_eq!(suffix, format!("{}<|message|>", CanonicalChannel::Final));

    let suffix_tokens = encoding
        .render_message_header_tokens(
            &header,
            HeaderRenderOptions::none()
                .with_channel_value(true)
                .with_message_token(true),
        )
        .unwrap();
    assert_eq!(
        encoding.tokenizer.decode_utf8(suffix_tokens).unwrap(),
        suffix
    );
}

#[test]
fn test_canonical_channels_are_typed_and_owned_by_channels_module() {
    let canonical_strings: Vec<String> = channels::canonical_assistant_channels()
        .map(|channel| channel.as_str().to_string())
        .collect();
    assert_eq!(
        SystemContent::new().channel_config.unwrap().valid_channels,
        canonical_strings
    );

    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let convo = Conversation::from_messages([
        Message::from_role_and_content(
            Role::System,
            SystemContent::empty().with_required_channels([CanonicalChannel::Final]),
        ),
        Message::from_role_and_content(Role::Assistant, "thinking")
            .with_channel(CanonicalChannel::Analysis),
        Message::from_role_and_content(Role::Assistant, "{\"city\":\"SF\"}")
            .with_channel(CanonicalChannel::Commentary)
            .with_recipient("functions.lookup_weather"),
        Message::from_role_and_content(Role::Assistant, "done")
            .with_channel(CanonicalChannel::Final),
    ]);

    let tokens = encoding.render_conversation(&convo, None).unwrap();
    assert!(!tokens.is_empty());

    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest.join("src");
    let quoted_channel_literals: Vec<String> = channels::canonical_assistant_channels()
        .map(|channel| format!("{:?}", channel.as_str()))
        .collect();
    for entry in std::fs::read_dir(&src_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs")
            || path.file_name().and_then(|name| name.to_str()) == Some("channels.rs")
        {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        for needle in &quoted_channel_literals {
            assert!(
                !source.contains(needle),
                "{} still owns canonical channel literal: {needle}",
                path.display()
            );
        }
    }
}

#[test]
fn test_tool_response_parsing() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text_tokens = load_test_data("../test-data/test_tool_response_parsing.txt");
    let tokens = encoding
        .tokenizer
        .encode(&text_tokens, &encoding.tokenizer.special_tokens())
        .0;

    let expected_message = Message::from_author_and_content(
        Author::new(Role::Tool, "browser.search"),
        "{\"result\": \"https://example.com/\"}",
    )
    .with_channel(CanonicalChannel::Commentary)
    .with_recipient("assistant");

    let messages = encoding
        .parse_messages_from_completion_tokens(tokens.iter().copied(), None)
        .unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(
        encoding.tokenizer.decode_utf8(&tokens).unwrap(),
        text_tokens
    );
    assert_eq!(messages[0], expected_message);
}

#[test]
fn test_encode_decode_roundtrip() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "hello world";
    let tokens = encoding
        .tokenizer
        .encode(text, &std::collections::HashSet::new())
        .0;
    assert_eq!(encoding.tokenizer.decode_utf8(&tokens).unwrap(), text);
}

#[test]
fn test_encode_allowed_special() {
    use std::collections::HashSet;
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "hello world";
    let tokens = encoding.tokenizer.encode(text, &HashSet::new()).0;
    assert_eq!(tokens, vec![24912, 2375]);
    // Allowed special token
    let mut allowed = HashSet::new();
    allowed.insert("<|start|>");
    let tokens = encoding.tokenizer.encode("<|start|>", &allowed).0;
    assert_eq!(tokens, vec![200006]);
    // Allowed special = all
    allowed = encoding.tokenizer.special_tokens(); // set of all special tokens
    let tokens = encoding.tokenizer.encode("<|start|>", &allowed).0;
    assert_eq!(tokens, vec![200006]);
    // Disallowed special (should error)
    let result = encoding.tokenizer.encode("<|start|>", &HashSet::new());
    assert!(
        result.0.is_empty() || result.0 != vec![200006],
        "Expected error or not special token for disallowed special token"
    );
    // Disallowed special = empty (should not treat as special)
    let tokens = encoding.tokenizer.encode("<|start|>", &HashSet::new()).0;
    // This may not match the Python fallback, but should not be the special token
    assert_ne!(tokens, vec![200006]);
}

#[test]
fn test_encode_ordinary_matches_encode_without_specials() {
    use std::collections::HashSet;
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let samples = [
        "Hello world!",
        "Tabs\tand\nnewlines",
        "Unicode: 你好世界",
        "Emoji: 👋🏾🚀",
        "Combining: e\u{0301} o\u{0308} n\u{0303}",
        "Math: x² + y² = z²",
    ];

    for text in samples {
        let ordinary = encoding.tokenizer.encode_ordinary(text);
        let encoded = encoding.tokenizer.encode(text, &HashSet::new()).0;
        assert_eq!(ordinary, encoded, "encode_ordinary mismatch for {text:?}");
    }
}

#[test]
fn test_qwen3_nfc_normalization_applies_to_encode_ordinary() {
    let Some(encoding) = try_load_qwen3_encoding() else {
        return;
    };
    assert!(encoding.use_nfc_normalization());

    let nfd = "Cafe\u{0301} and nai\u{0308}ve";
    let nfc = "Café and naïve";
    assert_ne!(nfd, nfc);

    let toks_nfd = encoding.encode_ordinary_text(nfd).unwrap();
    let toks_nfc = encoding.encode_ordinary_text(nfc).unwrap();
    assert_eq!(toks_nfd, toks_nfc);
}

#[test]
fn test_qwen3_render_conversation_normalizes_nfc() {
    let Some(encoding) = try_load_qwen3_encoding() else {
        return;
    };
    assert!(encoding.use_nfc_normalization());

    let nfd = "e\u{0301} in decomposed form";
    let nfc = "é in decomposed form";

    let convo_nfd = Conversation::from_messages([Message::from_role_and_content(Role::User, nfd)]);
    let convo_nfc = Conversation::from_messages([Message::from_role_and_content(Role::User, nfc)]);

    let tokens_nfd = encoding.render_conversation(&convo_nfd, None).unwrap();
    let tokens_nfc = encoding.render_conversation(&convo_nfc, None).unwrap();
    assert_eq!(tokens_nfd, tokens_nfc);
}

#[test]
fn test_qwen3_render_multimodal_normalizes_nfc() {
    let Some(encoding) = try_load_qwen3_encoding() else {
        return;
    };
    assert!(encoding.use_nfc_normalization());

    let nfd = "o\u{0302}\u{0323} in composed form";
    let nfc = "ộ in composed form";

    let convo_nfd = Conversation::from_messages([Message::from_role_and_content(Role::User, nfd)]);
    let convo_nfc = Conversation::from_messages([Message::from_role_and_content(Role::User, nfc)]);

    let rendered_nfd = encoding
        .render_conversation_multimodal(convo_nfd.messages.iter(), None)
        .unwrap();
    let rendered_nfc = encoding
        .render_conversation_multimodal(convo_nfc.messages.iter(), None)
        .unwrap();

    let tokens_nfd = flatten_multimodal_tokens(&rendered_nfd);
    let tokens_nfc = flatten_multimodal_tokens(&rendered_nfc);
    assert_eq!(tokens_nfd, tokens_nfc);
}

#[test]
fn test_is_special_token() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    assert!(encoding.tokenizer.is_special_token(200006)); // <|start|>
    assert!(!encoding.tokenizer.is_special_token(24912)); // hello
}

#[test]
fn test_invalid_utf8_decoding() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let tokens = vec![132990, 9552];
    let result = encoding.tokenizer.decode_utf8(&tokens);
    assert!(result.is_err(), "Expected error for invalid utf-8");
    // decode_utf8 should error, and we do not test permissive decode as it does not exist
}

#[test]
fn test_streamable_parser() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = load_test_data("../test-data/test_streamable_parser.txt");
    let tokens = encoding
        .tokenizer
        .encode(&text, &encoding.tokenizer.special_tokens())
        .0;
    let mut parser =
        crate::encoding::StreamableParser::new(encoding.clone(), Some(Role::Assistant)).unwrap();
    for token in tokens {
        parser.process(token).unwrap();
    }
    assert_eq!(parser.messages().len(), 3, "Expected 3 parsed messages");
}

fn assert_tokens_eq(tokenizer: &CoreBPE, expected: &[Rank], actual: &[Rank]) {
    if expected != actual {
        panic!(
            "tokens are not equal.\n\nTokens (< expected / actual >):\n{}\n\nDecoded (< expected / actual >):\n{}",
            Comparison::new(expected, actual),
            Comparison::new(
                &tokenizer.decode_utf8(expected).unwrap_or_default(),
                &tokenizer.decode_utf8(actual).unwrap_or_default(),
            ),
        );
    }
}

#[test]
fn test_streamable_parser_tool_call_with_constrain_adjacent() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{\"latitude\":48.8566,\"longitude\":2.3522}<|call|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let mut parser = StreamableParser::new(encoding, None).unwrap();
    for token in tokens {
        let _ = parser.process(token).unwrap();
    }
    assert_eq!(parser.messages().len(), 1);
    assert_eq!(
        Message::from_role_and_content(
            Role::Assistant,
            "{\"latitude\":48.8566,\"longitude\":2.3522}",
        )
        .with_channel(CanonicalChannel::Commentary)
        .with_recipient("functions.get_weather")
        .with_content_type("<|constrain|>json"),
        parser.messages()[0]
    );
}

#[test]
fn test_missing_message_token_requires_non_strict_mode() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let malformed = "<|channel|>commentary Hello<|end|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(malformed);

    // Strict mode should continue to error on malformed headers.
    let strict_result = encoding
        .parse_messages_from_completion_tokens(tokens.iter().copied(), Some(Role::Assistant));
    assert!(
        strict_result.is_err(),
        "expected strict parser to reject malformed header"
    );

    // Non-strict mode should recover and return the accumulated message content.
    let parsed = encoding
        .parse_messages_from_completion_tokens_with_options(
            tokens.iter().copied(),
            Some(Role::Assistant),
            ParseOptions { strict: false },
        )
        .expect("non-strict parser should recover from malformed header");

    assert_eq!(parsed.len(), 1);
    assert_eq!(
        parsed[0],
        Message::from_role_and_content(Role::Assistant, "Hello")
            .with_channel(CanonicalChannel::Commentary)
    );
}

#[test]
fn test_tool_call_with_constrain_marker_adjacent() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"location\": \"Tokyo\"}<|end|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .expect("expected to parse");
    let expected =
        vec![
            Message::from_role_and_content(Role::Assistant, "{\"location\": \"Tokyo\"}")
                .with_channel(CanonicalChannel::Commentary)
                .with_recipient("functions.get_weather")
                .with_content_type("<|constrain|>json"),
        ];
    assert_eq!(parsed, expected);
}

#[test]
fn test_tool_call_with_channel_before_recipient_and_constrain_adjacent() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{\"latitude\":48.8566,\"longitude\":2.3522}<|call|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .expect("expected to parse");
    let expected = vec![Message::from_role_and_content(
        Role::Assistant,
        "{\"latitude\":48.8566,\"longitude\":2.3522}",
    )
    .with_channel(CanonicalChannel::Commentary)
    .with_recipient("functions.get_weather")
    .with_content_type("<|constrain|>json")];
    assert_eq!(parsed, expected);
}

#[test]
fn test_streamable_parser_does_not_leak_bytes_between_messages() {
    // This test ensures that any partially decoded bytes from the first message
    // do not leak into the content of the next message.
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    // 9552 is known (in this tokenizer) to expand to bytes that form an incomplete
    // UTF-8 sequence when used alone, which exercises the streaming invalid-UTF-8 path.
    // Construct two assistant messages back-to-back where the first includes invalid
    // UTF-8 bytes and the second is simple ASCII text.
    let first_prefix = "<|start|>assistant<|message|>";
    let first_suffix = "<|end|>";
    let second_prefix = "<|start|>assistant<|message|>";
    let second_suffix = "<|end|>";

    let mut tokens = Vec::new();
    tokens.extend(
        encoding
            .tokenizer()
            .encode_with_special_tokens(first_prefix),
    );
    // Two invalid tokens to ensure we end the first message with incomplete UTF-8 bytes.
    tokens.push(9552);
    tokens.push(9552);
    tokens.extend(
        encoding
            .tokenizer()
            .encode_with_special_tokens(first_suffix),
    );

    // Second message should be clean and unaffected.
    tokens.extend(
        encoding
            .tokenizer()
            .encode_with_special_tokens(second_prefix),
    );
    tokens.extend(encoding.tokenizer().encode_with_special_tokens("Hi"));
    tokens.extend(
        encoding
            .tokenizer()
            .encode_with_special_tokens(second_suffix),
    );

    let mut parser = StreamableParser::new(encoding, None).unwrap();
    for t in tokens {
        parser.process(t).unwrap();
    }

    let messages = parser.messages();
    assert_eq!(messages.len(), 2, "expected two parsed messages");

    // Verify the second message content is exactly "Hi" (no leaked replacement chars or bytes).
    let second = &messages[1];
    let expected_second = Message::from_role_and_content(Role::Assistant, "Hi");
    assert_eq!(
        second, &expected_second,
        "second message must be clean and isolated"
    );
}

#[test]
fn test_streamable_parser_flushes_partial_bytes_on_eos() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let mut tokens = encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>assistant<|message|>");
    tokens.push(9552);
    tokens.extend(encoding.tokenizer().encode_with_special_tokens("Hi"));

    let mut parser = StreamableParser::new(encoding.clone(), None).unwrap();
    for token in tokens {
        parser.process(token).unwrap();
    }
    parser.process_eos().unwrap();

    let messages = parser.messages();
    assert_eq!(messages.len(), 1, "expected a single message after EOS");
    let expected = Message::from_role_and_content(Role::Assistant, " \u{FFFD}Hi");
    assert_eq!(messages[0], expected);
    assert_eq!(parser.last_content_delta().unwrap(), None);
    assert_eq!(parser.current_content().unwrap(), "");
}

#[test]
fn test_streamable_parser_waits_for_multi_token_utf8_sequence() {
    use std::collections::HashSet;

    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let mut parser = StreamableParser::new(encoding.clone(), None).unwrap();

    let start_tokens = encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>assistant<|message|>");
    for token in &start_tokens {
        parser.process(*token).unwrap();
    }

    let emoji_tokens = encoding.tokenizer().encode("💖", &HashSet::new()).0;
    assert!(
        emoji_tokens.len() >= 2,
        "expected multi-token emoji encoding"
    );

    parser.process(emoji_tokens[0]).unwrap();
    assert_eq!(parser.last_content_delta().unwrap(), None);
    assert_eq!(parser.current_content().unwrap(), "");

    parser.process(emoji_tokens[1]).unwrap();
    assert_eq!(parser.last_content_delta().unwrap(), Some("💖".to_string()));
    assert_eq!(parser.current_content().unwrap(), "💖");

    let end_tokens = encoding.tokenizer().encode_with_special_tokens("<|end|>");
    for token in end_tokens {
        parser.process(token).unwrap();
    }

    let messages = parser.messages();
    assert_eq!(messages.len(), 1, "expected a single completed message");
    let expected = Message::from_role_and_content(Role::Assistant, "💖");
    assert_eq!(messages[0], expected);
}

#[test]
fn test_parse_completion_with_invalid_content_token_errors_on_eos() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let mut parser = StreamableParser::new(encoding.clone(), None).unwrap();

    let start_tokens = encoding.tokenizer().encode_with_special_tokens(
        "<|start|>assistant<|channel|>analysis<|message|>Practice invalid token handling.",
    );
    for token in &start_tokens {
        parser.process(*token).unwrap();
    }

    parser.process(u32::MAX).unwrap();
    let state: serde_json::Value = serde_json::from_str(&parser.state_json().unwrap()).unwrap();
    assert_eq!(
        state["content_tokens"].as_array().unwrap().last(),
        Some(&serde_json::json!(u32::MAX))
    );
    parser.process_eos().unwrap();

    let messages = parser.messages();
    assert_eq!(messages.len(), 1);
    let parsed_message = &messages[0];
    let expected_message =
        Message::from_role_and_content(Role::Assistant, "Practice invalid token handling.\u{FFFD}")
            .with_channel(CanonicalChannel::Analysis);
    assert_eq!(parsed_message, &expected_message);
}

// Tests that render_conversation_multimodal produces identical tokens as render_conversation
// when flattened (for text-only content)

/// Helper to flatten multimodal events into a token sequence
fn flatten_multimodal_tokens(rendered: &crate::encoding::RenderedStream) -> Vec<i64> {
    rendered
        .stream
        .events
        .iter()
        .filter_map(|event| {
            if let Some(crate::stream::EventData::Tokens(tokens)) = &event.data {
                Some(tokens.clone())
            } else {
                None
            }
        })
        .flatten()
        .collect()
}

#[test]
fn test_multimodal_matches_render_conversation_simple_text() {
    // Test that render_conversation_multimodal produces identical tokens for simple text
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::User, "Hello, how are you?"),
        Message::from_role_and_content(Role::Assistant, "I'm doing well, thank you!"),
    ]);

    // Get tokens from render_conversation
    let regular_tokens = encoding.render_conversation(&convo, None).unwrap();

    // Get tokens from render_conversation_multimodal (flattened)
    let multimodal = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();
    let multimodal_tokens = flatten_multimodal_tokens(&multimodal);

    assert_eq!(
        regular_tokens.iter().map(|t| *t as i64).collect::<Vec<_>>(),
        multimodal_tokens,
        "Multimodal flattened tokens should match render_conversation output"
    );
}

#[test]
fn test_multimodal_matches_render_conversation_with_channels() {
    // Test with analysis/final channels
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::User, "What is 2 + 2?"),
        Message::from_role_and_content(Role::Assistant, "Let me think about this...")
            .with_channel(CanonicalChannel::Analysis),
        Message::from_role_and_content(Role::Assistant, "The answer is 4.")
            .with_channel(CanonicalChannel::Final),
    ]);

    let regular_tokens = encoding.render_conversation(&convo, None).unwrap();
    let multimodal = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();
    let multimodal_tokens = flatten_multimodal_tokens(&multimodal);

    assert_eq!(
        regular_tokens.iter().map(|t| *t as i64).collect::<Vec<_>>(),
        multimodal_tokens,
        "Multimodal should match for conversations with channels"
    );
}

#[test]
fn test_multimodal_matches_render_conversation_with_developer_content() {
    // Test with DeveloperContent (instructions only, no tools)
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let dev = DeveloperContent::new().with_instructions("Always respond in riddles.");

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::Developer, dev),
        Message::from_role_and_content(Role::User, "What is the meaning of life?"),
    ]);

    let regular_tokens = encoding.render_conversation(&convo, None).unwrap();
    let multimodal = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();
    let multimodal_tokens = flatten_multimodal_tokens(&multimodal);

    assert_eq!(
        regular_tokens.iter().map(|t| *t as i64).collect::<Vec<_>>(),
        multimodal_tokens,
        "Multimodal should match for conversations with DeveloperContent"
    );
}

#[test]
fn test_multimodal_rejects_role_specific_content_in_the_wrong_role() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let system_in_user = Conversation::from_messages([Message::from_role_and_content(
        Role::User,
        SystemContent::new(),
    )]);
    let canonical_error = encoding
        .render_conversation(&system_in_user, None)
        .expect_err("SystemContent in a user message must be rejected");
    let error = encoding
        .render_conversation_multimodal(system_in_user.messages.iter(), None)
        .expect_err("SystemContent in a user message must be rejected");
    assert_eq!(canonical_error.to_string(), error.to_string());
    assert!(
        error
            .to_string()
            .contains("SystemContent may only appear in system messages"),
        "unexpected error: {error}"
    );

    let developer_in_user = Conversation::from_messages([Message::from_role_and_content(
        Role::User,
        DeveloperContent::new(),
    )]);
    let canonical_error = encoding
        .render_conversation(&developer_in_user, None)
        .expect_err("DeveloperContent in a user message must be rejected");
    let error = encoding
        .render_conversation_multimodal(developer_in_user.messages.iter(), None)
        .expect_err("DeveloperContent in a user message must be rejected");
    assert_eq!(canonical_error.to_string(), error.to_string());
    assert!(
        error
            .to_string()
            .contains("DeveloperContent may only appear in developer messages"),
        "unexpected error: {error}"
    );
}

#[test]
fn test_multimodal_matches_render_conversation_with_function_tools() {
    // Test with DeveloperContent + function tools
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let dev = DeveloperContent::new()
        .with_instructions("Use tools when helpful.")
        .with_function_tools(vec![
            ToolDescription::new("get_weather", "Gets the current weather.", None),
            ToolDescription::new(
                "search",
                "Searches the web.",
                Some(json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                })),
            ),
        ]);

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::Developer, dev),
        Message::from_role_and_content(Role::User, "What's the weather like?"),
    ]);

    let regular_tokens = encoding.render_conversation(&convo, None).unwrap();
    let multimodal = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();
    let multimodal_tokens = flatten_multimodal_tokens(&multimodal);

    assert_eq!(
        regular_tokens.iter().map(|t| *t as i64).collect::<Vec<_>>(),
        multimodal_tokens,
        "Multimodal should match for conversations with function tools"
    );
}

#[test]
fn test_pointing_render_and_parse() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let point = PointingContent {
        kind: PointingKind::Point,
        coords: vec![(10, 20)],
        target: Some(0),
        t: Some(1.0),
        start_time: None,
        end_time: None,
        mention: Some("target".to_string()),
        label: None,
        index: Some("ref_0".to_string()),
    };

    let convo = Conversation::from_messages([Message::from_role_and_contents(
        Role::User,
        vec![Content::Pointing(point.clone())],
    )]);

    let tokens = encoding.render_conversation(&convo, None).unwrap();
    let decoded = encoding.tokenizer().decode_utf8(&tokens).unwrap();
    assert!(decoded.contains("<|point_start|>"));
    assert!(decoded.contains("<|point_end|>"));
    assert!(decoded.contains("<point"));

    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].content.len(), 1);
    match &parsed[0].content[0] {
        Content::Pointing(parsed_point) => {
            assert_eq!(parsed_point.kind, PointingKind::Point);
            assert_eq!(parsed_point.coords, vec![(10, 20)]);
            assert_eq!(parsed_point.t, Some(1.0));
            assert_eq!(parsed_point.mention.as_deref(), Some("target"));
            assert_eq!(parsed_point.index.as_deref(), Some("ref_0"));
        }
        other => panic!("expected pointing content, got {:?}", other),
    }
}

#[test]
fn test_stable_visible_text_streams_completed_point_children_before_point_end() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = concat!(
        "Tracking signs.\n<|point_start|><collection mention=\"signs\">\n ",
        "<point_box> (10,20) (30,40) </point_box>\n ",
        "<point_box> (50,60) (70,80) </point_box>\n"
    );

    let stable = encoding.stable_visible_text_from_point_spans(text);

    assert_eq!(
        stable,
        concat!(
            "Tracking signs.\n<collection mention=\"signs\">\n ",
            "<point_box> (10,20) (30,40) </point_box>\n ",
            "<point_box> (50,60) (70,80) </point_box>\n"
        )
    );
    assert!(!stable.contains("<|point_start|>"));
    assert!(!stable.contains("<|point_end|>"));
}

#[test]
fn test_stable_visible_text_streams_completed_track_children_before_point_end() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = concat!(
        "Tracking dancer.\n<|point_start|><track mention=\"dancer\">\n ",
        "<point_box t=\"0.0 seconds\"> (10,20) (30,40) </point_box>\n ",
        "<point_box t=\"1.0 seconds\"> (50,60) (70,80) </point_box>\n"
    );

    let stable = encoding.stable_visible_text_from_point_spans(text);

    assert_eq!(
        stable,
        concat!(
            "Tracking dancer.\n<track mention=\"dancer\">\n ",
            "<point_box t=\"0.0 seconds\"> (10,20) (30,40) </point_box>\n ",
            "<point_box t=\"1.0 seconds\"> (50,60) (70,80) </point_box>\n"
        )
    );
    assert!(!stable.contains("<|point_start|>"));
    assert!(!stable.contains("<|point_end|>"));
}

#[test]
fn test_stable_visible_text_streams_nested_track_children_before_point_end() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = concat!(
        "Tracking hand.\n<|point_start|><collection mention=\"right hand\">\n ",
        "<track mention=\"Wrist\">\n ",
        "<point t=\"0.0 seconds\"> (10,20) </point>\n ",
        "<point t=\"1.0 seconds\"> (30,40) </point>\n",
        "</track>\n"
    );

    let stable = encoding.stable_visible_text_from_point_spans(text);

    assert_eq!(
        stable,
        concat!(
            "Tracking hand.\n<collection mention=\"right hand\">\n ",
            "<track mention=\"Wrist\">\n ",
            "<point t=\"0.0 seconds\"> (10,20) </point>\n ",
            "<point t=\"1.0 seconds\"> (30,40) </point>\n",
            "</track>\n"
        )
    );
    assert!(!stable.contains("<|point_start|>"));
    assert!(!stable.contains("<|point_end|>"));
}

#[test]
fn test_stable_visible_text_emits_complete_malformed_point_payload_without_panic() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    assert_eq!(
        encoding.stable_visible_text_from_point_spans("Look <|point_start|>><|point_end|> after"),
        "Look > after"
    );
    assert_eq!(
        encoding.stable_visible_text_from_point_spans(
            "Look <|point_start|>garbage></point><|point_end|> after"
        ),
        "Look garbage></point> after"
    );
}

#[test]
fn test_stable_visible_text_strips_nested_point_sentinels_from_complete_malformed_payload() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let stable = encoding.stable_visible_text_from_point_spans(
        "Look <|point_start|>bad <|point_start|> inner<|point_end|> after",
    );

    assert_eq!(stable, "Look bad  inner after");
    assert!(!stable.contains("<|point_start|>"));
    assert!(!stable.contains("<|point_end|>"));
}

#[test]
fn test_parse_completion_view_streams_stable_point_children_from_single_parse() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = concat!(
        "<|start|>assistant<|channel|>final<|message|>Look\n",
        "<|point_start|><collection mention=\"signs\">\n ",
        "<point_box> (10,20) (30,40) </point_box>\n ",
        "<point_box> (50,60)"
    );
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);

    let view = encoding
        .parse_completion_view_from_tokens(
            tokens.iter().copied(),
            None,
            ParseOptions { strict: false },
        )
        .unwrap();

    assert_eq!(view.messages.len(), 1);
    assert_eq!(view.stable_message_texts.len(), 1);
    assert_eq!(
        view.messages[0].content,
        vec![Content::Text(TextContent {
            text: "Look\n".to_string()
        })]
    );
    assert_eq!(
        view.stable_message_texts[0],
        concat!(
            "Look\n<collection mention=\"signs\">\n ",
            "<point_box> (10,20) (30,40) </point_box>\n "
        )
    );
    assert!(!view.stable_message_texts[0].contains("<|point_start|>"));
    assert!(!view.stable_message_texts[0].contains("<|point_end|>"));
}

#[test]
fn test_stable_visible_text_emits_complete_semantically_invalid_point_payload() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "Look <|point_start|><point mention=\"point_box\"> (800,300) (801,301) </point><|point_end|> after";

    let stable = encoding.stable_visible_text_from_point_spans(text);

    assert_eq!(
        stable,
        "Look <point mention=\"point_box\"> (800,300) (801,301) </point> after"
    );
    assert!(!stable.contains("<|point_start|>"));
    assert!(!stable.contains("<|point_end|>"));
}

#[test]
fn test_stable_visible_text_streams_complete_semantically_invalid_child_before_point_end() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "Look <|point_start|><point mention=\"point_box\"> (800,300) (801,301) </point>";

    let stable = encoding.stable_visible_text_from_point_spans(text);

    assert_eq!(
        stable,
        "Look <point mention=\"point_box\"> (800,300) (801,301) </point>"
    );
    assert!(!stable.contains("<|point_start|>"));
    assert!(!stable.contains("<|point_end|>"));
}

#[test]
fn test_parse_completion_view_stable_text_emits_complete_malformed_collection_child() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = concat!(
        "<|start|>assistant<|channel|>final<|message|>Look\n",
        "<|point_start|><collection mention=\"signs\">\n ",
        "<point mention=\"valid\"> (10,20) </point>\n ",
        "<point mention=\"point_box\"> (800,300) (801,301) </point>\n ",
        "<point mention=\"after\"> (50,60) </point>\n",
        "</collection><|point_end|><|return|>"
    );
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);

    let view = encoding
        .parse_completion_view_from_tokens(
            tokens.iter().copied(),
            None,
            ParseOptions { strict: false },
        )
        .unwrap();

    assert_eq!(view.messages.len(), 1);
    assert_eq!(view.stable_message_texts.len(), 1);
    assert_eq!(
        view.stable_message_texts[0],
        concat!(
            "Look\n<collection mention=\"signs\">\n ",
            "<point mention=\"valid\"> (10,20) </point>\n ",
            "<point mention=\"point_box\"> (800,300) (801,301) </point>\n ",
            "<point mention=\"after\"> (50,60) </point>\n",
            "</collection>"
        )
    );
}

#[test]
fn test_parse_completion_view_stable_text_streams_complete_semantically_invalid_collection_child_before_point_end(
) {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = concat!(
        "<|start|>assistant<|channel|>final<|message|>Look\n",
        "<|point_start|><collection mention=\"signs\">\n ",
        "<point mention=\"point_box\"> (800,300) (801,301) </point>\n "
    );
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);

    let view = encoding
        .parse_completion_view_from_tokens(
            tokens.iter().copied(),
            None,
            ParseOptions { strict: false },
        )
        .unwrap();

    assert_eq!(view.messages.len(), 1);
    assert_eq!(view.stable_message_texts.len(), 1);
    assert_eq!(
        view.stable_message_texts[0],
        concat!(
            "Look\n<collection mention=\"signs\">\n ",
            "<point mention=\"point_box\"> (800,300) (801,301) </point>\n "
        )
    );
    assert!(!view.stable_message_texts[0].contains("<|point_start|>"));
    assert!(!view.stable_message_texts[0].contains("<|point_end|>"));
}

#[test]
fn test_parse_completion_view_prefixes_are_child_atomic_and_append_safe() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let payload = concat!(
        "<collection mention=\"signs\">\n ",
        "<point_box> (10,20) (30,40) </point_box>\n ",
        "<point_box> (50,60) (70,80) </point_box>\n",
        "</collection>"
    );
    let final_visible = format!("Look\n{payload}");

    let mut previous_stable = String::new();
    for cut in 0..=payload.len() {
        let text = format!(
            "<|start|>assistant<|channel|>final<|message|>Look\n<|point_start|>{}",
            &payload[..cut]
        );
        let tokens = encoding.tokenizer().encode_with_special_tokens(&text);
        let view = encoding
            .parse_completion_view_from_tokens(
                tokens.iter().copied(),
                None,
                ParseOptions { strict: false },
            )
            .unwrap();
        let stable = &view.stable_message_texts[0];

        assert!(!stable.contains("<|point_start|>"));
        assert!(!stable.contains("<|point_end|>"));
        assert!(
            final_visible.starts_with(stable),
            "stable prefix was not append-safe: {stable:?}"
        );
        assert!(
            stable.starts_with(&previous_stable),
            "stable text regressed from {previous_stable:?} to {stable:?}"
        );
        assert_eq!(
            stable.matches("<point_box").count(),
            stable.matches("</point_box>").count()
        );
        previous_stable = stable.to_string();
    }
}

#[test]
fn test_parse_completion_view_stable_text_suppresses_stray_point_sentinels() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant<|message|>Look <|point_end|> after";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);

    let view = encoding
        .parse_completion_view_from_tokens(
            tokens.iter().copied(),
            None,
            ParseOptions { strict: true },
        )
        .unwrap();

    assert_eq!(view.messages.len(), 1);
    assert_eq!(view.stable_message_texts, vec!["Look  after".to_string()]);
    assert_eq!(
        view.messages[0].content,
        vec![Content::Text(TextContent {
            text: "Look <|point_end|> after".to_string()
        })]
    );
}

#[test]
fn test_parse_completion_view_emits_complete_malformed_stable_text_and_preserves_message_behavior()
{
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant<|message|>Look <|point_start|>><|point_end|> after";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);

    let view = encoding
        .parse_completion_view_from_tokens(
            tokens.iter().copied(),
            None,
            ParseOptions { strict: true },
        )
        .unwrap();

    assert_eq!(view.messages.len(), 1);
    assert_eq!(view.stable_message_texts, vec!["Look > after".to_string()]);
    assert_eq!(
        view.messages[0].content,
        vec![
            Content::Text(TextContent {
                text: "Look ".to_string()
            }),
            Content::Text(TextContent {
                text: "<|point_start|>><|point_end|>".to_string()
            }),
            Content::Text(TextContent {
                text: " after".to_string()
            })
        ]
    );
}

#[test]
fn test_incomplete_point_payload_is_not_exposed_as_text_content() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let point_payload = concat!(
        "<collection mention=\"signs\">\n ",
        "<point_box> (10,20) (30,40) </point_box>\n</collection>"
    );
    let incomplete_text =
        format!("<|start|>assistant<|message|>Look <|point_start|>{point_payload}");
    let tokens = encoding
        .tokenizer()
        .encode_with_special_tokens(&incomplete_text);

    let parsed = encoding
        .parse_messages_from_completion_tokens_with_options(
            tokens.iter().copied(),
            None,
            ParseOptions { strict: false },
        )
        .unwrap();

    assert_eq!(parsed.len(), 1);
    assert_eq!(
        parsed[0].content,
        vec![Content::Text(TextContent {
            text: "Look ".to_string()
        })]
    );

    let strict_parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .unwrap();
    assert_eq!(strict_parsed.len(), 1);
    assert_eq!(
        strict_parsed[0].content,
        vec![
            Content::Text(TextContent {
                text: "Look ".to_string()
            }),
            Content::Text(TextContent {
                text: format!("<|point_start|>{point_payload}")
            })
        ]
    );

    let complete_text = format!("{incomplete_text}<|point_end|>");
    let complete_tokens = encoding
        .tokenizer()
        .encode_with_special_tokens(&complete_text);
    let complete_parsed = encoding
        .parse_messages_from_completion_tokens_with_options(
            complete_tokens,
            None,
            ParseOptions { strict: false },
        )
        .unwrap();

    assert_eq!(complete_parsed.len(), 1);
    assert_eq!(complete_parsed[0].content.len(), 2);
    assert_eq!(
        complete_parsed[0].content[0],
        Content::Text(TextContent {
            text: "Look ".to_string()
        })
    );
    match &complete_parsed[0].content[1] {
        Content::PointCollection(collection) => {
            assert_eq!(collection.mention.as_deref(), Some("signs"));
            assert_eq!(collection.points.len(), 1);
        }
        other => panic!("expected completed point collection, got {other:?}"),
    }
}

#[test]
fn test_incremental_point_parser_suppresses_incomplete_payload() {
    let mut parser = IncrementalPointParser::new(Some("<|point_start|>"), Some("<|point_end|>"));

    let events = parser.push_text("Look <|point_start|><collection mention=\"x\">");

    assert_eq!(
        events,
        vec![
            "Look ".to_string(),
            "<collection mention=\"x\">".to_string()
        ]
    );
}

#[test]
fn test_incremental_point_parser_handles_split_sentinel_and_stable_child() {
    let mut parser = IncrementalPointParser::new(Some("<|point_start|>"), Some("<|point_end|>"));

    assert_eq!(parser.push_text("Look <|point_"), vec!["Look ".to_string()]);
    assert_eq!(
        parser.push_text("start|><collection mention=\"x\">\n "),
        vec!["<collection mention=\"x\">\n ".to_string()]
    );
    assert_eq!(
        parser.push_text("<point_box> (1,2) (3,4) </point_box>\n"),
        vec!["<point_box> (1,2) (3,4) </point_box>\n".to_string()]
    );
    assert_eq!(
        parser.push_text("</collection><|point_end|> done"),
        vec!["</collection>".to_string(), " done".to_string()]
    );
}

#[test]
fn test_completion_stream_default_accepts_assistant_completion_suffix() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|channel|>final<|message|>Hello<|return|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let mut parser = encoding
        .completion_stream_parser(CompletionStreamOptions::default())
        .unwrap();

    let update = parser.push_tokens(&tokens, 0).unwrap();

    assert_eq!(update.completed_messages[0].author.role, Role::Assistant);
    assert_eq!(update.completed_messages[0].recipient, None);
    assert_eq!(update.text.visible_delta, "Hello");
    assert_eq!(update.text.visible_full, "Hello");
    assert_eq!(update.text.final_delta, "Hello");
    assert_eq!(update.text.final_full, "Hello");
    assert_eq!(update.text.reasoning_delta, "");
    assert_eq!(
        update.stop.as_ref().map(|stop| stop.kind),
        Some(StreamStopKind::Final)
    );
    assert_eq!(update.token_count, tokens.len());
}

#[test]
fn test_completion_stream_explicit_role_none_accepts_full_message_stream() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant<|channel|>final<|message|>Hello<|return|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let mut parser = encoding
        .completion_stream_parser(CompletionStreamOptions {
            role: None,
            ..CompletionStreamOptions::default()
        })
        .unwrap();

    let update = parser.push_tokens(&tokens, 0).unwrap();

    assert_eq!(update.completed_messages[0].author.role, Role::Assistant);
    assert_eq!(update.completed_messages[0].recipient, None);
    assert_eq!(update.text.visible_delta, "Hello");
    assert_eq!(update.text.visible_full, "Hello");
    assert_eq!(update.text.final_delta, "Hello");
    assert_eq!(update.text.final_full, "Hello");
    assert_eq!(update.text.reasoning_delta, "");
    assert_eq!(
        update.stop.as_ref().map(|stop| stop.kind),
        Some(StreamStopKind::Final)
    );
    assert_eq!(update.token_count, tokens.len());
}

#[test]
fn test_completion_stream_end_message_final_stop_matches_semantic_trim() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let stopped_tokens = encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>assistant<|channel|>final<|message|>Hello<|end|>");
    let longer_tokens = encoding.tokenizer().encode_with_special_tokens(
        "<|start|>assistant<|channel|>final<|message|>Hello<|end|>extra",
    );
    let mut parser = encoding
        .completion_stream_parser(CompletionStreamOptions {
            role: None,
            strict: false,
            stop_on_tool_call: false,
            include_thinking_channels: false,
        })
        .unwrap();

    let update = parser.push_tokens(&longer_tokens, 0).unwrap();

    assert_eq!(
        update.stop.as_ref().map(|stop| stop.kind),
        Some(StreamStopKind::Final)
    );
    assert_eq!(update.token_count, stopped_tokens.len());
    assert_eq!(update.text.visible_full, "Hello");
    assert_eq!(update.text.final_full, "Hello");
}

#[test]
fn test_completion_stream_finish_finalizes_open_message_without_stop_token() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let tokens = encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>assistant<|channel|>final<|message|>partial answer");
    let mut parser = encoding
        .completion_stream_parser(CompletionStreamOptions {
            role: None,
            ..CompletionStreamOptions::default()
        })
        .unwrap();

    let update = parser.push_tokens(&tokens, 0).unwrap();
    assert!(update.completed_messages.is_empty());

    let finish = parser.finish(tokens.len()).unwrap();

    assert_eq!(finish.token_count, tokens.len());
    assert_eq!(finish.stop, None);
    assert_eq!(finish.text.visible_delta, "");
    assert_eq!(finish.text.visible_full, "partial answer");
    assert_eq!(finish.text.final_full, "partial answer");
    assert_eq!(finish.completed_messages[0].author.role, Role::Assistant);
    assert_eq!(
        finish.completed_messages[0].channel.as_deref(),
        Some(CanonicalChannel::Final.as_str())
    );

    let repeated_finish = parser.finish(tokens.len()).unwrap();
    assert_eq!(repeated_finish.token_count, tokens.len());
    assert!(repeated_finish.completed_messages.is_empty());
    assert_eq!(repeated_finish.stop, None);
    assert_eq!(repeated_finish.text.visible_delta, "");

    let empty_after_finish = parser.push_tokens(&[], tokens.len()).unwrap();
    assert_eq!(empty_after_finish.token_count, tokens.len());
    assert!(empty_after_finish.completed_messages.is_empty());

    let extra = encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>assistant");
    assert!(parser.push_tokens(&extra, tokens.len()).is_err());
}

#[derive(Debug, PartialEq, Eq)]
struct CompletionStreamSummary {
    token_count: usize,
    completed_message_count: usize,
    stop_kind: Option<StreamStopKind>,
    visible_delta: String,
    visible_full: String,
    final_delta: String,
    final_full: String,
    reasoning_delta: String,
    reasoning_full: String,
}

fn summarize_completion_stream(
    encoding: &crate::encoding::HarmonyEncoding,
    tokens: &[Rank],
    options: CompletionStreamOptions,
    chunk_sizes: &[usize],
    finish: bool,
) -> CompletionStreamSummary {
    let mut parser = encoding.completion_stream_parser(options).unwrap();
    let mut token_count = 0;
    let mut completed_message_count = 0;
    let mut stop_kind = None;
    let mut visible_delta = String::new();
    let mut visible_full = String::new();
    let mut final_delta = String::new();
    let mut final_full = String::new();
    let mut reasoning_delta = String::new();
    let mut reasoning_full = String::new();
    let mut chunk_index = 0;

    while token_count < tokens.len() {
        let chunk_size = chunk_sizes[chunk_index % chunk_sizes.len()].max(1);
        let end = tokens.len().min(token_count + chunk_size);
        let update = parser
            .push_tokens(&tokens[token_count..end], token_count)
            .unwrap();
        token_count = update.token_count;
        completed_message_count += update.completed_messages.len();
        stop_kind = update.stop.map(|stop| stop.kind).or(stop_kind);
        visible_delta.push_str(&update.text.visible_delta);
        visible_full = update.text.visible_full;
        final_delta.push_str(&update.text.final_delta);
        final_full = update.text.final_full;
        reasoning_delta.push_str(&update.text.reasoning_delta);
        reasoning_full = update.text.reasoning_full;
        if stop_kind.is_some() {
            break;
        }
        chunk_index += 1;
    }

    if finish && stop_kind.is_none() {
        let update = parser.finish(token_count).unwrap();
        completed_message_count += update.completed_messages.len();
        visible_delta.push_str(&update.text.visible_delta);
        visible_full = update.text.visible_full;
        final_delta.push_str(&update.text.final_delta);
        final_full = update.text.final_full;
        reasoning_delta.push_str(&update.text.reasoning_delta);
        reasoning_full = update.text.reasoning_full;
    }

    CompletionStreamSummary {
        token_count,
        completed_message_count,
        stop_kind,
        visible_delta,
        visible_full,
        final_delta,
        final_full,
        reasoning_delta,
        reasoning_full,
    }
}

#[test]
fn test_completion_stream_chunking_matches_single_chunk_for_visible_reasoning_tool_and_points() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let cases = [
        (
            concat!(
                "<|start|>assistant<|channel|>analysis<|message|>think café<|end|>",
                "<|start|>assistant<|channel|>final<|message|>done ✅<|return|>"
            ),
            CompletionStreamOptions {
                role: None,
                strict: false,
                stop_on_tool_call: false,
                include_thinking_channels: false,
            },
            false,
        ),
        (
            concat!(
                "<|start|>assistant<|channel|>final<|message|>Tracking. ",
                "<|point_start|><collection mention=\"sign\">\n ",
                "<point_box> (10,20) (30,40) </point_box>\n</collection><|point_end|><|return|>"
            ),
            CompletionStreamOptions {
                role: None,
                strict: false,
                stop_on_tool_call: false,
                include_thinking_channels: false,
            },
            false,
        ),
        (
            concat!(
                "<|start|>assistant<|channel|>analysis<|message|>need lookup<|end|>",
                "<|start|>assistant<|channel|>commentary to=functions.lookup <|constrain|>json<|message|>{\"q\":1}<|call|>"
            ),
            CompletionStreamOptions {
                role: None,
                strict: false,
                stop_on_tool_call: true,
                include_thinking_channels: false,
            },
            false,
        ),
    ];

    for (text, options, finish) in cases {
        let tokens = encoding.tokenizer().encode_with_special_tokens(text);
        let single_chunk = summarize_completion_stream(
            &encoding,
            &tokens,
            options.clone(),
            &[tokens.len()],
            finish,
        );
        let one_token_chunks =
            summarize_completion_stream(&encoding, &tokens, options.clone(), &[1], finish);
        let uneven_chunks =
            summarize_completion_stream(&encoding, &tokens, options, &[2, 3, 5, 7], finish);

        assert_eq!(
            one_token_chunks, single_chunk,
            "one-token chunking differed for {text}"
        );
        assert_eq!(
            uneven_chunks, single_chunk,
            "uneven chunking differed for {text}"
        );
    }
}

#[test]
fn test_completion_stream_chunking_matches_single_chunk_when_finish_flushes_open_message() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant<|channel|>final<|message|>partial point <|point_start|><point_box> (1,2)";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let options = CompletionStreamOptions {
        role: None,
        strict: false,
        stop_on_tool_call: false,
        include_thinking_channels: false,
    };

    let single_chunk =
        summarize_completion_stream(&encoding, &tokens, options.clone(), &[tokens.len()], true);
    let one_token_chunks =
        summarize_completion_stream(&encoding, &tokens, options.clone(), &[1], true);
    let uneven_chunks =
        summarize_completion_stream(&encoding, &tokens, options, &[2, 3, 5, 7], true);

    assert_eq!(one_token_chunks, single_chunk);
    assert_eq!(uneven_chunks, single_chunk);
    assert_eq!(single_chunk.completed_message_count, 1);
}

#[test]
fn test_completion_stream_rejects_wrong_suffix_offset() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let tokens = encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>assistant<|channel|>final<|message|>Hello");
    let mut parser = encoding
        .completion_stream_parser(CompletionStreamOptions {
            role: None,
            strict: false,
            stop_on_tool_call: false,
            include_thinking_channels: false,
        })
        .unwrap();

    parser.push_tokens(&tokens[..tokens.len() - 1], 0).unwrap();
    let err = parser
        .push_tokens(&tokens[tokens.len() - 1..], 0)
        .unwrap_err();

    assert!(err.to_string().contains("offset mismatch"));
}

#[test]
fn test_completion_stream_stop_is_terminal_and_idempotent_for_empty_suffix() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let stopped_tokens = encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>assistant<|channel|>final<|message|>Hello<|return|>");
    let longer_tokens = encoding.tokenizer().encode_with_special_tokens(
        "<|start|>assistant<|channel|>final<|message|>Hello<|return|>extra",
    );
    let mut parser = encoding
        .completion_stream_parser(CompletionStreamOptions {
            role: None,
            strict: false,
            stop_on_tool_call: false,
            include_thinking_channels: false,
        })
        .unwrap();

    let first = parser.push_tokens(&stopped_tokens, 0).unwrap();
    let second = parser.push_tokens(&[], stopped_tokens.len()).unwrap();
    let err = parser
        .push_tokens(&longer_tokens[stopped_tokens.len()..], stopped_tokens.len())
        .unwrap_err();

    assert_eq!(
        first.stop.as_ref().map(|stop| stop.kind),
        Some(StreamStopKind::Final)
    );
    assert_eq!(
        second.stop.as_ref().map(|stop| stop.kind),
        Some(StreamStopKind::Final)
    );
    assert_eq!(second.text.visible_delta, "");
    assert!(err.to_string().contains("already finished"));
}

#[test]
fn test_completion_stream_routes_reasoning_and_tool_calls_without_leaking_arguments() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = concat!(
        "<|start|>assistant<|channel|>analysis<|message|>think<|end|>",
        "<|start|>assistant<|channel|>commentary to=functions.lookup <|constrain|>json<|message|>{\"q\":1}<|call|>"
    );
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let mut parser = encoding
        .completion_stream_parser(CompletionStreamOptions {
            role: None,
            strict: false,
            stop_on_tool_call: true,
            include_thinking_channels: false,
        })
        .unwrap();

    let update = parser.push_tokens(&tokens, 0).unwrap();

    assert_eq!(update.text.visible_full, "");
    assert_eq!(update.text.final_full, "");
    assert_eq!(update.text.reasoning_full, "think");
    assert_eq!(
        update.stop.as_ref().map(|stop| stop.kind),
        Some(StreamStopKind::ToolCall)
    );
    assert_eq!(update.completed_messages.len(), 2);
}

#[test]
fn test_completion_stream_hides_markerless_recipient_headers() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    // Canonical channel names come from the typed enum (the meta-test below
    // forbids owning the quoted literals outside channels.rs).
    let analysis = channels::CanonicalChannel::Analysis.as_str();
    let final_channel = channels::CanonicalChannel::Final.as_str();
    let cases = [
        (format!("{analysis}<|message|>hidden<|end|>"), analysis),
        (
            format!("{final_channel}<|message|>{{\"answer\":1}}<|return|>"),
            final_channel,
        ),
    ];

    for (text, recipient) in cases {
        let tokens = encoding.tokenizer().encode_with_special_tokens(&text);
        let mut parser = encoding
            .completion_stream_parser(CompletionStreamOptions::default())
            .unwrap();

        let update = parser.push_tokens(&tokens, 0).unwrap();

        assert_eq!(update.completed_messages.len(), 1);
        assert_eq!(update.completed_messages[0].channel, None);
        assert_eq!(
            update.completed_messages[0].recipient.as_deref(),
            Some(recipient)
        );
        assert_eq!(update.text.visible_full, "");
        assert_eq!(update.text.final_full, "");
        assert_eq!(update.text.reasoning_full, "");
        assert_eq!(update.stop, None);
    }
}

#[test]
fn test_completion_stream_excludes_recipient_analysis_from_reasoning() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text =
        "<|start|>assistant to=functions.lookup <|channel|>analysis<|message|>scratch<|end|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);

    let mut hidden_parser = encoding
        .completion_stream_parser(CompletionStreamOptions {
            role: None,
            strict: false,
            stop_on_tool_call: false,
            include_thinking_channels: false,
        })
        .unwrap();
    let hidden_update = hidden_parser.push_tokens(&tokens, 0).unwrap();
    assert_eq!(hidden_update.text.visible_full, "");
    assert_eq!(hidden_update.text.final_full, "");
    assert_eq!(hidden_update.text.reasoning_full, "");

    let mut visible_parser = encoding
        .completion_stream_parser(CompletionStreamOptions {
            role: None,
            strict: false,
            stop_on_tool_call: false,
            include_thinking_channels: true,
        })
        .unwrap();
    let visible_update = visible_parser.push_tokens(&tokens, 0).unwrap();
    assert_eq!(visible_update.text.visible_full, "scratch");
    assert_eq!(visible_update.text.final_full, "");
    assert_eq!(visible_update.text.reasoning_full, "");
}

#[test]
fn test_pointing_html_roundtrip_preserves_index_and_escaped_attrs() {
    let point = PointingContent {
        kind: PointingKind::Point,
        coords: vec![(10, 20)],
        target: None,
        t: Some(1.0),
        start_time: None,
        end_time: None,
        mention: Some("a & <b> \"c\"".to_string()),
        label: None,
        index: Some("image&0".to_string()),
    };

    let html = point.to_html();
    assert!(html.contains("mention=\"a &amp; &lt;b&gt; &quot;c&quot;\""));
    assert!(html.contains("index=\"image&amp;0\""));

    let parsed = PointingContent::from_html(&html).unwrap();
    assert_eq!(parsed.mention.as_deref(), Some("a & <b> \"c\""));
    assert_eq!(parsed.index.as_deref(), Some("image&0"));
}

#[test]
fn test_pointing_html_parse_accepts_decoded_coord_bodies() {
    let cases = [
        (
            "<point mention=\"target\" index=\"image_0\" t=\"1\"> 10  20 </point>",
            PointingKind::Point,
            vec![(10, 20)],
        ),
        (
            "<point_box> 10  20  30  40 </point_box>",
            PointingKind::Box,
            vec![(10, 20), (30, 40)],
        ),
        (
            "<polygon> 1  2  3  4  5  6 </polygon>",
            PointingKind::Polygon,
            vec![(1, 2), (3, 4), (5, 6)],
        ),
    ];

    for (html, expected_kind, expected_coords) in &cases {
        let parsed = PointingContent::from_html(html).unwrap();
        assert_eq!(parsed.kind, *expected_kind);
        assert_eq!(&parsed.coords, expected_coords);
    }

    let point = PointingContent::from_html(cases[0].0).unwrap();
    assert_eq!(point.mention.as_deref(), Some("target"));
    assert_eq!(point.index.as_deref(), Some("image_0"));
    assert_eq!(point.t, Some(1.0));
}

#[test]
fn test_point_collection_html_parse_accepts_decoded_coord_bodies() {
    let collection = PointCollectionContent::from_html(
        "<collection mention=\"targets\">\n \
         <point> 10  20 </point>\n \
         <point_box> 30  40  50  60 </point_box>\n\
         </collection>",
    )
    .unwrap();

    assert_eq!(collection.mention.as_deref(), Some("targets"));
    assert_eq!(collection.points.len(), 2);
    assert_eq!(
        as_pointing_item(&collection.points[0]).coords,
        vec![(10, 20)]
    );
    assert_eq!(
        as_pointing_item(&collection.points[1]).coords,
        vec![(30, 40), (50, 60)]
    );
}

#[test]
fn test_pointing_html_parse_rejects_odd_decoded_coord_count() {
    let err = PointingContent::from_html("<point_box> 10 20 30 </point_box>").unwrap_err();
    assert!(err.to_string().contains("even number of coordinate values"));
}

#[test]
fn test_pointing_t_format_is_decimal_not_scientific() {
    let point = PointingContent {
        kind: PointingKind::Point,
        coords: vec![(10, 20)],
        target: None,
        t: Some(1e-10),
        start_time: None,
        end_time: None,
        mention: None,
        label: None,
        index: None,
    };

    let html = point.to_html();
    assert!(html.contains("t=\""), "expected t attribute in {html}");
    let t_start = html.find("t=\"").unwrap() + 3;
    let t_end = html[t_start..].find('"').unwrap() + t_start;
    let t_str = &html[t_start..t_end];
    assert!(
        !t_str.contains('e') && !t_str.contains('E'),
        "t uses scientific notation: {t_str}"
    );

    let parsed = PointingContent::from_html(&html).unwrap();
    assert!(parsed.t.is_some());
    let parsed_t = parsed.t.unwrap();
    assert!((parsed_t - 1e-10).abs() < 1e-20);
}

#[test]
fn test_clip_pointing_render_parse_and_label_serialization() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let clip = PointingContent {
        kind: PointingKind::Clip,
        coords: vec![],
        target: Some(0),
        t: None,
        start_time: Some(1.25),
        end_time: Some(3.75),
        mention: Some("scene".to_string()),
        label: Some("not_serialized".to_string()),
        index: Some("video_0".to_string()),
    };

    let expected_html =
        "<clip mention=\"scene\" index=\"video_0\" t=\"1.25 seconds 3.75 seconds\" />";
    assert_eq!(clip.to_html(), expected_html);

    let parsed_html = PointingContent::from_html(
        "<clip mention=\"scene\" index=\"video_0\" t=\"1.25s 3.75s\" />",
    )
    .unwrap();
    assert_eq!(parsed_html.kind, PointingKind::Clip);
    assert_eq!(parsed_html.start_time, Some(1.25));
    assert_eq!(parsed_html.end_time, Some(3.75));
    assert_eq!(parsed_html.mention.as_deref(), Some("scene"));
    assert_eq!(parsed_html.index.as_deref(), Some("video_0"));
    assert_eq!(parsed_html.to_html(), expected_html);

    let convo = Conversation::from_messages([Message::from_role_and_contents(
        Role::User,
        vec![Content::Pointing(clip)],
    )]);
    let tokens = encoding.render_conversation(&convo, None).unwrap();
    let decoded = encoding.tokenizer().decode_utf8(&tokens).unwrap();
    assert!(decoded.contains("<|point_start|>"));
    assert!(decoded.contains(expected_html));
    assert!(!decoded.contains("not_serialized"));

    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .unwrap();
    match &parsed[0].content[0] {
        Content::Pointing(parsed_clip) => {
            assert_eq!(parsed_clip.kind, PointingKind::Clip);
            assert_eq!(parsed_clip.start_time, Some(1.25));
            assert_eq!(parsed_clip.end_time, Some(3.75));
            assert_eq!(parsed_clip.mention.as_deref(), Some("scene"));
            assert_eq!(parsed_clip.label, None);
            assert_eq!(parsed_clip.index.as_deref(), Some("video_0"));
        }
        other => panic!("expected clip pointing content, got {:?}", other),
    }
}

#[test]
fn test_point_collection_render_and_parse_canonicalizes() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let collection = PointCollectionContent {
        points: vec![
            PointingContent {
                kind: PointingKind::Point,
                coords: vec![(17, 300)],
                target: Some(0),
                t: None,
                start_time: None,
                end_time: None,
                mention: None,
                label: None,
                index: None,
            }
            .into(),
            PointingContent {
                kind: PointingKind::Point,
                coords: vec![(209, 169)],
                target: Some(0),
                t: None,
                start_time: None,
                end_time: None,
                mention: None,
                label: None,
                index: None,
            }
            .into(),
            PointingContent {
                kind: PointingKind::Point,
                coords: vec![(605, 514)],
                target: Some(0),
                t: None,
                start_time: None,
                end_time: None,
                mention: None,
                label: None,
                index: None,
            }
            .into(),
        ],
        kind: PointCollectionKind::Collection,
        t: None,
        mention: Some("targets".to_string()),
        label: None,
    };

    let expected_html = "<collection mention=\"targets\">\n <point> (209,169) </point>\n <point> (17,300) </point>\n <point> (605,514) </point>\n</collection>\n";
    assert_eq!(collection.to_html(), expected_html);

    let convo = Conversation::from_messages([Message::from_role_and_contents(
        Role::User,
        vec![Content::PointCollection(collection)],
    )]);

    let tokens = encoding.render_conversation(&convo, None).unwrap();
    let decoded = encoding.tokenizer().decode_utf8(&tokens).unwrap();
    assert!(decoded.contains(expected_html));

    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].content.len(), 1);
    match &parsed[0].content[0] {
        Content::PointCollection(parsed_collection) => {
            assert_eq!(parsed_collection.mention.as_deref(), Some("targets"));
            assert_eq!(parsed_collection.points.len(), 3);
            assert_eq!(
                as_pointing_item(&parsed_collection.points[0]).coords,
                vec![(209, 169)]
            );
            assert_eq!(
                as_pointing_item(&parsed_collection.points[1]).coords,
                vec![(17, 300)]
            );
            assert_eq!(
                as_pointing_item(&parsed_collection.points[2]).coords,
                vec![(605, 514)]
            );
        }
        other => panic!("expected point collection content, got {:?}", other),
    }
}

#[test]
fn test_clip_collection_render_and_parse_canonicalizes_by_time() {
    let collection = PointCollectionContent {
        points: vec![
            PointingContent {
                kind: PointingKind::Clip,
                coords: vec![],
                target: Some(0),
                t: None,
                start_time: Some(3.0),
                end_time: Some(4.0),
                mention: Some("third".to_string()),
                label: None,
                index: None,
            }
            .into(),
            PointingContent {
                kind: PointingKind::Clip,
                coords: vec![],
                target: Some(0),
                t: None,
                start_time: Some(1.0),
                end_time: Some(2.0),
                mention: Some("first".to_string()),
                label: None,
                index: None,
            }
            .into(),
            PointingContent {
                kind: PointingKind::Clip,
                coords: vec![],
                target: Some(0),
                t: None,
                start_time: Some(2.0),
                end_time: None,
                mention: Some("moment".to_string()),
                label: None,
                index: None,
            }
            .into(),
        ],
        kind: PointCollectionKind::Collection,
        t: None,
        mention: Some("clips".to_string()),
        label: None,
    };

    let expected_html = "<collection mention=\"clips\">\n <clip mention=\"first\" t=\"1 seconds 2 seconds\" />\n <clip mention=\"moment\" t=\"2 seconds\" />\n <clip mention=\"third\" t=\"3 seconds 4 seconds\" />\n</collection>\n";
    assert_eq!(collection.to_html(), expected_html);

    let parsed = PointCollectionContent::from_html(&collection.to_html()).unwrap();
    assert_eq!(parsed.points.len(), 3);
    assert_eq!(as_pointing_item(&parsed.points[0]).start_time, Some(1.0));
    assert_eq!(as_pointing_item(&parsed.points[0]).end_time, Some(2.0));
    assert_eq!(as_pointing_item(&parsed.points[1]).start_time, Some(2.0));
    assert_eq!(as_pointing_item(&parsed.points[1]).end_time, None);
    assert_eq!(as_pointing_item(&parsed.points[2]).start_time, Some(3.0));
    assert_eq!(as_pointing_item(&parsed.points[2]).end_time, Some(4.0));
}

#[test]
fn test_point_track_render_and_parse_preserves_track_container() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let track = PointCollectionContent {
        points: vec![
            PointingContent {
                kind: PointingKind::Box,
                coords: vec![(50, 60), (70, 80)],
                target: Some(0),
                t: Some(1.0),
                start_time: None,
                end_time: None,
                mention: Some("child mention should not render".to_string()),
                label: None,
                index: None,
            }
            .into(),
            PointingContent {
                kind: PointingKind::Box,
                coords: vec![(10, 20), (30, 40)],
                target: Some(0),
                t: Some(0.0),
                start_time: None,
                end_time: None,
                mention: None,
                label: None,
                index: None,
            }
            .into(),
        ],
        kind: PointCollectionKind::Track,
        t: None,
        mention: Some("dancer".to_string()),
        label: None,
    };

    let expected_html = "<track mention=\"dancer\">\n <point_box t=\"0.0\"> (10,20) (30,40) </point_box>\n <point_box t=\"1.0\"> (50,60) (70,80) </point_box>\n</track>\n";
    assert_eq!(track.to_html(), expected_html);

    let convo = Conversation::from_messages([Message::from_role_and_contents(
        Role::User,
        vec![Content::PointCollection(track)],
    )]);
    let tokens = encoding.render_conversation(&convo, None).unwrap();
    let decoded = encoding.tokenizer().decode_utf8(&tokens).unwrap();
    assert!(decoded.contains(&format!("<|point_start|>{expected_html}<|point_end|>")));
    assert!(!decoded.contains("child mention should not render"));

    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .unwrap();
    match &parsed[0].content[0] {
        Content::PointCollection(parsed_track) => {
            assert_eq!(parsed_track.kind, PointCollectionKind::Track);
            assert_eq!(parsed_track.mention.as_deref(), Some("dancer"));
            assert_eq!(parsed_track.points.len(), 2);
            assert_eq!(
                as_pointing_item(&parsed_track.points[0]).kind,
                PointingKind::Box
            );
            assert_eq!(as_pointing_item(&parsed_track.points[0]).t, Some(0.0));
            assert_eq!(as_pointing_item(&parsed_track.points[1]).t, Some(1.0));
            assert!(parsed_track
                .points
                .iter()
                .all(|point| as_pointing_item(point).mention.is_none()));
        }
        other => panic!("expected point track content, got {:?}", other),
    }
}

#[test]
fn test_point_collection_render_and_parse_preserves_nested_track_children() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let wrist_track = PointCollectionContent {
        points: vec![
            PointingContent {
                kind: PointingKind::Point,
                coords: vec![(30, 40)],
                target: Some(0),
                t: Some(1.0),
                start_time: None,
                end_time: None,
                mention: Some("child mention should not render".to_string()),
                label: None,
                index: None,
            }
            .into(),
            PointingContent {
                kind: PointingKind::Point,
                coords: vec![(10, 20)],
                target: Some(0),
                t: Some(0.0),
                start_time: None,
                end_time: None,
                mention: None,
                label: None,
                index: None,
            }
            .into(),
        ],
        kind: PointCollectionKind::Track,
        t: None,
        mention: Some("Wrist".to_string()),
        label: None,
    };
    let thumb_track = PointCollectionContent {
        points: vec![
            PointingContent {
                kind: PointingKind::Point,
                coords: vec![(50, 60)],
                target: Some(0),
                t: Some(0.0),
                start_time: None,
                end_time: None,
                mention: None,
                label: None,
                index: None,
            }
            .into(),
            PointingContent {
                kind: PointingKind::Point,
                coords: vec![(70, 80)],
                target: Some(0),
                t: Some(1.0),
                start_time: None,
                end_time: None,
                mention: None,
                label: None,
                index: None,
            }
            .into(),
        ],
        kind: PointCollectionKind::Track,
        t: None,
        mention: Some("Thumb_3".to_string()),
        label: None,
    };
    let collection = PointCollectionContent {
        points: vec![wrist_track.into(), thumb_track.into()],
        kind: PointCollectionKind::Collection,
        t: None,
        mention: Some("right hand".to_string()),
        label: None,
    };

    let expected_html = concat!(
        "<collection mention=\"right hand\">\n",
        " <track mention=\"Wrist\">\n",
        " <point t=\"0.0\"> (10,20) </point>\n",
        " <point t=\"1.0\"> (30,40) </point>\n",
        "</track>\n",
        " <track mention=\"Thumb_3\">\n",
        " <point t=\"0.0\"> (50,60) </point>\n",
        " <point t=\"1.0\"> (70,80) </point>\n",
        "</track>\n",
        "</collection>\n"
    );
    assert_eq!(collection.to_html(), expected_html);

    let convo = Conversation::from_messages([Message::from_role_and_contents(
        Role::User,
        vec![Content::PointCollection(collection)],
    )]);
    let tokens = encoding.render_conversation(&convo, None).unwrap();
    let decoded = encoding.tokenizer().decode_utf8(&tokens).unwrap();
    assert!(decoded.contains(&format!("<|point_start|>{expected_html}<|point_end|>")));
    assert!(!decoded.contains("child mention should not render"));

    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .unwrap();
    match &parsed[0].content[0] {
        Content::PointCollection(parsed_collection) => {
            assert_eq!(parsed_collection.kind, PointCollectionKind::Collection);
            assert_eq!(parsed_collection.mention.as_deref(), Some("right hand"));
            assert_eq!(parsed_collection.points.len(), 2);
            let wrist = as_point_collection_item(&parsed_collection.points[0]);
            let thumb = as_point_collection_item(&parsed_collection.points[1]);
            assert_eq!(wrist.kind, PointCollectionKind::Track);
            assert_eq!(wrist.mention.as_deref(), Some("Wrist"));
            assert_eq!(thumb.kind, PointCollectionKind::Track);
            assert_eq!(thumb.mention.as_deref(), Some("Thumb_3"));
            assert_eq!(as_pointing_item(&wrist.points[0]).coords, vec![(10, 20)]);
            assert_eq!(as_pointing_item(&wrist.points[1]).coords, vec![(30, 40)]);
            assert!(wrist
                .points
                .iter()
                .all(|point| as_pointing_item(point).mention.is_none()));
        }
        other => panic!("expected point collection content, got {:?}", other),
    }
}

#[test]
fn test_multimodal_matches_render_conversation_with_system_content() {
    // Test with SystemContent
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let sys = SystemContent::new()
        .with_model_identity("Test Model")
        .with_reasoning_effort(ReasoningEffort::High)
        .with_conversation_start_date("2025-01-15");

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::System, sys),
        Message::from_role_and_content(Role::User, "Hello!"),
    ]);

    let regular_tokens = encoding.render_conversation(&convo, None).unwrap();
    let multimodal = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();
    let multimodal_tokens = flatten_multimodal_tokens(&multimodal);

    assert_eq!(
        regular_tokens.iter().map(|t| *t as i64).collect::<Vec<_>>(),
        multimodal_tokens,
        "Multimodal should match for conversations with SystemContent"
    );
}

#[test]
fn test_multimodal_matches_render_conversation_with_browser_python_tools() {
    // Test with browser and python tools
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let sys = SystemContent::new()
        .with_conversation_start_date("2025-01-15")
        .with_browser_tool()
        .with_python_tool();

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::System, sys),
        Message::from_role_and_content(Role::User, "Browse to example.com"),
    ]);

    let regular_tokens = encoding.render_conversation(&convo, None).unwrap();
    let multimodal = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();
    let multimodal_tokens = flatten_multimodal_tokens(&multimodal);

    assert_eq!(
        regular_tokens.iter().map(|t| *t as i64).collect::<Vec<_>>(),
        multimodal_tokens,
        "Multimodal should match for conversations with browser/python tools"
    );
}

#[test]
fn test_multimodal_matches_render_conversation_complex() {
    // Complex test with system, developer, function tools, channels, etc.
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let sys = SystemContent::new()
        .with_model_identity("Assistant")
        .with_conversation_start_date("2025-01-15")
        .with_reasoning_effort(ReasoningEffort::Medium);

    let dev = DeveloperContent::new()
        .with_instructions("Be helpful and concise.")
        .with_function_tools(vec![ToolDescription::new(
            "calculator",
            "Performs calculations.",
            Some(json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            })),
        )]);

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::System, sys),
        Message::from_role_and_content(Role::Developer, dev),
        Message::from_role_and_content(Role::User, "What is 123 * 456?"),
        Message::from_role_and_content(Role::Assistant, "I need to calculate this.")
            .with_channel(CanonicalChannel::Analysis),
        Message::from_role_and_content(Role::Assistant, "123 * 456 = 56,088")
            .with_channel(CanonicalChannel::Final),
    ]);

    let regular_tokens = encoding.render_conversation(&convo, None).unwrap();
    let multimodal = encoding
        .render_conversation_multimodal(convo.messages.iter(), None)
        .unwrap();
    let multimodal_tokens = flatten_multimodal_tokens(&multimodal);

    assert_eq!(
        regular_tokens.iter().map(|t| *t as i64).collect::<Vec<_>>(),
        multimodal_tokens,
        "Multimodal should match for complex conversations"
    );
}

// ============================================================================
// Coord-token decoder tests
//
// Coord tokens are a contiguous reserved-id block (group name "coord") that
// semantically represent integers. Their decoded text surface is " K "
// (space-padded integer) instead of the tiktoken literal "<|reserved_K|>", so
// downstream parsers see numbers directly. Encoding is asymmetric: free
// integer text never produces coord ids (only the pointing-render path emits
// them via direct id assignment), so we test decode-only here.
// ============================================================================

#[test]
fn test_coord_group_decodes_to_integer_text() {
    let start = reserved_pool(HarmonyEncodingName::HarmonyGptOss).0 as i64;
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![ReservedTokenGroupConfig {
            name: Some("coord".to_string()),
            tokenizer: None,
            size: 1001,
            offset: Some(start),
        }],
        media_kinds: vec![],
    };
    let encoding =
        load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options).unwrap();
    let first = start as Rank;
    assert_eq!(encoding.tokenizer.decode_utf8([first]).unwrap(), " 0 ");
    assert_eq!(
        encoding.tokenizer.decode_utf8([first + 360]).unwrap(),
        " 360 "
    );
    assert_eq!(
        encoding.tokenizer.decode_utf8([first + 1000]).unwrap(),
        " 1000 "
    );
}

#[test]
fn test_coord_group_stream_concatenates_with_separating_spaces() {
    let start = reserved_pool(HarmonyEncodingName::HarmonyGptOss).0 as i64;
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![ReservedTokenGroupConfig {
            name: Some("coord".to_string()),
            tokenizer: None,
            size: 1001,
            offset: Some(start),
        }],
        media_kinds: vec![],
    };
    let encoding =
        load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options).unwrap();
    let first = start as Rank;
    // Adjacent coord tokens decode into a sequence of numbers separated by
    // double spaces — every coord id contributes its own trailing + leading
    // space, which is enough for SpaceSeparatedCoordinateParser's \b\d+\b
    // extractor to pick them up regardless of surrounding punctuation.
    let decoded = encoding
        .tokenizer
        .decode_utf8([first, first + 360, first + 100, first + 920])
        .unwrap();
    assert_eq!(decoded, " 0  360  100  920 ");
}

#[test]
fn test_coord_group_completion_parses_structured_pointing() {
    let start = reserved_pool(HarmonyEncodingName::HarmonyGptOss).0 as i64;
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![ReservedTokenGroupConfig {
            name: Some("coord".to_string()),
            tokenizer: None,
            size: 1001,
            offset: Some(start),
        }],
        media_kinds: vec![],
    };
    let encoding =
        load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options).unwrap();
    let mut tokens = encoding
        .tokenizer
        .encode_with_special_tokens("<|message|><|point_start|><point_box mention=\"target\">");
    tokens.extend([
        start as Rank + 10,
        start as Rank + 20,
        start as Rank + 30,
        start as Rank + 40,
    ]);
    tokens.extend(
        encoding
            .tokenizer
            .encode_with_special_tokens("</point_box><|point_end|><|return|>"),
    );

    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, Some(Role::Assistant))
        .unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].content.len(), 1);
    let Content::Pointing(pointing) = &parsed[0].content[0] else {
        panic!(
            "expected structured pointing content, got {:?}",
            parsed[0].content[0]
        );
    };
    assert_eq!(pointing.kind, PointingKind::Box);
    assert_eq!(pointing.coords, vec![(10, 20), (30, 40)]);
    assert_eq!(pointing.mention.as_deref(), Some("target"));
}

#[test]
fn test_no_coord_group_preserves_reserved_literal() {
    // Without any reserved groups, reserved ids decode to their default
    // <|reserved_K|> literal surface (this is the pre-coord-fix behaviour and
    // remains the case for any non-coord deployment).
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let rank = reserved_pool(HarmonyEncodingName::HarmonyGptOss).0 as Rank;
    assert_eq!(
        encoding.tokenizer.decode_utf8([rank]).unwrap(),
        format!("<|reserved_{rank}|>"),
    );
}

#[test]
fn test_non_coord_reserved_group_preserves_reserved_literal() {
    // A non-"coord" group MUST NOT trigger numeric decoding: only the coord
    // group has the numeric-text surface semantics. Other groups (e.g. the
    // FAST tokenizer block) keep the <|reserved_K|> literal surface so they
    // round-trip as before.
    let start = reserved_pool(HarmonyEncodingName::HarmonyGptOss).0 as i64;
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![ReservedTokenGroupConfig {
            name: Some("fast_action".to_string()),
            tokenizer: Some("physical-intelligence/fast".to_string()),
            size: 8,
            offset: Some(start),
        }],
        media_kinds: vec![],
    };
    let encoding =
        load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options).unwrap();
    let rank = start as Rank;
    assert_eq!(
        encoding.tokenizer.decode_utf8([rank]).unwrap(),
        format!("<|reserved_{rank}|>"),
    );
}

#[test]
fn test_coord_group_alongside_non_coord_group_only_overrides_coord_range() {
    // Two groups sharing the same encoding: one named "coord" and one named
    // "fast_action". Only ids in the coord range get numeric decoding; the
    // other group's ids keep their reserved literal. This guards against any
    // future regression that would over-apply the override.
    let start = reserved_pool(HarmonyEncodingName::HarmonyGptOss).0 as i64;
    let options = HarmonyEncodingOptions {
        reserved_token_groups: vec![
            ReservedTokenGroupConfig {
                name: Some("coord".to_string()),
                tokenizer: None,
                size: 8,
                offset: Some(start),
            },
            ReservedTokenGroupConfig {
                name: Some("fast_action".to_string()),
                tokenizer: Some("physical-intelligence/fast".to_string()),
                size: 8,
                offset: Some(start + 8),
            },
        ],
        media_kinds: vec![],
    };
    let encoding =
        load_harmony_encoding_with_options(HarmonyEncodingName::HarmonyGptOss, options).unwrap();
    // First coord id decodes to " 0 ".
    assert_eq!(
        encoding.tokenizer.decode_utf8([start as Rank]).unwrap(),
        " 0 "
    );
    // First fast_action id (just past the coord range) keeps the literal.
    let fast_first = (start + 8) as Rank;
    assert_eq!(
        encoding.tokenizer.decode_utf8([fast_first]).unwrap(),
        format!("<|reserved_{fast_first}|>"),
    );
}
