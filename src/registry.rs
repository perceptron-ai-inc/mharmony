// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    encoding::{
        FormattingToken, HarmonyEncoding, RegisteredMediaKind, RenderFormattingTokenError,
        ReservedTokenGroup,
    },
    tiktoken::{CoreBPE, Rank},
    tiktoken_ext,
    token_tables::encoding_table,
};

/// Name of the reserved-token group whose ids decode to ` K ` (the textual
/// integer) instead of the tiktoken literal `<|reserved_K|>`.
const COORD_GROUP_NAME: &str = "coord";

/// Build the override map that gives each coord-group id `coord_offset + K`
/// a space-padded textual integer (` K `) as its decoded surface text. The
/// space padding ensures consecutive coord tokens stay separable when
/// concatenated (e.g. ` 0  360  100  920 ` for four coord tokens in a row).
///
/// Returns an empty map when no `coord` group is registered, in which case
/// callers should skip the override step entirely. This guarantees zero
/// behaviour change for deployments without coord tokens.
fn coord_decoder_overrides(groups: &[ReservedTokenGroup]) -> HashMap<Rank, Vec<u8>> {
    let mut overrides: HashMap<Rank, Vec<u8>> = HashMap::new();
    for group in groups {
        if group.name.as_deref() != Some(COORD_GROUP_NAME) {
            continue;
        }
        for i in 0..group.size {
            let rank = (group.offset + i as i64) as Rank;
            overrides.insert(rank, format!(" {i} ").into_bytes());
        }
    }
    overrides
}

/// Apply coord-token decoder overrides to a freshly-loaded `CoreBPE` so that
/// downstream `decode_*` calls emit numeric text instead of `<|reserved_K|>`
/// literals for coord-group ids. No-op when `groups` contains no `coord`
/// group.
fn apply_coord_decoder_overrides(tokenizer: &mut CoreBPE, groups: &[ReservedTokenGroup]) {
    let overrides = coord_decoder_overrides(groups);
    if !overrides.is_empty() {
        tokenizer.apply_decoder_overrides(overrides);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum HarmonyEncodingName {
    HarmonyGptOss,
    Qwen3Harmony,
    Qwen35Harmony,
}

impl std::fmt::Display for HarmonyEncodingName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                HarmonyEncodingName::HarmonyGptOss => "HarmonyGptOss",
                HarmonyEncodingName::Qwen3Harmony => "Qwen3Harmony",
                HarmonyEncodingName::Qwen35Harmony => "Qwen35Harmony",
            }
        )
    }
}

impl std::str::FromStr for HarmonyEncodingName {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "HarmonyGptOss" | "HARMONY_GPT_OSS" => Ok(HarmonyEncodingName::HarmonyGptOss),
            "Qwen3Harmony" | "QWEN3_HARMONY" => Ok(HarmonyEncodingName::Qwen3Harmony),
            "Qwen35Harmony" | "QWEN35_HARMONY" => Ok(HarmonyEncodingName::Qwen35Harmony),
            _ => anyhow::bail!("Invalid HarmonyEncodingName: {}", s),
        }
    }
}

impl std::fmt::Debug for HarmonyEncodingName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct HarmonyEncodingOptions {
    #[serde(default)]
    pub reserved_token_groups: Vec<ReservedTokenGroupConfig>,
    #[serde(default)]
    pub media_kinds: Vec<MediaKindRegistrationConfig>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ReservedTokenGroupConfig {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tokenizer: Option<String>,
    pub size: usize,
    #[serde(default)]
    pub offset: Option<i64>,
}

fn empty_json_object() -> serde_json::Value {
    serde_json::json!({})
}

/// Runtime registration for a private or application-defined media kind.
///
/// The formatting token must already exist as one special token in the chosen
/// tokenizer. Processor and validator callbacks stay host-side.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct MediaKindRegistrationConfig {
    pub kind: String,
    pub formatting_token: String,
    pub modality: String,
    #[serde(default = "empty_json_object")]
    pub empty_payload: serde_json::Value,
}

/// (base_vocab_size, reserved_pool_start, reserved_pool_size) for encodings
/// with a tabulated coord vocab layout; None otherwise.
pub fn encoding_vocab_info(name: HarmonyEncodingName) -> Option<(u32, u32, u32)> {
    let table = encoding_table(encoding_spec(name).encoding.name()).ok()?;
    Some((
        table.base_vocab_size?,
        table.reserved_group_pool.as_ref()?.start,
        table.reserved_group_pool.as_ref()?.size,
    ))
}

/// Return the inclusive reserved-token pool bounds for an encoding.
pub fn reserved_token_group_pool(name: HarmonyEncodingName) -> anyhow::Result<(u32, u32)> {
    let table = encoding_table(encoding_spec(name).encoding.name())?;
    let pool = table
        .reserved_group_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("encoding {name} has no reserved token-group pool"))?;
    let end = pool
        .start
        .checked_add(pool.size - 1)
        .ok_or_else(|| anyhow::anyhow!("encoding {name} reserved pool overflows"))?;
    Ok((pool.start, end))
}

pub(crate) fn resolve_reserved_token_groups(
    name: HarmonyEncodingName,
    configs: Vec<ReservedTokenGroupConfig>,
) -> anyhow::Result<Vec<ReservedTokenGroup>> {
    if configs.is_empty() {
        return Ok(Vec::new());
    }

    let (pool_start, pool_end) = reserved_token_group_pool(name)?;
    let pool_start = i64::from(pool_start);
    let pool_end = i64::from(pool_end);

    let mut groups: Vec<ReservedTokenGroup> = Vec::new();
    let mut used_ranges: Vec<(i64, i64)> = Vec::new();
    let mut cursor = pool_start;

    for (idx, cfg) in configs.into_iter().enumerate() {
        if cfg.name.is_none() && cfg.tokenizer.is_none() {
            anyhow::bail!("reserved_token_groups[{idx}] must specify name or tokenizer");
        }
        if cfg.size == 0 {
            anyhow::bail!("reserved_token_groups[{idx}] size must be > 0");
        }
        let size = i64::try_from(cfg.size)
            .map_err(|_| anyhow::anyhow!("reserved_token_groups[{idx}] size is too large"))?;
        let offset = cfg.offset.unwrap_or(cursor);
        let end = offset.checked_add(size - 1).ok_or_else(|| {
            anyhow::anyhow!("reserved_token_groups[{idx}] offset range overflows")
        })?;
        if cfg.offset.is_none() {
            if end > pool_end {
                anyhow::bail!(
                    "reserved_token_groups[{idx}] size {size} exceeds reserved pool ({pool_start}..={pool_end})"
                );
            }
            cursor = end + 1;
        }
        if offset < pool_start || end > pool_end {
            anyhow::bail!(
                "reserved_token_groups[{idx}] offset range {offset}..={end} outside reserved pool ({pool_start}..={pool_end})"
            );
        }
        for (used_start, used_end) in &used_ranges {
            if offset <= *used_end && end >= *used_start {
                anyhow::bail!("reserved_token_groups[{idx}] overlaps reserved range {used_start}..={used_end}");
            }
        }
        used_ranges.push((offset, end));

        groups.push(ReservedTokenGroup {
            name: cfg.name,
            tokenizer: cfg.tokenizer,
            size: cfg.size,
            offset,
        });
    }

    Ok(groups)
}

#[derive(Clone, Copy)]
struct EncodingSpec {
    encoding: tiktoken_ext::Encoding,
    n_ctx: usize,
    max_action_length: usize,
    use_nfc_normalization: bool,
}

fn encoding_spec(name: HarmonyEncodingName) -> EncodingSpec {
    match name {
        HarmonyEncodingName::HarmonyGptOss => EncodingSpec {
            encoding: tiktoken_ext::Encoding::O200kHarmony,
            n_ctx: 1_048_576,
            max_action_length: 524_288,
            use_nfc_normalization: false,
        },
        HarmonyEncodingName::Qwen3Harmony => EncodingSpec {
            encoding: tiktoken_ext::Encoding::Qwen3Harmony,
            n_ctx: 131_072,
            max_action_length: 65_536,
            use_nfc_normalization: true,
        },
        HarmonyEncodingName::Qwen35Harmony => EncodingSpec {
            encoding: tiktoken_ext::Encoding::Qwen35Harmony,
            n_ctx: 131_072,
            max_action_length: 65_536,
            use_nfc_normalization: true,
        },
    }
}

fn formatting_token_mapping(
    name: HarmonyEncodingName,
) -> anyhow::Result<HashMap<FormattingToken, String>> {
    let spec = encoding_spec(name);
    encoding_table(spec.encoding.name())?
        .formatting_tokens
        .iter()
        .map(|(symbol, token)| {
            let formatting_token = FormattingToken::from_symbol(symbol).ok_or_else(|| {
                anyhow::anyhow!("encoding {name} has unknown formatting-token symbol {symbol:?}")
            })?;
            Ok((formatting_token, token.clone()))
        })
        .collect()
}

fn resolve_formatting_token_ids(
    tokenizer: &CoreBPE,
    formatting_tokens: &HashMap<FormattingToken, String>,
) -> Result<HashMap<FormattingToken, Rank>, RenderFormattingTokenError> {
    formatting_tokens
        .iter()
        .map(|(&token, text)| {
            let rank = match tokenizer.special_token_id(text) {
                Some(rank) => rank,
                None => {
                    // Some legacy mappings intentionally use an ordinary-token
                    // spelling. Preserve their first-token behavior, but do the
                    // general-purpose tokenization once while constructing the
                    // encoding instead of on every rendered marker.
                    let encoding = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        tokenizer.encode_with_special_tokens(text)
                    }))
                    .map_err(|_| RenderFormattingTokenError::TokenizerFailure { token })?;
                    match encoding.first().copied() {
                        Some(rank) => rank,
                        None => {
                            return Err(RenderFormattingTokenError::InvalidEncoding {
                                token,
                                encoding,
                            });
                        }
                    }
                }
            };
            Ok((token, rank))
        })
        .collect()
}

fn resolve_special_token(tokenizer: &CoreBPE, token: &str) -> anyhow::Result<Rank> {
    tokenizer.special_token_id(token).ok_or_else(|| {
        anyhow::anyhow!("registered formatting token {token:?} is not a tokenizer special token")
    })
}

fn media_kind_registrations(
    name: HarmonyEncodingName,
    tokenizer: &CoreBPE,
    formatting_tokens: &HashMap<FormattingToken, String>,
    additional: Vec<MediaKindRegistrationConfig>,
) -> anyhow::Result<HashMap<String, RegisteredMediaKind>> {
    let table = encoding_table(encoding_spec(name).encoding.name())?;
    let mut registrations = HashMap::new();
    for media in &table.media_kinds {
        let symbol = FormattingToken::from_symbol(&media.formatting_token).ok_or_else(|| {
            anyhow::anyhow!(
                "encoding {name} media kind {:?} has unknown formatting-token symbol {:?}",
                media.kind,
                media.formatting_token
            )
        })?;
        let formatting_token = formatting_tokens
            .get(&symbol)
            .ok_or_else(|| anyhow::anyhow!("encoding {name} does not map {symbol}"))?
            .clone();
        let token_id = match media.token_id {
            Some(token_id) => token_id,
            None => resolve_special_token(tokenizer, &formatting_token)?,
        };
        registrations.insert(
            media.kind.clone(),
            RegisteredMediaKind {
                token_id,
                modality: media.modality.clone(),
                empty_payload: media.empty_payload.clone(),
            },
        );
    }
    for media in additional {
        anyhow::ensure!(
            !media.kind.is_empty() && media.kind.trim() == media.kind,
            "media kind must be non-empty and contain no surrounding whitespace"
        );
        anyhow::ensure!(
            !media.modality.is_empty() && media.modality.trim() == media.modality,
            "media modality must be non-empty and contain no surrounding whitespace"
        );
        anyhow::ensure!(
            !media.formatting_token.is_empty()
                && media.formatting_token.trim() == media.formatting_token,
            "media formatting token must be non-empty and contain no surrounding whitespace"
        );
        anyhow::ensure!(
            !registrations.contains_key(&media.kind),
            "media kind {:?} is already registered",
            media.kind
        );
        registrations.insert(
            media.kind,
            RegisteredMediaKind {
                token_id: resolve_special_token(tokenizer, &media.formatting_token)?,
                modality: media.modality,
                empty_payload: media.empty_payload,
            },
        );
    }
    Ok(registrations)
}

fn finish_encoding(
    name: HarmonyEncodingName,
    reserved_token_groups: Vec<ReservedTokenGroup>,
    media_kinds: Vec<MediaKindRegistrationConfig>,
    mut tokenizer: CoreBPE,
) -> anyhow::Result<HarmonyEncoding> {
    let spec = encoding_spec(name);
    apply_coord_decoder_overrides(&mut tokenizer, &reserved_token_groups);
    let format_token_mapping = formatting_token_mapping(name)?;
    let format_token_ids = resolve_formatting_token_ids(&tokenizer, &format_token_mapping)?;
    let media_kinds =
        media_kind_registrations(name, &tokenizer, &format_token_mapping, media_kinds)?;
    Ok(HarmonyEncoding {
        name: name.to_string(),
        n_ctx: spec.n_ctx,
        tokenizer: Arc::new(tokenizer),
        tokenizer_name: spec.encoding.name().to_owned(),
        max_message_tokens: spec.n_ctx - spec.max_action_length,
        max_action_length: spec.max_action_length,
        format_token_mapping,
        format_token_ids,
        media_kinds,
        stop_formatting_tokens: HashSet::from([
            FormattingToken::EndMessageDoneSampling,
            FormattingToken::EndMessageAssistantToTool,
            FormattingToken::EndMessage,
        ]),
        stop_formatting_tokens_for_assistant_actions: HashSet::from([
            FormattingToken::EndMessageDoneSampling,
            FormattingToken::EndMessageAssistantToTool,
        ]),
        use_nfc_normalization: spec.use_nfc_normalization,
        reserved_token_groups,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_harmony_encoding(name: HarmonyEncodingName) -> anyhow::Result<HarmonyEncoding> {
    load_harmony_encoding_with_options(name, HarmonyEncodingOptions::default())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_harmony_encoding_with_options(
    name: HarmonyEncodingName,
    options: HarmonyEncodingOptions,
) -> anyhow::Result<HarmonyEncoding> {
    let reserved_token_groups = resolve_reserved_token_groups(name, options.reserved_token_groups)?;
    let tokenizer = encoding_spec(name).encoding.load()?;
    finish_encoding(name, reserved_token_groups, options.media_kinds, tokenizer)
}

#[cfg(target_arch = "wasm32")]
pub async fn load_harmony_encoding(name: HarmonyEncodingName) -> anyhow::Result<HarmonyEncoding> {
    load_harmony_encoding_with_options(name, HarmonyEncodingOptions::default()).await
}

#[cfg(target_arch = "wasm32")]
pub async fn load_harmony_encoding_with_options(
    name: HarmonyEncodingName,
    options: HarmonyEncodingOptions,
) -> anyhow::Result<HarmonyEncoding> {
    let reserved_token_groups = resolve_reserved_token_groups(name, options.reserved_token_groups)?;
    if name != HarmonyEncodingName::HarmonyGptOss {
        anyhow::bail!(
            "{name} is not available in wasm builds because its vocabulary is not bundled"
        );
    }
    let tokenizer = encoding_spec(name).encoding.load().await?;
    finish_encoding(name, reserved_token_groups, options.media_kinds, tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_with_one_special() -> CoreBPE {
        CoreBPE::new(
            [(b"x".to_vec(), 1)],
            [("<|special|>".to_string(), 100)],
            r"(?s).",
        )
        .unwrap()
    }

    #[test]
    fn pre_resolved_formatting_tokens_match_previous_encoding_ids() {
        let tokenizer = tokenizer_with_one_special();
        let mappings = HashMap::from([
            (FormattingToken::Start, "<|special|>".to_string()),
            // Legacy tables may intentionally map a formatting symbol to an
            // ordinary spelling. It must retain the old first-token result.
            (FormattingToken::Message, "x".to_string()),
        ]);

        let resolved = resolve_formatting_token_ids(&tokenizer, &mappings).unwrap();
        for (formatting_token, text) in &mappings {
            let previous = tokenizer.encode_with_special_tokens(text);
            assert_eq!(resolved[formatting_token], previous[0]);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn all_shipped_formatting_tokens_keep_their_previous_ids() {
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        assert_eq!(
            encoding.format_token_ids.len(),
            encoding.format_token_mapping.len()
        );
        for (&formatting_token, text) in &encoding.format_token_mapping {
            let previous = encoding.tokenizer.encode_with_special_tokens(text);
            assert_eq!(
                encoding.render_formatting_token(formatting_token).unwrap(),
                previous[0],
                "formatting token {formatting_token} changed id"
            );
        }
    }

    #[test]
    fn invalid_formatting_mapping_fails_during_resolution() {
        let tokenizer = tokenizer_with_one_special();
        let mappings = HashMap::from([(FormattingToken::Start, String::new())]);

        let error = resolve_formatting_token_ids(&tokenizer, &mappings).unwrap_err();
        assert!(matches!(
            error,
            RenderFormattingTokenError::InvalidEncoding {
                token: FormattingToken::Start,
                ref encoding,
            } if encoding.is_empty()
        ));
    }

    #[test]
    fn registered_special_tokens_use_exact_lookup_and_error() {
        let tokenizer = tokenizer_with_one_special();
        assert_eq!(
            resolve_special_token(&tokenizer, "<|special|>").unwrap(),
            100
        );
        assert_eq!(
            resolve_special_token(&tokenizer, "x")
                .unwrap_err()
                .to_string(),
            "registered formatting token \"x\" is not a tokenizer special token"
        );
    }
}
