// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
//! Validated token and shipped-media tables bundled as release data.

use std::{collections::HashMap, sync::OnceLock};

use serde::Deserialize;

use crate::tiktoken::Rank;

const TOKEN_TABLES_JSON: &str = include_str!("../data/token_tables.json");

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct SpecialTokenEntry {
    pub(crate) text: String,
    pub(crate) id: Rank,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ReservedPool {
    pub(crate) start: Rank,
    pub(crate) size: Rank,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ReservedSpecialFill {
    pub(crate) start: Rank,
    pub(crate) end: Rank,
    #[serde(default)]
    pub(crate) exclude: Vec<Rank>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ShippedMediaKind {
    pub(crate) kind: String,
    pub(crate) formatting_token: String,
    pub(crate) token_id: Option<Rank>,
    pub(crate) modality: String,
    #[serde(default)]
    pub(crate) empty_payload: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct EncodingTokenTable {
    #[serde(default)]
    pub(crate) special_tokens: Vec<SpecialTokenEntry>,
    #[serde(default)]
    pub(crate) formatting_tokens: HashMap<String, String>,
    #[serde(default)]
    pub(crate) media_kinds: Vec<ShippedMediaKind>,
    pub(crate) base_vocab_size: Option<Rank>,
    pub(crate) reserved_group_pool: Option<ReservedPool>,
    pub(crate) reserved_special_fill: Option<ReservedSpecialFill>,
}

#[derive(Debug, Deserialize)]
struct TokenTablesFile {
    schema_version: u32,
    encodings: HashMap<String, EncodingTokenTable>,
}

fn parse_tables() -> TokenTablesFile {
    let tables: TokenTablesFile =
        serde_json::from_str(TOKEN_TABLES_JSON).expect("bundled token tables must be valid JSON");
    assert_eq!(tables.schema_version, 1, "unsupported token-table schema");
    for (name, table) in &tables.encodings {
        validate_table(name, table)
            .unwrap_or_else(|error| panic!("invalid {name} token table: {error}"));
    }
    tables
}

fn validate_table(name: &str, table: &EncodingTokenTable) -> anyhow::Result<()> {
    let mut token_texts = std::collections::HashSet::new();
    for token in &table.special_tokens {
        anyhow::ensure!(!token.text.is_empty(), "special token text is empty");
        anyhow::ensure!(
            token_texts.insert(token.text.as_str()),
            "duplicate special token {:?}",
            token.text
        );
    }
    for (symbol, token) in &table.formatting_tokens {
        anyhow::ensure!(
            !symbol.is_empty() && !token.is_empty(),
            "empty formatting-token mapping"
        );
    }
    let mut media_kinds = std::collections::HashSet::new();
    for media in &table.media_kinds {
        anyhow::ensure!(!media.kind.trim().is_empty(), "media kind is empty");
        anyhow::ensure!(
            media_kinds.insert(media.kind.as_str()),
            "duplicate media kind {:?}",
            media.kind
        );
        anyhow::ensure!(!media.modality.trim().is_empty(), "media modality is empty");
        anyhow::ensure!(
            table
                .formatting_tokens
                .contains_key(&media.formatting_token),
            "media kind {:?} uses unknown formatting token {:?}",
            media.kind,
            media.formatting_token
        );
    }
    if let Some(pool) = &table.reserved_group_pool {
        anyhow::ensure!(pool.size > 0, "reserved group pool is empty");
        pool.start
            .checked_add(pool.size - 1)
            .ok_or_else(|| anyhow::anyhow!("reserved group pool overflows"))?;
    }
    if let Some(fill) = &table.reserved_special_fill {
        anyhow::ensure!(
            fill.start <= fill.end,
            "reserved special fill range is reversed"
        );
    }
    anyhow::ensure!(!name.trim().is_empty(), "encoding table name is empty");
    Ok(())
}

pub(crate) fn encoding_table(name: &str) -> anyhow::Result<&'static EncodingTokenTable> {
    static TABLES: OnceLock<TokenTablesFile> = OnceLock::new();
    TABLES
        .get_or_init(parse_tables)
        .encodings
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("no shipped token table for encoding {name:?}"))
}

pub fn shipped_token_tables_json() -> &'static str {
    TOKEN_TABLES_JSON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_tables_validate_and_contain_only_shipped_media_kinds() {
        let tables = parse_tables();
        assert_eq!(tables.encodings.len(), 5);
        let media_kinds = tables
            .encodings
            .values()
            .flat_map(|table| table.media_kinds.iter())
            .map(|media| media.kind.as_str())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            media_kinds,
            std::collections::HashSet::from(["image", "video"])
        );
    }
}
