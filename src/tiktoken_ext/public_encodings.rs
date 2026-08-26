// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read as _,
    path::Path,
    sync::OnceLock,
};

#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs::OpenOptions,
    io::{BufReader, BufWriter, Write as _},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use base64::{prelude::BASE64_STANDARD, Engine as _};
use serde::de::{self, MapAccess, Visitor};

use crate::{
    tiktoken::{CoreBPE, Rank},
    token_tables::{encoding_table, EncodingTokenTable},
};
use sha2::{Digest as _, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("the env var TIKTOKEN_ENCODINGS_BASE is not set, or invalid")]
    InvalidEncodingBaseDirEnvVar,

    #[error("unknown encoding name: {0}")]
    UnknownEncodingName(String),

    #[error("invalid tiktoken vocab file: {0}")]
    InvalidTiktokenVocabFile(#[source] std::io::Error),

    #[error("{0} must contain exactly 64 hexadecimal SHA-256 characters")]
    InvalidVocabDigest(String),

    #[error("vocabulary SHA-256 mismatch (expected {expected_hash}, computed {computed_hash})")]
    VocabHashMismatch {
        expected_hash: String,
        computed_hash: String,
    },

    #[error("vocabulary SHA-256 verification is required; set {0}")]
    MissingRequiredVocabDigest(String),

    #[error("{0} must be one of 1, true, yes, on, 0, false, no, or off")]
    InvalidBooleanEnvironmentVariable(String),

    #[error("failed to create CoreBPE: {0}")]
    CoreBPECreationFailed(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("error downloading or loading vocab file: {0}")]
    DownloadOrLoadVocabFile(
        #[source]
        #[from]
        RemoteVocabFileError,
    ),

    #[error("failed to extend encoding")]
    FailedToExtendEncoding(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("Qwen3 vocabulary location is not configured; set QWEN3_VOCAB_PATH, HF_HOME, or HOME")]
    MissingQwen3VocabLocation,
}

#[derive(Debug, thiserror::Error)]
pub enum RemoteVocabFileError {
    #[error("failed to download or load vocab file")]
    FailedToDownloadOrLoadVocabFile(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("an underlying IO error occurred while {0}: {1}")]
    IOError(String, #[source] std::io::Error),

    #[error(
        "remote vocabulary SHA-256 mismatch (expected {expected_hash}, computed {computed_hash})"
    )]
    HashMismatch {
        expected_hash: String,
        computed_hash: String,
    },

    #[error("vocabulary exceeds the {limit_bytes}-byte resource limit")]
    ResourceLimitExceeded { limit_bytes: u64 },

    #[error("remote vocabulary response did not declare a bounded content length")]
    MissingContentLength,

    #[error("unsafe vocabulary cache entry: {0}")]
    UnsafeCacheEntry(String),
}

const TIKTOKEN_ENCODINGS_BASE_VAR: &str = "TIKTOKEN_ENCODINGS_BASE";
const TIKTOKEN_CACHE_DIR_VAR: &str = "TIKTOKEN_RS_CACHE_DIR";
const QWEN3_VOCAB_DIGEST_VAR: &str = "QWEN3_VOCAB_SHA256";
const QWEN35_VOCAB_DIGEST_VAR: &str = "QWEN35_VOCAB_SHA256";
const REQUIRE_QWEN_VOCAB_DIGEST_VAR: &str = "MHARMONY_REQUIRE_QWEN_VOCAB_SHA256";
const QWEN3_OFFICIAL_VOCAB_SHA256: &str =
    "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910";
#[cfg(test)]
const QWEN3_OFFICIAL_VOCAB_BYTES: u64 = 2_776_833;
#[cfg(not(target_arch = "wasm32"))]
const QWEN3_OFFICIAL_REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";
const QWEN35_OFFICIAL_VOCAB_SHA256: &str =
    "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003";
const MAX_VOCAB_BYTES: u64 = 64 * 1024 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
#[cfg(not(target_arch = "wasm32"))]
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_TIKTOKEN_BASE_URL: &str = concat!(
    "https://",
    "open",
    "ai",
    "public",
    ".blob.core.windows.net/encodings/"
);

static TIKTOKEN_BASE_URL_OVERRIDE: OnceLock<String> = OnceLock::new();
#[cfg(not(target_arch = "wasm32"))]
static CACHE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn set_tiktoken_base_url(base_url: impl Into<String>) {
    let mut base = base_url.into();
    if !base.ends_with('/') {
        base.push('/');
    }
    // ignore error if already set
    let _ = TIKTOKEN_BASE_URL_OVERRIDE.set(base);
}

fn tiktoken_base_url() -> &'static str {
    TIKTOKEN_BASE_URL_OVERRIDE
        .get()
        .map(|s| s.as_str())
        .unwrap_or(DEFAULT_TIKTOKEN_BASE_URL)
}

fn invalid_vocab_data(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}

fn normalize_sha256(value: &str, setting_name: &str) -> Result<String, LoadError> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LoadError::InvalidVocabDigest(setting_name.to_string()));
    }
    Ok(value.to_ascii_lowercase())
}

#[cfg(not(target_arch = "wasm32"))]
fn environment_flag(name: &str) -> Result<bool, LoadError> {
    let value = match std::env::var(name) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(LoadError::InvalidBooleanEnvironmentVariable(
                name.to_string(),
            ));
        }
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        _ => Err(LoadError::InvalidBooleanEnvironmentVariable(
            name.to_string(),
        )),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn qwen_vocab_digest_from_env(
    digest_var: &str,
    default_digest: Option<&str>,
) -> Result<Option<String>, LoadError> {
    let require_digest = environment_flag(REQUIRE_QWEN_VOCAB_DIGEST_VAR)?;
    let digest = match std::env::var(digest_var) {
        Ok(value) => Some(normalize_sha256(&value, digest_var)?),
        Err(std::env::VarError::NotPresent) => default_digest.map(str::to_string),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(LoadError::InvalidVocabDigest(digest_var.to_string()));
        }
    };
    if digest.is_none() && require_digest {
        return Err(LoadError::MissingRequiredVocabDigest(
            digest_var.to_string(),
        ));
    }
    Ok(digest)
}

#[cfg(not(target_arch = "wasm32"))]
fn qwen3_vocab_path_from_locations(
    explicit_path: Option<PathBuf>,
    hf_home: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Result<PathBuf, LoadError> {
    let nonempty = |path: PathBuf| (!path.as_os_str().is_empty()).then_some(path);
    if let Some(path) = explicit_path.and_then(nonempty) {
        return Ok(path);
    }

    let snapshot_path = Path::new("hub")
        .join("models--Qwen--Qwen3-0.6B")
        .join("snapshots")
        .join(QWEN3_OFFICIAL_REVISION)
        .join("vocab.json");
    if let Some(path) = hf_home.and_then(nonempty) {
        return Ok(path.join(snapshot_path));
    }
    if let Some(path) = home.and_then(nonempty) {
        return Ok(path.join(".cache/huggingface").join(snapshot_path));
    }

    Err(LoadError::MissingQwen3VocabLocation)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    O200kBase,
    O200kHarmony,
    Cl100kBase,
    Qwen3Harmony,
    Qwen35Harmony,
}

impl Encoding {
    pub fn all() -> &'static [Self] {
        &[
            Self::O200kBase,
            Self::O200kHarmony,
            Self::Cl100kBase,
            Self::Qwen3Harmony,
            Self::Qwen35Harmony,
        ]
    }

    pub fn from_name(name: impl AsRef<str>) -> Option<Self> {
        let name_str = name.as_ref();
        for encoding in Self::all() {
            if encoding.name() == name_str {
                return Some(*encoding);
            }
        }
        None
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_name(name: impl AsRef<str>) -> Result<CoreBPE, LoadError> {
        let name = name.as_ref();
        Self::from_name(name)
            .ok_or_else(|| LoadError::UnknownEncodingName(name.to_string()))?
            .load()
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load_from_name(name: impl AsRef<str>) -> Result<CoreBPE, LoadError> {
        let name = name.as_ref();
        Self::from_name(name)
            .ok_or_else(|| LoadError::UnknownEncodingName(name.to_string()))?
            .load()
            .await
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::O200kBase => "o200k_base",
            Self::O200kHarmony => "o200k_harmony",
            Self::Cl100kBase => "cl100k_base",
            Self::Qwen3Harmony => "qwen3_harmony",
            Self::Qwen35Harmony => "qwen35_harmony",
        }
    }

    /// The pinned digest used automatically for an official local vocabulary.
    /// A caller can pin a different authenticated checkpoint with
    /// `QWEN3_VOCAB_SHA256`, `QWEN35_VOCAB_SHA256`, or
    /// [`Self::load_qwen_with_vocab_sha256`].
    pub fn official_vocab_sha256(&self) -> Option<&'static str> {
        match self {
            Self::Qwen3Harmony => Some(QWEN3_OFFICIAL_VOCAB_SHA256),
            Self::Qwen35Harmony => Some(QWEN35_OFFICIAL_VOCAB_SHA256),
            _ => None,
        }
    }

    fn token_table(&self) -> &'static EncodingTokenTable {
        encoding_table(self.name()).expect("every built-in encoding must have a token table")
    }

    fn special_tokens(&self) -> Vec<(String, Rank)> {
        self.token_table()
            .special_tokens
            .iter()
            .map(|token| (token.text.clone(), token.id))
            .collect()
    }

    fn special_tokens_with_reserved_fill(&self) -> Vec<(String, Rank)> {
        let table = self.token_table();
        let mut specials = self.special_tokens();
        if let Some(fill) = &table.reserved_special_fill {
            let occupied: std::collections::HashSet<Rank> = specials
                .iter()
                .map(|(_, id)| *id)
                .chain(fill.exclude.iter().copied())
                .collect();
            specials.extend(
                (fill.start..=fill.end)
                    .filter(|id| !occupied.contains(id))
                    .map(|id| (format!("<|reserved_{id}|>"), id)),
            );
        }
        specials
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(&self) -> Result<CoreBPE, LoadError> {
        // Qwen3Harmony and Qwen35Harmony use HuggingFace cache format, handle separately.
        if matches!(self, Self::Qwen3Harmony) {
            let expected_hash =
                qwen_vocab_digest_from_env(QWEN3_VOCAB_DIGEST_VAR, self.official_vocab_sha256())?;
            return self.load_qwen3(expected_hash.as_deref());
        }
        if matches!(self, Self::Qwen35Harmony) {
            let expected_hash =
                qwen_vocab_digest_from_env(QWEN35_VOCAB_DIGEST_VAR, self.official_vocab_sha256())?;
            return self.load_qwen35(expected_hash.as_deref());
        }

        #[cfg(not(target_arch = "wasm32"))]
        let (vocab_file_path, check_hash) =
            if let Ok(base_dir) = std::env::var(TIKTOKEN_ENCODINGS_BASE_VAR) {
                (PathBuf::from(base_dir).join(self.vocab_file_name()), true)
            } else {
                let url = self.public_vocab_file_url();
                (
                    download_or_find_cached_file(&url, Some(self.expected_hash()))
                        .map_err(LoadError::DownloadOrLoadVocabFile)?,
                    false,
                )
            };

        match self {
            Self::O200kHarmony => {
                let specials = self.special_tokens_with_reserved_fill();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    load_encoding_from_file(
                        vocab_file_path,
                        check_hash.then(|| self.expected_hash()),
                        specials,
                        &self.pattern(),
                    )
                }
                #[cfg(target_arch = "wasm32")]
                {
                    load_encoding_from_bytes(&vocab_bytes, None, specials, &self.pattern())
                }
            }
            Self::O200kBase => {
                let specials = self.special_tokens_with_reserved_fill();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    load_encoding_from_file(
                        vocab_file_path,
                        check_hash.then(|| self.expected_hash()),
                        specials,
                        &self.pattern(),
                    )
                }
                #[cfg(target_arch = "wasm32")]
                {
                    load_encoding_from_bytes(&vocab_bytes, None, specials, &self.pattern())
                }
            }
            _ => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    load_encoding_from_file(
                        vocab_file_path,
                        check_hash.then(|| self.expected_hash()),
                        self.special_tokens(),
                        &self.pattern(),
                    )
                }
                #[cfg(target_arch = "wasm32")]
                {
                    load_encoding_from_bytes(
                        &vocab_bytes,
                        None,
                        self.special_tokens(),
                        &self.pattern(),
                    )
                }
            }
        }
    }

    /// Load a Qwen encoding while requiring a specific SHA-256 digest for its
    /// checkpoint-local `vocab.json`.
    ///
    /// This is the strongest option for callers that already authenticate a
    /// model checkpoint. [`Self::load`] also verifies a digest when the matching
    /// `QWEN3_VOCAB_SHA256` or `QWEN35_VOCAB_SHA256` environment variable is
    /// set. Both built-in Qwen encodings authenticate their official
    /// vocabulary by default.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_qwen_with_vocab_sha256(&self, expected_sha256: &str) -> Result<CoreBPE, LoadError> {
        let expected_sha256 = normalize_sha256(expected_sha256, "expected_sha256")?;
        match self {
            Self::Qwen3Harmony => self.load_qwen3(Some(&expected_sha256)),
            Self::Qwen35Harmony => self.load_qwen35(Some(&expected_sha256)),
            _ => Err(LoadError::InvalidTiktokenVocabFile(invalid_vocab_data(
                "explicit Qwen vocabulary integrity is only supported for Qwen encodings",
            ))),
        }
    }

    /// Load Qwen3 tokenizer from HuggingFace cache or environment variable.
    #[cfg(not(target_arch = "wasm32"))]
    fn load_qwen3(&self, expected_sha256: Option<&str>) -> Result<CoreBPE, LoadError> {
        let vocab_path = qwen3_vocab_path_from_locations(
            std::env::var_os("QWEN3_VOCAB_PATH").map(PathBuf::from),
            std::env::var_os("HF_HOME").map(PathBuf::from),
            std::env::var_os("HOME").map(PathBuf::from),
        )?;

        if !vocab_path.is_file() {
            return Err(LoadError::InvalidTiktokenVocabFile(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Qwen3 vocab.json not found at {:?}. Please download the model with: \
                     `python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', revision='{QWEN3_OFFICIAL_REVISION}')\"`",
                    vocab_path
                ),
            )));
        }

        // Add reserved tokens for future use (reserved pool)
        let specials = self.special_tokens_with_reserved_fill();

        load_encoding_from_huggingface_verified(
            vocab_path,
            expected_sha256,
            specials,
            &self.pattern(),
        )
    }

    /// Load Qwen3.5 tokenizer from HuggingFace cache or environment variable.
    #[cfg(not(target_arch = "wasm32"))]
    fn load_qwen35(&self, expected_sha256: Option<&str>) -> Result<CoreBPE, LoadError> {
        fn find_vocab_in_hf_cache(cache_root: &Path) -> Option<PathBuf> {
            let repos = ["Qwen3.5-27B", "Qwen3.5-4B", "Qwen3.5-2B", "Qwen3.5-0.8B"];
            for repo in repos {
                let snapshots_dir = cache_root
                    .join("hub")
                    .join(format!("models--Qwen--{repo}"))
                    .join("snapshots");
                let entries = match std::fs::read_dir(&snapshots_dir) {
                    Ok(entries) => entries,
                    Err(_) => continue,
                };
                let mut snapshots: Vec<PathBuf> = entries
                    .filter_map(|entry| entry.ok().map(|e| e.path()))
                    .collect();
                snapshots.sort();
                for snapshot in snapshots.into_iter().rev() {
                    let vocab = snapshot.join("vocab.json");
                    if vocab.is_file() {
                        return Some(vocab);
                    }
                }
            }
            None
        }

        let vocab_path = if let Ok(qwen35_path) = std::env::var("QWEN35_VOCAB_PATH") {
            PathBuf::from(qwen35_path)
        } else if let Ok(hf_home) = std::env::var("HF_HOME") {
            find_vocab_in_hf_cache(Path::new(&hf_home)).unwrap_or_else(|| {
                PathBuf::from(hf_home).join("hub/models--Qwen--Qwen3.5-0.8B/snapshots")
            })
        } else {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            let default_cache_root = PathBuf::from(home).join(".cache/huggingface");
            find_vocab_in_hf_cache(&default_cache_root).unwrap_or_else(|| {
                default_cache_root.join("hub/models--Qwen--Qwen3.5-0.8B/snapshots")
            })
        };

        if !vocab_path.is_file() {
            return Err(LoadError::InvalidTiktokenVocabFile(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Qwen3.5 vocab.json not found at {:?}. Please download the model with: \
                     `python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B')\"`",
                    vocab_path
                ),
            )));
        }

        let specials = self.special_tokens_with_reserved_fill();

        load_encoding_from_huggingface_verified(
            vocab_path,
            expected_sha256,
            specials,
            &self.pattern(),
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load(&self) -> Result<CoreBPE, LoadError> {
        let url = self.public_vocab_file_url();
        let vocab_bytes = download_or_find_cached_file_bytes(&url, Some(self.expected_hash()))
            .await
            .map_err(LoadError::DownloadOrLoadVocabFile)?;

        match self {
            Self::O200kHarmony => {
                let specials = self.special_tokens_with_reserved_fill();
                load_encoding_from_bytes(&vocab_bytes, None, specials, &self.pattern())
            }
            Self::O200kBase => {
                let specials = self.special_tokens_with_reserved_fill();
                load_encoding_from_bytes(&vocab_bytes, None, specials, &self.pattern())
            }
            _ => {
                load_encoding_from_bytes(&vocab_bytes, None, self.special_tokens(), &self.pattern())
            }
        }
    }

    fn public_vocab_file_url(&self) -> String {
        let base = tiktoken_base_url();
        match self {
            Self::O200kBase => format!("{base}o200k_base.tiktoken"),
            Self::O200kHarmony => format!("{base}o200k_base.tiktoken"),
            Self::Cl100kBase => format!("{base}cl100k_base.tiktoken"),
            Self::Qwen3Harmony => String::new(), // Loaded from HuggingFace cache
            Self::Qwen35Harmony => String::new(), // Loaded from HuggingFace cache
        }
    }

    fn vocab_file_name(&self) -> &'static str {
        match self {
            Self::O200kBase => "o200k_base.tiktoken",
            Self::O200kHarmony => "o200k_base.tiktoken",
            Self::Cl100kBase => "cl100k_base.tiktoken",
            Self::Qwen3Harmony => "vocab.json",
            Self::Qwen35Harmony => "vocab.json",
        }
    }

    fn expected_hash(&self) -> &'static str {
        match self {
            Self::O200kBase => "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
            Self::O200kHarmony => {
                "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
            }
            Self::Cl100kBase => "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
            Self::Qwen3Harmony => "",  // No hash check for local HF files
            Self::Qwen35Harmony => "", // No hash check for local HF files
        }
    }

    fn pattern(&self) -> String {
        match self {
            Self::O200kBase => {
                [
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\s+",
                ].join("|")
            }
            Self::O200kHarmony => {
                [
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\s+",
                ].join("|")
            }
            Self::Cl100kBase => {
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_string()
            }
            // Qwen3 uses the same pattern as HuggingFace tokenizers (single digit \p{N})
            Self::Qwen3Harmony | Self::Qwen35Harmony => {
                [
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+",
                    "\\p{N}",  // Single digit, matches HuggingFace exactly
                    " ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\s+",
                ].join("|")
            }
        }
    }
}

fn read_bounded<R>(reader: R, limit_bytes: u64) -> std::io::Result<Vec<u8>>
where
    R: std::io::Read,
{
    let mut bytes = Vec::new();
    reader.take(limit_bytes + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit_bytes {
        return Err(invalid_vocab_data(format!(
            "vocabulary exceeds the {limit_bytes}-byte resource limit"
        )));
    }
    Ok(bytes)
}

struct UniqueHuggingFaceVocab(Vec<(String, Rank)>);

impl<'de> serde::Deserialize<'de> for UniqueHuggingFaceVocab {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct UniqueVocabVisitor;

        impl<'de> Visitor<'de> for UniqueVocabVisitor {
            type Value = UniqueHuggingFaceVocab;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a vocabulary object with unique token keys")
            }

            fn visit_map<M>(self, mut entries: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut vocab = Vec::with_capacity(entries.size_hint().unwrap_or(0));
                let mut tokens = HashSet::new();
                while let Some((token, rank)) = entries.next_entry::<String, Rank>()? {
                    if !tokens.insert(token.clone()) {
                        return Err(de::Error::custom(
                            "vocabulary contains a duplicate token key",
                        ));
                    }
                    vocab.push((token, rank));
                }
                Ok(UniqueHuggingFaceVocab(vocab))
            }
        }

        deserializer.deserialize_map(UniqueVocabVisitor)
    }
}

/// Load a HuggingFace vocab.json file into a BPE encoder map.
fn load_huggingface_vocab<P>(
    path: P,
    expected_sha256: Option<&str>,
) -> Result<HashMap<Vec<u8>, Rank>, LoadError>
where
    P: AsRef<Path>,
{
    // Hash and parse bytes read through the same file descriptor. This avoids a
    // check-then-open race for checkpoint paths (which are commonly symlinks in
    // the Hugging Face cache) while keeping memory use bounded.
    let file = File::open(path).map_err(LoadError::InvalidTiktokenVocabFile)?;
    let bytes = read_bounded(file, MAX_VOCAB_BYTES).map_err(LoadError::InvalidTiktokenVocabFile)?;
    if let Some(expected_sha256) = expected_sha256 {
        let computed_hash = format!("{:x}", Sha256::digest(&bytes));
        if computed_hash != expected_sha256 {
            return Err(LoadError::VocabHashMismatch {
                expected_hash: expected_sha256.to_string(),
                computed_hash,
            });
        }
    }
    let vocab: UniqueHuggingFaceVocab = serde_json::from_slice(&bytes).map_err(|error| {
        LoadError::InvalidTiktokenVocabFile(invalid_vocab_data(error.to_string()))
    })?;

    // Convert string keys to bytes, handling the GPT-2 byte encoding
    // (where bytes 0-255 are mapped to unicode characters)
    let mut encoder = HashMap::default();
    let mut rank_owners: HashMap<Rank, Vec<u8>> = HashMap::default();
    for (token, rank) in vocab.0 {
        let bytes = gpt2_bytes_decode(&token);
        if let Some(previous_rank) = encoder.insert(bytes.clone(), rank) {
            if previous_rank != rank {
                return Err(LoadError::InvalidTiktokenVocabFile(invalid_vocab_data(
                    "multiple vocabulary entries decode to the same byte sequence with different ranks",
                )));
            }
        }
        if let Some(previous_bytes) = rank_owners.insert(rank, bytes.clone()) {
            if previous_bytes != bytes {
                return Err(LoadError::InvalidTiktokenVocabFile(invalid_vocab_data(
                    format!("ordinary vocabulary rank {rank} is assigned to multiple tokens"),
                )));
            }
        }
    }
    Ok(encoder)
}

/// Decode a GPT-2 style encoded string to bytes.
/// GPT-2 tokenizers use a special encoding where bytes are mapped to unicode characters.
///
/// The GPT-2 byte encoder maps:
/// - Bytes 33-126 ('!' to '~') -> themselves
/// - Bytes 161-172 ('¡' to '¬') -> themselves
/// - Bytes 174-255 ('®' to 'ÿ') -> themselves
/// - Non-printable bytes (0-32, 127-160, 173) -> Unicode chars starting at U+0100
fn gpt2_bytes_decode(s: &str) -> Vec<u8> {
    s.chars()
        .map(|c| {
            let code = c as u32;
            match code {
                // Printable ASCII that maps to itself
                33..=126 => code as u8,
                // Latin-1 supplement that maps to itself
                161..=172 => code as u8,
                174..=255 => code as u8,
                // Non-printable bytes 0-32 mapped to U+0100 - U+0120
                0x100..=0x120 => (code - 0x100) as u8,
                // Non-printable bytes 127-160 mapped to U+0121 - U+0142
                0x121..=0x142 => (code - 0x121 + 127) as u8,
                // Non-printable byte 173 mapped to U+0143
                0x143 => 173,
                // Fallback - try to use as-is (for actual unicode chars in vocab)
                _ => code as u8,
            }
        })
        .collect()
}

fn load_tiktoken_vocab<R>(
    mut reader: R,
    expected_hash: Option<&str>,
) -> std::result::Result<HashMap<Vec<u8>, Rank>, std::io::Error>
where
    R: std::io::BufRead,
{
    let mut hasher = expected_hash.map(|_| Sha256::new());
    let mut bpe_ranks = HashMap::new();
    // using readline here so that the line returned includes the newline bytes for the hasher
    let mut lin_no = 0;
    let mut line_buffer = String::new();
    while reader.read_line(&mut line_buffer)? > 0 {
        lin_no += 1;
        if let Some(hasher) = hasher.as_mut() {
            hasher.update(line_buffer.as_bytes());
        }
        let line = line_buffer.trim_end();
        let (token, rank) = line.split_once(' ').ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected token and rank, could not split on ' ' at line {lin_no}"),
            )
        })?;
        let bytes = BASE64_STANDARD.decode(token).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to decode base64 token at line {lin_no}: {e}",),
            )
        })?;
        let rank = rank.parse().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to parse rank at line {lin_no}: {e}"),
            )
        })?;
        bpe_ranks.insert(bytes, rank);
        line_buffer.clear();
    }
    if let Some(hasher) = hasher {
        let expected_hash = expected_hash.unwrap();
        let computed_hash = format!("{:x}", hasher.finalize());
        if computed_hash != expected_hash {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("hash mismatch: computed={computed_hash}, expected={expected_hash}"),
            ));
        }
    }
    Ok(bpe_ranks)
}

pub fn load_tiktoken_vocab_file<P>(
    path: P,
    expected_hash: Option<&str>,
) -> std::result::Result<HashMap<Vec<u8>, Rank>, std::io::Error>
where
    P: AsRef<Path>,
{
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    load_tiktoken_vocab(reader, expected_hash)
}

fn create_core_bpe<S, TS>(
    mut encoder: HashMap<Vec<u8>, Rank>,
    special_tokens: S,
    pattern: &str,
) -> Result<CoreBPE, LoadError>
where
    S: IntoIterator<Item = (TS, Rank)>,
    TS: Into<String>,
{
    let special_tokens: Vec<(String, Rank)> = special_tokens
        .into_iter()
        .map(|(token, rank)| (token.into(), rank))
        .collect();
    let mut special_token_ranks: HashMap<String, Rank> = HashMap::new();
    let mut control_ranks = HashSet::new();

    // Qwen vocabularies include their canonical control-token spellings in
    // vocab.json. Remove an exact spelling/rank match from the ordinary BPE
    // map, then pass it exclusively through CoreBPE's special-token map. Any
    // other overlap is ambiguous and could make decoding depend on hash-map
    // insertion order, so reject it.
    for (token, rank) in &special_tokens {
        if let Some(previous_rank) = special_token_ranks.insert(token.clone(), *rank) {
            if previous_rank != *rank {
                return Err(LoadError::InvalidTiktokenVocabFile(invalid_vocab_data(
                    format!("control token {token:?} is assigned conflicting ranks"),
                )));
            }
        }
        control_ranks.insert(*rank);
        if let Some(ordinary_rank) = encoder.get(token.as_bytes()).copied() {
            if ordinary_rank != *rank {
                return Err(LoadError::InvalidTiktokenVocabFile(invalid_vocab_data(
                    format!(
                        "control token {token:?} uses rank {rank}, but the ordinary vocabulary assigns it rank {ordinary_rank}"
                    ),
                )));
            }
            encoder.remove(token.as_bytes());
        }
    }
    if let Some((_, rank)) = encoder
        .iter()
        .find(|(_, rank)| control_ranks.contains(rank))
    {
        return Err(LoadError::InvalidTiktokenVocabFile(invalid_vocab_data(
            format!("ordinary vocabulary rank {rank} collides with a control-token rank"),
        )));
    }

    CoreBPE::new(encoder, special_tokens, pattern).map_err(LoadError::CoreBPECreationFailed)
}

pub fn load_encoding_from_file<P, S, TS>(
    file_path: P,
    expected_hash: Option<&str>,
    special_tokens: S,
    pattern: &str,
) -> Result<CoreBPE, LoadError>
where
    P: AsRef<Path>,
    S: IntoIterator<Item = (TS, Rank)>,
    TS: Into<String>,
{
    let encoder = load_tiktoken_vocab_file(file_path, expected_hash)
        .map_err(LoadError::InvalidTiktokenVocabFile)?;
    create_core_bpe(encoder, special_tokens, pattern)
}

#[cfg(target_arch = "wasm32")]
fn load_encoding_from_bytes<S, TS>(
    vocab_bytes: &[u8],
    expected_hash: Option<&str>,
    special_tokens: S,
    pattern: &str,
) -> Result<CoreBPE, LoadError>
where
    S: IntoIterator<Item = (TS, Rank)>,
    TS: Into<String>,
{
    let encoder = load_tiktoken_vocab(vocab_bytes, expected_hash)
        .map_err(LoadError::InvalidTiktokenVocabFile)?;
    create_core_bpe(encoder, special_tokens, pattern)
}

/// Load a Hugging Face `vocab.json`, requiring it to match `expected_sha256`
/// when supplied. The hash is calculated over exactly the bytes that are
/// parsed, and the input is capped at [`MAX_VOCAB_BYTES`].
pub fn load_encoding_from_huggingface_verified<P, S, TS>(
    vocab_path: P,
    expected_sha256: Option<&str>,
    special_tokens: S,
    pattern: &str,
) -> Result<CoreBPE, LoadError>
where
    P: AsRef<Path>,
    S: IntoIterator<Item = (TS, Rank)>,
    TS: Into<String>,
{
    let expected_sha256 = expected_sha256
        .map(|digest| normalize_sha256(digest, "expected_sha256"))
        .transpose()?;
    let encoder = load_huggingface_vocab(vocab_path, expected_sha256.as_deref())?;
    create_core_bpe(encoder, special_tokens, pattern)
}

/// This returns the path to a file containing the data at `url`. If the file is
/// cached, it is used. Otherwise, the file is downloaded and cached.
#[cfg(not(target_arch = "wasm32"))]
fn download_or_find_cached_file(
    url: &str,
    expected_hash: Option<&str>,
) -> Result<PathBuf, RemoteVocabFileError> {
    let cache_dir = resolve_cache_dir()?;
    let cache_path = resolve_cache_path(&cache_dir, url);
    if cache_entry_metadata(&cache_path)?.is_some() {
        if verify_file_hash(&cache_path, expected_hash)? {
            return Ok(cache_path);
        }
        remove_regular_cache_file(&cache_path)?;
    }

    // Download into an unpredictable, create-new file in the private cache
    // directory. Only a complete, hash-verified file is atomically published.
    let (temporary_path, mut temporary_file) = create_cache_temp_file(&cache_dir, &cache_path)?;
    let download_result = (|| {
        let computed_hash = load_remote_file(url, &mut temporary_file)?;
        if let Some(expected_hash) = expected_hash {
            if computed_hash != expected_hash {
                return Err(RemoteVocabFileError::HashMismatch {
                    expected_hash: expected_hash.to_string(),
                    computed_hash,
                });
            }
        }
        temporary_file.sync_all().map_err(|error| {
            RemoteVocabFileError::IOError("syncing temporary cache file".to_string(), error)
        })?;
        std::fs::rename(&temporary_path, &cache_path).or_else(|error| {
            // On platforms where rename does not replace an entry, another
            // process may have won the same verified download race.
            if error.kind() == std::io::ErrorKind::AlreadyExists
                && verify_file_hash(&cache_path, expected_hash)?
            {
                Ok(())
            } else {
                Err(RemoteVocabFileError::IOError(
                    "publishing cache file".to_string(),
                    error,
                ))
            }
        })?;
        Ok(cache_path.clone())
    })();
    if download_result.is_err() || temporary_path.exists() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    download_result
}

#[cfg(target_arch = "wasm32")]
async fn download_or_find_cached_file_bytes(
    url: &str,
    expected_hash: Option<&str>,
) -> Result<Vec<u8>, RemoteVocabFileError> {
    let bytes = load_remote_file_bytes(url).await?;
    if let Some(expected_hash) = expected_hash {
        let computed_hash = format!("{:x}", Sha256::digest(&bytes));
        if computed_hash != expected_hash {
            return Err(RemoteVocabFileError::HashMismatch {
                expected_hash: expected_hash.to_string(),
                computed_hash,
            });
        }
    }
    Ok(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_cache_dir() -> Result<PathBuf, RemoteVocabFileError> {
    let cache_dir = if let Some(cache_dir_override) = std::env::var_os(TIKTOKEN_CACHE_DIR_VAR) {
        PathBuf::from(cache_dir_override)
    } else if let Some(xdg_cache_home) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(xdg_cache_home).join("mharmony/tiktoken-rs")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache/mharmony/tiktoken-rs")
    } else {
        return Err(RemoteVocabFileError::IOError(
            "resolving a private cache directory".to_string(),
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "set TIKTOKEN_RS_CACHE_DIR, XDG_CACHE_HOME, or HOME",
            ),
        ));
    };
    create_private_cache_dir(&cache_dir)?;
    Ok(cache_dir)
}

#[cfg(not(target_arch = "wasm32"))]
fn create_private_cache_dir(cache_dir: &Path) -> Result<(), RemoteVocabFileError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder.create(cache_dir).map_err(|error| {
            RemoteVocabFileError::IOError("creating private cache directory".to_string(), error)
        })?;
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(cache_dir).map_err(|error| {
        RemoteVocabFileError::IOError("creating private cache directory".to_string(), error)
    })?;

    let metadata = std::fs::symlink_metadata(cache_dir).map_err(|error| {
        RemoteVocabFileError::IOError("inspecting private cache directory".to_string(), error)
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(RemoteVocabFileError::UnsafeCacheEntry(
            "cache directory must be a real directory, not a symlink".to_string(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.mode() & 0o077 != 0 {
            std::fs::set_permissions(cache_dir, std::fs::Permissions::from_mode(0o700)).map_err(
                |error| {
                    RemoteVocabFileError::IOError(
                        "restricting private cache directory permissions".to_string(),
                        error,
                    )
                },
            )?;
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_cache_path(cache_dir: &Path, url: &str) -> PathBuf {
    let cache_key = format!("{:x}", Sha256::digest(url.as_bytes()));
    cache_dir.join(cache_key)
}

#[cfg(not(target_arch = "wasm32"))]
fn cache_entry_metadata(path: &Path) -> Result<Option<std::fs::Metadata>, RemoteVocabFileError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err(RemoteVocabFileError::UnsafeCacheEntry(
                "cache entry must be a regular file, not a symlink".to_string(),
            ))
        }
        Ok(metadata) if metadata.len() > MAX_VOCAB_BYTES => {
            Err(RemoteVocabFileError::ResourceLimitExceeded {
                limit_bytes: MAX_VOCAB_BYTES,
            })
        }
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(RemoteVocabFileError::IOError(
            "inspecting cache entry".to_string(),
            error,
        )),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn remove_regular_cache_file(path: &Path) -> Result<(), RemoteVocabFileError> {
    if cache_entry_metadata(path)?.is_none() {
        return Ok(());
    }
    std::fs::remove_file(path).map_err(|error| {
        RemoteVocabFileError::IOError("removing invalid cache file".to_string(), error)
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn open_regular_cache_file(path: &Path) -> Result<Option<File>, RemoteVocabFileError> {
    let Some(before_open) = cache_entry_metadata(path)? else {
        return Ok(None);
    };
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(RemoteVocabFileError::IOError(
                "opening cache file".to_string(),
                error,
            ));
        }
    };
    let opened = file.metadata().map_err(|error| {
        RemoteVocabFileError::IOError("inspecting opened cache file".to_string(), error)
    })?;
    let Some(after_open) = cache_entry_metadata(path)? else {
        return Err(RemoteVocabFileError::UnsafeCacheEntry(
            "cache entry changed while it was opened".to_string(),
        ));
    };
    if !opened.is_file() || opened.len() > MAX_VOCAB_BYTES {
        return Err(RemoteVocabFileError::UnsafeCacheEntry(
            "opened cache entry is not a bounded regular file".to_string(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if (before_open.dev(), before_open.ino()) != (opened.dev(), opened.ino())
            || (after_open.dev(), after_open.ino()) != (opened.dev(), opened.ino())
        {
            return Err(RemoteVocabFileError::UnsafeCacheEntry(
                "cache entry changed while it was opened".to_string(),
            ));
        }
    }
    #[cfg(not(unix))]
    if before_open.len() != opened.len() || after_open.len() != opened.len() {
        return Err(RemoteVocabFileError::UnsafeCacheEntry(
            "cache entry changed while it was opened".to_string(),
        ));
    }
    Ok(Some(file))
}

#[cfg(not(target_arch = "wasm32"))]
fn create_cache_temp_file(
    cache_dir: &Path,
    cache_path: &Path,
) -> Result<(PathBuf, File), RemoteVocabFileError> {
    let cache_key = cache_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            RemoteVocabFileError::UnsafeCacheEntry("cache key is not valid UTF-8".to_string())
        })?;
    for _ in 0..128 {
        let counter = CACHE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temporary_path = cache_dir.join(format!(
            ".{cache_key}.{}.{}.{counter}.tmp",
            std::process::id(),
            timestamp
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        match options.open(&temporary_path) {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(RemoteVocabFileError::IOError(
                    "creating temporary cache file".to_string(),
                    error,
                ));
            }
        }
    }
    Err(RemoteVocabFileError::IOError(
        "creating temporary cache file".to_string(),
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique cache file",
        ),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_file_hash(
    file_path: &Path,
    expected_hash: Option<&str>,
) -> Result<bool, RemoteVocabFileError> {
    let Some(file) = open_regular_cache_file(file_path)? else {
        return Ok(false);
    };
    let Some(expected_hash) = expected_hash else {
        return Ok(true);
    };
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut total_bytes = 0_u64;
    let mut buffer = [0_u8; 8192];
    loop {
        let bytes_read = reader.read(&mut buffer).map_err(|error| {
            RemoteVocabFileError::IOError("reading cache file".to_string(), error)
        })?;
        if bytes_read == 0 {
            break;
        }
        total_bytes += bytes_read as u64;
        if total_bytes > MAX_VOCAB_BYTES {
            return Err(RemoteVocabFileError::ResourceLimitExceeded {
                limit_bytes: MAX_VOCAB_BYTES,
            });
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let computed_hash = format!("{:x}", hasher.finalize());
    Ok(computed_hash == expected_hash)
}

/// Loads a remote file into a newly-created cache file and returns its digest.
#[cfg(not(target_arch = "wasm32"))]
fn load_remote_file(url: &str, destination: &mut File) -> Result<String, RemoteVocabFileError> {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .timeout(HTTP_REQUEST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(3))
        .build()
        .map_err(|error| {
            RemoteVocabFileError::FailedToDownloadOrLoadVocabFile(Box::new(error.without_url()))
        })?;
    let mut response = client
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|error| {
            RemoteVocabFileError::FailedToDownloadOrLoadVocabFile(Box::new(error.without_url()))
        })?;
    if response
        .content_length()
        .is_some_and(|length| length > MAX_VOCAB_BYTES)
    {
        return Err(RemoteVocabFileError::ResourceLimitExceeded {
            limit_bytes: MAX_VOCAB_BYTES,
        });
    }

    let mut dest = BufWriter::new(destination);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    let mut total_bytes = 0_u64;
    loop {
        let bytes_read = response.read(&mut buffer).map_err(|error| {
            RemoteVocabFileError::IOError("reading response body".to_string(), error)
        })?;
        if bytes_read == 0 {
            break;
        }
        total_bytes += bytes_read as u64;
        if total_bytes > MAX_VOCAB_BYTES {
            return Err(RemoteVocabFileError::ResourceLimitExceeded {
                limit_bytes: MAX_VOCAB_BYTES,
            });
        }
        dest.write_all(&buffer[..bytes_read]).map_err(|error| {
            RemoteVocabFileError::IOError("writing temporary cache file".to_string(), error)
        })?;
        hasher.update(&buffer[..bytes_read]);
    }
    dest.flush().map_err(|error| {
        RemoteVocabFileError::IOError("flushing temporary cache file".to_string(), error)
    })?;
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(target_arch = "wasm32")]
async fn load_remote_file_bytes(url: &str) -> Result<Vec<u8>, RemoteVocabFileError> {
    use reqwest::Client;

    let client = Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|error| {
            RemoteVocabFileError::FailedToDownloadOrLoadVocabFile(Box::new(error.without_url()))
        })?;
    let content_length = response
        .content_length()
        .ok_or(RemoteVocabFileError::MissingContentLength)?;
    if content_length > MAX_VOCAB_BYTES {
        return Err(RemoteVocabFileError::ResourceLimitExceeded {
            limit_bytes: MAX_VOCAB_BYTES,
        });
    }
    let bytes = response.bytes().await.map_err(|error| {
        RemoteVocabFileError::FailedToDownloadOrLoadVocabFile(Box::new(error.without_url()))
    })?;
    if bytes.len() as u64 > MAX_VOCAB_BYTES {
        return Err(RemoteVocabFileError::ResourceLimitExceeded {
            limit_bytes: MAX_VOCAB_BYTES,
        });
    }
    Ok(bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TemporaryVocab {
        directory: PathBuf,
        path: PathBuf,
    }

    impl TemporaryVocab {
        fn new(contents: &str) -> Self {
            let directory = std::env::temp_dir().join(format!(
                "mharmony-vocab-test-{}-{}",
                std::process::id(),
                CACHE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&directory).unwrap();
            let path = directory.join("vocab.json");
            std::fs::write(&path, contents).unwrap();
            Self { directory, path }
        }
    }

    impl Drop for TemporaryVocab {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    fn load_test_vocab(contents: &str, specials: Vec<(&str, Rank)>) -> Result<CoreBPE, LoadError> {
        let vocab = TemporaryVocab::new(contents);
        load_encoding_from_huggingface_verified(&vocab.path, None, specials, r"\S+")
    }

    #[test]
    fn test_load_bundled_encodings() {
        for encoding in [
            Encoding::O200kBase,
            Encoding::O200kHarmony,
            Encoding::Cl100kBase,
        ] {
            let _ = encoding.load().unwrap();
        }
    }

    #[test]
    fn exact_control_token_entries_are_removed_from_the_ordinary_vocab() {
        let encoding = load_test_vocab(r#"{"a":0,"<|ctrl|>":1}"#, vec![("<|ctrl|>", 1)]).unwrap();
        assert_eq!(encoding.encode_ordinary("a"), vec![0]);
        assert_eq!(encoding.encode_with_special_tokens("<|ctrl|>"), vec![1]);
    }

    #[test]
    fn rejects_ordinary_control_rank_collisions() {
        let error = load_test_vocab(r#"{"a":0,"b":1}"#, vec![("<|ctrl|>", 1)])
            .err()
            .unwrap();
        assert!(error.to_string().contains("collides with a control-token"));
    }

    #[test]
    fn rejects_control_token_spelling_with_a_different_rank() {
        let error = load_test_vocab(r#"{"a":0,"<|ctrl|>":2}"#, vec![("<|ctrl|>", 1)])
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("ordinary vocabulary assigns it rank 2"));
    }

    #[test]
    fn rejects_duplicate_ordinary_ranks() {
        let error = load_test_vocab(r#"{"a":0,"b":0}"#, vec![("<|ctrl|>", 1)])
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("rank 0 is assigned to multiple tokens"));
    }

    #[test]
    fn rejects_duplicate_json_token_keys() {
        let error = load_test_vocab(r#"{"a":0,"a":1}"#, vec![("<|ctrl|>", 2)])
            .err()
            .unwrap();
        assert!(error.to_string().contains("duplicate token key"));
    }

    #[test]
    fn rejects_tokens_that_decode_to_the_same_bytes() {
        let error = load_test_vocab(r#"{"Ā":0,"\u0000":1}"#, vec![("<|ctrl|>", 2)])
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("decode to the same byte sequence"));
    }

    #[test]
    fn verifies_the_exact_bytes_that_are_parsed() {
        let vocab = TemporaryVocab::new(r#"{"a":0}"#);
        let expected = format!("{:x}", Sha256::digest(std::fs::read(&vocab.path).unwrap()));
        load_encoding_from_huggingface_verified(
            &vocab.path,
            Some(&expected),
            [("<|ctrl|>", 1)],
            r"\S+",
        )
        .unwrap();

        let error = load_encoding_from_huggingface_verified(
            &vocab.path,
            Some(&"0".repeat(64)),
            [("<|ctrl|>", 1)],
            r"\S+",
        )
        .err()
        .unwrap();
        assert!(matches!(error, LoadError::VocabHashMismatch { .. }));
    }

    #[test]
    fn validates_digest_syntax_and_bounded_reads() {
        assert!(matches!(
            normalize_sha256("not-a-digest", "test"),
            Err(LoadError::InvalidVocabDigest(_))
        ));
        assert!(read_bounded(std::io::Cursor::new(vec![0_u8; 5]), 4).is_err());
    }

    #[test]
    fn cache_keys_and_errors_do_not_expose_source_urls() {
        let source = "https://user:secret@example.invalid/vocab?token=sensitive";
        let cache_path = resolve_cache_path(Path::new("cache"), source);
        let cache_key = cache_path.file_name().unwrap().to_str().unwrap();
        assert_eq!(cache_key.len(), 64);
        assert!(!cache_path.to_string_lossy().contains("secret"));

        let error = RemoteVocabFileError::HashMismatch {
            expected_hash: "a".repeat(64),
            computed_hash: "b".repeat(64),
        };
        assert!(!error.to_string().contains(source));
        assert!(!error.to_string().contains("sensitive"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_cache_entries() {
        use std::os::unix::fs::symlink;

        let temporary = TemporaryVocab::new(r#"{"a":0}"#);
        let link = temporary.directory.join("cache-link");
        symlink(&temporary.path, &link).unwrap();
        assert!(matches!(
            cache_entry_metadata(&link),
            Err(RemoteVocabFileError::UnsafeCacheEntry(_))
        ));
    }

    #[test]
    fn qwen3_official_vocab_digest_is_pinned() {
        assert_eq!(
            Encoding::Qwen3Harmony.official_vocab_sha256(),
            Some("ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910")
        );
    }

    #[test]
    fn qwen3_vocab_path_requires_an_explicit_location_source() {
        assert!(matches!(
            qwen3_vocab_path_from_locations(None, None, None),
            Err(LoadError::MissingQwen3VocabLocation)
        ));
        assert_eq!(
            qwen3_vocab_path_from_locations(
                Some(PathBuf::from("/checkpoint/vocab.json")),
                Some(PathBuf::from("/hf")),
                Some(PathBuf::from("/home")),
            )
            .unwrap(),
            PathBuf::from("/checkpoint/vocab.json")
        );
        assert_eq!(
            qwen3_vocab_path_from_locations(None, Some(PathBuf::from("/hf")), None).unwrap(),
            PathBuf::from("/hf")
                .join("hub/models--Qwen--Qwen3-0.6B/snapshots")
                .join(QWEN3_OFFICIAL_REVISION)
                .join("vocab.json")
        );
        assert_eq!(
            qwen3_vocab_path_from_locations(None, None, Some(PathBuf::from("/opt/user"))).unwrap(),
            PathBuf::from("/opt/user/.cache/huggingface")
                .join("hub/models--Qwen--Qwen3-0.6B/snapshots")
                .join(QWEN3_OFFICIAL_REVISION)
                .join("vocab.json")
        );
    }

    #[test]
    fn qwen35_official_vocab_digest_is_pinned() {
        assert_eq!(
            Encoding::Qwen35Harmony.official_vocab_sha256(),
            Some("ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003")
        );
    }

    #[test]
    #[ignore = "CI supplies the pinned Qwen3 vocab through QWEN3_VOCAB_PATH"]
    fn test_load_qwen3_pinned_vocab() {
        let path = std::env::var_os("QWEN3_VOCAB_PATH")
            .expect("QWEN3_VOCAB_PATH must point to the pinned vocab.json");
        let metadata = std::fs::metadata(&path).expect("the pinned Qwen3 vocab must be a file");
        assert!(metadata.is_file());
        assert_eq!(metadata.len(), QWEN3_OFFICIAL_VOCAB_BYTES);
        let configured_digest = std::env::var(QWEN3_VOCAB_DIGEST_VAR)
            .unwrap_or_else(|_| QWEN3_OFFICIAL_VOCAB_SHA256.to_string());
        assert_eq!(
            normalize_sha256(&configured_digest, QWEN3_VOCAB_DIGEST_VAR).unwrap(),
            QWEN3_OFFICIAL_VOCAB_SHA256
        );
        let encoding = Encoding::Qwen3Harmony.load().unwrap();
        assert!(!encoding.encode_ordinary("hello, robot").is_empty());
    }

    #[test]
    #[ignore = "CI supplies the pinned Qwen3.5 vocab through QWEN35_VOCAB_PATH"]
    fn test_load_qwen35_pinned_vocab() {
        let path = std::env::var_os("QWEN35_VOCAB_PATH")
            .expect("QWEN35_VOCAB_PATH must point to the pinned vocab.json");
        assert!(Path::new(&path).is_file());
        let configured_digest = std::env::var(QWEN35_VOCAB_DIGEST_VAR)
            .unwrap_or_else(|_| QWEN35_OFFICIAL_VOCAB_SHA256.to_string());
        assert_eq!(
            normalize_sha256(&configured_digest, QWEN35_VOCAB_DIGEST_VAR).unwrap(),
            QWEN35_OFFICIAL_VOCAB_SHA256
        );
        let encoding = Encoding::Qwen35Harmony.load().unwrap();
        assert!(!encoding.encode_ordinary("hello, robot").is_empty());
    }

    #[test]
    #[ignore = "requires Qwen vocab.json files in the local Hugging Face cache"]
    fn test_load_cache_dependent_encodings() {
        for encoding in [Encoding::Qwen3Harmony, Encoding::Qwen35Harmony] {
            let _ = encoding.load().unwrap();
        }
    }
}
