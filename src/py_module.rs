// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
//! Python bindings for the harmony crate.
//!
//! The bindings are kept intentionally small: we expose the `HarmonyEncoding` type
//! together with the operations that are required by the original Rust test
//! suite (rendering a conversation for completion, parsing messages from
//! completion tokens and decoding tokens back into UTF-8). All higher-level
//! data-structures (Conversation, Message, SystemContent, DeveloperContent, …) are passed across the FFI
//! boundary as JSON.  This allows us to keep the Rust ↔ Python interface very
//! light-weight while still re-using the exact same logic that is implemented
//! in Rust.
//!
//! A thin, typed, user-facing Python wrapper around these low-level bindings is
//! provided in `harmony/__init__.py`.

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

// We need the `Python` type later on.
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::Python;

use pyo3::types::{PyAny, PyBool, PyDict, PyList, PyModule, PyTuple};
use pyo3::Bound;
use serde_json::Value as JsonValue;

// Define a custom Python exception so users can catch Harmony specific errors.
create_exception!(mharmony, HarmonyError, PyRuntimeError);

use crate::{
    chat::{Message, Role, ToolNamespaceConfig},
    encoding::{
        CompletionStreamOptions, CompletionStreamParser, CompletionStreamUpdate, HarmonyEncoding,
        HeaderRenderOptions, MessageHeader, ParseOptions, StreamableParser,
    },
    load_harmony_encoding_with_options, HarmonyEncodingName, HarmonyEncodingOptions,
    MediaKindRegistrationConfig, ReservedTokenGroupConfig,
};

fn json_value_to_py(py: Python<'_>, value: &JsonValue) -> PyResult<PyObject> {
    match value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => (*b).into_py_any(py),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py_any(py)
            } else if let Some(u) = n.as_u64() {
                u.into_py_any(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py_any(py)
            } else {
                Ok(py.None())
            }
        }
        JsonValue::String(s) => s.into_py_any(py),
        JsonValue::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_value_to_py(py, item)?)?;
            }
            list.into_py_any(py)
        }
        JsonValue::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_value_to_py(py, v)?)?;
            }
            dict.into_py_any(py)
        }
    }
}

fn parse_reserved_token_groups(
    value: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<ReservedTokenGroupConfig>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_none() {
        return Ok(Vec::new());
    }
    let list = value.downcast::<PyList>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "reserved_token_groups must be a list of dicts",
        )
    })?;

    let mut configs: Vec<ReservedTokenGroupConfig> = Vec::with_capacity(list.len());
    for (idx, item) in list.iter().enumerate() {
        let dict = item.downcast::<PyDict>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "reserved_token_groups[{idx}] must be a dict"
            ))
        })?;

        let optional_string = |field: &str| -> PyResult<Option<String>> {
            let Some(value) = dict.get_item(field)? else {
                return Ok(None);
            };
            if value.is_none() {
                return Ok(None);
            }
            value.extract::<String>().map(Some).map_err(|_| {
                py_value_error(format!(
                    "reserved_token_groups[{idx}].{field} must be a string or None"
                ))
            })
        };
        let name = optional_string("name")?;
        let tokenizer = optional_string("tokenizer")?;

        let size_value = dict.get_item("size")?.ok_or_else(|| {
            py_value_error(format!(
                "reserved_token_groups[{idx}] missing required 'size'"
            ))
        })?;
        let size = if size_value.is_instance_of::<PyBool>() {
            Err(py_value_error(format!(
                "reserved_token_groups[{idx}].size must be a non-negative integer"
            )))
        } else {
            size_value.extract::<usize>().map_err(|_| {
                py_value_error(format!(
                    "reserved_token_groups[{idx}].size must be a non-negative integer"
                ))
            })
        }?;

        let offset = match dict.get_item("offset")? {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) if value.is_instance_of::<PyBool>() => {
                return Err(py_value_error(format!(
                    "reserved_token_groups[{idx}].offset must be an integer or None"
                )));
            }
            Some(value) => Some(value.extract::<i64>().map_err(|_| {
                py_value_error(format!(
                    "reserved_token_groups[{idx}].offset must be an integer or None"
                ))
            })?),
        };

        configs.push(ReservedTokenGroupConfig {
            name,
            tokenizer,
            size,
            offset,
        });
    }
    Ok(configs)
}

fn py_json_value(value: &Bound<'_, PyAny>, context: &str) -> PyResult<JsonValue> {
    let json = PyModule::import(value.py(), "json")?;
    let serialized: String = json
        .call_method1("dumps", (value,))
        .and_then(|encoded| encoded.extract())
        .map_err(|error| py_value_error(format!("{context} must be JSON serializable: {error}")))?;
    serde_json::from_str(&serialized)
        .map_err(|error| py_value_error(format!("{context} must be valid JSON: {error}")))
}

fn parse_media_kind_registrations(
    value: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<MediaKindRegistrationConfig>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_none() {
        return Ok(Vec::new());
    }
    let list = value
        .downcast::<PyList>()
        .map_err(|_| py_value_error("media_kinds must be a list of dicts"))?;

    list.iter()
        .enumerate()
        .map(|(index, item)| {
            let dict = item
                .downcast::<PyDict>()
                .map_err(|_| py_value_error(format!("media_kinds[{index}] must be a dict")))?;
            let required_string = |name: &str| -> PyResult<String> {
                dict.get_item(name)?
                    .ok_or_else(|| {
                        py_value_error(format!("media_kinds[{index}] missing required {name:?}"))
                    })?
                    .extract::<String>()
                    .map_err(|_| {
                        py_value_error(format!("media_kinds[{index}].{name} must be a string"))
                    })
            };
            let empty_payload = dict
                .get_item("empty_payload")?
                .map(|payload| {
                    py_json_value(&payload, &format!("media_kinds[{index}].empty_payload"))
                })
                .transpose()?
                .unwrap_or_else(|| serde_json::json!({}));

            Ok(MediaKindRegistrationConfig {
                kind: required_string("kind")?,
                formatting_token: required_string("formatting_token")?,
                modality: required_string("modality")?,
                empty_payload,
            })
        })
        .collect()
}

fn py_value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

const DEFAULT_MAX_INLINE_MEDIA_BYTES: usize = 64 * 1024 * 1024;

fn config_bool(
    values: &Bound<'_, PyDict>,
    name: &str,
    default: bool,
    context: &str,
) -> PyResult<bool> {
    let Some(value) = values.get_item(name)? else {
        return Ok(default);
    };
    if !value.is_instance_of::<PyBool>() {
        return Err(py_value_error(format!(
            "{context} {name:?} must be a boolean"
        )));
    }
    value
        .extract::<bool>()
        .map_err(|_| py_value_error(format!("{context} {name:?} must be a boolean")))
}

fn parse_render_config(
    values: Option<&Bound<'_, PyDict>>,
) -> PyResult<crate::encoding::RenderConversationConfig> {
    Ok(crate::encoding::RenderConversationConfig {
        auto_drop_analysis: match values {
            Some(values) => config_bool(values, "auto_drop_analysis", true, "render config")?,
            None => true,
        },
    })
}

fn preprocess_u32(value: &Bound<'_, PyAny>, name: &str) -> PyResult<u32> {
    if value.is_instance_of::<PyBool>() {
        return Err(py_value_error(format!(
            "preprocess config {name:?} must be an integer"
        )));
    }
    value
        .extract::<u32>()
        .map_err(|_| py_value_error(format!("preprocess config {name:?} must be an integer")))
}

fn preprocess_optional_u32(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<u32>> {
    if value.is_none() {
        Ok(None)
    } else {
        preprocess_u32(value, name).map(Some)
    }
}

fn parse_preprocess_config(
    values: Option<&Bound<'_, PyDict>>,
) -> PyResult<crate::preprocess::ImagePreprocessConfig> {
    let mut config = crate::preprocess::ImagePreprocessConfig::default();
    let Some(values) = values else {
        return Ok(config);
    };

    if let Some(value) = values.get_item("patch_size")? {
        config.patch_size = preprocess_u32(&value, "patch_size")?;
    }
    if let Some(value) = values.get_item("max_num_patches")? {
        config.max_num_patches = preprocess_optional_u32(&value, "max_num_patches")?;
    }
    if let Some(value) = values.get_item("min_num_patches")? {
        config.min_num_patches = preprocess_optional_u32(&value, "min_num_patches")?;
    }
    if let Some(value) = values.get_item("pixel_shuffle_scale")? {
        config.pixel_shuffle_scale = preprocess_u32(&value, "pixel_shuffle_scale")?;
    }

    Ok(config)
}

fn parse_max_inline_media_bytes(values: Option<&Bound<'_, PyDict>>) -> PyResult<usize> {
    let Some(value) = values
        .map(|values| values.get_item("max_inline_media_bytes"))
        .transpose()?
        .flatten()
    else {
        return Ok(DEFAULT_MAX_INLINE_MEDIA_BYTES);
    };
    if value.is_instance_of::<PyBool>() {
        return Err(py_value_error(
            "preprocess config \"max_inline_media_bytes\" must be a positive integer",
        ));
    }
    let limit = value.extract::<usize>().map_err(|_| {
        py_value_error("preprocess config \"max_inline_media_bytes\" must be a positive integer")
    })?;
    if limit == 0 {
        return Err(py_value_error(
            "preprocess config \"max_inline_media_bytes\" must be a positive integer",
        ));
    }
    Ok(limit)
}

fn decode_base64_media(
    payload_b64: &str,
    error_prefix: &str,
    max_decoded_bytes: usize,
) -> PyResult<Vec<u8>> {
    let max_encoded_bytes = max_decoded_bytes.div_ceil(3).saturating_mul(4);
    if payload_b64.len() > max_encoded_bytes {
        return Err(py_value_error(format!(
            "{error_prefix}: inline media exceeds the {max_decoded_bytes}-byte limit"
        )));
    }
    let decoded = B64
        .decode(payload_b64)
        .map_err(|e| py_value_error(format!("{error_prefix}: {e}")))?;
    if decoded.len() > max_decoded_bytes {
        return Err(py_value_error(format!(
            "{error_prefix}: inline media exceeds the {max_decoded_bytes}-byte limit"
        )));
    }
    Ok(decoded)
}

fn media_processor_config<'py>(
    py: Python<'py>,
    base_config: &Bound<'py, PyDict>,
    metadata: &Option<JsonValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let media_config = base_config.copy()?;
    if let Some(meta) = metadata {
        media_config.set_item("metadata", json_value_to_py(py, meta)?)?;
    }
    Ok(media_config)
}

fn media_source_to_py(
    py: Python<'_>,
    media: &crate::chat::MediaContent,
    max_inline_media_bytes: usize,
) -> PyResult<Option<PyObject>> {
    if let Some(source) = &media.source {
        return json_value_to_py(py, source).map(Some);
    }
    let Some(payload_b64) = media.media_ref.bytes_b64.as_deref() else {
        return Ok(None);
    };
    let bytes = py.allow_threads(|| {
        decode_base64_media(payload_b64, "invalid media base64", max_inline_media_bytes)
    })?;
    Ok(Some(pyo3::types::PyBytes::new(py, &bytes).unbind().into()))
}

fn registered_processor(
    py: Python<'_>,
    kind: &str,
    media_processors: &Option<Bound<'_, PyDict>>,
    image_processor: &Option<PyObject>,
) -> PyResult<Option<PyObject>> {
    if let Some(processors) = media_processors {
        if let Some(processor) = processors.get_item(kind)? {
            return Ok(Some(processor.unbind()));
        }
    }
    Ok((kind == "image")
        .then(|| {
            image_processor
                .as_ref()
                .map(|processor| processor.clone_ref(py))
        })
        .flatten())
}

fn validate_callback_mapping(
    callbacks: &Option<Bound<'_, PyDict>>,
    callback_type: &str,
) -> PyResult<()> {
    let Some(callbacks) = callbacks else {
        return Ok(());
    };
    for (kind, callback) in callbacks.iter() {
        let kind = kind
            .extract::<String>()
            .map_err(|_| py_value_error(format!("media {callback_type} keys must be strings")))?;
        if !callback.is_callable() {
            return Err(py_value_error(format!(
                "media {callback_type} registered for kind {kind:?} must be callable"
            )));
        }
    }
    Ok(())
}

fn registered_validator(
    kind: &str,
    media_validators: &Option<Bound<'_, PyDict>>,
) -> PyResult<Option<PyObject>> {
    let Some(validators) = media_validators else {
        return Ok(None);
    };
    Ok(validators.get_item(kind)?.map(pyo3::Bound::unbind))
}

fn result_field<'py>(result: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(mapping) = result.downcast::<PyDict>() {
        return mapping
            .get_item(name)?
            .ok_or_else(|| py_value_error(format!("media processor result is missing {name:?}")));
    }
    result
        .getattr(name)
        .map_err(|_| py_value_error(format!("media processor result is missing {name:?}")))
}

fn numeric_array_len(data: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(array) = data.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(array.as_array().len());
    }
    if let Ok(values) = data.extract::<Vec<f32>>() {
        return Ok(values.len());
    }

    let py = data.py();
    let numpy = PyModule::import(py, "numpy")
        .map_err(|_| py_value_error("image processor data must be a numeric array or sequence"))?;
    let array = numpy
        .call_method1("asarray", (data,))
        .map_err(|_| py_value_error("image processor data must be a numeric array or sequence"))?;
    let dtype_kind: String = array
        .getattr("dtype")
        .and_then(|dtype| dtype.getattr("kind"))
        .and_then(|kind| kind.extract())
        .map_err(|_| py_value_error("image processor data must be a numeric array or sequence"))?;
    if !matches!(dtype_kind.as_str(), "i" | "u" | "f") {
        return Err(py_value_error(
            "image processor data must be a numeric array or sequence",
        ));
    }
    let array = array
        .call_method1("astype", ("float32",))
        .map_err(|_| py_value_error("image processor data must be a numeric array or sequence"))?;
    let array = array
        .extract::<PyReadonlyArrayDyn<f32>>()
        .map_err(|_| py_value_error("image processor data must be a numeric array or sequence"))?;
    Ok(array.as_array().len())
}

fn image_processor_payload<'py>(
    py: Python<'py>,
    result: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = result_field(result, "data")?;
    let shape: Vec<i64> = match result_field(result, "shape") {
        Ok(shape) => shape.extract()?,
        Err(_) => data
            .getattr("shape")
            .and_then(|shape| shape.extract())
            .map_err(|_| {
                py_value_error("image processor result must provide shape or data.shape")
            })?,
    };
    if shape.len() != 3 {
        return Err(py_value_error(format!(
            "image processor shape must have three dimensions, got {shape:?}"
        )));
    }
    if shape.iter().any(|dimension| *dimension < 0) {
        return Err(py_value_error(format!(
            "image processor shape dimensions must be non-negative, got {shape:?}"
        )));
    }
    let has_zero = shape.contains(&0);
    if has_zero && shape != [0, 0, 0] {
        return Err(py_value_error(format!(
            "image processor shape must be all positive or [0, 0, 0], got {shape:?}"
        )));
    }
    let expected_len = if has_zero {
        0
    } else {
        shape.iter().try_fold(1_usize, |product, dimension| {
            let dimension = usize::try_from(*dimension).map_err(|_| {
                py_value_error(format!(
                    "image processor shape element count overflows for {shape:?}"
                ))
            })?;
            product.checked_mul(dimension).ok_or_else(|| {
                py_value_error(format!(
                    "image processor shape element count overflows for {shape:?}"
                ))
            })
        })?
    };

    let payload = PyDict::new(py);
    let array = result
        .getattr("into_numpy")
        .and_then(|method| method.call0())
        .or_else(|_| result.getattr("to_numpy").and_then(|method| method.call0()))
        .ok();
    let payload_data = array.as_ref().unwrap_or(&data);
    let actual_len = numeric_array_len(payload_data)?;
    if actual_len != expected_len {
        return Err(py_value_error(format!(
            "image processor data length {actual_len} does not match shape product {expected_len}"
        )));
    }
    payload.set_item("data", payload_data)?;
    payload.set_item("shape", shape)?;
    Ok(payload)
}

fn set_media_tags(event: &Bound<'_, PyDict>, media: &crate::chat::MediaContent) -> PyResult<()> {
    let tags = PyDict::new(event.py());
    tags.set_item("media_ref_id", media.media_ref.id.as_str())?;
    tags.set_item("media_ref_type", media.kind.as_str())?;
    event.set_item("tags", tags)
}

/// A thin PyO3 wrapper around the Rust `HarmonyEncoding` struct.
#[pyclass]
struct PyHarmonyEncoding {
    inner: HarmonyEncoding,
}

/// Streaming parser exposed to Python.
#[pyclass]
struct PyStreamableParser {
    inner: StreamableParser,
}

#[pyclass]
struct PyCompletionStreamParser {
    inner: CompletionStreamParser,
}

#[pymethods]
impl PyHarmonyEncoding {
    /// Create a new `HarmonyEncoding` by name.
    #[new]
    #[pyo3(signature = (name, reserved_token_groups=None, media_kinds=None))]
    fn new(
        name: &str,
        reserved_token_groups: Option<Bound<'_, PyAny>>,
        media_kinds: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let parsed: HarmonyEncodingName = name
            .parse::<HarmonyEncodingName>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let reserved_token_groups = parse_reserved_token_groups(reserved_token_groups)?;
        let media_kinds = parse_media_kind_registrations(media_kinds)?;
        let options = HarmonyEncodingOptions {
            reserved_token_groups,
            media_kinds,
        };
        let encoding = load_harmony_encoding_with_options(parsed, options)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        Ok(Self { inner: encoding })
    }

    /// Return the name of the encoding.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Render a conversation (in JSON format) for completion.
    ///
    /// Parameters
    /// ----------
    /// conversation_json : str
    ///     A JSON encoded `Conversation` (as produced by `serde_json`).
    /// next_turn_role : str
    ///     The role of the *next* turn (e.g. "assistant").
    /// config : dict (optional)
    ///     Optional config dict. Only supports 'auto_drop_analysis' (bool).
    ///
    /// Returns
    /// -------
    /// List[int]
    ///     The encoded token sequence.
    fn render_conversation_for_completion(
        &self,
        py: Python<'_>,
        conversation_json: &str,
        next_turn_role: &str,
        config: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        let rust_config = parse_render_config(config.as_ref())?;

        // Convert role string
        let role = Role::try_from(next_turn_role).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown role: {next_turn_role}"
            ))
        })?;

        let json_owned = conversation_json.to_owned();
        let inner = &self.inner;
        let conversation: crate::chat::Conversation = py
            .allow_threads(|| serde_json::from_str(&json_owned).map_err(|e| e.to_string()))
            .map_err(|e| py_value_error(format!("invalid conversation JSON: {e}")))?;

        py.allow_threads(|| {
            inner
                .render_conversation_for_completion(&conversation, role, Some(&rust_config))
                .map_err(|e| e.to_string())
        })
        .map_err(|e: String| PyErr::new::<HarmonyError, _>(e))
    }

    /// Render a conversation without appending a new role.
    fn render_conversation(
        &self,
        py: Python<'_>,
        conversation_json: &str,
        config: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        let rust_config = parse_render_config(config.as_ref())?;

        let json_owned = conversation_json.to_owned();
        let inner = &self.inner;
        let conversation: crate::chat::Conversation = py
            .allow_threads(|| serde_json::from_str(&json_owned).map_err(|e| e.to_string()))
            .map_err(|e| py_value_error(format!("invalid conversation JSON: {e}")))?;
        // Release GIL for rendering after input validation has succeeded.
        py.allow_threads(|| {
            inner
                .render_conversation(&conversation, Some(&rust_config))
                .map_err(|e| e.to_string())
        })
        .map_err(|e: String| PyErr::new::<HarmonyError, _>(e))
    }

    /// Render a conversation into a multimodal RenderedStream (JSON string).
    ///
    /// Parameters
    /// ----------
    /// conversation_json : str
    ///     A JSON encoded `Conversation`.
    /// config : dict (optional)
    ///     Optional preprocess config with keys:
    ///     - patch_size: int (default 16)
    ///     - max_num_patches: int or None (default 3600)
    ///     - min_num_patches: int or None (default None)
    ///     - pixel_shuffle_scale: int (default 1)
    fn render_conversation_multimodal(
        &self,
        py: Python<'_>,
        conversation_json: &str,
        config: Option<Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let image_config = parse_preprocess_config(config.as_ref())?;

        let preprocess_config = crate::preprocess::PreprocessConfig {
            image: image_config,
        };

        preprocess_config
            .validate()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let json_owned = conversation_json.to_owned();
        let inner = &self.inner;

        // Release GIL for heavy Rust work (JSON parse + rendering + image preprocessing + JSON serialize)
        py.allow_threads(move || {
            let conversation: crate::chat::Conversation = serde_json::from_str(&json_owned)
                .map_err(|e| format!("invalid conversation JSON: {e}"))?;

            let rendered = inner
                .render_conversation_multimodal(&conversation, Some(&preprocess_config))
                .map_err(|e| e.to_string())?;

            serde_json::to_string(&rendered).map_err(|e| e.to_string())
        })
        .map_err(|e: String| PyErr::new::<HarmonyError, _>(e))
    }

    /// Render a conversation with host-provided processors registered by media kind.
    ///
    /// Each processor receives `(source, config_dict)`. `source` is the opaque
    /// `MediaContent.source` value when present, otherwise decoded inline bytes.
    /// The image compatibility callback is used only when the generic mapping has
    /// no `image` entry.
    #[pyo3(signature = (conversation_json, config=None, media_processors=None, image_processor=None, timestamp_processor=None, media_validators=None))]
    #[allow(clippy::too_many_arguments)]
    fn render_conversation_multimodal_with_processors(
        &self,
        py: Python<'_>,
        conversation_json: &str,
        config: Option<Bound<'_, PyDict>>,
        media_processors: Option<Bound<'_, PyDict>>,
        image_processor: Option<PyObject>,
        timestamp_processor: Option<PyObject>,
        media_validators: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        use crate::chat::{Content, TokensContent};

        let json_owned = conversation_json.to_owned();
        let mut conversation: crate::chat::Conversation = py
            .allow_threads(|| serde_json::from_str(&json_owned).map_err(|e| e.to_string()))
            .map_err(|e| py_value_error(format!("invalid conversation JSON: {e}")))?;

        let image_config = parse_preprocess_config(config.as_ref())?;
        let max_inline_media_bytes = parse_max_inline_media_bytes(config.as_ref())?;
        let preprocess_config = crate::preprocess::PreprocessConfig {
            image: image_config.clone(),
        };
        preprocess_config
            .validate()
            .map_err(|error| py_value_error(error.to_string()))?;

        let processor_config = if let Some(cfg) = &config {
            cfg.copy()?
        } else {
            PyDict::new(py)
        };
        processor_config.set_item("patch_size", image_config.patch_size)?;
        processor_config.set_item("max_num_patches", image_config.max_num_patches)?;
        processor_config.set_item("min_num_patches", image_config.min_num_patches)?;
        processor_config.set_item("pixel_shuffle_scale", image_config.pixel_shuffle_scale)?;

        // Validation runs before rendering and before any processor callback. The
        // core first enforces that every kind has a registration; host validators
        // then own kind-specific payload and metadata invariants.
        validate_callback_mapping(&media_processors, "processor")?;
        validate_callback_mapping(&media_validators, "validator")?;
        if image_processor
            .as_ref()
            .is_some_and(|processor| !processor.bind(py).is_callable())
        {
            return Err(py_value_error("image processor must be callable"));
        }
        for message in &conversation.messages {
            for content in &message.content {
                let Content::Media(media) = content else {
                    continue;
                };
                self.inner
                    .registered_media_kind_for(media)
                    .map_err(|error| py_value_error(error.to_string()))?;
                let Some(validator) = registered_validator(&media.kind, &media_validators)? else {
                    continue;
                };

                let media_value = serde_json::to_value(media)
                    .map_err(|error| py_value_error(error.to_string()))?;
                let media_object = json_value_to_py(py, &media_value)?;
                let media_dict = media_object
                    .bind(py)
                    .downcast::<PyDict>()
                    .map_err(|_| py_value_error("serialized media content must be a mapping"))?;
                media_dict.set_item("type", "media")?;
                let item_config = media_processor_config(py, &processor_config, &media.metadata)?;
                let verdict = validator.call1(py, (media_object, item_config))?;
                if let Ok(verdict) = verdict.bind(py).downcast::<PyBool>() {
                    if !verdict.extract::<bool>()? {
                        return Err(py_value_error(format!(
                            "media validator for kind {:?} rejected media ref {:?}",
                            media.kind, media.media_ref.id
                        )));
                    }
                }
            }
        }

        // Timestamp rewriting is performed before the shared Rust renderer. Pointing
        // payloads become pre-rendered token content so their framing remains exact.
        if let Some(timestamp_processor) = &timestamp_processor {
            for message in &mut conversation.messages {
                for content in &mut message.content {
                    let role = message.author.role.to_string();
                    let process = |text: &str| -> PyResult<String> {
                        let context = PyDict::new(py);
                        context.set_item("valid_times", PyTuple::empty(py))?;
                        context.set_item("video_fps", 0.0)?;
                        timestamp_processor
                            .call1(py, (text, role.as_str(), context))?
                            .extract::<String>(py)
                    };

                    let pointing_markup = match content {
                        Content::Text(text) => {
                            text.text = process(&text.text)?;
                            None
                        }
                        Content::Pointing(pointing) => Some(pointing.to_html()),
                        Content::PointCollection(collection) => Some(collection.to_html()),
                        _ => None,
                    };
                    if let Some(markup) = pointing_markup {
                        let processed = process(&markup)?;
                        let mut tokens = vec![self
                            .inner
                            .render_formatting_token(crate::encoding::FormattingToken::PointStart)
                            .map_err(|error| PyErr::new::<HarmonyError, _>(error.to_string()))?
                            as i64];
                        tokens.extend(
                            self.inner
                                .encode_ordinary_text(&processed)
                                .map_err(|error| PyErr::new::<HarmonyError, _>(error.to_string()))?
                                .into_iter()
                                .map(i64::from),
                        );
                        tokens.push(
                            self.inner
                                .render_formatting_token(crate::encoding::FormattingToken::PointEnd)
                                .map_err(|error| PyErr::new::<HarmonyError, _>(error.to_string()))?
                                as i64,
                        );
                        *content = Content::Tokens(TokensContent {
                            tokens,
                            tags: Default::default(),
                        });
                    }
                }
            }
        }

        let rendered = self
            .inner
            .render_conversation_multimodal(conversation.messages.iter(), Some(&preprocess_config))
            .map_err(|error| PyErr::new::<HarmonyError, _>(error.to_string()))?;
        let rendered_value = serde_json::to_value(rendered)
            .map_err(|error| PyErr::new::<HarmonyError, _>(error.to_string()))?;
        let result = json_value_to_py(py, &rendered_value)?;
        let result_dict = result
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|_| PyErr::new::<HarmonyError, _>("rendered stream must be a mapping"))?;
        let stream = result_dict
            .get_item("stream")?
            .ok_or_else(|| PyErr::new::<HarmonyError, _>("rendered stream is missing stream"))?;
        let stream = stream.downcast::<PyDict>().map_err(|_| {
            PyErr::new::<HarmonyError, _>("rendered stream.stream must be a mapping")
        })?;
        let events = stream
            .get_item("events")?
            .ok_or_else(|| PyErr::new::<HarmonyError, _>("rendered stream is missing events"))?;
        let events = events
            .downcast::<PyList>()
            .map_err(|_| PyErr::new::<HarmonyError, _>("rendered stream.events must be a list"))?;

        let mut media_by_key = std::collections::HashMap::new();
        let message_refs: Vec<_> = conversation.messages.iter().collect();
        let render_config = crate::encoding::RenderConversationConfig::default();
        for message in
            crate::encoding::conversation_messages_to_render(&message_refs, &render_config)
        {
            for content in &message.content {
                if let Content::Media(media) = content {
                    media_by_key
                        .entry((media.kind.clone(), media.media_ref.id.clone()))
                        .or_insert_with(std::collections::VecDeque::new)
                        .push_back(media);
                }
            }
        }

        for event in events.iter() {
            let event = event
                .downcast::<PyDict>()
                .map_err(|_| PyErr::new::<HarmonyError, _>("rendered event must be a mapping"))?;
            let modality: String = event
                .get_item("modality")?
                .ok_or_else(|| PyErr::new::<HarmonyError, _>("rendered event is missing modality"))?
                .extract()?;
            let Some(tags) = event.get_item("tags")? else {
                continue;
            };
            let Ok(tags) = tags.downcast::<PyDict>() else {
                continue;
            };
            let Some(kind) = tags.get_item("media_ref_type")? else {
                continue;
            };
            let Some(media_id) = tags.get_item("media_ref_id")? else {
                continue;
            };
            let kind: String = kind.extract()?;
            let media_id: String = media_id.extract()?;
            let registration = self
                .inner
                .registered_media_kind(&kind)
                .map_err(|error| py_value_error(error.to_string()))?;
            if modality != registration.modality {
                continue;
            }
            let Some(media) = media_by_key
                .get_mut(&(kind.clone(), media_id))
                .and_then(std::collections::VecDeque::pop_front)
            else {
                continue;
            };
            let Some(source) = media_source_to_py(py, media, max_inline_media_bytes)? else {
                continue;
            };
            let processor = registered_processor(py, &kind, &media_processors, &image_processor)?
                .ok_or_else(|| {
                py_value_error(format!(
                    "media kind {kind:?} has a source but no processor callback"
                ))
            })?;
            let item_config = media_processor_config(py, &processor_config, &media.metadata)?;
            let processed = processor.call1(py, (source, item_config))?;
            let processed = processed.bind(py);

            let data = PyDict::new(py);
            match kind.as_str() {
                "image" => data.set_item("Image", image_processor_payload(py, processed)?)?,
                "video" => data.set_item("Video", processed)?,
                _ => {
                    let envelope = PyDict::new(py);
                    envelope.set_item("kind", kind.as_str())?;
                    envelope.set_item("payload", processed)?;
                    data.set_item("Media", envelope)?;
                }
            }
            event.set_item("data", data)?;
            set_media_tags(event, media)?;
        }

        Ok(result)
    }

    /// Render a conversation for training.
    fn render_conversation_for_training(
        &self,
        conversation_json: &str,
        config: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        let conversation: crate::chat::Conversation = serde_json::from_str(conversation_json)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "invalid conversation JSON: {e}"
                ))
            })?;

        let rust_config = parse_render_config(config.as_ref())?;

        self.inner
            .render_conversation_for_training(&conversation, Some(&rust_config))
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    fn render_message_header_text(
        &self,
        header_json: &str,
        options_json: &str,
    ) -> PyResult<String> {
        let header: MessageHeader = serde_json::from_str(header_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid message header JSON: {e}"
            ))
        })?;
        let options: HeaderRenderOptions = serde_json::from_str(options_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid header render options JSON: {e}"
            ))
        })?;

        self.inner
            .render_message_header_text(&header, options)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn render_message_header_tokens(
        &self,
        header_json: &str,
        options_json: &str,
    ) -> PyResult<Vec<u32>> {
        let header: MessageHeader = serde_json::from_str(header_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid message header JSON: {e}"
            ))
        })?;
        let options: HeaderRenderOptions = serde_json::from_str(options_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid header render options JSON: {e}"
            ))
        })?;

        self.inner
            .render_message_header_tokens(&header, options)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Render a single message into tokens.
    fn render(
        &self,
        message_json: &str,
        render_options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        let message: crate::chat::Message = serde_json::from_str(message_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid message JSON: {e}"))
        })?;

        let rust_options = if let Some(options_dict) = render_options {
            let conversation_has_function_tools = config_bool(
                &options_dict,
                "conversation_has_function_tools",
                false,
                "render options",
            )?;
            Some(crate::encoding::RenderOptions {
                conversation_has_function_tools,
            })
        } else {
            None
        };

        self.inner
            .render(&message, rust_options.as_ref())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    /// Given a list of completion tokens, parse them back into a sequence of
    /// messages.  The result is returned as a JSON string which can be
    /// deserialised on the Python side.
    #[allow(clippy::needless_pass_by_value)]
    fn parse_messages_from_completion_tokens(
        &self,
        tokens: Vec<u32>,
        role: Option<&str>,
        strict: Option<bool>,
    ) -> PyResult<String> {
        let role_parsed = if let Some(r) = role {
            Some(Role::try_from(r).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("unknown role: {r}"))
            })?)
        } else {
            None
        };

        let options = ParseOptions {
            strict: strict.unwrap_or(true),
        };

        let messages: Vec<Message> = self
            .inner
            .parse_messages_from_completion_tokens_with_options(tokens, role_parsed, options)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;

        serde_json::to_string(&messages).map_err(|e| {
            PyErr::new::<HarmonyError, _>(format!("failed to serialise messages to JSON: {e}"))
        })
    }

    /// Given completion tokens, parse them once into structured messages plus
    /// aligned append-safe stable display text for each message. The result is
    /// returned as JSON for the Python wrapper.
    #[allow(clippy::needless_pass_by_value)]
    fn parse_completion_view_from_tokens(
        &self,
        tokens: Vec<u32>,
        role: Option<&str>,
        strict: Option<bool>,
    ) -> PyResult<String> {
        let role_parsed = if let Some(r) = role {
            Some(Role::try_from(r).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("unknown role: {r}"))
            })?)
        } else {
            None
        };

        let options = ParseOptions {
            strict: strict.unwrap_or(true),
        };

        let view = self
            .inner
            .parse_completion_view_from_tokens(tokens, role_parsed, options)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;

        serde_json::to_string(&view).map_err(|e| {
            PyErr::new::<HarmonyError, _>(format!("failed to serialise parse view to JSON: {e}"))
        })
    }

    fn stable_visible_text_from_point_spans(&self, text: &str) -> String {
        self.inner.stable_visible_text_from_point_spans(text)
    }

    #[pyo3(signature = (role="assistant", strict=None, stop_on_tool_call=None, include_thinking_channels=None))]
    fn completion_stream_parser(
        &self,
        role: Option<&str>,
        strict: Option<bool>,
        stop_on_tool_call: Option<bool>,
        include_thinking_channels: Option<bool>,
    ) -> PyResult<PyCompletionStreamParser> {
        let parsed_role = role
            .map(|r| {
                Role::try_from(r).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("unknown role: {r}"))
                })
            })
            .transpose()?;
        let inner = self
            .inner
            .completion_stream_parser(CompletionStreamOptions {
                role: parsed_role,
                strict: strict.unwrap_or(false),
                stop_on_tool_call: stop_on_tool_call.unwrap_or(false),
                include_thinking_channels: include_thinking_channels.unwrap_or(false),
            })
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        Ok(PyCompletionStreamParser { inner })
    }

    /// Decode a sequence of tokens into text using the underlying tokenizer.
    fn decode_utf8(&self, py: Python<'_>, tokens: Vec<u32>) -> PyResult<String> {
        let inner = &self.inner;
        py.allow_threads(|| {
            inner
                .tokenizer()
                .decode_utf8(tokens)
                .map_err(|e| e.to_string())
        })
        .map_err(|e: String| PyErr::new::<HarmonyError, _>(e))
    }

    /// Decode a sequence of tokens into raw bytes using the underlying tokenizer.
    fn decode_bytes(&self, py: Python<'_>, tokens: Vec<u32>) -> PyResult<Vec<u8>> {
        let inner = &self.inner;
        py.allow_threads(|| {
            inner
                .tokenizer()
                .decode_bytes(tokens)
                .map_err(|e| e.to_string())
        })
        .map_err(|e: String| PyErr::new::<HarmonyError, _>(e))
    }

    /// Encode text into tokens using the underlying tokenizer with a set of allowed special tokens.
    fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<u32>> {
        // Extract allowed_special while holding GIL
        let allowed_vec: Vec<String> = match allowed_special {
            Some(obj) => obj.extract::<Vec<String>>().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "invalid allowed_special: {e}"
                ))
            })?,
            None => Vec::new(),
        };

        let text_owned = text.to_owned();
        let inner = &self.inner;

        // Release GIL for encoding
        let tokens = py.allow_threads(move || {
            let allowed_set: std::collections::HashSet<&str> =
                allowed_vec.iter().map(|s| s.as_str()).collect();
            inner.encode_text(&text_owned, &allowed_set)
        });

        tokens.map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    /// Encode timestamps (Qwen format) using Python-equivalent rounding.
    #[pyo3(signature = (timestamps, precision=None))]
    fn encode_timestamps_qwen(
        &self,
        py: Python<'_>,
        timestamps: Vec<f64>,
        precision: Option<usize>,
    ) -> PyResult<Vec<Vec<i64>>> {
        let inner = &self.inner;
        py.allow_threads(move || {
            inner
                .encode_timestamps_qwen(&timestamps, precision)
                .map_err(|e| e.to_string())
        })
        .map_err(|e: String| PyErr::new::<HarmonyError, _>(e))
    }

    /// Return the list of special tokens for this tokenizer.
    fn special_tokens(&self) -> Vec<String> {
        self.inner
            .tokenizer()
            .special_tokens()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Check whether a token id corresponds to a special token.
    fn is_special_token(&self, token: u32) -> bool {
        self.inner.tokenizer().is_special_token(token)
    }

    /// Return the stop tokens for the encoding.
    fn stop_tokens(&self) -> PyResult<Vec<u32>> {
        self.inner
            .stop_tokens()
            .map(|set| set.into_iter().collect())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    /// Return the stop tokens for assistant actions.
    fn stop_tokens_for_assistant_actions(&self) -> PyResult<Vec<u32>> {
        self.inner
            .stop_tokens_for_assistant_actions()
            .map(|set| set.into_iter().collect())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }
}

#[pymethods]
impl PyStreamableParser {
    #[new]
    fn new(
        encoding: &PyHarmonyEncoding,
        role: Option<&str>,
        strict: Option<bool>,
    ) -> PyResult<Self> {
        let parsed_role = role
            .map(|r| {
                Role::try_from(r).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("unknown role: {r}"))
                })
            })
            .transpose()?;
        let options = ParseOptions {
            strict: strict.unwrap_or(true),
        };
        let inner =
            StreamableParser::new_with_options(encoding.inner.clone(), parsed_role, options)
                .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    fn process(&mut self, token: u32) -> PyResult<()> {
        self.inner
            .process(token)
            .map(|_| ())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    fn process_eos(&mut self) -> PyResult<()> {
        self.inner
            .process_eos()
            .map(|_| ())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    #[getter]
    fn current_content(&self) -> PyResult<String> {
        self.inner
            .current_content()
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    #[getter]
    fn current_role(&self) -> Option<String> {
        self.inner.current_role().map(|r| r.as_str().to_string())
    }

    #[getter]
    fn current_content_type(&self) -> Option<String> {
        self.inner.current_content_type()
    }

    #[getter]
    fn last_content_delta(&self) -> PyResult<Option<String>> {
        self.inner
            .last_content_delta()
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    #[getter]
    fn messages(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.messages()).map_err(|e| {
            PyErr::new::<HarmonyError, _>(format!("failed to serialise messages to JSON: {e}"))
        })
    }

    #[getter]
    fn tokens(&self) -> Vec<u32> {
        self.inner.tokens().to_vec()
    }

    #[getter]
    fn state(&self) -> PyResult<String> {
        self.inner
            .state_json()
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    #[getter]
    fn current_recipient(&self) -> Option<String> {
        self.inner.current_recipient()
    }

    #[getter]
    fn current_channel(&self) -> Option<String> {
        self.inner.current_channel()
    }
}

fn completion_stream_update_to_py_dict(
    py: Python<'_>,
    update: CompletionStreamUpdate,
) -> PyResult<PyObject> {
    let value = serde_json::to_value(update).map_err(|e| {
        PyErr::new::<HarmonyError, _>(format!("failed to serialize stream update: {e}"))
    })?;
    json_value_to_py(py, &value)
}

#[pymethods]
impl PyCompletionStreamParser {
    #[getter]
    fn current_recipient(&self) -> Option<String> {
        self.inner.current_recipient()
    }

    #[getter]
    fn current_channel(&self) -> Option<String> {
        self.inner.current_channel()
    }

    fn push_tokens(
        &mut self,
        py: Python<'_>,
        token_ids_delta: Vec<u32>,
        absolute_start: usize,
    ) -> PyResult<PyObject> {
        let update = py
            .allow_threads(|| self.inner.push_tokens(&token_ids_delta, absolute_start))
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        completion_stream_update_to_py_dict(py, update)
    }

    fn finish(&mut self, py: Python<'_>, absolute_start: usize) -> PyResult<PyObject> {
        let update = py
            .allow_threads(|| self.inner.finish(absolute_start))
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        completion_stream_update_to_py_dict(py, update)
    }
}

/// Python module definition.
#[pymodule]
fn mharmony(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the PyHarmonyEncoding class.
    crate::channels::register_python_bindings(_py, m)?;
    m.add_class::<PyHarmonyEncoding>()?;
    m.add_class::<PyStreamableParser>()?;
    m.add_class::<PyCompletionStreamParser>()?;
    m.add("HarmonyError", _py.get_type::<HarmonyError>())?;

    // Register preprocessing types with DLPack support
    m.add_class::<crate::preprocess::PyProcessedImage>()?;

    // Convenience function mirroring the Rust-side `load_harmony_encoding` but
    // returning an *instance* of `PyHarmonyEncoding`.
    #[pyfunction(name = "load_harmony_encoding")]
    #[pyo3(signature = (name, reserved_token_groups=None, media_kinds=None))]
    fn load_harmony_encoding_py(
        py: Python<'_>,
        name: &str,
        reserved_token_groups: Option<Bound<'_, PyAny>>,
        media_kinds: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyHarmonyEncoding>> {
        let enc = PyHarmonyEncoding::new(name, reserved_token_groups, media_kinds)?;
        Py::new(py, enc)
    }
    m.add_function(pyo3::wrap_pyfunction!(load_harmony_encoding_py, m)?)?;

    // Convenience functions to get the tool configs for the browser and python tools.
    #[pyfunction]
    fn get_tool_namespace_config(py: Python<'_>, tool: &str) -> PyResult<PyObject> {
        let cfg = match tool {
            "browser" => ToolNamespaceConfig::browser(),
            "python" => ToolNamespaceConfig::python(),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown tool namespace: {tool}"
                )));
            }
        };
        let py_cfg =
            serde_json::to_value(&cfg).map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        let json_str = serde_json::to_string(&py_cfg)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        let json_mod = PyModule::import(py, "json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        Ok(py_obj.into())
    }
    m.add_function(pyo3::wrap_pyfunction!(get_tool_namespace_config, m)?)?;

    // Build the opening markup tag (`<{tag}{ATTRS}>`) for a pointing primitive
    // or collection. Mirrors `chat::format_markup_open_tag` so Python callers
    // that emit pointing markup themselves stay byte-identical with the
    // Rust-side text emitter.
    #[pyfunction]
    #[pyo3(signature = (tag, mention=None, index=None, t=None))]
    fn format_markup_open_tag(
        tag: &str,
        mention: Option<&str>,
        index: Option<&str>,
        t: Option<f64>,
    ) -> String {
        crate::chat::format_markup_open_tag(tag, mention, index, t)
    }
    m.add_function(pyo3::wrap_pyfunction!(format_markup_open_tag, m)?)?;

    // Variant for temporally pre-aligned primitives: `t` is a pre-formatted
    // display string (e.g. "1.2 seconds") emitted verbatim (attr-escaped) in
    // the same attribute position.
    #[pyfunction]
    #[pyo3(signature = (tag, mention=None, index=None, t=None))]
    fn format_markup_open_tag_display_t(
        tag: &str,
        mention: Option<&str>,
        index: Option<&str>,
        t: Option<&str>,
    ) -> String {
        crate::chat::format_markup_open_tag_display_t(tag, mention, index, t)
    }
    m.add_function(pyo3::wrap_pyfunction!(format_markup_open_tag_display_t, m)?)?;

    // (base_vocab_size, reserved_pool_start, reserved_pool_size) for the named
    // encoding, or None when it has no tabulated coord vocab layout. Sourced
    // from the same tiktoken_ext constants the reserved-pool allocator uses.
    #[pyfunction]
    fn encoding_vocab_info(name: &str) -> PyResult<Option<(u32, u32, u32)>> {
        let parsed: HarmonyEncodingName = name
            .parse()
            .map_err(|e: anyhow::Error| py_value_error(e.to_string()))?;
        Ok(crate::registry::encoding_vocab_info(parsed))
    }
    m.add_function(pyo3::wrap_pyfunction!(encoding_vocab_info, m)?)?;

    // Resolve reserved-token groups to absolute (offset, size) pairs, in input
    // order, using the allocator that actually runs at encoding load
    // (`registry.rs::resolve_reserved_token_groups`). Raises ValueError on
    // layouts that allocator rejects (missing name/tokenizer, zero size, pool
    // overflow, overlapping ranges).
    #[pyfunction]
    #[pyo3(signature = (name, reserved_token_groups=None))]
    fn resolve_reserved_token_group_offsets(
        name: &str,
        reserved_token_groups: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(i64, u64)>> {
        let parsed: HarmonyEncodingName = name
            .parse()
            .map_err(|e: anyhow::Error| py_value_error(e.to_string()))?;
        let configs = parse_reserved_token_groups(reserved_token_groups)?;
        let groups = crate::registry::resolve_reserved_token_groups(parsed, configs)
            .map_err(|e| py_value_error(e.to_string()))?;
        Ok(groups
            .into_iter()
            .map(|g| (g.offset, g.size as u64))
            .collect())
    }
    m.add_function(pyo3::wrap_pyfunction!(
        resolve_reserved_token_group_offsets,
        m
    )?)?;

    Ok(())
}
