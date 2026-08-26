// Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
//! mharmony preprocessing types for host-provided media callbacks.
//!
//! This module defines the built-in image configuration and result types.
//! All actual preprocessing must be done via user-provided Python callbacks.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImagePreprocessConfig {
    pub resize: (u32, u32), // (height, width) - used only if max_num_patches is None
    pub normalize_mean: [f32; 3],
    pub normalize_std: [f32; 3],
    pub patch_size: u32,
    pub max_num_patches: Option<u32>,
    pub min_num_patches: Option<u32>,
    pub pixel_shuffle_scale: u32,
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        Self {
            resize: (224, 224),
            normalize_mean: [0.5, 0.5, 0.5],
            normalize_std: [0.5, 0.5, 0.5],
            patch_size: 16,
            max_num_patches: Some(3600), // 60x60 patches
            min_num_patches: None,
            pixel_shuffle_scale: 1,
        }
    }
}

impl ImagePreprocessConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.patch_size > 0, "patch_size must be positive");
        ensure!(
            self.pixel_shuffle_scale > 0,
            "pixel_shuffle_scale must be positive"
        );
        ensure!(
            self.resize.0 > 0 && self.resize.1 > 0,
            "resize dimensions must be positive"
        );
        if let Some(max) = self.max_num_patches {
            ensure!(max > 0, "max_num_patches must be positive");
        }
        if let Some(min) = self.min_num_patches {
            ensure!(min > 0, "min_num_patches must be positive");
        }
        if let (Some(min), Some(max)) = (self.min_num_patches, self.max_num_patches) {
            ensure!(min <= max, "min_num_patches must be <= max_num_patches");
        }
        Ok(())
    }
    /// CLIP ViT-L/14 preset
    pub fn clip_vit_l() -> Self {
        Self {
            resize: (224, 224),
            normalize_mean: [0.48145466, 0.4578275, 0.40821073],
            normalize_std: [0.26862954, 0.261_302_6, 0.275_777_1],
            patch_size: 14,
            max_num_patches: Some(3600),
            min_num_patches: None,
            pixel_shuffle_scale: 1,
        }
    }

    /// SigLIP SO400M preset
    pub fn siglip_so400m() -> Self {
        Self {
            resize: (384, 384),
            normalize_mean: [0.5, 0.5, 0.5],
            normalize_std: [0.5, 0.5, 0.5],
            patch_size: 14,
            max_num_patches: Some(3600),
            min_num_patches: None,
            pixel_shuffle_scale: 1,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PreprocessConfig {
    pub image: ImagePreprocessConfig,
}

impl PreprocessConfig {
    pub fn validate(&self) -> Result<()> {
        self.image.validate()
    }
}

// ---------------------------------------------------------------------------
// Processed output types (returned by user callbacks)
// ---------------------------------------------------------------------------

/// Preprocessed image ready for model consumption.
/// Data is stored as flattened patches with shape [h_patches, w_patches, patch_dim]
/// where patch_dim = 3 * patch_size^2.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessedImage {
    pub data: Vec<f32>,
    pub shape: Vec<i64>, // [h_patches, w_patches, patch_dim]
}

impl ProcessedImage {
    /// Get the number of elements in the tensor
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }

    /// Get strides for the tensor (assuming C-contiguous layout)
    pub fn strides(&self) -> Vec<i64> {
        if self.shape.is_empty() {
            return Vec::new();
        }
        let mut strides = vec![1i64; self.shape.len()];
        for i in (0..self.shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }
}

/// Preprocessed vector payload (flattened data + original shape).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessedVector {
    pub data: Vec<f32>,
    pub shape: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Python bindings with tensor data access (zero-copy via numpy crate)
// ---------------------------------------------------------------------------

#[cfg(feature = "python-binding")]
mod py_impl {
    use super::*;
    use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn};
    use pyo3::prelude::*;
    use pyo3::types::PyAnyMethods;

    fn coerce_processed_data(
        data: &Bound<'_, PyAny>,
        shape: Option<Vec<i64>>,
    ) -> PyResult<(Vec<f32>, Vec<i64>)> {
        if data.is_none() {
            return Ok((Vec::new(), shape.unwrap_or_default()));
        }

        if let Ok(array) = data.extract::<PyReadonlyArrayDyn<f32>>() {
            let arr = array.as_array();
            let out_shape =
                shape.unwrap_or_else(|| arr.shape().iter().map(|&d| d as i64).collect());
            let data_vec: Vec<f32> = arr.iter().copied().collect();
            return Ok((data_vec, out_shape));
        }

        let py = data.py();
        if let Ok(np) = py.import("numpy") {
            if let Ok(arr) = np.call_method1("asarray", (data,)) {
                let arr = arr.call_method1("astype", ("float32",))?;
                if let Ok(array) = arr.extract::<PyReadonlyArrayDyn<f32>>() {
                    let arr = array.as_array();
                    let out_shape =
                        shape.unwrap_or_else(|| arr.shape().iter().map(|&d| d as i64).collect());
                    let data_vec: Vec<f32> = arr.iter().copied().collect();
                    return Ok((data_vec, out_shape));
                }
            }
        }

        if data.hasattr("detach")? {
            let mut tensor = data.call_method0("detach")?;
            if tensor.hasattr("cpu")? {
                tensor = tensor.call_method0("cpu")?;
            }
            if tensor.hasattr("numpy")? {
                let arr = tensor.call_method0("numpy")?;
                if let Ok(array) = arr.extract::<PyReadonlyArrayDyn<f32>>() {
                    let arr = array.as_array();
                    let out_shape =
                        shape.unwrap_or_else(|| arr.shape().iter().map(|&d| d as i64).collect());
                    let data_vec: Vec<f32> = arr.iter().copied().collect();
                    return Ok((data_vec, out_shape));
                }
            }
        }

        let vec: Vec<f32> = data.extract()?;
        let out_shape = shape.unwrap_or_else(|| vec![vec.len() as i64]);
        Ok((vec, out_shape))
    }
    /// Wrapper for ProcessedImage with tensor data access
    #[pyclass(name = "ProcessedImage")]
    pub struct PyProcessedImage {
        pub inner: ProcessedImage,
    }

    #[pymethods]
    impl PyProcessedImage {
        #[new]
        fn new(data: Bound<'_, PyAny>, shape: Option<Vec<i64>>) -> PyResult<Self> {
            let (data, shape) = coerce_processed_data(&data, shape)?;
            Ok(Self {
                inner: ProcessedImage { data, shape },
            })
        }

        #[getter]
        fn shape(&self) -> Vec<i64> {
            self.inner.shape.clone()
        }

        #[getter]
        fn numel(&self) -> usize {
            self.inner.numel()
        }

        /// Get the raw data as a flat list of f32 values (copies)
        #[getter]
        fn data(&self) -> Vec<f32> {
            self.inner.data.clone()
        }

        /// Convert to numpy array with efficient memory transfer.
        fn into_numpy(slf: Py<Self>, py: Python<'_>) -> PyResult<PyObject> {
            let mut this = slf.borrow_mut(py);
            let shape = this.inner.shape.clone();
            let data = std::mem::take(&mut this.inner.data);
            drop(this);

            let flat_arr = data.into_pyarray(py);
            let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            let reshaped = flat_arr.reshape(shape_usize)?;
            Ok(reshaped.unbind().into())
        }

        /// Convert to numpy array with efficient memory transfer.
        fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let flat_arr = self.inner.data.clone().into_pyarray(py);
            let shape_usize: Vec<usize> = self.inner.shape.iter().map(|&x| x as usize).collect();
            let reshaped = flat_arr.reshape(shape_usize)?;
            Ok(reshaped.unbind().into())
        }

        /// Convert to torch tensor with efficient memory transfer.
        fn into_torch(slf: Py<Self>, py: Python<'_>) -> PyResult<PyObject> {
            let np_arr = Self::into_numpy(slf, py)?;
            let torch = py.import("torch")?;
            let tensor = torch.call_method1("from_numpy", (np_arr,))?;
            Ok(tensor.unbind())
        }

        /// Convert to torch tensor with efficient memory transfer.
        fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
            let torch = py.import("torch")?;
            let np_arr = self.to_numpy(py)?;
            let tensor = torch.call_method1("from_numpy", (np_arr,))?;
            Ok(tensor.unbind())
        }
    }

    impl From<ProcessedImage> for PyProcessedImage {
        fn from(inner: ProcessedImage) -> Self {
            Self { inner }
        }
    }
}

#[cfg(feature = "python-binding")]
pub use py_impl::PyProcessedImage;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_preprocess_config_presets() {
        let clip = ImagePreprocessConfig::clip_vit_l();
        assert_eq!(clip.resize, (224, 224));
        assert_eq!(clip.patch_size, 14);

        let siglip = ImagePreprocessConfig::siglip_so400m();
        assert_eq!(siglip.resize, (384, 384));
        assert_eq!(siglip.patch_size, 14);
    }

    #[test]
    fn test_processed_image_strides() {
        // After patchification, shape is [H//ps, W//ps, C*ps*ps]
        // For 224x224 with patch_size=16: [14, 14, 768]
        let img = ProcessedImage {
            data: vec![0.0; 14 * 14 * 768],
            shape: vec![14, 14, 768],
        };
        let strides = img.strides();
        assert_eq!(strides, vec![14 * 768, 768, 1]);
        assert!(ProcessedImage {
            data: Vec::new(),
            shape: Vec::new(),
        }
        .strides()
        .is_empty());
    }
}
