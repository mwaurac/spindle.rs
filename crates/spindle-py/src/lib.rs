use pyo3::{exceptions::PyValueError, prelude::*};
use spindle_core::{device::Device, dtype::Dtype, tensor::Tensor as CoreTensor};

fn parse_dtype(name: &str) -> PyResult<Dtype> {
    Dtype::from_name(name).ok_or_else(|| PyValueError::new_err(format!("unknown dtype {name:?}")))
}

fn parse_device(name: &str) -> PyResult<Device> {
    match name {
        "cpu" => Ok(Device::CPU),
        _ => Err(PyValueError::new_err(format!(
            "unsupported device {name:?}"
        ))),
    }
}

#[pyclass(frozen)]
struct Tensor {
    inner: CoreTensor,
}

#[pymethods]
impl Tensor {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    #[getter]
    fn device(&self) -> &'static str {
        self.inner.device().as_str()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.layout().ndim()
    }

    fn numel(&self) -> usize {
        self.inner.layout().numel()
    }

    fn __repr__(&self) -> String {
        format!(
            "tensor(shape={:?}, dtype={}, device={})",
            self.shape(),
            self.dtype(),
            self.device()
        )
    }

    // ctors
    #[new]
    #[pyo3(signature = (shape, dtype = "f32", device = "cpu"))]
    fn new(shape: Vec<usize>, dtype: &str, device: &str) -> PyResult<Self> {
        let inner = CoreTensor::zeros(&shape, parse_dtype(dtype)?, parse_device(device)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype = "f32", device = "cpu"))]
    fn zeros(shape: Vec<usize>, dtype: &str, device: &str) -> PyResult<Self> {
        let inner = CoreTensor::zeros(&shape, parse_dtype(dtype)?, parse_device(device)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype = "f32", device = "cpu"))]
    fn ones(shape: Vec<usize>, dtype: &str, device: &str) -> PyResult<Self> {
        let inner = CoreTensor::ones(&shape, parse_dtype(dtype)?, parse_device(device)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    Ok(())
}
