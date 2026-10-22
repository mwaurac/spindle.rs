use crate::dtype::Dtype;
use crate::error::Result;
use crate::layout::Layout;
use std::{fmt, sync::Arc};

use crate::{device::Device, storage::Storage};

pub struct Tensor(Arc<TensorImpl_>);

struct TensorImpl_ {
    storage: Arc<Storage>, // TODO: prop change this when there are multple writers. NOT SAFE now
    layout: Layout,
    device: Device,
}

impl Tensor {
    // accessors
    pub fn device(&self) -> Device {
        self.0.device
    }

    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    pub fn dtype(&self) -> Dtype {
        self.0.layout.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        self.0.layout.shape()
    }

    pub fn strides(&self) -> &[usize] {
        self.0.layout.strides()
    }

    /// Size in bytes of the underlying storage, which views may share
    pub fn storage_len(&self) -> usize {
        self.0.storage.len()
    }
    // ctors
    pub fn from_storage(storage: Storage, layout: Layout, device: Device) -> Self {
        Tensor(Arc::new(TensorImpl_ {
            storage: Arc::new(storage),
            layout,
            device,
        }))
    }
    pub fn zeros(shape: &[usize], dtype: Dtype, device: Device) -> Result<Self> {
        let layout = Layout::contiguous(shape, dtype)?;
        let storage = Storage::zeros(device, layout.nbytes())?;
        Ok(Self::from_storage(storage, layout, device))
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, device={:?})",
            self.shape(),
            self.dtype().as_str(),
            self.device()
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn zeros() {
        let t = Tensor::zeros(&[4, 64], Dtype::Q8_0, Device::CPU).unwrap();
        assert_eq!(t.shape(), [4, 64]);
        assert_eq!(t.storage_len(), 4 * 2 * 34);
        assert_eq!(
            format!("{t:?}"),
            "Tensor(shape=[4, 64], dtype=q8_0, device=CPU)"
        );
    }
}
