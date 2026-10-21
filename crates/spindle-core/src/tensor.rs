use crate::dtype::Dtype;
use crate::error::Result;
use crate::layout::Layout;
use std::sync::Arc;

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
