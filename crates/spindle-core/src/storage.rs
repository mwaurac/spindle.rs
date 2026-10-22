use crate::error::Result;
use crate::{buffer::Buffer, device::Device};
pub struct Storage {
    buffer: Buffer,
}

impl Storage {
    pub fn zeros(device: Device, len: usize) -> Result<Self> {
        Ok(Self {
            buffer: Buffer::alloc_zeroed(device, len)?,
        })
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}
