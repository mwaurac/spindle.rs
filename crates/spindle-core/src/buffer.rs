use crate::error::Result;
use crate::{cpu::CpuBuffer, device::Device};

pub enum Buffer {
    Cpu(CpuBuffer),
    //TODO: impl other device buffers
}

impl Buffer {
    pub fn alloc_zeroed(device: Device, len: usize) -> Result<Self> {
        Ok(match device {
            Device::CPU => Buffer::Cpu(CpuBuffer::alloc_zeroed(len)?),
        })
    }

    pub fn alloc_uninit(device: Device, len: usize) -> Result<Self> {
        Ok(match device {
            Device::CPU => Buffer::Cpu(unsafe { CpuBuffer::alloc_uninit(len) }?),
        })
    }

    pub fn len(&self) -> usize {
        match self {
            Buffer::Cpu(b) => b.len(),
        }
    }
}
