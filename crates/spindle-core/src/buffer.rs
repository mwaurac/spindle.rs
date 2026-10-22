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

    /// MOves host data onto `device`
    pub fn from_host(device: Device, data: CpuBuffer) -> Self {
        match device {
            Device::CPU => Buffer::Cpu(data),
        }
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

    /// The buffer's bytes in host memory
    pub fn host_bytes(&self) -> &[u8] {
        match self {
            Buffer::Cpu(b) => b.as_bytes(),
        }
    }
}
