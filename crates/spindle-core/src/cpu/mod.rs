use crate::device::Device;
use crate::error::{Error, Result};

use std::{
    alloc::{self},
    ptr::NonNull,
};

pub const ALIGN: usize = 64;

pub struct CpuBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    fn alloc(len: usize, zeroed: bool) -> Result<Self> {
        if len == 0 {
            // zero-sized allocations are not allowerd; nothing is ever read from
            // this pointer since the buffer is empty
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
            });
        }

        let oom = || Error::OutOfMemory {
            device: Device::CPU,
            bytes: len,
        };

        let layout = alloc::Layout::from_size_align(len, ALIGN).map_err(|_| oom())?;
        let ptr = unsafe {
            if zeroed {
                alloc::alloc_zeroed(layout)
            } else {
                alloc::alloc(layout)
            }
        };

        let ptr = NonNull::new(ptr).ok_or_else(oom)?;
        Ok(Self { ptr, len })
    }

    pub fn alloc_uninit(len: usize) -> Result<Self> {
        Self::alloc(len, false)
    }

    pub fn alloc_zeroed(len: usize) -> Result<Self> {
        Self::alloc(len, true)
    }
}
