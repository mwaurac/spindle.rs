pub mod binary;
pub mod compare;
pub mod num;
pub mod reduce;
pub mod unary;

use bytemuck::Pod;

use crate::device::Device;
use crate::error::{Error, Result};

use std::{
    alloc::{self},
    ptr::NonNull,
};
/// Reads a `T` at byte `offset`, which need not be aligned for `T`.
fn read<T: Pod>(bytes: &[u8], offset: usize) -> T {
    bytemuck::pod_read_unaligned(&bytes[offset..offset + size_of::<T>()])
}

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

    pub unsafe fn alloc_uninit(len: usize) -> Result<Self> {
        Self::alloc(len, false)
    }

    pub fn alloc_zeroed(len: usize) -> Result<Self> {
        Self::alloc(len, true)
    }

    /// Allocates a buffer holding a copy of `data`.
    pub fn from_slice<T: Pod>(data: &[T]) -> Result<Self> {
        let src: &[u8] = bytemuck::cast_slice(data);
        // SAFETY: every byte is written by the copy below.
        let mut buf = unsafe { Self::alloc_uninit(src.len()) }?;
        buf.as_bytes_mut().copy_from_slice(src);
        Ok(buf)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` bytes (or dangling with `len == 0`)
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_bytes`, and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: allocated in `alloc` with this exact layout
            unsafe {
                alloc::dealloc(
                    self.ptr.as_ptr(),
                    alloc::Layout::from_size_align_unchecked(self.len, ALIGN),
                );
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn zeroed() {
        let buf = CpuBuffer::alloc_zeroed(100).unwrap();
        assert_eq!(buf.len(), 100);
        assert!(buf.as_bytes().iter().all(|&b| b == 0));
        assert_eq!(buf.as_ptr() as usize % ALIGN, 0)
    }

    #[test]
    fn from_slice() {
        let buf = CpuBuffer::from_slice(&[1.0f32, -2.0, 3.5]).unwrap();
        assert_eq!(buf.len(), 12);
        assert_eq!(buf.as_ptr() as usize % ALIGN, 0);
        let values: &[f32] = bytemuck::cast_slice(buf.as_bytes());
        assert_eq!(values, [1.0, -2.0, 3.5]);
    }

    #[test]
    fn empty() {
        let buf = CpuBuffer::alloc_zeroed(0).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.as_bytes(), [] as [u8; 0]);
    }

    #[test]
    fn too_large() {
        assert!(matches!(
            CpuBuffer::alloc_zeroed(usize::MAX),
            Err(Error::OutOfMemory { .. })
        ));
    }
}
