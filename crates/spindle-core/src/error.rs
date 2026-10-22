use std::fmt;

use crate::{device::Device, dtype::Dtype};

#[derive(Debug)]
pub enum Error {
    TooManyDims { ndim: usize, max: usize },
    OutOfMemory { device: Device, bytes: usize },
    BlockMisaligned { dtype: Dtype, shape: Vec<usize> },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::TooManyDims { ndim, max } => {
                write!(f, "Tensor has {ndim} dims, at most {max} are supported")
            }
            Error::OutOfMemory { device, bytes } => {
                write!(f, "Out of memory allocating {bytes} on {device:?}")
            }

            Error::BlockMisaligned { dtype, shape } => write!(
                f,
                "shape {shape:?} is invalid for {}: the innermost dim must be a multiple of {}",
                dtype.as_str(),
                dtype.block_size()
            ),
        }
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
