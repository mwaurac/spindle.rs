use std::fmt;

use crate::device::Device;

#[derive(Debug)]
pub enum Error {
    TooManyDims { ndim: usize, max: usize },
    OutOfMemory { device: Device, bytes: usize },
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
        }
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
