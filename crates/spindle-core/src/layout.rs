use crate::dtype::Dtype;
use crate::error::{Error, Result};

const MAX_DIMS: usize = 6;

pub struct Layout {
    ndim: u8,
    shape: [usize; MAX_DIMS],
    strides: [usize; MAX_DIMS],
    dtype: Dtype,
    offset: usize,
}

fn check_dims(ndim: usize) -> Result<()> {
    if ndim > MAX_DIMS {
        return Err(Error::TooManyDims {
            ndim,
            max: MAX_DIMS,
        });
    }
    Ok(())
}

fn padded(shape: &[usize], fill: usize) -> [usize; MAX_DIMS] {
    let mut slots = [fill; MAX_DIMS];
    slots[MAX_DIMS - shape.len()..].copy_from_slice(shape);
    slots
}

impl Layout {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.ndim as usize
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape[MAX_DIMS - self.ndim()..]
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides[MAX_DIMS - self.ndim()..]
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn nbytes(&self) -> usize {
        self.dtype().row_size(self.numel())
    }

    pub fn contiguous(shape: &[usize], dtype: Dtype) -> Result<Self> {
        let ndim = shape.len();
        check_dims(ndim)?;

        let mut layout = Self {
            ndim: ndim as u8,
            shape: padded(shape, 1),
            strides: [0; MAX_DIMS],
            dtype,
            offset: 0,
        };

        layout.strides = layout.contiguous_strides();
        Ok(layout)
    }

    pub fn contiguous_strides(&self) -> [usize; MAX_DIMS] {
        let mut strides = [0; MAX_DIMS];

        if self.ndim == 0 {
            return strides;
        }
        let last = MAX_DIMS - 1;
        strides[last] = self.dtype.type_size();
        let mut acc = self.dtype.type_size() * (self.shape[last] / self.dtype.block_size()).max(1);
        for i in (MAX_DIMS - self.ndim()..last).rev() {
            strides[i] = acc;
            acc *= self.shape[i].max(1);
        }
        strides
    }
}
