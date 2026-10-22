use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::strided::StridedOffsets;

pub const MAX_DIMS: usize = 6;

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

    /// All `MAX_DIMS` shape slots, padding included
    pub fn padded_shape(&self) -> &[usize; MAX_DIMS] {
        &self.shape
    }

    /// Byte strides, one per dim
    pub fn strides(&self) -> &[usize] {
        &self.strides[MAX_DIMS - self.ndim()..]
    }

    /// All `MAX_DIMS` strides slots, padding included
    pub fn padded_strides(&self) -> &[usize; MAX_DIMS] {
        &self.strides
    }

    /// Byte offset of the first element with the underlying buffer
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Bytes need to store this shape contiguously
    pub fn nbytes(&self) -> usize {
        self.dtype().row_size(self.numel())
    }

    /// ctor
    /// contiguous (row-major) layout for `shape` with elements of `dtype`
    pub fn contiguous(shape: &[usize], dtype: Dtype) -> Result<Self> {
        let ndim = shape.len();
        check_dims(ndim)?;

        if dtype.is_quantized() && shape.last().is_none_or(|&n| n % dtype.block_size() != 0) {
            return Err(Error::BlockMisaligned {
                dtype,
                shape: shape.to_vec(),
            });
        }
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

    /// Layout with explicit byte strides and offset, as produced by views.
    ///
    /// Only the rank is validated; callers must ensure every offset the layout
    /// reaches lies inside the buffer it is used with.
    #[cfg_attr(not(test), allow(dead_code))] // used by the view ops
    pub(crate) fn from_parts(
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        dtype: Dtype,
    ) -> Result<Self> {
        let ndim = shape.len();
        check_dims(ndim)?;
        assert_eq!(strides.len(), ndim, "one stride per dim");
        let mut layout = Self {
            ndim: ndim as u8,
            shape: padded(shape, 1),
            strides: [0; MAX_DIMS],
            dtype,
            offset,
        };
        layout.strides[MAX_DIMS - ndim..].copy_from_slice(strides);
        Ok(layout)
    }

    /// Byte offsets of every element (block, for quantized dtypes) in
    /// row-major order.
    pub fn offsets(&self) -> StridedOffsets {
        StridedOffsets::new(self)
    }
    /// Byte strides this shape would have if it were contiguous, in slot form
    /// (padding slots are 0)
    /// Zero-sized dims are treated as size 1 so strides stay non-zero, like Pytorch
    pub fn contiguous_strides(&self) -> [usize; MAX_DIMS] {
        let mut strides = [0; MAX_DIMS];

        if self.ndim == 0 {
            return strides;
        }

        // The innermost dim steps one block (one element for non-quantized types)
        // and holds shape /block_size blocks.
        let last = MAX_DIMS - 1;
        strides[last] = self.dtype.type_size();
        let mut acc = self.dtype.type_size() * (self.shape[last] / self.dtype.block_size()).max(1);
        for i in (MAX_DIMS - self.ndim()..last).rev() {
            strides[i] = acc;
            acc *= self.shape[i].max(1);
        }
        strides
    }

    /// Whether the elements are laid out densely in row-major order.
    ///
    /// Strides of size-1 dims (padding included) are ignored since they are
    /// never stepped, adn empty layouts are always contiguous
    pub fn is_contiguous(&self) -> bool {
        if self.numel() == 0 {
            return true;
        }

        let expected = self.contiguous_strides();
        (0..MAX_DIMS).all(|i| self.shape[i] == 1 || self.strides[i] == expected[i])
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn padding_is_leading() {
        let l = Layout::contiguous(&[2, 3], Dtype::F32).unwrap();
        assert_eq!(l.padded_shape(), &[1, 1, 1, 1, 2, 3]);
        assert_eq!(l.padded_strides(), &[0, 0, 0, 0, 12, 4]);
        assert_eq!(l.shape(), [2, 3]);
        assert_eq!(l.strides(), [12, 4]);

        let s = Layout::contiguous(&[], Dtype::F32).unwrap();
        assert_eq!(s.padded_shape(), &[1; MAX_DIMS]);
    }

    #[test]
    fn contiguous_f32() {
        let l = Layout::contiguous(&[4, 3, 2], Dtype::F32).unwrap();
        assert_eq!(l.ndim(), 3);
        assert_eq!(l.shape(), [4, 3, 2]);
        assert_eq!(l.strides(), [24, 8, 4]);
        assert_eq!(l.offset(), 0);
        assert_eq!(l.numel(), 24);
        assert_eq!(l.nbytes(), 96);
        assert!(l.is_contiguous());
    }

    #[test]
    fn size_one_dims() {
        let l = Layout::contiguous(&[4, 3, 1], Dtype::F32).unwrap();
        assert_eq!(l.strides(), [12, 4, 4]);
    }

    #[test]
    fn scalar() {
        let l = Layout::contiguous(&[], Dtype::F32).unwrap();
        assert_eq!(l.ndim(), 0);
        assert_eq!(l.shape(), [] as [usize; 0]);
        assert_eq!(l.numel(), 1);
        assert_eq!(l.nbytes(), 4);
        assert!(l.is_contiguous());
    }

    #[test]
    fn quantized_rows() {
        // A 4096-wide Q8_0 row is 128 blocks of 34 bytes.
        let l = Layout::contiguous(&[2, 11008, 4096], Dtype::Q8_0).unwrap();
        assert_eq!(l.strides(), [11008 * 4352, 4352, 34]);
        assert_eq!(l.nbytes(), 2 * 11008 * 4352);
        assert!(l.is_contiguous());
    }

    #[test]
    fn quantized_misaligned() {
        assert!(matches!(
            Layout::contiguous(&[4, 33], Dtype::Q8_0),
            Err(Error::BlockMisaligned { .. })
        ));
        assert!(matches!(
            Layout::contiguous(&[], Dtype::Q8_K),
            Err(Error::BlockMisaligned { .. })
        ));
    }

    #[test]
    fn too_many_dims() {
        assert!(Layout::contiguous(&[1; MAX_DIMS], Dtype::F32).is_ok());
        assert!(matches!(
            Layout::contiguous(&[1; MAX_DIMS + 1], Dtype::F32),
            Err(Error::TooManyDims {
                ndim: 7,
                max: MAX_DIMS
            })
        ));
    }

    #[test]
    fn empty_dims() {
        let l = Layout::contiguous(&[0, 3], Dtype::F32).unwrap();
        assert_eq!(l.numel(), 0);
        assert_eq!(l.nbytes(), 0);
        assert_eq!(l.strides(), [12, 4]);
        assert!(l.is_contiguous());
    }
}
