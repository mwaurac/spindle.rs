use crate::layout::{Layout, MAX_DIMS};
/// Iterator over the byte offsets of a layout's units in row-major order.
///
/// A unit is one element or one block for a quantized dtype. Offsets are
/// absolute withing the buffer (they include the layout;s offset)
///
/// Size-1 dims are dropped and adjacent dims that are contiguous with each
/// other are merged before iterating, so a contiguous layout works as a single
/// flat dim.
///
#[derive(Debug, Clone)]
pub struct StridedOffsets {
    ndim: usize,
    shape: [usize; MAX_DIMS],
    strides: [usize; MAX_DIMS],
    index: [usize; MAX_DIMS],
    next: usize,
    remaining: usize,
}

impl StridedOffsets {
    pub fn new(layout: &Layout) -> Self {
        let dtype = layout.dtype();
        let ndim = layout.ndim();

        // Slots are padded on the left, matching `Layout`.
        let mut units = [1; MAX_DIMS];
        units[MAX_DIMS - ndim..].copy_from_slice(layout.shape());
        if ndim > 0 {
            units[MAX_DIMS - 1] /= dtype.block_size();
        }
        let remaining = units[MAX_DIMS - ndim..].iter().product();

        // Coalesce, walking outermost to innermost and appending to `shape`
        let mut shape = [1; MAX_DIMS];
        let mut strides = [0; MAX_DIMS];
        let mut ndim = 0;
        for (&n, &stride) in units[MAX_DIMS - layout.ndim()..]
            .iter()
            .zip(layout.strides())
        {
            if n == 1 {
                continue;
            }
            // the previous dim steps exactly over this whole dim: merge them.
            if ndim > 0 && strides[ndim - 1] == stride * n {
                shape[ndim - 1] *= n;
                strides[ndim - 1] = stride;
            } else {
                shape[ndim] = n;
                strides[ndim] = stride;
                ndim += 1;
            }
        }

        Self {
            ndim,
            shape,
            strides,
            index: [0; MAX_DIMS],
            next: layout.offset(),
            remaining,
        }
    }

    pub fn coalesced_ndim(&self) -> usize {
        self.ndim
    }
}

impl Iterator for StridedOffsets {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let current = self.next;
        if self.remaining > 0 {
            // Odometer increment from the innermost dim, carrying outwards.
            for d in (0..self.ndim).rev() {
                self.index[d] += 1;
                self.next += self.strides[d];
                if self.index[d] < self.shape[d] {
                    break;
                }
                self.next -= self.strides[d] * self.shape[d];
                self.index[d] = 0;
            }
        }
        Some(current)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StridedOffsets {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dtype::Dtype;

    fn offsets(layout: &Layout) -> Vec<usize> {
        StridedOffsets::new(layout).collect()
    }

    #[test]
    fn contiguous_is_flat() {
        let l = Layout::contiguous(&[2, 3, 2], Dtype::F32).unwrap();
        let it = StridedOffsets::new(&l);
        assert_eq!(it.coalesced_ndim(), 1);
        assert_eq!(it.len(), 12);
        assert_eq!(
            it.collect::<Vec<_>>(),
            (0..12).map(|i| i * 4).collect::<Vec<_>>()
        );
    }

    #[test]
    fn transposed() {
        // [3, 2] f32 transposed to [2, 3].
        let l = Layout::from_parts(&[2, 3], &[4, 8], 0, Dtype::F32).unwrap();
        assert_eq!(offsets(&l), [0, 8, 16, 4, 12, 20]);
    }

    #[test]
    fn broadcast() {
        // A row of 3 f32 repeated twice.
        let l = Layout::from_parts(&[2, 3], &[0, 4], 0, Dtype::F32).unwrap();
        assert_eq!(offsets(&l), [0, 4, 8, 0, 4, 8]);
    }

    #[test]
    fn sliced_with_offset() {
        // Columns 1..3 of a [2, 4] f32 tensor.
        let l = Layout::from_parts(&[2, 2], &[16, 4], 4, Dtype::F32).unwrap();
        let it = StridedOffsets::new(&l);
        assert_eq!(it.coalesced_ndim(), 2);
        assert_eq!(it.collect::<Vec<_>>(), [4, 8, 20, 24]);
    }

    #[test]
    fn partial_coalescing() {
        // Batch-strided [2, 2, 3] f32: each [2, 3] matrix is contiguous, but the
        // matrices are 32 bytes apart instead of 24.
        let l = Layout::from_parts(&[2, 2, 3], &[32, 12, 4], 0, Dtype::F32).unwrap();
        let it = StridedOffsets::new(&l);
        assert_eq!(it.coalesced_ndim(), 2);
        assert_eq!(
            it.collect::<Vec<_>>(),
            [0, 4, 8, 12, 16, 20, 32, 36, 40, 44, 48, 52]
        );
    }

    #[test]
    fn size_one_dims_ignored() {
        let l = Layout::from_parts(&[1, 3, 1], &[999, 4, 777], 0, Dtype::F32).unwrap();
        let it = StridedOffsets::new(&l);
        assert_eq!(it.coalesced_ndim(), 1);
        assert_eq!(it.collect::<Vec<_>>(), [0, 4, 8]);
    }

    #[test]
    fn scalar_and_empty() {
        let l = Layout::from_parts(&[], &[], 12, Dtype::F32).unwrap();
        assert_eq!(offsets(&l), [12]);

        let l = Layout::contiguous(&[3, 0, 2], Dtype::F32).unwrap();
        assert_eq!(offsets(&l), [] as [usize; 0]);
    }

    #[test]
    fn quantized_blocks() {
        // Two rows of 64 Q8_0 elements = 2 blocks of 34 bytes per row.
        let l = Layout::contiguous(&[2, 64], Dtype::Q8_0).unwrap();
        assert_eq!(offsets(&l), [0, 34, 68, 102]);

        // Only the second row.
        let l = Layout::from_parts(&[1, 64], &[68, 34], 68, Dtype::Q8_0).unwrap();
        assert_eq!(offsets(&l), [68, 102]);
    }
}
