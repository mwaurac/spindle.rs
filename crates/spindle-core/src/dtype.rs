#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F64,
}

impl Dtype {
    pub fn as_str(&self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
        }
    }

    pub fn type_size(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F64 => 8,
        }
    }

    pub fn block_size(&self) -> usize {
        // for now we have no block quantized dtypes so the blcok size is 1
        1
    }

    pub fn row_size(&self, numel: usize) -> usize {
        debug_assert_eq!(
            numel % self.block_size(),
            0,
            "{self:?}: {numel} is not a multiple of the block size "
        );
        numel / self.block_size() * self.type_size()
    }
}
