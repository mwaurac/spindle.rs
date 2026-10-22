pub const QK: usize = 32;
pub const QK_K: usize = 256;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    I32,
    F32,
    F64,
    Q8_0,
    Q8_K,
}

impl Dtype {
    pub fn as_str(&self) -> &'static str {
        match self {
            Dtype::I32 => "i32",
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::Q8_0 => "q8_0",
            Dtype::Q8_K => "q8_k",
        }
    }

    pub const fn type_size(&self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F64 => 8,
            // d: f16, qs: [i8; 32]
            Dtype::Q8_0 => 2 + QK,
            // d: f32, qs: [i8; 256], bsums: [i16; 16]
            Dtype::Q8_K => 4 + QK_K + QK_K / 16 * 2,
        }
    }

    pub const fn block_size(&self) -> usize {
        match self {
            Dtype::Q8_0 => QK,
            Dtype::Q8_K => QK_K,
            _ => 1,
        }
    }

    pub fn row_size(&self, numel: usize) -> usize {
        debug_assert_eq!(
            numel % self.block_size(),
            0,
            "{self:?}: {numel} is not a multiple of the block size "
        );
        numel / self.block_size() * self.type_size()
    }

    pub const fn is_quantized(&self) -> bool {
        self.block_size() > 1
    }

    pub const fn is_float(&self) -> bool {
        matches!(self, Dtype::F32 | Dtype::F64)
    }

    pub const fn is_int(&self) -> bool {
        matches!(self, Dtype::I32)
    }
}
