pub const QK: usize = 32;
pub const QK_K: usize = 256;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Dtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
    U8 = 256,
    Bool = 257,
}

impl Dtype {
    pub fn from_u32(v: u32) -> Option<Dtype> {
        Some(match v {
            0 => Dtype::F32,
            1 => Dtype::F16,
            2 => Dtype::Q4_0,
            3 => Dtype::Q4_1,
            6 => Dtype::Q5_0,
            7 => Dtype::Q5_1,
            8 => Dtype::Q8_0,
            9 => Dtype::Q8_1,
            10 => Dtype::Q2_K,
            11 => Dtype::Q3_K,
            12 => Dtype::Q4_K,
            13 => Dtype::Q5_K,
            14 => Dtype::Q6_K,
            15 => Dtype::Q8_K,
            24 => Dtype::I8,
            25 => Dtype::I16,
            26 => Dtype::I32,
            27 => Dtype::I64,
            28 => Dtype::F64,
            30 => Dtype::BF16,
            256 => Dtype::U8,
            257 => Dtype::Bool,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::Q4_0 => "q4_0",
            Dtype::Q4_1 => "q4_1",
            Dtype::Q5_0 => "q5_0",
            Dtype::Q5_1 => "q5_1",
            Dtype::Q8_0 => "q8_0",
            Dtype::Q8_1 => "q8_1",
            Dtype::Q2_K => "q2_k",
            Dtype::Q3_K => "q3_k",
            Dtype::Q4_K => "q4_k",
            Dtype::Q5_K => "q5_k",
            Dtype::Q6_K => "q6_k",
            Dtype::Q8_K => "q8_k",
            Dtype::I8 => "i8",
            Dtype::I16 => "i16",
            Dtype::I32 => "i32",
            Dtype::I64 => "i64",
            Dtype::F64 => "f64",
            Dtype::BF16 => "bf16",
            Dtype::U8 => "u8",
            Dtype::Bool => "bool",
        }
    }

    /// Size in bytes of one block (one element for a non-quantized type)
    pub fn type_size(&self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F64 | Dtype::I64 => 8,
            // d: f16, qs: [u8; 16]
            Dtype::Q4_0 => 2 + QK / 2,
            // d, m: f16, qs: [u8; 16]
            Dtype::Q4_1 => 2 * 2 + QK / 2,
            // d: f16, qh: [u8; 4], qs: [u8; 16]
            Dtype::Q5_0 => 2 + 4 + QK / 2,
            // d, m: f16, qh: [u8; 4], qs: [u8; 16]
            Dtype::Q5_1 => 2 * 2 + 4 + QK / 2,
            // d: f16, qs: [i8; 32]
            Dtype::Q8_0 => 2 + QK,
            // d, s: f16, qs: [i8; 32]
            Dtype::Q8_1 => 2 * 2 + QK,
            // scales: [u8; 16], qs: [u8; 64], d, dmin: f16
            Dtype::Q2_K => QK_K / 16 + QK_K / 4 + 2 * 2,
            // hmask: [u8; 32], qs: [u8; 64], scales: [u8; 12], d: f16
            Dtype::Q3_K => QK_K / 8 + QK_K / 4 + 12 + 2,
            // d, dmin: f16, scales: [u8; 12], qs: [u8; 128]
            Dtype::Q4_K => 2 * 2 + 12 + QK_K / 2,
            // d, dmin: f16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128]
            Dtype::Q5_K => 2 * 2 + 12 + QK_K / 8 + QK_K / 2,
            // ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16
            Dtype::Q6_K => QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
            // d: f32, qs: [i8; 256], bsums: [i16; 16]
            Dtype::Q8_K => 4 + QK_K + QK_K / 16 * 2,
        }
    }

    /// Number of elements stored together in one block. 1 for non-quantized types.
    pub const fn block_size(&self) -> usize {
        match self {
            Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 | Dtype::Q8_0 | Dtype::Q8_1 => QK,
            Dtype::Q2_K | Dtype::Q3_K | Dtype::Q4_K | Dtype::Q5_K | Dtype::Q6_K | Dtype::Q8_K => {
                QK_K
            }
            _ => 1,
        }
    }

    /// BYtes needed to store `numel` contiguous elements. `numel` must be a multiple of `block_size`
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
        matches!(self, Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64)
    }

    pub const fn is_int(&self) -> bool {
        matches!(
            self,
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::U8
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const ALL: [Dtype; 22] = [
        Dtype::F32,
        Dtype::F16,
        Dtype::Q4_0,
        Dtype::Q4_1,
        Dtype::Q5_0,
        Dtype::Q5_1,
        Dtype::Q8_0,
        Dtype::Q8_1,
        Dtype::Q2_K,
        Dtype::Q3_K,
        Dtype::Q4_K,
        Dtype::Q5_K,
        Dtype::Q6_K,
        Dtype::Q8_K,
        Dtype::I8,
        Dtype::I16,
        Dtype::I32,
        Dtype::I64,
        Dtype::F64,
        Dtype::BF16,
        Dtype::U8,
        Dtype::Bool,
    ];

    #[test]
    fn id_roundtrip() {
        for dt in ALL {
            assert_eq!(Dtype::from_u32(dt as u32), Some(dt));
        }
        assert_eq!(Dtype::from_u32(4), None);
    }

    #[test]
    fn block_sizes_match_ggml() {
        let expected = [
            (Dtype::Q4_0, 18),
            (Dtype::Q4_1, 20),
            (Dtype::Q5_0, 22),
            (Dtype::Q5_1, 24),
            (Dtype::Q8_0, 34),
            (Dtype::Q8_1, 36),
            (Dtype::Q2_K, 84),
            (Dtype::Q3_K, 110),
            (Dtype::Q4_K, 144),
            (Dtype::Q5_K, 176),
            (Dtype::Q6_K, 210),
            (Dtype::Q8_K, 292),
        ];
        for (dt, size) in expected {
            assert_eq!(dt.type_size(), size, "{dt:?}");
        }
    }

    #[test]
    fn categories_are_disjoint() {
        for dt in ALL {
            let n = [dt.is_float(), dt.is_int(), dt.is_quantized()]
                .iter()
                .filter(|&&b| b)
                .count();
            let expected = if dt == Dtype::Bool { 0 } else { 1 };
            assert_eq!(n, expected, "{dt:?}");
        }
    }

    #[test]
    fn row_size() {
        assert_eq!(Dtype::F32.row_size(10), 40);
        assert_eq!(Dtype::Q8_0.row_size(4096), 4096 / 32 * 34);
        assert_eq!(Dtype::Q4_K.row_size(4096), 16 * 144);
    }
}
