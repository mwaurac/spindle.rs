use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::dtype::{Dtype, QK, QK_K};

/// A block of a quantized dtype.
pub trait QuantBlock: Pod {
    const DTYPE: Dtype;
}

macro_rules! quant_blocks {
    ($($block:ident => $dtype:ident),* $(,)?) => {$(
        impl QuantBlock for $block {
            const DTYPE: Dtype = Dtype::$dtype;
        }
        const _: () = assert!(size_of::<$block>() == Dtype::$dtype.type_size());
        const _: () = assert!(Dtype::$dtype.is_quantized());
    )*};
}

quant_blocks! {
    BlockQ4_0 => Q4_0,
    BlockQ4_1 => Q4_1,
    BlockQ5_0 => Q5_0,
    BlockQ5_1 => Q5_1,
    BlockQ8_0 => Q8_0,
    BlockQ8_1 => Q8_1,
    BlockQ2_K => Q2_K,
    BlockQ3_K => Q3_K,
    BlockQ4_K => Q4_K,
    BlockQ5_K => Q5_K,
    BlockQ6_K => Q6_K,
    BlockQ8_K => Q8_K,
}

/// 4-bit, symmetric: `x = d * (q - 8)`.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: f16,
    /// Two 4-bit quants per byte.
    pub qs: [u8; QK / 2],
}

/// 4-bit with minimum: `x = d * q + m`.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub d: f16,
    pub m: f16,
    pub qs: [u8; QK / 2],
}

/// 5-bit, symmetric: `x = d * (q - 16)`.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub d: f16,
    /// 5th bit of each quant.
    pub qh: [u8; 4],
    /// Low 4 bits, two quants per byte.
    pub qs: [u8; QK / 2],
}

/// 5-bit with minimum: `x = d * q + m`.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub d: f16,
    pub m: f16,
    pub qh: [u8; 4],
    pub qs: [u8; QK / 2],
}

/// 8-bit, symmetric: `x = d * q`.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; QK],
}

/// 8-bit with precomputed `s = d * sum(qs)`, used for activations in
/// Q4_1/Q5_1 dot products.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub d: f16,
    pub s: f16,
    pub qs: [i8; QK],
}

/// 2-bit K-quant: 16 sub-blocks of 16, with 4-bit scales and mins.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ2_K {
    /// Low 4 bits: scale, high 4 bits: min, per sub-block.
    pub scales: [u8; QK_K / 16],
    pub qs: [u8; QK_K / 4],
    pub d: f16,
    pub dmin: f16,
}

/// 3-bit K-quant: 16 sub-blocks of 16, with 6-bit scales.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ3_K {
    /// High bit of each quant.
    pub hmask: [u8; QK_K / 8],
    /// Low 2 bits of each quant.
    pub qs: [u8; QK_K / 4],
    /// 16 packed 6-bit scales.
    pub scales: [u8; 12],
    pub d: f16,
}

/// 4-bit K-quant: 8 sub-blocks of 32, with 6-bit scales and mins.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4_K {
    pub d: f16,
    pub dmin: f16,
    /// 8 scales and 8 mins, packed 6 bits each.
    pub scales: [u8; 12],
    pub qs: [u8; QK_K / 2],
}

/// 5-bit K-quant: 8 sub-blocks of 32, with 6-bit scales and mins.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5_K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],
    /// High bit of each quant.
    pub qh: [u8; QK_K / 8],
    /// Low 4 bits of each quant.
    pub qs: [u8; QK_K / 2],
}

/// 6-bit K-quant: 16 sub-blocks of 16, with 8-bit scales.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ6_K {
    /// Low 4 bits of each quant.
    pub ql: [u8; QK_K / 2],
    /// High 2 bits of each quant.
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: f16,
}

/// 8-bit K-quant, used only for activations in K-quant dot products.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ8_K {
    pub d: f32,
    pub qs: [i8; QK_K],
    /// Sums of each group of 16 quants.
    pub bsums: [i16; QK_K / 16],
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cast_from_bytes() {
        // Two Q8_0 blocks as they would appear in a file.
        let mut bytes = vec![0u8; 2 * Dtype::Q8_0.type_size()];
        bytes[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        bytes[2] = 3;
        bytes[34 + 33] = (-1i8) as u8;

        let blocks: &[BlockQ8_0] = bytemuck::cast_slice(&bytes);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].d.to_f32(), 0.5);
        assert_eq!(blocks[0].qs[0], 3);
        assert_eq!(blocks[1].qs[31], -1);
    }
}
