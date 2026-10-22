use bytemuck::Pod;
use half::{bf16, f16};

use crate::dtype::Dtype;

/// A rust type that is stored as the element of a non-quantized dtype
pub trait Element: Pod {
    const DTYPE: Dtype;

    /// Converts with `as` semantics: floats round, ints truncate and saturate.
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

macro_rules! primitive_elements {
    ($($ty:ty => $dtype:ident),* $(,)?) => {$(
        impl Element for $ty {
            const DTYPE: Dtype = Dtype::$dtype;

            fn from_f64(v: f64) -> Self {
                v as $ty
            }

            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    )*};
}

primitive_elements! {
    f32 => F32,
    f64 => F64,
    i8 => I8,
    i16 => I16,
    i32 => I32,
    i64 => I64,
    u8 => U8,
}

macro_rules! half_elements {
    ($($ty:ty => $dtype:ident),* $(,)?) => {$(
        impl Element for $ty {
            const DTYPE: Dtype = Dtype::$dtype;

            fn from_f64(v: f64) -> Self {
                <$ty>::from_f64(v)
            }

            fn to_f64(self) -> f64 {
                <$ty>::to_f64(self)
            }
        }
    )*};
}

half_elements! {
    f16 => F16,
    bf16 => BF16,
}
