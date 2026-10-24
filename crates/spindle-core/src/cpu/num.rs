//! Numeric traits shared by the reference kernels.

use std::ops::{Add, Div, Mul, Neg, Sub};

pub(super) trait Real:
    Copy
    + PartialOrd
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const HALF: Self;
    /// `sqrt(2 / π)`
    const SQRT_2_OVER_PI: Self;
    const GELU_COEF: Self;

    fn abs(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn is_nan(self) -> bool;
}

macro_rules! impl_real {
    ($($t:ident),*) => {$(
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const HALF: Self = 0.5;
            // 2/√π · 1/√2 = √(2/π)
            const SQRT_2_OVER_PI: Self =
                std::$t::consts::FRAC_2_SQRT_PI * std::$t::consts::FRAC_1_SQRT_2;
            const GELU_COEF: Self = 0.044_715;

            fn abs(self) -> Self { <$t>::abs(self) }
            fn exp(self) -> Self { <$t>::exp(self) }
            fn ln(self) -> Self { <$t>::ln(self) }
            fn sqrt(self) -> Self { <$t>::sqrt(self) }
            fn tanh(self) -> Self { <$t>::tanh(self) }
            fn powf(self, n: Self) -> Self { <$t>::powf(self, n) }
            fn is_nan(self) -> bool { <$t>::is_nan(self) }
        }
    )*};
}

impl_real!(f32, f64);

pub(super) trait Int: Copy + Ord {
    const ZERO: Self;
    fn wrapping_neg(self) -> Self;
    fn wrapping_abs(self) -> Self;
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn wrapping_mul(self, rhs: Self) -> Self;
}

macro_rules! impl_int {
    ($($t:ty),*) => {$(
        impl Int for $t {
            const ZERO: Self = 0;
            fn wrapping_neg(self) -> Self { <$t>::wrapping_neg(self) }
            fn wrapping_abs(self) -> Self { <$t>::wrapping_abs(self) }
            fn wrapping_add(self, rhs: Self) -> Self { <$t>::wrapping_add(self, rhs) }
            fn wrapping_sub(self, rhs: Self) -> Self { <$t>::wrapping_sub(self, rhs) }
            fn wrapping_mul(self, rhs: Self) -> Self { <$t>::wrapping_mul(self, rhs) }
        }
    )*};
}

impl_int!(i8, i16, i32, i64);

impl Int for u8 {
    const ZERO: Self = 0;
    fn wrapping_neg(self) -> Self {
        u8::wrapping_neg(self)
    }
    fn wrapping_abs(self) -> Self {
        self
    }
    fn wrapping_add(self, rhs: Self) -> Self {
        u8::wrapping_add(self, rhs)
    }
    fn wrapping_sub(self, rhs: Self) -> Self {
        u8::wrapping_sub(self, rhs)
    }
    fn wrapping_mul(self, rhs: Self) -> Self {
        u8::wrapping_mul(self, rhs)
    }
}
