use std::ops::{Add, Div, Mul, Neg};

use bytemuck::Pod;
use half::{bf16, f16};

use crate::{
    cpu::CpuBuffer,
    device::Device,
    dtype::Dtype,
    error::{Error, Result},
    layout::Layout,
    ops::UnaryOp,
};

pub fn unary(op: UnaryOp, bytes: &[u8], layout: &Layout) -> Result<CpuBuffer> {
    let unsupported = || Error::UnsupportedOp {
        op: op.name(),
        dtype: layout.dtype(),
        device: Device::CPU,
    };

    match layout.dtype() {
        Dtype::F32 => map(bytes, layout, float_fn::<f32>(op)),

        Dtype::F64 => map(bytes, layout, float_fn::<f64>(op)),
        Dtype::F16 => {
            let f = float_fn::<f32>(op);
            map(bytes, layout, |x: f16| f16::from_f32(f(x.to_f32())))
        }
        Dtype::BF16 => {
            let f = float_fn::<f32>(op);
            map(bytes, layout, |x: bf16| bf16::from_f32(f(x.to_f32())))
        }
        Dtype::I8 => map(bytes, layout, int_fn::<i8>(op).ok_or_else(unsupported)?),
        Dtype::I16 => map(bytes, layout, int_fn::<i16>(op).ok_or_else(unsupported)?),
        Dtype::I32 => map(bytes, layout, int_fn::<i32>(op).ok_or_else(unsupported)?),
        Dtype::I64 => map(bytes, layout, int_fn::<i64>(op).ok_or_else(unsupported)?),
        Dtype::U8 => map(bytes, layout, int_fn::<u8>(op).ok_or_else(unsupported)?),
        _ => Err(unsupported()),
    }
}

fn map<T: Pod>(bytes: &[u8], layout: &Layout, f: impl Fn(T) -> T) -> Result<CpuBuffer> {
    let size = size_of::<T>();
    let out: Vec<T> = layout
        .offsets()
        .map(|o| f(bytemuck::pod_read_unaligned(&bytes[o..o + size])))
        .collect();
    CpuBuffer::from_slice(&out)
}

trait Real:
    Copy
    + PartialOrd
    + Neg<Output = Self>
    + Add<Output = Self>
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
        }
    )*};
}

impl_real!(f32, f64);

fn float_fn<F: Real>(op: UnaryOp) -> fn(F) -> F {
    fn sigmoid<F: Real>(x: F) -> F {
        F::ONE / (F::ONE + (-x).exp())
    }
    match op {
        UnaryOp::Neg => |x: F| -x,
        UnaryOp::Abs => |x: F| x.abs(),
        UnaryOp::Exp => |x: F| x.exp(),
        UnaryOp::Log => |x: F| x.ln(),
        UnaryOp::Sqrt => |x: F| x.sqrt(),
        UnaryOp::Rsqrt => |x: F| F::ONE / x.sqrt(),
        UnaryOp::Tanh => |x: F| x.tanh(),
        UnaryOp::Sigmoid => sigmoid,
        UnaryOp::Relu => |x: F| if x < F::ZERO { F::ZERO } else { x },
        UnaryOp::Silu => |x: F| x * sigmoid(x),
        UnaryOp::Gelu => |x: F| {
            let inner = F::SQRT_2_OVER_PI * (x + F::GELU_COEF * x * x * x);
            F::HALF * x * (F::ONE + inner.tanh())
        },
    }
}

trait Int: Copy + Ord {
    const ZERO: Self;
    fn wrapping_neg(self) -> Self;
    fn wrapping_abs(self) -> Self;
}

macro_rules! impl_signed {
    ($($t:ty),*) => {$(
        impl Int for $t {
            const ZERO: Self = 0;
            fn wrapping_neg(self) -> Self { <$t>::wrapping_neg(self) }
            fn wrapping_abs(self) -> Self { <$t>::wrapping_abs(self) }
        }
    )*};
}

impl_signed!(i8, i16, i32, i64);

impl Int for u8 {
    const ZERO: Self = 0;
    fn wrapping_neg(self) -> Self {
        u8::wrapping_neg(self)
    }
    fn wrapping_abs(self) -> Self {
        self
    }
}

/// The integer version of `op`, or `None` if it is float-only.
fn int_fn<I: Int>(op: UnaryOp) -> Option<fn(I) -> I> {
    Some(match op {
        UnaryOp::Neg => |x: I| x.wrapping_neg(),
        UnaryOp::Abs => |x: I| x.wrapping_abs(),
        UnaryOp::Relu => |x: I| x.max(I::ZERO),
        _ => return None,
    })
}

#[cfg(test)]
mod test {
    use crate::{
        device::Device, dtype::Dtype, error::Error, layout::Layout, ops::UnaryOp, tensor::Tensor,
    };
    use half::{bf16, f16};

    const CPU: Device = Device::CPU;
    const INPUT: [f64; 7] = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];

    fn reference(op: UnaryOp, x: f64) -> f64 {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        match op {
            UnaryOp::Neg => -x,
            UnaryOp::Abs => x.abs(),
            UnaryOp::Exp => x.exp(),
            UnaryOp::Log => x.ln(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Rsqrt => 1.0 / x.sqrt(),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Sigmoid => sigmoid,
            UnaryOp::Relu => x.max(0.0),
            UnaryOp::Silu => x * sigmoid,
            UnaryOp::Gelu => {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }
        }
    }

    const ALL_OPS: [UnaryOp; 11] = [
        UnaryOp::Neg,
        UnaryOp::Abs,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sqrt,
        UnaryOp::Rsqrt,
        UnaryOp::Tanh,
        UnaryOp::Sigmoid,
        UnaryOp::Relu,
        UnaryOp::Silu,
        UnaryOp::Gelu,
    ];

    /// Compares within `tol`, treating NaN == NaN and equal infinities as equal.
    fn assert_close(op: UnaryOp, got: f64, want: f64, tol: f64) {
        let ok = (got.is_nan() && want.is_nan())
            || got == want
            || (got - want).abs() <= tol * want.abs().max(1.0);
        assert!(ok, "{}: got {got}, want {want}", op.name());
    }

    #[test]
    fn float_ops_match_reference() {
        let f32_in: Vec<f32> = INPUT.iter().map(|&x| x as f32).collect();
        let t32 = Tensor::from_slice(&f32_in, &[7], CPU).unwrap();
        let t64 = Tensor::from_slice(&INPUT, &[7], CPU).unwrap();
        let t16: Vec<f16> = INPUT.iter().map(|&x| f16::from_f64(x)).collect();
        let t16 = Tensor::from_slice(&t16, &[7], CPU).unwrap();
        let tb16: Vec<bf16> = INPUT.iter().map(|&x| bf16::from_f64(x)).collect();
        let tb16 = Tensor::from_slice(&tb16, &[7], CPU).unwrap();

        for op in ALL_OPS {
            let out32 = t32.unary(op).unwrap().to_vec::<f32>().unwrap();
            let out64 = t64.unary(op).unwrap().to_vec::<f64>().unwrap();
            let out16 = t16.unary(op).unwrap().to_vec::<f16>().unwrap();
            let outb16 = tb16.unary(op).unwrap().to_vec::<bf16>().unwrap();
            for (i, &x) in INPUT.iter().enumerate() {
                let want = reference(op, x);
                assert_close(op, out64[i], want, 1e-12);
                assert_close(op, out32[i] as f64, want, 1e-6);
                assert_close(op, out16[i].to_f64(), want, 1e-3);
                assert_close(op, outb16[i].to_f64(), want, 1e-2);
            }
        }
    }

    #[test]
    fn method_shorthands() {
        let t = Tensor::from_slice(&[-1.0f32, 4.0], &[2], CPU).unwrap();
        assert_eq!(t.neg().unwrap().to_vec::<f32>().unwrap(), [1.0, -4.0]);
        assert_eq!(t.relu().unwrap().to_vec::<f32>().unwrap(), [0.0, 4.0]);
        assert_eq!(
            t.abs().unwrap().sqrt().unwrap().to_vec::<f32>().unwrap(),
            [1.0, 2.0]
        );
    }

    #[test]
    fn extremes_stay_finite() {
        let t = Tensor::from_slice(&[-1000.0f32, 1000.0, f32::NAN], &[3], CPU).unwrap();
        assert_eq!(
            t.sigmoid().unwrap().to_vec::<f32>().unwrap()[..2],
            [0.0, 1.0]
        );
        let silu = t.silu().unwrap().to_vec::<f32>().unwrap();
        assert_eq!(silu[..2], [-0.0, 1000.0]);
        assert!(t.relu().unwrap().to_vec::<f32>().unwrap()[2].is_nan());
    }

    #[test]
    fn int_ops() {
        let t = Tensor::from_slice(&[-128i8, -3, 0, 5, 127], &[5], CPU).unwrap();
        assert_eq!(
            t.neg().unwrap().to_vec::<i8>().unwrap(),
            [-128, 3, 0, -5, -127]
        );
        assert_eq!(
            t.abs().unwrap().to_vec::<i8>().unwrap(),
            [-128, 3, 0, 5, 127]
        );
        assert_eq!(t.relu().unwrap().to_vec::<i8>().unwrap(), [0, 0, 0, 5, 127]);

        let t = Tensor::from_slice(&[-7i64, 7], &[2], CPU).unwrap();
        assert_eq!(t.abs().unwrap().to_vec::<i64>().unwrap(), [7, 7]);

        let t = Tensor::from_slice(&[0u8, 1, 255], &[3], CPU).unwrap();
        assert_eq!(t.neg().unwrap().to_vec::<u8>().unwrap(), [0, 255, 1]);
        assert_eq!(t.abs().unwrap().to_vec::<u8>().unwrap(), [0, 1, 255]);
    }

    #[test]
    fn unsupported_dtypes() {
        let t = Tensor::from_slice(&[1i32, 2], &[2], CPU).unwrap();
        assert!(matches!(
            t.exp(),
            Err(Error::UnsupportedOp {
                op: "exp",
                dtype: Dtype::I32,
                ..
            })
        ));

        let q = Tensor::zeros(&[32], Dtype::Q8_0, CPU).unwrap();
        assert!(matches!(q.neg(), Err(Error::UnsupportedOp { .. })));

        let b = Tensor::full(&[2], 1.0, Dtype::Bool, CPU).unwrap();
        assert!(matches!(b.abs(), Err(Error::UnsupportedOp { .. })));
    }

    #[test]
    fn strided_input_gives_contiguous_output() {
        // [[0, 1, 2], [3, 4, 5]] transposed to [[0, 3], [1, 4], [2, 5]].
        let t = Tensor::arange(0.0, 6.0, 1.0, Dtype::F32, CPU).unwrap();
        let tt = t.with_layout(Layout::from_parts(&[3, 2], &[4, 12], 0, Dtype::F32).unwrap());
        let out = tt.neg().unwrap();
        assert_eq!(out.shape(), [3, 2]);
        assert!(out.layout().is_contiguous());
        assert_eq!(
            out.to_vec::<f32>().unwrap(),
            [-0.0, -3.0, -1.0, -4.0, -2.0, -5.0]
        );
    }

    #[test]
    fn scalar_and_empty() {
        let s = Tensor::full(&[], 4.0, Dtype::F32, CPU).unwrap();
        let out = s.rsqrt().unwrap();
        assert_eq!(out.shape(), [] as [usize; 0]);
        assert_eq!(out.to_vec::<f32>().unwrap(), [0.5]);

        let e = Tensor::zeros(&[0, 3], Dtype::F32, CPU).unwrap();
        assert_eq!(e.exp().unwrap().shape(), [0, 3]);
    }
}
