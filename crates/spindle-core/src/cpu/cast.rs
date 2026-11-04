use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};

use super::{CpuBuffer, read};
use crate::{
    device::Device,
    dtype::Dtype,
    error::{Error, Result},
    layout::Layout,
};

/// Converts every element of the tensor described by `bytes` and `layout` from
/// `src` to `dst`, producing a contiguous buffer.
///
/// Floats convert through `f64` and integers through `i64`, with `as`
/// semantics: float → int truncates toward zero and saturates (NaN becomes 0),
/// int → int wraps, and anything → `Bool` is `x != 0`.
pub fn cast(bytes: &[u8], layout: &Layout, src: Dtype, dst: Dtype) -> Result<CpuBuffer> {
    let input = Input { bytes, layout, src };
    match src {
        Dtype::F32 => input.to::<f32>(dst),
        Dtype::F64 => input.to::<f64>(dst),
        Dtype::F16 => input.to::<f16>(dst),
        Dtype::BF16 => input.to::<bf16>(dst),
        Dtype::I8 => input.to::<i8>(dst),
        Dtype::I16 => input.to::<i16>(dst),
        Dtype::I32 => input.to::<i32>(dst),
        Dtype::I64 => input.to::<i64>(dst),
        Dtype::U8 => input.to::<u8>(dst),
        Dtype::Bool => input.to::<BoolByte>(dst),
        _ => Err(unsupported(src)),
    }
}

fn unsupported(dtype: Dtype) -> Error {
    Error::UnsupportedOp {
        op: "cast",
        dtype,
        device: Device::CPU,
    }
}

struct Input<'a> {
    bytes: &'a [u8],
    layout: &'a Layout,
    src: Dtype,
}

impl Input<'_> {
    fn to<S: Scalar>(&self, dst: Dtype) -> Result<CpuBuffer> {
        match dst {
            Dtype::F32 => self.convert::<S, f32>(),
            Dtype::F64 => self.convert::<S, f64>(),
            Dtype::F16 => self.convert::<S, f16>(),
            Dtype::BF16 => self.convert::<S, bf16>(),
            Dtype::I8 => self.convert::<S, i8>(),
            Dtype::I16 => self.convert::<S, i16>(),
            Dtype::I32 => self.convert::<S, i32>(),
            Dtype::I64 => self.convert::<S, i64>(),
            Dtype::U8 => self.convert::<S, u8>(),
            Dtype::Bool => self.convert::<S, BoolByte>(),
            _ => Err(unsupported(dst)),
        }
    }

    fn convert<S: Scalar, D: Scalar>(&self) -> Result<CpuBuffer> {
        let out: Vec<D> = self
            .layout
            .offsets()
            .map(|o| D::from_value(read::<S>(self.bytes, o).to_value()))
            .collect();
        CpuBuffer::from_slice(&out)
    }
}

/// An element widened to the widest type of its kind.
#[derive(Clone, Copy)]
enum Value {
    Float(f64),
    Int(i64),
}

trait Scalar: Pod {
    fn to_value(self) -> Value;
    fn from_value(v: Value) -> Self;
}

macro_rules! float_scalar {
    ($($t:ty: $to:expr, $from_f64:expr, $from_i64:expr);* $(;)?) => {$(
        impl Scalar for $t {
            fn to_value(self) -> Value {
                Value::Float($to(self))
            }
            fn from_value(v: Value) -> Self {
                match v {
                    Value::Float(x) => $from_f64(x),
                    Value::Int(x) => $from_i64(x),
                }
            }
        }
    )*};
}

float_scalar! {
    f32: |x: f32| x as f64, |x: f64| x as f32, |x: i64| x as f32;
    f64: |x: f64| x, |x: f64| x, |x: i64| x as f64;
    f16: |x: f16| x.to_f64(), f16::from_f64, |x: i64| f16::from_f64(x as f64);
    bf16: |x: bf16| x.to_f64(), bf16::from_f64, |x: i64| bf16::from_f64(x as f64);
}

macro_rules! int_scalar {
    ($($t:ty),*) => {$(
        impl Scalar for $t {
            fn to_value(self) -> Value {
                Value::Int(self as i64)
            }
            fn from_value(v: Value) -> Self {
                match v {
                    Value::Float(x) => x as $t,
                    Value::Int(x) => x as $t,
                }
            }
        }
    )*};
}

int_scalar!(i8, i16, i32, i64, u8);

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
struct BoolByte(u8);

impl Scalar for BoolByte {
    fn to_value(self) -> Value {
        Value::Int(self.0 as i64)
    }
    fn from_value(v: Value) -> Self {
        BoolByte(match v {
            Value::Float(x) => x != 0.0,
            Value::Int(x) => x != 0,
        } as u8)
    }
}

#[cfg(test)]
mod test {
    use half::{bf16, f16};

    use crate::{device::Device, dtype::Dtype, error::Error, layout::Layout, tensor::Tensor};

    const CPU: Device = Device::CPU;

    #[test]
    fn float_to_int() {
        let t = Tensor::from_slice(
            &[2.9f32, -2.9, 0.5, f32::NAN, 1e10, -1e10, f32::INFINITY],
            &[7],
            CPU,
        )
        .unwrap();
        let i = t.cast(Dtype::I32).unwrap();
        assert_eq!(i.dtype(), Dtype::I32);
        assert_eq!(
            i.to_vec::<i32>().unwrap(),
            [2, -2, 0, 0, i32::MAX, i32::MIN, i32::MAX]
        );
        assert_eq!(
            t.cast(Dtype::U8).unwrap().to_vec::<u8>().unwrap(),
            [2, 0, 0, 0, 255, 0, 255]
        );
    }

    #[test]
    fn int_to_int_wraps() {
        let t = Tensor::from_slice(&[300i64, -129, i64::MAX], &[3], CPU).unwrap();
        assert_eq!(
            t.cast(Dtype::I8).unwrap().to_vec::<i8>().unwrap(),
            [44, 127, -1]
        );

        let u = Tensor::from_slice(&[200u8, 255], &[2], CPU).unwrap();
        assert_eq!(
            u.cast(Dtype::I8).unwrap().to_vec::<i8>().unwrap(),
            [-56, -1]
        );
        assert_eq!(
            u.cast(Dtype::I64).unwrap().to_vec::<i64>().unwrap(),
            [200, 255]
        );

        // Large i64 values survive i64 → i64-sized paths exactly.
        let big = Tensor::from_slice(&[(1i64 << 62) + 1], &[1], CPU).unwrap();
        let back = big.cast(Dtype::I32).unwrap().cast(Dtype::I64).unwrap();
        assert_eq!(back.to_vec::<i64>().unwrap(), [1]);
    }

    #[test]
    fn int_to_float() {
        let t = Tensor::from_slice(&[-3i32, 0, 16_777_217], &[3], CPU).unwrap();
        assert_eq!(
            t.cast(Dtype::F64).unwrap().to_vec::<f64>().unwrap(),
            [-3.0, 0.0, 16_777_217.0]
        );
        // 2^24 + 1 is not representable in f32 and rounds to even.
        assert_eq!(
            t.cast(Dtype::F32).unwrap().to_vec::<f32>().unwrap(),
            [-3.0, 0.0, 16_777_216.0]
        );
    }

    #[test]
    fn between_floats() {
        let t = Tensor::from_slice(&[1.0f32 / 3.0, 65_520.0, 1e-8], &[3], CPU).unwrap();
        let h = t.cast(Dtype::F16).unwrap().to_vec::<f16>().unwrap();
        assert_eq!(h[0], f16::from_f32(1.0 / 3.0));
        assert!(h[1].is_infinite()); // above f16::MAX after rounding
        assert_eq!(h[2], f16::from_f32(1e-8));

        let b = t.cast(Dtype::BF16).unwrap().cast(Dtype::F64).unwrap();
        let expected = bf16::from_f32(1.0 / 3.0).to_f64();
        assert_eq!(b.to_vec::<f64>().unwrap()[0], expected);
    }

    #[test]
    fn bool_conversions() {
        let t = Tensor::from_slice(&[0.0f32, -0.0, 2.5, f32::NAN], &[4], CPU).unwrap();
        let b = t.cast(Dtype::Bool).unwrap();
        assert_eq!(b.dtype(), Dtype::Bool);
        assert_eq!(b.to_vec::<u8>().unwrap(), [0, 0, 1, 1]);
        assert_eq!(
            b.cast(Dtype::F32).unwrap().to_vec::<f32>().unwrap(),
            [0.0, 0.0, 1.0, 1.0]
        );

        let i = Tensor::from_slice(&[0i64, 256, -1], &[3], CPU).unwrap();
        // 256 is non-zero, even though its low byte is 0.
        assert_eq!(
            i.cast(Dtype::Bool).unwrap().to_vec::<u8>().unwrap(),
            [0, 1, 1]
        );
    }

    #[test]
    fn strided_and_same_dtype() {
        // Transpose of [[0, 1, 2], [3, 4, 5]].
        let t = Tensor::arange(0.0, 6.0, 1.0, Dtype::F32, CPU).unwrap();
        let tt = t.with_layout(Layout::from_parts(&[3, 2], &[4, 12], 0, Dtype::F32).unwrap());
        let c = tt.cast(Dtype::I16).unwrap();
        assert!(c.layout().is_contiguous());
        assert_eq!(c.to_vec::<i16>().unwrap(), [0, 3, 1, 4, 2, 5]);

        // Same dtype keeps the view as is.
        let same = tt.cast(Dtype::F32).unwrap();
        assert_eq!(same.layout(), tt.layout());
        assert_eq!(
            same.to_vec::<f32>().unwrap(),
            [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn quantized_unsupported() {
        let q = Tensor::zeros(&[32], Dtype::Q8_0, CPU).unwrap();
        assert!(matches!(
            q.cast(Dtype::F32),
            Err(Error::UnsupportedOp {
                op: "cast",
                dtype: Dtype::Q8_0,
                ..
            })
        ));
        let f = Tensor::zeros(&[32], Dtype::F32, CPU).unwrap();
        assert!(matches!(
            f.cast(Dtype::Q4_K),
            Err(Error::UnsupportedOp { op: "cast", .. })
        ));
    }
}
