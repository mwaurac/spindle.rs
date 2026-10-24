//! Reference kernels for elementwise binary ops.

use bytemuck::Pod;
use half::{bf16, f16};

use super::{
    CpuBuffer,
    num::{Int, Real},
    read,
};
use crate::{
    device::Device,
    dtype::Dtype,
    error::{Error, Result},
    layout::Layout,
    ops::BinaryOp,
};

/// Applies `op` elementwise to two tensors of the same dtype, producing a
/// contiguous buffer. Both layouts must already be broadcast to the same shape.
pub fn binary(op: BinaryOp, lhs: (&[u8], &Layout), rhs: (&[u8], &Layout)) -> Result<CpuBuffer> {
    debug_assert_eq!(lhs.1.shape(), rhs.1.shape());
    let unsupported = || Error::UnsupportedOp {
        op: op.name(),
        dtype: lhs.1.dtype(),
        device: Device::CPU,
    };
    match lhs.1.dtype() {
        Dtype::F32 => zip_map(lhs, rhs, float_fn::<f32>(op)),
        Dtype::F64 => zip_map(lhs, rhs, float_fn::<f64>(op)),
        Dtype::F16 => {
            let f = float_fn::<f32>(op);
            zip_map(lhs, rhs, |a: f16, b: f16| {
                f16::from_f32(f(a.to_f32(), b.to_f32()))
            })
        }
        Dtype::BF16 => {
            let f = float_fn::<f32>(op);
            zip_map(lhs, rhs, |a: bf16, b: bf16| {
                bf16::from_f32(f(a.to_f32(), b.to_f32()))
            })
        }
        Dtype::I8 => zip_map(lhs, rhs, int_fn::<i8>(op).ok_or_else(unsupported)?),
        Dtype::I16 => zip_map(lhs, rhs, int_fn::<i16>(op).ok_or_else(unsupported)?),
        Dtype::I32 => zip_map(lhs, rhs, int_fn::<i32>(op).ok_or_else(unsupported)?),
        Dtype::I64 => zip_map(lhs, rhs, int_fn::<i64>(op).ok_or_else(unsupported)?),
        Dtype::U8 => zip_map(lhs, rhs, int_fn::<u8>(op).ok_or_else(unsupported)?),
        _ => Err(unsupported()),
    }
}

fn zip_map<T: Pod>(
    (lhs, lhs_layout): (&[u8], &Layout),
    (rhs, rhs_layout): (&[u8], &Layout),
    f: impl Fn(T, T) -> T,
) -> Result<CpuBuffer> {
    let out: Vec<T> = lhs_layout
        .offsets()
        .zip(rhs_layout.offsets())
        .map(|(a, b)| f(read(lhs, a), read(rhs, b)))
        .collect();
    CpuBuffer::from_slice(&out)
}

fn float_fn<F: Real>(op: BinaryOp) -> fn(F, F) -> F {
    match op {
        BinaryOp::Add => |a: F, b: F| a + b,
        BinaryOp::Sub => |a: F, b: F| a - b,
        BinaryOp::Mul => |a: F, b: F| a * b,
        BinaryOp::Div => |a: F, b: F| a / b,
        BinaryOp::Pow => |a: F, b: F| a.powf(b),
        // `a > NaN` and `a < NaN` are false, so a NaN `b` is returned too.
        BinaryOp::Maximum => |a: F, b: F| if a.is_nan() || a > b { a } else { b },
        BinaryOp::Minimum => |a: F, b: F| if a.is_nan() || a < b { a } else { b },
    }
}

/// The integer version of `op`, or `None` if it is float-only.
fn int_fn<I: Int>(op: BinaryOp) -> Option<fn(I, I) -> I> {
    Some(match op {
        BinaryOp::Add => |a: I, b: I| a.wrapping_add(b),
        BinaryOp::Sub => |a: I, b: I| a.wrapping_sub(b),
        BinaryOp::Mul => |a: I, b: I| a.wrapping_mul(b),
        BinaryOp::Maximum => |a: I, b: I| a.max(b),
        BinaryOp::Minimum => |a: I, b: I| a.min(b),
        BinaryOp::Div | BinaryOp::Pow => return None,
    })
}

#[cfg(test)]
mod test {
    use half::{bf16, f16};

    use crate::{
        device::Device, dtype::Dtype, error::Error, layout::Layout, ops::BinaryOp, tensor::Tensor,
    };

    const CPU: Device = Device::CPU;
    const LHS: [f64; 6] = [-2.0, -0.5, 0.0, 0.5, 1.5, 3.0];
    const RHS: [f64; 6] = [3.0, 2.0, -1.0, 0.25, 1.5, -2.0];

    const ALL_OPS: [BinaryOp; 7] = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Pow,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
    ];

    fn reference(op: BinaryOp, a: f64, b: f64) -> f64 {
        match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Pow => a.powf(b),
            BinaryOp::Maximum => a.max(b),
            BinaryOp::Minimum => a.min(b),
        }
    }

    fn assert_close(op: BinaryOp, got: f64, want: f64, tol: f64) {
        let ok = (got.is_nan() && want.is_nan())
            || got == want
            || (got - want).abs() <= tol * want.abs().max(1.0);
        assert!(ok, "{}: got {got}, want {want}", op.name());
    }

    fn tensor<T: crate::element::Element>(f: impl Fn(f64) -> T, data: &[f64]) -> Tensor {
        let data: Vec<T> = data.iter().map(|&x| f(x)).collect();
        Tensor::from_slice(&data, &[data.len()], CPU).unwrap()
    }

    #[test]
    fn float_ops_match_reference() {
        let pairs = [
            (tensor(|x| x as f32, &LHS), tensor(|x| x as f32, &RHS)),
            (tensor(|x| x, &LHS), tensor(|x| x, &RHS)),
            (tensor(f16::from_f64, &LHS), tensor(f16::from_f64, &RHS)),
            (tensor(bf16::from_f64, &LHS), tensor(bf16::from_f64, &RHS)),
        ];
        for op in ALL_OPS {
            for (a, b) in &pairs {
                let out = a.binary(b, op).unwrap();
                let (got, tol): (Vec<f64>, f64) = match out.dtype() {
                    Dtype::F32 => (
                        out.to_vec::<f32>()
                            .unwrap()
                            .iter()
                            .map(|&x| x as f64)
                            .collect(),
                        1e-6,
                    ),
                    Dtype::F64 => (out.to_vec::<f64>().unwrap(), 1e-12),
                    Dtype::F16 => (
                        out.to_vec::<f16>()
                            .unwrap()
                            .iter()
                            .map(|x| x.to_f64())
                            .collect(),
                        1e-3,
                    ),
                    Dtype::BF16 => (
                        out.to_vec::<bf16>()
                            .unwrap()
                            .iter()
                            .map(|x| x.to_f64())
                            .collect(),
                        1e-2,
                    ),
                    _ => unreachable!(),
                };
                for i in 0..LHS.len() {
                    assert_close(op, got[i], reference(op, LHS[i], RHS[i]), tol);
                }
            }
        }
    }

    #[test]
    fn broadcasting() {
        // [2, 3] + [3]
        let a = Tensor::arange(0.0, 6.0, 1.0, Dtype::F32, CPU).unwrap();
        let a = a.with_layout(Layout::contiguous(&[2, 3], Dtype::F32).unwrap());
        let row = Tensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], CPU).unwrap();
        let out = a.add(&row).unwrap();
        assert_eq!(out.shape(), [2, 3]);
        assert_eq!(
            out.to_vec::<f32>().unwrap(),
            [10.0, 21.0, 32.0, 13.0, 24.0, 35.0]
        );

        // [2, 1] * [1, 3]: outer product.
        let col = Tensor::from_slice(&[1i32, 2], &[2, 1], CPU).unwrap();
        let row = Tensor::from_slice(&[1i32, 10, 100], &[1, 3], CPU).unwrap();
        let out = col.mul(&row).unwrap();
        assert_eq!(out.shape(), [2, 3]);
        assert_eq!(out.to_vec::<i32>().unwrap(), [1, 10, 100, 2, 20, 200]);

        // Scalar on either side.
        let s = Tensor::full(&[], 2.0, Dtype::F32, CPU).unwrap();
        let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], CPU).unwrap();
        assert_eq!(s.pow(&v).unwrap().to_vec::<f32>().unwrap(), [2.0, 4.0, 8.0]);
        assert_eq!(v.pow(&s).unwrap().to_vec::<f32>().unwrap(), [1.0, 4.0, 9.0]);

        // Empty dims broadcast like any other.
        let e = Tensor::zeros(&[0, 3], Dtype::F32, CPU).unwrap();
        assert_eq!(e.sub(&v).unwrap().shape(), [0, 3]);
    }

    #[test]
    fn strided_inputs() {
        // [[0, 1, 2], [3, 4, 5]] transposed to [3, 2], plus a [2] row.
        let t = Tensor::arange(0.0, 6.0, 1.0, Dtype::F64, CPU).unwrap();
        let tt = t.with_layout(Layout::from_parts(&[3, 2], &[8, 24], 0, Dtype::F64).unwrap());
        let row = Tensor::from_slice(&[100.0f64, 200.0], &[2], CPU).unwrap();
        let out = tt.add(&row).unwrap();
        assert!(out.layout().is_contiguous());
        assert_eq!(
            out.to_vec::<f64>().unwrap(),
            [100.0, 203.0, 101.0, 204.0, 102.0, 205.0]
        );
    }

    #[test]
    fn int_ops_wrap() {
        let a = Tensor::from_slice(&[127i8, -128, 5], &[3], CPU).unwrap();
        let b = Tensor::from_slice(&[1i8, 1, -7], &[3], CPU).unwrap();
        assert_eq!(a.add(&b).unwrap().to_vec::<i8>().unwrap(), [-128, -127, -2]);
        assert_eq!(a.sub(&b).unwrap().to_vec::<i8>().unwrap(), [126, 127, 12]);
        assert_eq!(a.mul(&b).unwrap().to_vec::<i8>().unwrap(), [127, -128, -35]);
        assert_eq!(a.maximum(&b).unwrap().to_vec::<i8>().unwrap(), [127, 1, 5]);
        assert_eq!(
            a.minimum(&b).unwrap().to_vec::<i8>().unwrap(),
            [1, -128, -7]
        );

        let a = Tensor::from_slice(&[0u8, 200], &[2], CPU).unwrap();
        let b = Tensor::from_slice(&[1u8, 100], &[2], CPU).unwrap();
        assert_eq!(a.sub(&b).unwrap().to_vec::<u8>().unwrap(), [255, 100]);
        assert_eq!(a.add(&b).unwrap().to_vec::<u8>().unwrap(), [1, 44]);
    }

    #[test]
    fn maximum_minimum_propagate_nan() {
        let a = Tensor::from_slice(&[f32::NAN, 1.0, 2.0], &[3], CPU).unwrap();
        let b = Tensor::from_slice(&[1.0f32, f32::NAN, 3.0], &[3], CPU).unwrap();
        for out in [a.maximum(&b).unwrap(), a.minimum(&b).unwrap()] {
            let v = out.to_vec::<f32>().unwrap();
            assert!(v[0].is_nan() && v[1].is_nan());
        }
        assert_eq!(a.maximum(&b).unwrap().to_vec::<f32>().unwrap()[2], 3.0);
        assert_eq!(a.minimum(&b).unwrap().to_vec::<f32>().unwrap()[2], 2.0);
    }

    #[test]
    fn errors() {
        let f = Tensor::from_slice(&[1.0f32, 2.0], &[2], CPU).unwrap();
        let i = Tensor::from_slice(&[1i32, 2], &[2], CPU).unwrap();
        assert!(matches!(
            f.add(&i),
            Err(Error::DtypeMismatch {
                op: "add",
                lhs: Dtype::F32,
                rhs: Dtype::I32
            })
        ));

        let g = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], CPU).unwrap();
        assert!(matches!(
            f.mul(&g),
            Err(Error::ShapeMismatch { op: "mul", .. })
        ));

        assert!(matches!(
            i.div(&i),
            Err(Error::UnsupportedOp {
                op: "div",
                dtype: Dtype::I32,
                ..
            })
        ));
        assert!(matches!(
            i.pow(&i),
            Err(Error::UnsupportedOp { op: "pow", .. })
        ));

        let q = Tensor::zeros(&[32], Dtype::Q8_0, CPU).unwrap();
        assert!(matches!(q.add(&q), Err(Error::UnsupportedOp { .. })));
    }
}
