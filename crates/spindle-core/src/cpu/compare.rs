use bytemuck::Pod;
use half::{bf16, f16};

use crate::{
    cpu::CpuBuffer,
    device::Device,
    dtype::Dtype,
    error::{Error, Result},
    layout::Layout,
    ops::CmpOp,
};

pub fn compare(op: CmpOp, lhs: (&[u8], &Layout), rhs: (&[u8], &Layout)) -> Result<CpuBuffer> {
    debug_assert_eq!(lhs.1.shape(), rhs.1.shape());
    match lhs.1.dtype() {
        Dtype::F32 => cmp::<f32>(op, lhs, rhs),
        Dtype::F64 => cmp::<f64>(op, lhs, rhs),
        Dtype::F16 => cmp::<f16>(op, lhs, rhs),
        Dtype::BF16 => cmp::<bf16>(op, lhs, rhs),
        Dtype::I8 => cmp::<i8>(op, lhs, rhs),
        Dtype::I16 => cmp::<i16>(op, lhs, rhs),
        Dtype::I32 => cmp::<i32>(op, lhs, rhs),
        Dtype::I64 => cmp::<i64>(op, lhs, rhs),
        Dtype::U8 | Dtype::Bool => cmp::<u8>(op, lhs, rhs),
        _ => Err(Error::UnsupportedOp {
            op: op.name(),
            device: Device::CPU,
            dtype: lhs.1.dtype(),
        }),
    }
}

fn cmp<T: Pod + PartialOrd>(
    op: CmpOp,
    lhs: (&[u8], &Layout),
    rhs: (&[u8], &Layout),
) -> Result<CpuBuffer> {
    let f: fn(T, T) -> bool = match op {
        CmpOp::Eq => |a, b| a == b,
        CmpOp::Ne => |a, b| a != b,
        CmpOp::Lt => |a, b| a < b,
        CmpOp::Le => |a, b| a <= b,
        CmpOp::Gt => |a, b| a > b,
        CmpOp::Ge => |a, b| a >= b,
    };
    super::binary::zip_map(lhs, rhs, |a: T, b: T| f(a, b) as u8)
}

/// Picks each element from `on_true` where `cond` is non-zero and from
/// `on_false` elsewhere. `cond` is `Bool`; the other two share `dtype`. All
/// three layouts must already be broadcast to the same shape.
///
/// Elements are copied as raw bytes, so every non-quantized dtype works.
pub fn where_cond(
    (cond, cond_layout): (&[u8], &Layout),
    (on_true, true_layout): (&[u8], &Layout),
    (on_false, false_layout): (&[u8], &Layout),
    dtype: Dtype,
) -> Result<CpuBuffer> {
    debug_assert_eq!(cond_layout.shape(), true_layout.shape());
    debug_assert_eq!(cond_layout.shape(), false_layout.shape());
    if dtype.is_quantized() {
        return Err(Error::UnsupportedOp {
            op: "where_cond",
            dtype,
            device: Device::CPU,
        });
    }
    let size = dtype.type_size();
    let mut out = Vec::with_capacity(true_layout.nbytes());
    let offsets = cond_layout
        .offsets()
        .zip(true_layout.offsets())
        .zip(false_layout.offsets());
    for ((c, t), f) in offsets {
        let src = if cond[c] != 0 {
            &on_true[t..t + size]
        } else {
            &on_false[f..f + size]
        };
        out.extend_from_slice(src);
    }
    CpuBuffer::from_slice(&out)
}

#[cfg(test)]
mod test {
    use half::f16;

    use crate::{
        device::Device, dtype::Dtype, error::Error, layout::Layout, ops::CmpOp, tensor::Tensor,
    };

    const CPU: Device = Device::CPU;

    fn bools(t: &Tensor) -> Vec<u8> {
        assert_eq!(t.dtype(), Dtype::Bool);
        t.to_vec::<u8>().unwrap()
    }

    #[test]
    fn float_comparisons() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, f32::NAN], &[4], CPU).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 2.0, 2.0, f32::NAN], &[4], CPU).unwrap();
        assert_eq!(bools(&a.eq(&b).unwrap()), [0, 1, 0, 0]);
        assert_eq!(bools(&a.ne(&b).unwrap()), [1, 0, 1, 1]);
        assert_eq!(bools(&a.lt(&b).unwrap()), [1, 0, 0, 0]);
        assert_eq!(bools(&a.le(&b).unwrap()), [1, 1, 0, 0]);
        assert_eq!(bools(&a.gt(&b).unwrap()), [0, 0, 1, 0]);
        assert_eq!(bools(&a.ge(&b).unwrap()), [0, 1, 1, 0]);
    }

    #[test]
    fn other_dtypes() {
        let a = Tensor::from_slice(&[f16::from_f32(-1.0), f16::ONE], &[2], CPU).unwrap();
        let b = Tensor::from_slice(&[f16::ZERO], &[1], CPU).unwrap();
        assert_eq!(bools(&a.gt(&b).unwrap()), [0, 1]);

        let a = Tensor::from_slice(&[-5i64, 7], &[2], CPU).unwrap();
        let b = Tensor::from_slice(&[7i64, 7], &[2], CPU).unwrap();
        assert_eq!(bools(&a.lt(&b).unwrap()), [1, 0]);

        // Bool compares as 0 < 1.
        let t = Tensor::full(&[2], 1.0, Dtype::Bool, CPU).unwrap();
        let f = Tensor::zeros(&[2], Dtype::Bool, CPU).unwrap();
        assert_eq!(bools(&f.lt(&t).unwrap()), [1, 1]);
        assert_eq!(bools(&t.eq(&t).unwrap()), [1, 1]);
    }

    #[test]
    fn comparison_broadcasts() {
        // Column [0, 1, 2] vs row [0, 1, 2]: lower-triangular mask without the
        // diagonal, the shape of a causal attention mask.
        let col = Tensor::arange(0.0, 3.0, 1.0, Dtype::I32, CPU).unwrap();
        let col = col.with_layout(Layout::contiguous(&[3, 1], Dtype::I32).unwrap());
        let row = Tensor::arange(0.0, 3.0, 1.0, Dtype::I32, CPU).unwrap();
        let mask = row.gt(&col).unwrap();
        assert_eq!(mask.shape(), [3, 3]);
        assert_eq!(bools(&mask), [0, 1, 1, 0, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn comparison_errors() {
        let f = Tensor::from_slice(&[1.0f32], &[1], CPU).unwrap();
        let i = Tensor::from_slice(&[1i32], &[1], CPU).unwrap();
        assert!(matches!(
            f.eq(&i),
            Err(Error::DtypeMismatch { op: "eq", .. })
        ));

        let g = Tensor::from_slice(&[1.0f32, 2.0], &[2], CPU).unwrap();
        let h = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], CPU).unwrap();
        assert!(matches!(
            g.lt(&h),
            Err(Error::ShapeMismatch { op: "lt", .. })
        ));

        let q = Tensor::zeros(&[32], Dtype::Q4_0, CPU).unwrap();
        assert!(matches!(
            q.compare(&q, CmpOp::Eq),
            Err(Error::UnsupportedOp { .. })
        ));
    }

    #[test]
    fn where_selects() {
        let cond = Tensor::from_slice(&[1u8, 0, 1], &[3], CPU).unwrap();
        let cond = cond
            .ne(&Tensor::zeros(&[], Dtype::U8, CPU).unwrap())
            .unwrap();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], CPU).unwrap();
        let y = Tensor::from_slice(&[-1.0f32, -2.0, -3.0], &[3], CPU).unwrap();
        let out = cond.where_cond(&x, &y).unwrap();
        assert_eq!(out.dtype(), Dtype::F32);
        assert_eq!(out.to_vec::<f32>().unwrap(), [1.0, -2.0, 3.0]);
    }

    #[test]
    fn where_broadcasts() {
        // Masking scores with -inf above the diagonal, as in causal attention.
        let idx = Tensor::arange(0.0, 3.0, 1.0, Dtype::I64, CPU).unwrap();
        let col = idx.with_layout(Layout::contiguous(&[3, 1], Dtype::I64).unwrap());
        let keep = idx.le(&col).unwrap();
        let scores = Tensor::full(&[2, 3, 3], 1.0, Dtype::F32, CPU).unwrap();
        let neg_inf = Tensor::full(&[], f64::NEG_INFINITY, Dtype::F32, CPU).unwrap();
        let out = keep.where_cond(&scores, &neg_inf).unwrap();
        assert_eq!(out.shape(), [2, 3, 3]);
        let inf = f32::NEG_INFINITY;
        let one_head = [1.0, inf, inf, 1.0, 1.0, inf, 1.0, 1.0, 1.0];
        assert_eq!(out.to_vec::<f32>().unwrap(), [one_head, one_head].concat());
    }

    #[test]
    fn where_strided_and_int() {
        // on_true is a transposed view.
        let t = Tensor::arange(0.0, 4.0, 1.0, Dtype::I16, CPU).unwrap();
        let tt = t.with_layout(Layout::from_parts(&[2, 2], &[2, 4], 0, Dtype::I16).unwrap());
        let cond = Tensor::full(&[2, 2], 1.0, Dtype::Bool, CPU).unwrap();
        let zero = Tensor::zeros(&[], Dtype::I16, CPU).unwrap();
        let out = cond.where_cond(&tt, &zero).unwrap();
        assert_eq!(out.to_vec::<i16>().unwrap(), [0, 2, 1, 3]);
    }

    #[test]
    fn where_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], &[2], CPU).unwrap();
        let not_bool = Tensor::from_slice(&[1u8, 0], &[2], CPU).unwrap();
        assert!(matches!(
            not_bool.where_cond(&x, &x),
            Err(Error::DtypeMismatch {
                op: "where_cond",
                lhs: Dtype::U8,
                rhs: Dtype::Bool
            })
        ));

        let cond = Tensor::full(&[2], 1.0, Dtype::Bool, CPU).unwrap();
        let y = Tensor::from_slice(&[1.0f64, 2.0], &[2], CPU).unwrap();
        assert!(matches!(
            cond.where_cond(&x, &y),
            Err(Error::DtypeMismatch {
                op: "where_cond",
                lhs: Dtype::F32,
                rhs: Dtype::F64
            })
        ));

        let z = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], CPU).unwrap();
        assert!(matches!(
            cond.where_cond(&x, &z),
            Err(Error::ShapeMismatch {
                op: "where_cond",
                ..
            })
        ));

        let q = Tensor::zeros(&[2, 32], Dtype::Q8_0, CPU).unwrap();
        let cond = Tensor::full(&[2, 32], 1.0, Dtype::Bool, CPU).unwrap();
        assert!(matches!(
            cond.where_cond(&q, &q),
            Err(Error::UnsupportedOp { .. })
        ));
    }
}
