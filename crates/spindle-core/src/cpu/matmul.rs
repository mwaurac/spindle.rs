use crate::{
    cpu::CpuBuffer,
    device::Device,
    dtype::Dtype,
    element::Element,
    error::{Error, Result},
    layout::Layout,
};

pub fn matmul(lhs: (&[u8], &Layout), rhs: (&[u8], &Layout), dtype: Dtype) -> Result<CpuBuffer> {
    match dtype {
        Dtype::F32 => matmul_typed::<f32>(lhs, rhs),
        Dtype::F64 => matmul_typed::<f64>(lhs, rhs),
        Dtype::F16 => matmul_typed::<half::f16>(lhs, rhs),
        Dtype::BF16 => matmul_typed::<half::bf16>(lhs, rhs),
        _ => Err(Error::UnsupportedOp {
            op: "matmul",
            dtype,
            device: Device::CPU,
        }),
    }
}

fn matmul_typed<T: Element>(
    (lhs, lhs_layout): (&[u8], &Layout),
    (rhs, rhs_layout): (&[u8], &Layout),
) -> Result<CpuBuffer> {
    let nd = lhs_layout.ndim();
    debug_assert_eq!(nd, rhs_layout.ndim());
    debug_assert_eq!(lhs_layout.shape()[..nd - 2], rhs_layout.shape()[..nd - 2]);
    let (m, k) = (lhs_layout.shape()[nd - 2], lhs_layout.shape()[nd - 1]);
    let n = rhs_layout.shape()[nd - 1];
    let (lhs_row, lhs_col) = (lhs_layout.strides()[nd - 2], lhs_layout.strides()[nd - 1]);
    let (rhs_row, rhs_col) = (rhs_layout.strides()[nd - 2], rhs_layout.strides()[nd - 1]);

    // Byte offset of each batch's matrix, in row-major batch order.
    let batches = |layout: &Layout| {
        Layout::from_parts(
            &layout.shape()[..nd - 2],
            &layout.strides()[..nd - 2],
            layout.offset(),
            layout.dtype(),
        )
        .expect("batch rank fits")
        .offsets()
    };

    let mut out = Vec::with_capacity(lhs_layout.numel() / k.max(1) * n);
    for (lhs_base, rhs_base) in batches(lhs_layout).zip(batches(rhs_layout)) {
        for i in 0..m {
            for j in 0..n {
                // Accumulate in f64 so the reference result is as exact as possible.
                let mut acc = 0.0f64;
                for p in 0..k {
                    let a: T = super::read(lhs, lhs_base + i * lhs_row + p * lhs_col);
                    let b: T = super::read(rhs, rhs_base + p * rhs_row + j * rhs_col);
                    acc += a.to_f64() * b.to_f64();
                }
                out.push(T::from_f64(acc));
            }
        }
    }
    CpuBuffer::from_slice(&out)
}

#[cfg(test)]
mod test {
    use half::f16;

    use crate::{device::Device, dtype::Dtype, error::Error, layout::Layout, tensor::Tensor};

    const CPU: Device = Device::CPU;

    fn range(shape: &[usize], dtype: Dtype) -> Tensor {
        let n = shape.iter().product::<usize>() as f64;
        let t = Tensor::arange(0.0, n, 1.0, dtype, CPU).unwrap();
        t.with_layout(Layout::contiguous(shape, dtype).unwrap())
    }

    /// Plain triple loop over row-major f64 data.
    fn reference(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
            }
        }
        out
    }

    #[test]
    fn two_by_two() {
        // [[0, 1, 2], [3, 4, 5]] @ [[0, 1], [2, 3], [4, 5]]
        let a = range(&[2, 3], Dtype::F32);
        let b = range(&[3, 2], Dtype::F32);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c.to_vec::<f32>().unwrap(), [10.0, 13.0, 28.0, 40.0]);
    }

    #[test]
    fn batched() {
        let a = range(&[2, 2, 3], Dtype::F64);
        let b = range(&[2, 3, 4], Dtype::F64);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), [2, 2, 4]);
        let (av, bv) = (a.to_vec::<f64>().unwrap(), b.to_vec::<f64>().unwrap());
        let expected = [
            reference(&av[..6], &bv[..12], 2, 3, 4),
            reference(&av[6..], &bv[12..], 2, 3, 4),
        ]
        .concat();
        assert_eq!(c.to_vec::<f64>().unwrap(), expected);
    }

    #[test]
    fn batch_broadcasting() {
        // Shared weight: [2, 3, 4] @ [4, 5].
        let x = range(&[2, 3, 4], Dtype::F64);
        let w = range(&[4, 5], Dtype::F64);
        let y = x.matmul(&w).unwrap();
        assert_eq!(y.shape(), [2, 3, 5]);
        let (xv, wv) = (x.to_vec::<f64>().unwrap(), w.to_vec::<f64>().unwrap());
        let expected = [
            reference(&xv[..12], &wv, 3, 4, 5),
            reference(&xv[12..], &wv, 3, 4, 5),
        ]
        .concat();
        assert_eq!(y.to_vec::<f64>().unwrap(), expected);

        // [2, 1, 2, 3] @ [3, 3, 2] → [2, 3, 2, 2].
        let a = range(&[2, 1, 2, 3], Dtype::F32);
        let b = range(&[3, 3, 2], Dtype::F32);
        assert_eq!(a.matmul(&b).unwrap().shape(), [2, 3, 2, 2]);
    }

    #[test]
    fn transposed_weight() {
        // x @ W^T with W stored as [n, k], the layout of a Linear weight.
        let x = range(&[2, 3], Dtype::F32);
        let w = range(&[4, 3], Dtype::F32);
        let wt = w.with_layout(Layout::from_parts(&[3, 4], &[4, 12], 0, Dtype::F32).unwrap());
        let y = x.matmul(&wt).unwrap();
        assert_eq!(y.shape(), [2, 4]);
        let xv: Vec<f64> = x
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let wtv: Vec<f64> = wt
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected: Vec<f32> = reference(&xv, &wtv, 2, 3, 4)
            .iter()
            .map(|&v| v as f32)
            .collect();
        assert_eq!(y.to_vec::<f32>().unwrap(), expected);
    }

    #[test]
    fn vectors() {
        let v = range(&[3], Dtype::F32);
        let m = range(&[3, 2], Dtype::F32);

        // Dot product → scalar.
        let d = v.matmul(&v).unwrap();
        assert_eq!(d.shape(), [] as [usize; 0]);
        assert_eq!(d.to_vec::<f32>().unwrap(), [5.0]);

        // [3] @ [3, 2] → [2].
        let vm = v.matmul(&m).unwrap();
        assert_eq!(vm.shape(), [2]);
        assert_eq!(vm.to_vec::<f32>().unwrap(), [10.0, 13.0]);

        // [2, 3] @ [3] → [2].
        let mv = range(&[2, 3], Dtype::F32).matmul(&v).unwrap();
        assert_eq!(mv.shape(), [2]);
        assert_eq!(mv.to_vec::<f32>().unwrap(), [5.0, 14.0]);
    }

    #[test]
    fn half_precision_and_empty() {
        let a: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let a = Tensor::from_slice(&a, &[2, 2], CPU).unwrap();
        let c = a.matmul(&a).unwrap();
        let expected: Vec<f16> = [7.0f32, 10.0, 15.0, 22.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        assert_eq!(c.to_vec::<f16>().unwrap(), expected);

        // k = 0 gives zeros, like PyTorch.
        let a = Tensor::zeros(&[2, 0], Dtype::F32, CPU).unwrap();
        let b = Tensor::zeros(&[0, 3], Dtype::F32, CPU).unwrap();
        assert_eq!(a.matmul(&b).unwrap().to_vec::<f32>().unwrap(), [0.0; 6]);
    }

    #[test]
    fn errors() {
        let a = range(&[2, 3], Dtype::F32);
        let b = range(&[2, 3], Dtype::F32);
        assert!(matches!(
            a.matmul(&b),
            Err(Error::ShapeMismatch { op: "matmul", .. })
        ));

        let a = range(&[2, 2, 3], Dtype::F32);
        let b = range(&[3, 3, 4], Dtype::F32);
        assert!(matches!(
            a.matmul(&b),
            Err(Error::ShapeMismatch { op: "matmul", .. })
        ));

        let f = range(&[2, 2], Dtype::F32);
        let d = range(&[2, 2], Dtype::F64);
        assert!(matches!(
            f.matmul(&d),
            Err(Error::DtypeMismatch { op: "matmul", .. })
        ));

        let i = range(&[2, 2], Dtype::I32);
        assert!(matches!(
            i.matmul(&i),
            Err(Error::UnsupportedOp {
                op: "matmul",
                dtype: Dtype::I32,
                ..
            })
        ));

        let s = Tensor::full(&[], 1.0, Dtype::F32, CPU).unwrap();
        assert!(matches!(
            s.matmul(&f),
            Err(Error::InvalidArgument { op: "matmul", .. })
        ));
    }
}
