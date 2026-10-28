use bytemuck::Pod;
use half::{bf16, f16};

use crate::{
    cpu::CpuBuffer,
    device::Device,
    dtype::Dtype,
    element::Element,
    error::{Error, Result},
    layout::Layout,
    ops::ReduceOp,
};

pub fn reduce(op: ReduceOp, bytes: &[u8], layout: &Layout, dim: usize) -> Result<CpuBuffer> {
    let unsupported = || Error::UnsupportedOp {
        op: op.name(),
        dtype: layout.dtype(),
        device: Device::CPU,
    };

    if op.output_dtype(layout.dtype()).is_none() {
        return Err(unsupported());
    }

    let lanes = Lanes::new(bytes, layout, dim, layout.dtype());
    match layout.dtype() {
        Dtype::F32 => lanes.float::<f32>(op),
        Dtype::F64 => lanes.float::<f64>(op),
        Dtype::F16 => lanes.float::<f16>(op),
        Dtype::BF16 => lanes.float::<bf16>(op),
        Dtype::I8 => lanes.int::<i8>(op),
        Dtype::I16 => lanes.int::<i16>(op),
        Dtype::I32 => lanes.int::<i32>(op),
        Dtype::I64 => lanes.int::<i64>(op),
        Dtype::U8 | Dtype::Bool => lanes.int::<u8>(op),
        _ => Err(unsupported()),
    }
}

struct Lanes<'a> {
    bytes: &'a [u8],
    layout: Layout,
    dtype: Dtype,
    count: usize,
    len: usize,
}

impl<'a> Lanes<'a> {
    fn new(bytes: &'a [u8], layout: &Layout, dim: usize, dtype: Dtype) -> Self {
        // Move `dim` to the end so each output's inputs are consecutive in
        // row-major order.
        let (mut shape, mut strides) = (Vec::new(), Vec::new());
        for (i, (&n, &s)) in layout.shape().iter().zip(layout.strides()).enumerate() {
            if i != dim {
                shape.push(n);
                strides.push(s);
            }
        }
        let count = shape.iter().product();
        let len = layout.shape()[dim];
        shape.push(len);
        strides.push(layout.strides()[dim]);
        let layout = Layout::from_parts(&shape, &strides, layout.offset(), dtype)
            .expect("rank is unchanged");
        Self {
            bytes,
            layout,
            dtype,
            count,
            len,
        }
    }

    /// Gathers the input, then calls `f` on each lane.
    fn map<T: Pod, U: Pod>(&self, f: impl Fn(&[T]) -> U) -> Result<CpuBuffer> {
        let values: Vec<T> = self
            .layout
            .offsets()
            .map(|o| super::read(self.bytes, o))
            .collect();
        let out: Vec<U> = (0..self.count)
            .map(|i| f(&values[i * self.len..(i + 1) * self.len]))
            .collect();
        CpuBuffer::from_slice(&out)
    }

    fn float<T: Element + PartialOrd>(&self, op: ReduceOp) -> Result<CpuBuffer> {
        // Accumulate in f64 so the reference sum is as exact as possible.
        let sum = |lane: &[T]| lane.iter().map(|x| x.to_f64()).sum::<f64>();
        match op {
            ReduceOp::Sum => self.map(|lane: &[T]| T::from_f64(sum(lane))),
            ReduceOp::Mean => self.map(|lane: &[T]| T::from_f64(sum(lane) / lane.len() as f64)),
            _ => self.extremum::<T>(op),
        }
    }

    fn int<T: Pod + PartialOrd + Into<i64>>(&self, op: ReduceOp) -> Result<CpuBuffer> {
        match op {
            ReduceOp::Sum => {
                self.map(|lane: &[T]| lane.iter().fold(0i64, |acc, &x| acc.wrapping_add(x.into())))
            }
            _ => self.extremum::<T>(op),
        }
    }

    fn extremum<T: Pod + PartialOrd>(&self, op: ReduceOp) -> Result<CpuBuffer> {
        match op {
            ReduceOp::Max => self.map(|lane: &[T]| lane[best(lane, true)]),
            ReduceOp::Min => self.map(|lane: &[T]| lane[best(lane, false)]),
            ReduceOp::Argmax => self.map(|lane: &[T]| best(lane, true) as i64),
            ReduceOp::Argmin => self.map(|lane: &[T]| best(lane, false) as i64),
            ReduceOp::Sum | ReduceOp::Mean => unreachable!("handled by the caller"),
        }
    }
}

/// Index of the first maximum (or minimum) of a non-empty lane. The first NaN,
/// if any, wins, so NaN propagates through max/min.
fn best<T: PartialOrd>(lane: &[T], greater: bool) -> usize {
    #[allow(clippy::eq_op)] // NaN is the only value not equal to itself
    let is_nan = |x: &T| x != x;
    let mut best = 0;
    for (i, x) in lane.iter().enumerate().skip(1) {
        if is_nan(&lane[best]) {
            break;
        }
        let better = if greater {
            x > &lane[best]
        } else {
            x < &lane[best]
        };
        if is_nan(x) || better {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod test {
    use half::f16;

    use crate::{
        device::Device, dtype::Dtype, error::Error, layout::Layout, ops::ReduceOp, tensor::Tensor,
    };

    const CPU: Device = Device::CPU;

    /// [[0, 1, 2], [3, 4, 5]] as f32.
    fn matrix() -> Tensor {
        Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], CPU).unwrap()
    }

    #[test]
    fn sum_and_mean() {
        let m = matrix();
        let s = m.sum(0, false).unwrap();
        assert_eq!(s.shape(), [3]);
        assert_eq!(s.to_vec::<f32>().unwrap(), [3.0, 5.0, 7.0]);

        let s = m.sum(1, true).unwrap();
        assert_eq!(s.shape(), [2, 1]);
        assert_eq!(s.to_vec::<f32>().unwrap(), [3.0, 12.0]);

        let s = m.mean(1, false).unwrap();
        assert_eq!(s.to_vec::<f32>().unwrap(), [1.0, 4.0]);

        let h: Vec<f16> = (0..4).map(|i| f16::from_f32(i as f32 * 0.5)).collect();
        let h = Tensor::from_slice(&h, &[4], CPU).unwrap();
        assert_eq!(
            h.sum(0, false).unwrap().to_vec::<f16>().unwrap(),
            [f16::from_f32(3.0)]
        );
    }

    #[test]
    fn middle_dim_of_3d() {
        // [2, 3, 2] with values 0..12, reduce dim 1.
        let t = Tensor::arange(0.0, 12.0, 1.0, Dtype::F64, CPU).unwrap();
        let t = t.with_layout(Layout::contiguous(&[2, 3, 2], Dtype::F64).unwrap());
        let s = t.sum(1, false).unwrap();
        assert_eq!(s.shape(), [2, 2]);
        assert_eq!(s.to_vec::<f64>().unwrap(), [6.0, 9.0, 24.0, 27.0]);
        assert_eq!(t.argmax(1, true).unwrap().shape(), [2, 1, 2]);
    }

    #[test]
    fn int_and_bool_sum_into_i64() {
        let t = Tensor::from_slice(&[i32::MAX, i32::MAX, 2], &[3], CPU).unwrap();
        let s = t.sum(0, false).unwrap();
        assert_eq!(s.dtype(), Dtype::I64);
        assert_eq!(s.to_vec::<i64>().unwrap(), [2 * i32::MAX as i64 + 2]);

        let b = Tensor::from_slice(&[1.0f32, -1.0, 2.0], &[3], CPU).unwrap();
        let positive = b.gt(&Tensor::zeros(&[], Dtype::F32, CPU).unwrap()).unwrap();
        assert_eq!(
            positive.sum(0, false).unwrap().to_vec::<i64>().unwrap(),
            [2]
        );

        let u = Tensor::from_slice(&[200u8, 200], &[2], CPU).unwrap();
        assert_eq!(u.sum(0, false).unwrap().to_vec::<i64>().unwrap(), [400]);
    }

    #[test]
    fn max_min_arg() {
        let t = Tensor::from_slice(&[3i32, 7, 7, -1, 0, -1], &[2, 3], CPU).unwrap();
        assert_eq!(t.max(1, false).unwrap().to_vec::<i32>().unwrap(), [7, 0]);
        assert_eq!(t.min(1, false).unwrap().to_vec::<i32>().unwrap(), [3, -1]);
        // Ties pick the first index.
        assert_eq!(t.argmax(1, false).unwrap().to_vec::<i64>().unwrap(), [1, 1]);
        assert_eq!(t.argmin(1, false).unwrap().to_vec::<i64>().unwrap(), [0, 0]);
        assert_eq!(
            t.argmax(0, false).unwrap().to_vec::<i64>().unwrap(),
            [0, 0, 0]
        );
    }

    #[test]
    fn nan_propagates() {
        let t = Tensor::from_slice(&[1.0f32, f32::NAN, 5.0, f32::NAN], &[4], CPU).unwrap();
        assert!(t.max(0, false).unwrap().to_vec::<f32>().unwrap()[0].is_nan());
        assert!(t.min(0, false).unwrap().to_vec::<f32>().unwrap()[0].is_nan());
        assert_eq!(t.argmax(0, false).unwrap().to_vec::<i64>().unwrap(), [1]);
        assert_eq!(t.argmin(0, false).unwrap().to_vec::<i64>().unwrap(), [1]);
    }

    #[test]
    fn strided_input() {
        // Transpose of [[0, 1, 2], [3, 4, 5]]: [[0, 3], [1, 4], [2, 5]].
        let t = matrix().with_layout(Layout::from_parts(&[3, 2], &[4, 12], 0, Dtype::F32).unwrap());
        assert_eq!(
            t.sum(1, false).unwrap().to_vec::<f32>().unwrap(),
            [3.0, 5.0, 7.0]
        );
        assert_eq!(t.argmax(0, false).unwrap().to_vec::<i64>().unwrap(), [2, 2]);
    }

    #[test]
    fn empty_dims() {
        let e = Tensor::zeros(&[2, 0], Dtype::F32, CPU).unwrap();
        assert_eq!(
            e.sum(1, false).unwrap().to_vec::<f32>().unwrap(),
            [0.0, 0.0]
        );
        assert!(e.mean(1, false).unwrap().to_vec::<f32>().unwrap()[0].is_nan());
        assert!(matches!(
            e.max(1, false),
            Err(Error::InvalidArgument { op: "max", .. })
        ));

        // Reducing a non-empty dim of an empty tensor gives an empty result.
        let s = e.sum(0, false).unwrap();
        assert_eq!(s.shape(), [0]);
    }

    #[test]
    fn errors() {
        let m = matrix();
        assert!(matches!(
            m.sum(2, false),
            Err(Error::InvalidArgument { op: "sum", .. })
        ));

        let s = Tensor::full(&[], 1.0, Dtype::F32, CPU).unwrap();
        assert!(matches!(
            s.sum(0, false),
            Err(Error::InvalidArgument { .. })
        ));

        let i = Tensor::from_slice(&[1i32, 2], &[2], CPU).unwrap();
        assert!(matches!(
            i.mean(0, false),
            Err(Error::UnsupportedOp {
                op: "mean",
                dtype: Dtype::I32,
                ..
            })
        ));

        let q = Tensor::zeros(&[2, 32], Dtype::Q8_0, CPU).unwrap();
        assert!(matches!(
            q.reduce(ReduceOp::Sum, 0, false),
            Err(Error::UnsupportedOp { .. })
        ));
    }
}
