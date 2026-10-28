use half::{bf16, f16};

use crate::buffer::Buffer;
use crate::cpu::{self, CpuBuffer};
use crate::dtype::Dtype;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::ops::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use std::{fmt, sync::Arc};

use crate::{device::Device, storage::Storage};

pub struct Tensor(Arc<TensorImpl_>);

struct TensorImpl_ {
    storage: Arc<Storage>, // TODO: prop change this when there are multple writers. NOT SAFE now
    layout: Layout,
    device: Device,
}

impl Tensor {
    // accessors
    pub fn device(&self) -> Device {
        self.0.device
    }

    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    pub fn dtype(&self) -> Dtype {
        self.0.layout.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        self.0.layout.shape()
    }

    pub fn strides(&self) -> &[usize] {
        self.0.layout.strides()
    }

    /// Size in bytes of the underlying storage, which views may share
    pub fn storage_len(&self) -> usize {
        self.0.storage.len()
    }
    // ctors
    pub fn from_storage(storage: Storage, layout: Layout, device: Device) -> Self {
        Tensor(Arc::new(TensorImpl_ {
            storage: Arc::new(storage),
            layout,
            device,
        }))
    }

    /// Contiguous tensor on `device` with the contents of a host buffer.
    fn from_host(data: CpuBuffer, layout: Layout, device: Device) -> Self {
        let storage = Storage::from_buffer(Buffer::from_host(device, data));
        Self::from_storage(storage, layout, device)
    }

    /// Contiguous tensor holding a copy of `data`.
    pub fn from_slice<T: Element>(data: &[T], shape: &[usize], device: Device) -> Result<Self> {
        let layout = Layout::contiguous(shape, T::DTYPE)?;
        if layout.numel() != data.len() {
            return Err(Error::ShapeMismatch {
                op: "from_slice",
                lhs: shape.to_vec(),
                rhs: vec![data.len()],
            });
        }
        let data = CpuBuffer::from_slice(data)?;
        Ok(Self::from_host(data, layout, device))
    }

    /// Tensor filled with zeros. WOrks for every dtype including quantized ones
    pub fn zeros(shape: &[usize], dtype: Dtype, device: Device) -> Result<Self> {
        let layout = Layout::contiguous(shape, dtype)?;
        let storage = Storage::zeros(device, layout.nbytes())?;
        Ok(Self::from_storage(storage, layout, device))
    }

    pub fn ones(shape: &[usize], dtype: Dtype, device: Device) -> Result<Self> {
        Self::full(shape, 1.0, dtype, device)
    }

    /// Tensor filled with `value`, converted to `dtype` with `as` semantics.
    /// For `Bool`, any non-zero value is `true`.
    pub fn full(shape: &[usize], value: f64, dtype: Dtype, device: Device) -> Result<Self> {
        let layout = Layout::contiguous(shape, dtype)?;
        let data = generate("full", dtype, layout.numel(), |_| value)?;
        Ok(Self::from_host(data, layout, device))
    }

    /// 1-D tensor of `start, start + step, …` up to but excluding `end`.
    ///
    /// Values are computed in f64 as `start + i * step`, so integer ranges are
    /// exact up to 2^53.
    pub fn arange(start: f64, end: f64, step: f64, dtype: Dtype, device: Device) -> Result<Self> {
        if step == 0.0 || !step.is_finite() || !start.is_finite() || !end.is_finite() {
            return Err(Error::InvalidArgument {
                op: "arange",
                msg: format!("invalid range {start}..{end} with step {step}"),
            });
        }
        let n = ((end - start) / step).ceil().max(0.0) as usize;
        let layout = Layout::contiguous(&[n], dtype)?;
        let data = generate("arange", dtype, n, |i| start + i as f64 * step)?;

        Ok(Self::from_host(data, layout, device))
    }

    /// Applies an elementwise unary op, returning a new contiguous tensor.
    pub fn unary(&self, op: UnaryOp) -> Result<Self> {
        let (dtype, device) = (self.dtype(), self.device());
        let data = match device {
            Device::CPU => cpu::unary::unary(op, self.0.storage.host_bytes(), self.layout())?,
        };
        let layout = Layout::contiguous(self.shape(), dtype)?;
        Ok(Self::from_host(data, layout, device))
    }

    /// Checks that `self` and `rhs` share a dtype and broadcast together, and
    /// returns the broadcast shape with both layouts expanded to it.
    fn broadcast_pair(
        &self,
        rhs: &Tensor,
        op: &'static str,
    ) -> Result<(Vec<usize>, Layout, Layout)> {
        if rhs.dtype() != self.dtype() {
            return Err(Error::DtypeMismatch {
                op,
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        let shape =
            crate::layout::broadcast_shapes(self.shape(), rhs.shape()).ok_or_else(|| {
                Error::ShapeMismatch {
                    op,
                    lhs: self.shape().to_vec(),
                    rhs: rhs.shape().to_vec(),
                }
            })?;
        let lhs = self
            .layout()
            .broadcast_to(&shape)
            .expect("shapes broadcast");
        let rhs = rhs.layout().broadcast_to(&shape).expect("shapes broadcast");
        Ok((shape, lhs, rhs))
    }
    /// Applies an elementwise binary op, broadcasting both operands to a common
    /// shape and returning a new contiguous tensor.
    /// Applies an elementwise binary op, broadcasting both operands to a common
    /// shape and returning a new contiguous tensor.
    pub fn binary(&self, rhs: &Tensor, op: BinaryOp) -> Result<Self> {
        let (dtype, device) = (self.dtype(), self.device());
        let (shape, lhs_layout, rhs_layout) = self.broadcast_pair(rhs, op.name())?;
        let data = match device {
            Device::CPU => cpu::binary::binary(
                op,
                (self.0.storage.host_bytes(), &lhs_layout),
                (rhs.0.storage.host_bytes(), &rhs_layout),
            )?,
        };
        let layout = Layout::contiguous(&shape, dtype)?;
        Ok(Self::from_host(data, layout, device))
    }

    /// Compares elementwise with broadcasting, returning a `Bool` tensor.
    pub fn compare(&self, rhs: &Tensor, op: CmpOp) -> Result<Self> {
        let device = self.device();
        let (shape, lhs_layout, rhs_layout) = self.broadcast_pair(rhs, op.name())?;
        let data = match device {
            Device::CPU => cpu::compare::compare(
                op,
                (self.0.storage.host_bytes(), &lhs_layout),
                (rhs.0.storage.host_bytes(), &rhs_layout),
            )?,
        };
        let layout = Layout::contiguous(&shape, Dtype::Bool)?;
        Ok(Self::from_host(data, layout, device))
    }

    /// Elementwise select: `on_true` where this `Bool` tensor is true and
    /// `on_false` elsewhere. All three broadcast to a common shape.
    pub fn where_cond(&self, on_true: &Tensor, on_false: &Tensor) -> Result<Self> {
        const OP: &str = "where_cond";
        if self.dtype() != Dtype::Bool {
            return Err(Error::DtypeMismatch {
                op: OP,
                lhs: self.dtype(),
                rhs: Dtype::Bool,
            });
        }
        let (values_shape, _, _) = on_true.broadcast_pair(on_false, OP)?;
        let shape =
            crate::layout::broadcast_shapes(self.shape(), &values_shape).ok_or_else(|| {
                Error::ShapeMismatch {
                    op: OP,
                    lhs: self.shape().to_vec(),
                    rhs: values_shape.clone(),
                }
            })?;
        let expand = |t: &Tensor| t.layout().broadcast_to(&shape).expect("shapes broadcast");
        let (dtype, device) = (on_true.dtype(), self.device());
        let data = match device {
            Device::CPU => cpu::compare::where_cond(
                (self.0.storage.host_bytes(), &expand(self)),
                (on_true.0.storage.host_bytes(), &expand(on_true)),
                (on_false.0.storage.host_bytes(), &expand(on_false)),
                dtype,
            )?,
        };
        let layout = Layout::contiguous(&shape, dtype)?;
        Ok(Self::from_host(data, layout, device))
    }

    /// Reduces over `dim`. With `keepdim` the reduced dim stays as size 1,
    /// otherwise it is removed.
    pub fn reduce(&self, op: ReduceOp, dim: usize, keepdim: bool) -> Result<Self> {
        let (dtype, device, shape) = (self.dtype(), self.device(), self.shape());
        if dim >= shape.len() {
            return Err(Error::InvalidArgument {
                op: op.name(),
                msg: format!(
                    "dim {dim} is out of range for a tensor with {} dims",
                    shape.len()
                ),
            });
        }
        let out_dtype = op.output_dtype(dtype).ok_or(Error::UnsupportedOp {
            op: op.name(),
            dtype,
            device,
        })?;
        if shape[dim] == 0 && op.needs_elements() {
            return Err(Error::InvalidArgument {
                op: op.name(),
                msg: format!("cannot reduce empty dim {dim}"),
            });
        }
        let data = match device {
            Device::CPU => {
                cpu::reduce::reduce(op, self.0.storage.host_bytes(), self.layout(), dim)?
            }
        };
        let mut out_shape = shape.to_vec();
        if keepdim {
            out_shape[dim] = 1;
        } else {
            out_shape.remove(dim);
        }
        let layout = Layout::contiguous(&out_shape, out_dtype)?;
        Ok(Self::from_host(data, layout, device))
    }

    /// Copies the elements out in row-major order. `T` must match the dtype;
    /// `Bool` tensors read as `u8` (0 or 1).
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        let dtype = self.dtype();
        if T::DTYPE != dtype && !(dtype == Dtype::Bool && T::DTYPE == Dtype::U8) {
            return Err(Error::DtypeMismatch {
                op: "to_vec",
                lhs: dtype,
                rhs: T::DTYPE,
            });
        }
        let layout = self.layout();
        let bytes = self.0.storage.host_bytes();
        if layout.is_contiguous() {
            let start = layout.offset();
            // Copies rather than casts: the offset may leave the bytes misaligned for `T`.
            return Ok(bytemuck::pod_collect_to_vec(
                &bytes[start..start + layout.nbytes()],
            ));
        }
        let size = dtype.type_size();
        let mut out = Vec::with_capacity(layout.nbytes());
        for offset in layout.offsets() {
            out.extend_from_slice(&bytes[offset..offset + size]);
        }
        Ok(bytemuck::pod_collect_to_vec(&out))
    }

    // View of the same storage with a different layout.
    #[cfg_attr(not(test), allow(dead_code))] // used by the view ops
    pub(crate) fn with_layout(&self, layout: Layout) -> Self {
        Tensor(Arc::new(TensorImpl_ {
            storage: Arc::clone(&self.0.storage),
            layout,
            device: self.0.device,
        }))
    }
}

/// Host buffer of `n` elements of `dtype`, where element `i` is `f(i)`.
fn generate(
    op: &'static str,
    dtype: Dtype,
    n: usize,
    f: impl Fn(usize) -> f64,
) -> Result<CpuBuffer> {
    fn collect<T: Element>(n: usize, f: impl Fn(usize) -> f64) -> Result<CpuBuffer> {
        let data: Vec<T> = (0..n).map(|i| T::from_f64(f(i))).collect();
        CpuBuffer::from_slice(&data)
    }
    match dtype {
        Dtype::F32 => collect::<f32>(n, f),
        Dtype::F16 => collect::<f16>(n, f),
        Dtype::BF16 => collect::<bf16>(n, f),
        Dtype::F64 => collect::<f64>(n, f),
        Dtype::I8 => collect::<i8>(n, f),
        Dtype::I16 => collect::<i16>(n, f),
        Dtype::I32 => collect::<i32>(n, f),
        Dtype::I64 => collect::<i64>(n, f),
        Dtype::U8 => collect::<u8>(n, f),
        Dtype::Bool => collect::<u8>(n, |i| if f(i) != 0.0 { 1.0 } else { 0.0 }),
        _ => Err(Error::UnsupportedOp {
            op,
            dtype,
            device: Device::CPU,
        }),
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, device={:?})",
            self.shape(),
            self.dtype().as_str(),
            self.device()
        )
    }
}

macro_rules! unary_methods {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl Tensor {$(
            #[doc = concat!("Elementwise [`UnaryOp::", stringify!($op), "`].")]
            pub fn $name(&self) -> Result<Self> {
                self.unary(UnaryOp::$op)
            }
        )*}
    };
}

unary_methods! {
    neg => Neg,
    abs => Abs,
    exp => Exp,
    log => Log,
    sqrt => Sqrt,
    rsqrt => Rsqrt,
    tanh => Tanh,
    sigmoid => Sigmoid,
    relu => Relu,
    silu => Silu,
    gelu => Gelu,
}

macro_rules! binary_methods {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl Tensor {$(
            #[doc = concat!("Elementwise [`BinaryOp::", stringify!($op), "`], with broadcasting.")]
            pub fn $name(&self, rhs: &Tensor) -> Result<Self> {
                self.binary(rhs, BinaryOp::$op)
            }
        )*}
    };
}

binary_methods! {
    add => Add,
    sub => Sub,
    mul => Mul,
    div => Div,
    pow => Pow,
    maximum => Maximum,
    minimum => Minimum,
}

macro_rules! compare_methods {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl Tensor {$(
            #[doc = concat!("Elementwise [`CmpOp::", stringify!($op), "`], with broadcasting.")]
            pub fn $name(&self, rhs: &Tensor) -> Result<Self> {
                self.compare(rhs, CmpOp::$op)
            }
        )*}
    };
}

compare_methods! {
    eq => Eq,
    ne => Ne,
    lt => Lt,
    le => Le,
    gt => Gt,
    ge => Ge,
}

macro_rules! reduce_methods {
    ($($name:ident => $op:ident),* $(,)?) => {
        impl Tensor {$(
            #[doc = concat!("[`ReduceOp::", stringify!($op), "`] over `dim`.")]
            pub fn $name(&self, dim: usize, keepdim: bool) -> Result<Self> {
                self.reduce(ReduceOp::$op, dim, keepdim)
            }
        )*}
    };
}

reduce_methods! {
    sum => Sum,
    mean => Mean,
    max => Max,
    min => Min,
    argmax => Argmax,
    argmin => Argmin,
}

#[cfg(test)]
mod test {
    use super::*;

    const CPU: Device = Device::CPU;

    #[test]
    fn from_slice_roundtrip() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], CPU).unwrap();
        assert_eq!(t.shape(), [2, 3]);
        assert_eq!(t.dtype(), Dtype::F32);
        assert_eq!(t.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let t = Tensor::from_slice(&[-7i64], &[], CPU).unwrap();
        assert_eq!(t.to_vec::<i64>().unwrap(), [-7]);

        let t = Tensor::from_slice(&[f16::from_f32(1.5)], &[1], CPU).unwrap();
        assert_eq!(t.dtype(), Dtype::F16);
        assert_eq!(t.to_vec::<f16>().unwrap(), [f16::from_f32(1.5)]);
    }

    #[test]
    fn from_slice_wrong_len() {
        assert!(matches!(
            Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[2, 2], CPU),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn to_vec_strided() {
        // [[0, 1, 2], [3, 4, 5]]
        let t = Tensor::arange(0.0, 6.0, 1.0, Dtype::I32, CPU).unwrap();

        // Transpose: [[0, 3], [1, 4], [2, 5]]
        let tt = t.with_layout(Layout::from_parts(&[3, 2], &[4, 12], 0, Dtype::I32).unwrap());
        assert_eq!(tt.to_vec::<i32>().unwrap(), [0, 3, 1, 4, 2, 5]);

        // Last column, broadcast to [2, 2]: [[2, 2], [5, 5]]
        let col = t.with_layout(Layout::from_parts(&[2, 2], &[12, 0], 8, Dtype::I32).unwrap());
        assert_eq!(col.to_vec::<i32>().unwrap(), [2, 2, 5, 5]);

        // Contiguous slice with an offset: the second row.
        let row = t.with_layout(Layout::from_parts(&[3], &[4], 12, Dtype::I32).unwrap());
        assert_eq!(row.to_vec::<i32>().unwrap(), [3, 4, 5]);
    }

    #[test]
    fn to_vec_wrong_type() {
        let t = Tensor::from_slice(&[1i32, 2], &[2], CPU).unwrap();
        assert!(matches!(
            t.to_vec::<f32>(),
            Err(Error::DtypeMismatch {
                lhs: Dtype::I32,
                rhs: Dtype::F32,
                ..
            })
        ));
    }

    #[test]
    fn zeros() {
        let t = Tensor::zeros(&[2, 2], Dtype::BF16, CPU).unwrap();
        assert_eq!(t.to_vec::<bf16>().unwrap(), [bf16::ZERO; 4]);

        let t = Tensor::zeros(&[4, 64], Dtype::Q8_0, CPU).unwrap();
        assert_eq!(t.storage_len(), 4 * 2 * 34);
        assert_eq!(
            format!("{t:?}"),
            "Tensor(shape=[4, 64], dtype=q8_0, device=CPU)"
        );
    }

    #[test]
    fn ones_and_full() {
        let t = Tensor::ones(&[3], Dtype::F16, CPU).unwrap();
        assert_eq!(t.to_vec::<f16>().unwrap(), [f16::ONE; 3]);

        let t = Tensor::full(&[2, 2], -3.7, Dtype::I32, CPU).unwrap();
        assert_eq!(t.to_vec::<i32>().unwrap(), [-3; 4]);

        let t = Tensor::full(&[], 2.5, Dtype::F64, CPU).unwrap();
        assert_eq!(t.to_vec::<f64>().unwrap(), [2.5]);

        let t = Tensor::full(&[2], 0.5, Dtype::Bool, CPU).unwrap();
        assert_eq!(t.to_vec::<u8>().unwrap(), [1, 1]);
    }

    #[test]
    fn full_quantized_unsupported() {
        assert!(matches!(
            Tensor::full(&[32], 1.0, Dtype::Q4_0, CPU),
            Err(Error::UnsupportedOp { op: "full", .. })
        ));
    }

    #[test]
    fn arange() {
        let t = Tensor::arange(0.0, 5.0, 1.0, Dtype::I64, CPU).unwrap();
        assert_eq!(t.shape(), [5]);
        assert_eq!(t.to_vec::<i64>().unwrap(), [0, 1, 2, 3, 4]);

        let t = Tensor::arange(0.0, 1.0, 0.25, Dtype::F32, CPU).unwrap();
        assert_eq!(t.to_vec::<f32>().unwrap(), [0.0, 0.25, 0.5, 0.75]);

        let t = Tensor::arange(3.0, 0.0, -1.0, Dtype::I32, CPU).unwrap();
        assert_eq!(t.to_vec::<i32>().unwrap(), [3, 2, 1]);

        // Non-integral span rounds the count up.
        let t = Tensor::arange(0.0, 1.0, 0.3, Dtype::F64, CPU).unwrap();
        assert_eq!(t.shape(), [4]);
    }

    #[test]
    fn arange_empty_and_invalid() {
        let t = Tensor::arange(5.0, 0.0, 1.0, Dtype::F32, CPU).unwrap();
        assert_eq!(t.shape(), [0]);
        assert_eq!(t.to_vec::<f32>().unwrap(), [] as [f32; 0]);

        assert!(matches!(
            Tensor::arange(0.0, 5.0, 0.0, Dtype::F32, CPU),
            Err(Error::InvalidArgument { op: "arange", .. })
        ));
        assert!(matches!(
            Tensor::arange(0.0, f64::INFINITY, 1.0, Dtype::F32, CPU),
            Err(Error::InvalidArgument { .. })
        ));
    }
}
