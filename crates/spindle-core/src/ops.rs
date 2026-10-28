use crate::dtype::Dtype;

/// Elementwise unary operations.
///
/// Every op is defined on float dtypes. `Neg`, `Abs` and `Relu` are also
/// defined on integer dtypes, where `Neg` and `Abs` wrap on overflow like
/// PyTorch (`-(-128i8) == -128`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    /// `1 / sqrt(x)`.
    Rsqrt,
    Tanh,
    /// `1 / (1 + exp(-x))`.
    Sigmoid,
    Relu,
    /// `x * sigmoid(x)`, aka swish
    Silu,
    /// Tanh approximation `0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x³)))`
    Gelu,
}

impl UnaryOp {
    pub fn name(&self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Abs => "abs",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Rsqrt => "rsqrt",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Relu => "relu",
            UnaryOp::Silu => "silu",
            UnaryOp::Gelu => "gelu",
        }
    }
}

/// Elementwise binary operations, with broadcasting.
///
/// Every op is defined on float dtypes. `Add`, `Sub`, `Mul`, `Maximum` and
/// `Minimum` are also defined on integer dtypes, where arithmetic wraps on
/// overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    /// Elementwise maximum; NaN if either input is NaN.
    Maximum,
    /// Elementwise minimum; NaN if either input is NaN.
    Minimum,
}

impl BinaryOp {
    pub fn name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Pow => "pow",
            BinaryOp::Maximum => "maximum",
            BinaryOp::Minimum => "minimum",
        }
    }
}

/// Elementwise comparisons, with broadcasting. The result is a `Bool` tensor.
///
/// Floats compare with IEEE semantics: every comparison with NaN is false
/// except `Ne`. `Bool` inputs compare as 0 and 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    pub fn name(&self) -> &'static str {
        match self {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        }
    }
}

/// Reductions over one dim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Floats keep their dtype; integers and `Bool` sum into `I64`.
    Sum,
    /// Float dtypes only. NaN over an empty dim.
    Mean,
    /// NaN if any element is NaN.
    Max,
    /// NaN if any element is NaN.
    Min,
    /// `I64` index of the first maximum; a NaN counts as the maximum.
    Argmax,
    /// `I64` index of the first minimum; a NaN counts as the minimum.
    Argmin,
}

impl ReduceOp {
    pub fn name(&self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Mean => "mean",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Argmax => "argmax",
            ReduceOp::Argmin => "argmin",
        }
    }

    /// Dtype of the result for an input of `dtype`, or `None` if unsupported.
    pub fn output_dtype(&self, dtype: Dtype) -> Option<Dtype> {
        if dtype.is_quantized() {
            return None;
        }
        match self {
            ReduceOp::Sum if dtype.is_float() => Some(dtype),
            ReduceOp::Sum => Some(Dtype::I64),
            ReduceOp::Mean => dtype.is_float().then_some(dtype),
            ReduceOp::Max | ReduceOp::Min => Some(dtype),
            ReduceOp::Argmax | ReduceOp::Argmin => Some(Dtype::I64),
        }
    }

    /// Whether the op has no value over an empty dim.
    pub fn needs_elements(&self) -> bool {
        !matches!(self, ReduceOp::Sum | ReduceOp::Mean)
    }
}
