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
