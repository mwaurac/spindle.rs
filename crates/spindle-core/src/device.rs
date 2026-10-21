#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
}

impl Device {
    pub fn as_str(&self) -> &'static str {
        match self {
            Device::CPU => "cpu",
        }
    }
}
