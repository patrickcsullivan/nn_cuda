use cust_core::DeviceCopy;

#[derive(Clone, Copy, Debug, DeviceCopy)]
#[repr(C)]
pub enum NodeIndex {
    Internal(usize),
    Leaf(usize),
}

impl Default for NodeIndex {
    fn default() -> Self {
        NodeIndex::Internal(0)
    }
}

#[derive(Clone, Copy, Debug, DeviceCopy)]
#[repr(C)]
pub struct ObjectIndex(pub usize);
