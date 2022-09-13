use cust_core::DeviceCopy;

#[derive(Clone, Copy, Debug, DeviceCopy)]
#[repr(C)]
pub enum NodeIndex {
    Internal(usize),
    Leaf(usize),
}

#[derive(Clone, Copy, Debug, DeviceCopy)]
#[repr(C)]
pub struct ObjectIndex(pub usize);
