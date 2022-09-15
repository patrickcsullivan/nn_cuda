use cust_core::DeviceCopy;

#[derive(Clone, Copy, Debug, DeviceCopy)]
#[repr(C)]
pub enum NodeIndex {
    Internal(InternalNodeIndex),
    Leaf(LeafNodeIndex),
}

impl Default for NodeIndex {
    fn default() -> Self {
        NodeIndex::Internal(InternalNodeIndex(0))
    }
}

#[derive(Clone, Copy, Debug, DeviceCopy)]
#[repr(C)]
pub struct InternalNodeIndex(pub usize);

#[derive(Clone, Copy, Debug, DeviceCopy)]
#[repr(C)]
pub struct LeafNodeIndex(pub usize);

#[derive(Clone, Copy, Debug, DeviceCopy)]
#[repr(C)]
pub struct ObjectIndex(pub usize);
