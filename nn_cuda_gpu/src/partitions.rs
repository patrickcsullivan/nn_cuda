use cust_core::DeviceCopy;

pub const PARTITION_BITS_COUNT: usize = 9;
pub const MAX_PARTITIONS_COUNT: usize = 512; // 2^9 = 512

/// Contains a the start and end index and the AABB for each partition.
///
/// Each partition uses 40 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, DeviceCopy)]
pub struct Partitions {
    pub count: usize,
    pub starts: [usize; MAX_PARTITIONS_COUNT],
    pub ends: [usize; MAX_PARTITIONS_COUNT],
    pub min_xs: [f32; MAX_PARTITIONS_COUNT],
    pub min_ys: [f32; MAX_PARTITIONS_COUNT],
    pub min_zs: [f32; MAX_PARTITIONS_COUNT],
    pub max_xs: [f32; MAX_PARTITIONS_COUNT],
    pub max_ys: [f32; MAX_PARTITIONS_COUNT],
    pub max_zs: [f32; MAX_PARTITIONS_COUNT],
}
