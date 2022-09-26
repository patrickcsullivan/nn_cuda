use cust_core::DeviceCopy;

pub const PARTITIONS_COUNT: usize = 400; // 24 bytes per partition

#[repr(C)]
#[derive(Clone, Copy, Debug, DeviceCopy)]
pub struct Partitions {
    pub partition_size: usize,
    pub min_xs: [f32; PARTITIONS_COUNT],
    pub min_ys: [f32; PARTITIONS_COUNT],
    pub min_zs: [f32; PARTITIONS_COUNT],
    pub max_xs: [f32; PARTITIONS_COUNT],
    pub max_ys: [f32; PARTITIONS_COUNT],
    pub max_zs: [f32; PARTITIONS_COUNT],
}
