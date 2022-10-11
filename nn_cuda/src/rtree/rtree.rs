use super::{
    data::{DeviceData, HostData},
    geometry::points_aabb,
};
use crate::{morton::map_to_morton_codes, point::Point3};
use cuda_std::vek::Vec3;
use cust::prelude::*;
use itertools::Itertools;
use std::{error::Error, time::Instant};

static PTX: &str = include_str!("../../../resources/bit_partition_gpu.ptx");

pub struct RTree<'a, T> {
    objects: &'a [T],
    module: Module,
    device_data: DeviceData,
}

impl<'a, T> RTree<'a, T>
where
    T: Point3,
{
    pub fn new(objects: &'a [T]) -> Result<Self, Box<dyn Error>> {
        let module = Module::from_ptx(PTX, &[])?;
        let host_data: HostData<4, 10> = HostData::new(objects);
        let device_data = host_data.copy_to_device()?;
        Ok(Self {
            objects,
            module,
            device_data,
        })
    }
}
