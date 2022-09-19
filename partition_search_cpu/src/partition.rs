use crate::morton::map_to_morton_codes_tmp;
use cuda_std::vek::{Aabb, Vec3};
use cust::prelude::*;
use itertools::Itertools;
use partition_search_gpu::kernels::PARTITIONS_COUNT;
use std::error::Error;

static PTX: &str = include_str!("../../resources/partition_search_gpu.ptx");

pub struct Partitions {
    pub sorted_object_indices: Vec<usize>,
    pub sorted_object_xs: Vec<f32>,
    pub sorted_object_ys: Vec<f32>,
    pub sorted_object_zs: Vec<f32>,

    pub partition_size: usize,
    pub partition_min_xs: Vec<f32>,
    pub partition_min_ys: Vec<f32>,
    pub partition_min_zs: Vec<f32>,
    pub partition_max_xs: Vec<f32>,
    pub partition_max_ys: Vec<f32>,
    pub partition_max_zs: Vec<f32>,
}

impl Partitions {
    pub fn new<T>(objects: &[T], aabb: &Aabb<f32>) -> Self
    where
        T: HasVec3,
    {
        let vecs = objects.iter().map(|o| o.vec3()).collect_vec();
        let morton_codes = map_to_morton_codes_tmp(&vecs, &aabb);

        let mut sorted_object_indices = (0..objects.len()).collect_vec();
        sorted_object_indices.sort_by_key(|&i| morton_codes[i]);
        let sorted_object_vecs = sorted_object_indices.iter().map(|&i| vecs[i]).collect_vec();
        let sorted_object_xs = sorted_object_vecs.iter().map(|v| v.x).collect_vec();
        let sorted_object_ys = sorted_object_vecs.iter().map(|v| v.y).collect_vec();
        let sorted_object_zs = sorted_object_vecs.iter().map(|v| v.z).collect_vec();

        let partition_size = objects.len() / PARTITIONS_COUNT;
        let mut partition_min_xs = Vec::with_capacity(PARTITIONS_COUNT);
        let mut partition_min_ys = Vec::with_capacity(PARTITIONS_COUNT);
        let mut partition_min_zs = Vec::with_capacity(PARTITIONS_COUNT);
        let mut partition_max_xs = Vec::with_capacity(PARTITIONS_COUNT);
        let mut partition_max_ys = Vec::with_capacity(PARTITIONS_COUNT);
        let mut partition_max_zs = Vec::with_capacity(PARTITIONS_COUNT);

        for i in 0..PARTITIONS_COUNT {
            let from = i * partition_size;
            let to = ((i + 1) * partition_size).min(sorted_object_vecs.len());
            let aabb = get_aabb(&sorted_object_vecs[from..to]);
            partition_min_xs.push(aabb.min.x);
            partition_min_ys.push(aabb.min.y);
            partition_min_zs.push(aabb.min.z);
            partition_max_xs.push(aabb.max.x);
            partition_max_ys.push(aabb.max.y);
            partition_max_zs.push(aabb.max.z);
        }

        Self {
            sorted_object_indices,
            sorted_object_xs,
            sorted_object_ys,
            sorted_object_zs,

            partition_size,
            partition_min_xs,
            partition_min_ys,
            partition_min_zs,
            partition_max_xs,
            partition_max_ys,
            partition_max_zs,
        }
    }

    pub fn find_nns(
        &self,
        queries: &[Vec3<f32>],
    ) -> Result<Vec<Option<(usize, f32)>>, Box<dyn Error>> {
        // Allocate memory on the CPU.
        let mut result_object_indices = vec![0usize; queries.len()];
        let mut result_dist2s = vec![0.0f32; queries.len()];

        let _ctx = cust::quick_init()?;
        let module = Module::from_ptx(PTX, &[])?;
        let kernel = module.get_function("partition_search_for_queries")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Allocate memory on the GPU.
        let dev_sorted_object_indices = self.sorted_object_indices.as_slice().as_dbuf()?;
        let dev_sorted_object_xs = self.sorted_object_xs.as_slice().as_dbuf()?;
        let dev_sorted_object_ys = self.sorted_object_ys.as_slice().as_dbuf()?;
        let dev_sorted_object_zs = self.sorted_object_zs.as_slice().as_dbuf()?;
        let dev_partition_min_xs = self.partition_min_xs.as_slice().as_dbuf()?;
        let dev_partition_min_ys = self.partition_min_ys.as_slice().as_dbuf()?;
        let dev_partition_min_zs = self.partition_min_zs.as_slice().as_dbuf()?;
        let dev_partition_max_xs = self.partition_max_xs.as_slice().as_dbuf()?;
        let dev_partition_max_ys = self.partition_max_ys.as_slice().as_dbuf()?;
        let dev_partition_max_zs = self.partition_max_zs.as_slice().as_dbuf()?;
        let dev_queries = queries.as_dbuf()?;
        let dev_result_object_indices = result_object_indices.as_slice().as_dbuf()?;
        let dev_result_dist2s = result_dist2s.as_slice().as_dbuf()?;

        let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
        let grid_size = (queries.len() as u32 + block_size - 1) / block_size;
        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    dev_sorted_object_indices.as_device_ptr(),
                    dev_sorted_object_indices.len(),
                    dev_sorted_object_xs.as_device_ptr(),
                    dev_sorted_object_xs.len(),
                    dev_sorted_object_ys.as_device_ptr(),
                    dev_sorted_object_ys.len(),
                    dev_sorted_object_zs.as_device_ptr(),
                    dev_sorted_object_zs.len(),
                    //-----
                    dev_partition_min_xs.as_device_ptr(),
                    dev_partition_min_xs.len(),
                    dev_partition_min_ys.as_device_ptr(),
                    dev_partition_min_ys.len(),
                    dev_partition_min_zs.as_device_ptr(),
                    dev_partition_min_zs.len(),
                    dev_partition_max_xs.as_device_ptr(),
                    dev_partition_max_xs.len(),
                    dev_partition_max_ys.as_device_ptr(),
                    dev_partition_max_ys.len(),
                    dev_partition_max_zs.as_device_ptr(),
                    dev_partition_max_zs.len(),
                    //-----
                    dev_queries.as_device_ptr(),
                    dev_queries.len(),
                    dev_result_object_indices.as_device_ptr(),
                    dev_result_dist2s.as_device_ptr(),
                )
            )?;
        }
        stream.synchronize()?;

        // Copy results from GPU back to CPU.
        dev_result_object_indices.copy_to(&mut result_object_indices)?;
        dev_result_dist2s.copy_to(&mut result_dist2s)?;

        let results = result_object_indices
            .into_iter()
            .zip(result_dist2s)
            .map(|(i, d)| if d.is_finite() { Some((i, d)) } else { None })
            .collect_vec();
        Ok(results)
    }
}

fn get_aabb(vecs: &[Vec3<f32>]) -> Aabb<f32> {
    let mut aabb = Aabb::new_empty(vecs[0]);
    for v in vecs {
        aabb.expand_to_contain_point(*v)
    }
    aabb
}

pub trait HasVec3 {
    fn vec3(&self) -> Vec3<f32>;
}
