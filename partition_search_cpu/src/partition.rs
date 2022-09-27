use crate::morton::map_to_morton_codes_tmp;
use cuda_std::vek::{Aabb, Vec3};
use cust::prelude::*;
use itertools::Itertools;
use partition_search_gpu::partitions::{Partitions, PARTITIONS_COUNT};
use std::{error::Error, ffi::CString, time::Instant};

static PTX: &str = include_str!("../../resources/partition_search_gpu.ptx");

pub struct PartitionSearch {
    pub sorted_object_indices: Vec<usize>,
    pub sorted_object_xs: Vec<f32>,
    pub sorted_object_ys: Vec<f32>,
    pub sorted_object_zs: Vec<f32>,
    pub partitions: Partitions,
}

impl PartitionSearch {
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

        // When the objects cannot be divided into even partitions, make all partitions
        // but the last even.
        let partition_size = if objects.len() % PARTITIONS_COUNT == 0 {
            objects.len() / PARTITIONS_COUNT
        } else {
            let last_partition_size = objects.len() % (PARTITIONS_COUNT - 1);
            (objects.len() - last_partition_size) / (PARTITIONS_COUNT - 1)
        };

        println!("Partition size: {}", partition_size);

        let mut partition_min_xs = [f32::NAN; PARTITIONS_COUNT];
        let mut partition_min_ys = [f32::NAN; PARTITIONS_COUNT];
        let mut partition_min_zs = [f32::NAN; PARTITIONS_COUNT];
        let mut partition_max_xs = [f32::NAN; PARTITIONS_COUNT];
        let mut partition_max_ys = [f32::NAN; PARTITIONS_COUNT];
        let mut partition_max_zs = [f32::NAN; PARTITIONS_COUNT];

        for i in 0..PARTITIONS_COUNT {
            let from = i * partition_size;
            let to = ((i + 1) * partition_size).min(sorted_object_vecs.len());
            let aabb = get_aabb(&sorted_object_vecs[from..to]);
            partition_min_xs[i] = aabb.min.x;
            partition_min_ys[i] = aabb.min.y;
            partition_min_zs[i] = aabb.min.z;
            partition_max_xs[i] = aabb.max.x;
            partition_max_ys[i] = aabb.max.y;
            partition_max_zs[i] = aabb.max.z;
        }

        Self {
            sorted_object_indices,
            sorted_object_xs,
            sorted_object_ys,
            sorted_object_zs,
            partitions: Partitions {
                partition_size,
                min_xs: partition_min_xs,
                min_ys: partition_min_ys,
                min_zs: partition_min_zs,
                max_xs: partition_max_xs,
                max_ys: partition_max_ys,
                max_zs: partition_max_zs,
            },
        }
    }

    pub fn find_nns(
        &self,
        queries: &[Vec3<f32>],
    ) -> Result<Vec<Option<(usize, f32)>>, Box<dyn Error>> {
        // Allocate memory on the CPU.
        let mut result_object_indices = vec![0usize; queries.len()];
        let mut result_dist2s = vec![f32::INFINITY; queries.len()];

        let now = Instant::now();
        let _ctx = cust::quick_init()?;
        let module = Module::from_ptx(PTX, &[])?;
        let kernel = module.get_function("partition_search_for_queries")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let elapsed = now.elapsed();
        println!("\tstarting CUDA:\t{:.2?}", elapsed);

        // Allocate constant memory on the GPU.
        let symbol_name = CString::new("PARTITIONS")?;
        let mut symbol = module.get_global::<Partitions>(symbol_name.as_c_str())?;
        symbol.copy_from(&self.partitions)?;

        // Allocate memory on the GPU.
        let now = Instant::now();
        let dev_sorted_object_indices = self.sorted_object_indices.as_slice().as_dbuf()?;
        let dev_sorted_object_xs = self.sorted_object_xs.as_slice().as_dbuf()?;
        let dev_sorted_object_ys = self.sorted_object_ys.as_slice().as_dbuf()?;
        let dev_sorted_object_zs = self.sorted_object_zs.as_slice().as_dbuf()?;
        let dev_queries = queries.as_dbuf()?;
        let dev_result_object_indices = result_object_indices.as_slice().as_dbuf()?;
        let dev_result_dist2s = result_dist2s.as_slice().as_dbuf()?;
        let elapsed = now.elapsed();
        println!("\thost -> device:\t{:.2?}", elapsed);

        let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
        let grid_size = (queries.len() as u32 + block_size - 1) / block_size;
        let now = Instant::now();
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
                    dev_queries.as_device_ptr(),
                    dev_queries.len(),
                    dev_result_object_indices.as_device_ptr(),
                    dev_result_dist2s.as_device_ptr(),
                )
            )?;
        }
        stream.synchronize()?;
        let elapsed = now.elapsed();
        println!("\trunning kernel:\t{:.2?}", elapsed);

        // Copy results from GPU back to CPU.
        let now = Instant::now();
        dev_result_object_indices.copy_to(&mut result_object_indices)?;
        dev_result_dist2s.copy_to(&mut result_dist2s)?;
        let elapsed = now.elapsed();
        println!("\tdevice -> host:\t{:.2?}", elapsed);

        let now = Instant::now();
        let results = result_object_indices
            .into_iter()
            .zip(result_dist2s)
            .map(|(i, d)| if d.is_finite() { Some((i, d)) } else { None })
            .collect_vec();
        let elapsed = now.elapsed();
        println!("\tpost process:\t{:.2?}", elapsed);

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

fn div_ceil(numerator: usize, denominator: usize) -> usize {
    (numerator + denominator - 1) / denominator
}

pub trait HasVec3 {
    fn vec3(&self) -> Vec3<f32>;
}
