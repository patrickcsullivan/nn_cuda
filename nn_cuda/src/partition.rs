use crate::morton::map_to_morton_codes_tmp;
use cuda_std::vek::{Aabb, Vec3};
use cust::prelude::*;
use itertools::Itertools;
use nn_cuda_gpu::partitions::{Partitions, MAX_PARTITIONS_COUNT, PARTITION_BITS_COUNT};
use std::{error::Error, ffi::CString, time::Instant};

static PTX: &str = include_str!("../../resources/nn_cuda_gpu.ptx");

pub struct BitPartitionSearch {
    pub sorted_object_indices: Vec<usize>,
    pub sorted_object_xs: Vec<f32>,
    pub sorted_object_ys: Vec<f32>,
    pub sorted_object_zs: Vec<f32>,

    pub module: Module,
    pub dev_sorted_object_indices: DeviceBuffer<usize>,
    pub dev_sorted_object_xs: DeviceBuffer<f32>,
    pub dev_sorted_object_ys: DeviceBuffer<f32>,
    pub dev_sorted_object_zs: DeviceBuffer<f32>,
}

impl BitPartitionSearch {
    pub fn new<T>(objects: &[T], aabb: &Aabb<f32>) -> Result<Self, Box<dyn Error>>
    where
        T: HasVec3,
    {
        // Sort the objects along the Z-curve.
        let vecs = objects.iter().map(|o| o.vec3()).collect_vec();
        let morton_codes = map_to_morton_codes_tmp(&vecs, &aabb);
        let mut sorted_object_indices = (0..objects.len()).collect_vec();
        sorted_object_indices.sort_by_key(|&i| morton_codes[i]);

        let sorted_object_vecs = sorted_object_indices.iter().map(|&i| vecs[i]).collect_vec();
        let sorted_object_xs = sorted_object_vecs.iter().map(|v| v.x).collect_vec();
        let sorted_object_ys = sorted_object_vecs.iter().map(|v| v.y).collect_vec();
        let sorted_object_zs = sorted_object_vecs.iter().map(|v| v.z).collect_vec();

        // Divide the objects into partitions by their most siginificant Morton code
        // bits.
        let sorted_morton_codes = sorted_object_indices
            .iter()
            .map(|&i| morton_codes[i])
            .collect_vec();
        let sizes = partition_by_bit_prefix(PARTITION_BITS_COUNT as u32, &sorted_morton_codes);
        let partition_count = sizes.len();
        let mut starts = sizes
            .iter()
            .scan(0, |sum, i| {
                *sum += i;
                Some(*sum)
            })
            .collect_vec();
        starts.pop();
        starts.insert(0, 0);
        let ends = starts
            .iter()
            .zip(sizes)
            .map(|(start, size)| start + size - 1)
            .collect_vec();

        // Calculate the AABBs for each partition.
        let mut partition_starts = [usize::MAX; MAX_PARTITIONS_COUNT];
        let mut partition_ends = [usize::MAX; MAX_PARTITIONS_COUNT];
        let mut partition_min_xs = [f32::NAN; MAX_PARTITIONS_COUNT];
        let mut partition_min_ys = [f32::NAN; MAX_PARTITIONS_COUNT];
        let mut partition_min_zs = [f32::NAN; MAX_PARTITIONS_COUNT];
        let mut partition_max_xs = [f32::NAN; MAX_PARTITIONS_COUNT];
        let mut partition_max_ys = [f32::NAN; MAX_PARTITIONS_COUNT];
        let mut partition_max_zs = [f32::NAN; MAX_PARTITIONS_COUNT];
        for i in 0..partition_count {
            partition_starts[i] = starts[i];
            partition_ends[i] = ends[i];
            let aabb = get_aabb(&sorted_object_vecs[starts[i]..=ends[i]]);
            partition_min_xs[i] = aabb.min.x;
            partition_min_ys[i] = aabb.min.y;
            partition_min_zs[i] = aabb.min.z;
            partition_max_xs[i] = aabb.max.x;
            partition_max_ys[i] = aabb.max.y;
            partition_max_zs[i] = aabb.max.z;
        }
        let partitions = Partitions {
            count: partition_count,
            starts: partition_starts,
            ends: partition_ends,
            min_xs: partition_min_xs,
            min_ys: partition_min_ys,
            min_zs: partition_min_zs,
            max_xs: partition_max_xs,
            max_ys: partition_max_ys,
            max_zs: partition_max_zs,
        };

        // Load the CUDA module and constant memory symbol.
        let module = Module::from_ptx(PTX, &[])?;
        let partitions_symbol_name = CString::new("PARTITIONS")?;
        let mut partitions_symbol =
            module.get_global::<Partitions>(partitions_symbol_name.as_c_str())?;

        // Load the partitions into device constant memory.
        partitions_symbol.copy_from(&partitions)?;

        // Load the sorted object indices and positions into device main memory.
        let dev_sorted_object_indices = sorted_object_indices.as_slice().as_dbuf()?;
        let dev_sorted_object_xs = sorted_object_xs.as_slice().as_dbuf()?;
        let dev_sorted_object_ys = sorted_object_ys.as_slice().as_dbuf()?;
        let dev_sorted_object_zs = sorted_object_zs.as_slice().as_dbuf()?;

        Ok(Self {
            sorted_object_indices,
            sorted_object_xs,
            sorted_object_ys,
            sorted_object_zs,
            module,
            dev_sorted_object_indices,
            dev_sorted_object_xs,
            dev_sorted_object_ys,
            dev_sorted_object_zs,
        })
    }

    pub fn find_nns(
        &self,
        stream: Stream,
        queries: &[Vec3<f32>],
    ) -> Result<Vec<Option<(usize, f32)>>, Box<dyn Error>> {
        // Allocate memory on the CPU.
        let mut result_object_indices = vec![0usize; queries.len()];
        let mut result_dist2s = vec![f32::INFINITY; queries.len()];

        // let _ctx = cust::quick_init()?;
        // let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Allocate global memory for queries and results on the GPU.
        let dev_queries = queries.as_dbuf()?;
        let dev_result_object_indices = result_object_indices.as_slice().as_dbuf()?;
        let dev_result_dist2s = result_dist2s.as_slice().as_dbuf()?;

        let kernel = self.module.get_function("partition_search_for_queries")?;
        let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
        let grid_size = (queries.len() as u32 + block_size - 1) / block_size;
        let now = Instant::now();
        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    self.dev_sorted_object_indices.as_device_ptr(),
                    self.dev_sorted_object_indices.len(),
                    self.dev_sorted_object_xs.as_device_ptr(),
                    self.dev_sorted_object_xs.len(),
                    self.dev_sorted_object_ys.as_device_ptr(),
                    self.dev_sorted_object_ys.len(),
                    self.dev_sorted_object_zs.as_device_ptr(),
                    self.dev_sorted_object_zs.len(),
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

fn partition_by_bit_prefix(bits_count: u32, sorted_morton_codes: &[u32]) -> Vec<usize> {
    let right_bits = 30 - bits_count;
    let bit_mask = (2_u32.pow(bits_count) - 1) << right_bits;
    let most_sigs = sorted_morton_codes
        .iter()
        .map(|&mc| (mc & bit_mask) >> right_bits)
        .collect_vec();

    let mut partition_sizes = vec![];
    let mut last_most_sig = u32::MAX;

    for next_most_sig in most_sigs {
        if next_most_sig != last_most_sig {
            last_most_sig = next_most_sig;
            partition_sizes.push(1);
        } else {
            let last_idx = partition_sizes.len() - 1;
            partition_sizes[last_idx] += 1;
        }
    }

    partition_sizes
}

pub trait HasVec3 {
    fn vec3(&self) -> Vec3<f32>;
}
