use crate::{morton::map_to_morton_codes, point::Point3};
use bit_partition_gpu::PARTITION_BITS_COUNT;
use cuda_std::vek::{Aabb, Vec3};
use cust::prelude::*;
use itertools::Itertools;
use std::{error::Error, time::Instant};

static PTX: &str = include_str!("../../resources/bit_partition_gpu.ptx");

pub struct BitPartitions<'a, T> {
    objects: &'a [T],
    module: Module,
    dev_sorted_object_indices: DeviceBuffer<usize>,
    dev_sorted_object_xs: DeviceBuffer<f32>,
    dev_sorted_object_ys: DeviceBuffer<f32>,
    dev_sorted_object_zs: DeviceBuffer<f32>,
    dev_partition_starts: DeviceBuffer<usize>,
    dev_partition_ends: DeviceBuffer<usize>,
    dev_partition_min_xs: DeviceBuffer<f32>,
    dev_partition_min_ys: DeviceBuffer<f32>,
    dev_partition_min_zs: DeviceBuffer<f32>,
    dev_partition_max_xs: DeviceBuffer<f32>,
    dev_partition_max_ys: DeviceBuffer<f32>,
    dev_partition_max_zs: DeviceBuffer<f32>,
}

impl<'a, T> BitPartitions<'a, T>
where
    &'a T: Point3,
{
    pub fn new(objects: &'a [T], aabb: &Aabb<f32>) -> Result<Self, Box<dyn Error>> {
        // Sort the objects along the Z-curve.
        let vecs = objects
            .iter()
            .map(|o| {
                let xyz = o.xyz();
                Vec3::new(xyz[0], xyz[1], xyz[2])
            })
            .collect_vec();
        let morton_codes = map_to_morton_codes(&vecs, &aabb);
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
        let mut partition_starts = vec![usize::MAX; partition_count];
        let mut partition_ends = vec![usize::MAX; partition_count];
        let mut partition_min_xs = vec![f32::NAN; partition_count];
        let mut partition_min_ys = vec![f32::NAN; partition_count];
        let mut partition_min_zs = vec![f32::NAN; partition_count];
        let mut partition_max_xs = vec![f32::NAN; partition_count];
        let mut partition_max_ys = vec![f32::NAN; partition_count];
        let mut partition_max_zs = vec![f32::NAN; partition_count];
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

        // Load the CUDA module.
        let module = Module::from_ptx(PTX, &[])?;

        // Load the sorted object indices and positions into device main memory.
        let dev_sorted_object_indices = sorted_object_indices.as_slice().as_dbuf()?;
        let dev_sorted_object_xs = sorted_object_xs.as_slice().as_dbuf()?;
        let dev_sorted_object_ys = sorted_object_ys.as_slice().as_dbuf()?;
        let dev_sorted_object_zs = sorted_object_zs.as_slice().as_dbuf()?;

        // Load the partitions into device main memory.
        let dev_partition_starts = partition_starts.as_slice().as_dbuf()?;
        let dev_partition_ends = partition_ends.as_slice().as_dbuf()?;
        let dev_partition_min_xs = partition_min_xs.as_slice().as_dbuf()?;
        let dev_partition_min_ys = partition_min_ys.as_slice().as_dbuf()?;
        let dev_partition_min_zs = partition_min_zs.as_slice().as_dbuf()?;
        let dev_partition_max_xs = partition_max_xs.as_slice().as_dbuf()?;
        let dev_partition_max_ys = partition_max_ys.as_slice().as_dbuf()?;
        let dev_partition_max_zs = partition_max_zs.as_slice().as_dbuf()?;

        Ok(Self {
            objects,
            module,
            dev_sorted_object_indices,
            dev_sorted_object_xs,
            dev_sorted_object_ys,
            dev_sorted_object_zs,
            dev_partition_starts,
            dev_partition_ends,
            dev_partition_min_xs,
            dev_partition_min_ys,
            dev_partition_min_zs,
            dev_partition_max_xs,
            dev_partition_max_ys,
            dev_partition_max_zs,
        })
    }

    pub fn find_nns<'b, Q>(
        &self,
        stream: Stream,
        queries: &'b [Q],
        queries_aabb: Option<Aabb<f32>>,
    ) -> Result<Vec<&'a T>, Box<dyn Error>>
    where
        &'b Q: Point3,
    {
        let queries: Vec<Vec3<f32>> = queries
            .into_iter()
            .map(|q| {
                let xyz = q.xyz();
                Vec3::new(xyz[0], xyz[1], xyz[2])
            })
            .collect_vec();
        let queries_aabb = queries_aabb.unwrap_or_else(|| get_aabb(&queries));

        // Radix sort queries.
        let now = Instant::now();
        let morton_codes = map_to_morton_codes(&queries, &queries_aabb);
        let mut sorted_query_indices = (0..queries.len()).collect_vec();
        sorted_query_indices.sort_by_key(|&i| morton_codes[i]);
        let sorted_queries = sorted_query_indices
            .iter()
            .map(|&i| queries[i])
            .collect_vec();
        let elapsed = now.elapsed();
        //println!("\tradix sorting queries:\t{:.2?}", elapsed);

        // Allocate memory on the CPU for results.
        let mut result_object_indices = vec![0usize; queries.len()];
        let mut result_dist2s = vec![f32::INFINITY; queries.len()];

        // Allocate global memory for queries and results on the GPU.
        let dev_sorted_queries = sorted_queries.as_slice().as_dbuf()?;
        let dev_result_object_indices = result_object_indices.as_slice().as_dbuf()?;
        let dev_result_dist2s = result_dist2s.as_slice().as_dbuf()?;

        let kernel = self.module.get_function("partition_search_for_queries")?;
        let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
        let grid_size = (sorted_queries.len() as u32 + block_size - 1) / block_size;
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
                    self.dev_partition_starts.as_device_ptr(),
                    self.dev_partition_starts.len(),
                    self.dev_partition_ends.as_device_ptr(),
                    self.dev_partition_ends.len(),
                    self.dev_partition_min_xs.as_device_ptr(),
                    self.dev_partition_min_xs.len(),
                    self.dev_partition_min_ys.as_device_ptr(),
                    self.dev_partition_min_ys.len(),
                    self.dev_partition_min_zs.as_device_ptr(),
                    self.dev_partition_min_zs.len(),
                    self.dev_partition_max_xs.as_device_ptr(),
                    self.dev_partition_max_xs.len(),
                    self.dev_partition_max_ys.as_device_ptr(),
                    self.dev_partition_max_ys.len(),
                    self.dev_partition_max_zs.as_device_ptr(),
                    self.dev_partition_max_zs.len(),
                    //-----
                    dev_sorted_queries.as_device_ptr(),
                    dev_sorted_queries.len(),
                    dev_result_object_indices.as_device_ptr(),
                    dev_result_dist2s.as_device_ptr(),
                )
            )?;
        }
        stream.synchronize()?;
        let elapsed = now.elapsed();
        //println!("\trunning kernel:\t{:.2?}", elapsed);

        // Copy results from GPU back to CPU.
        let now = Instant::now();
        dev_result_object_indices.copy_to(&mut result_object_indices)?;
        dev_result_dist2s.copy_to(&mut result_dist2s)?;
        let elapsed = now.elapsed();
        //println!("\tdevice -> host:\t{:.2?}", elapsed);

        // Results are generated in the same order as the sorted queries. We should
        // unsort them so that they are in the same order as the original
        // queries.
        let now = Instant::now();

        let mut original_order_nn_object_indices = vec![0; queries.len()];
        sorted_query_indices
            .into_iter()
            .zip(result_object_indices)
            .for_each(|(qi, oi)| original_order_nn_object_indices[qi] = oi);
        let original_order_nns = original_order_nn_object_indices
            .into_iter()
            .map(|oi| &self.objects[oi])
            .collect_vec();
        // let sorted_results = result_object_indices
        //     .into_iter()
        //     .zip(result_dist2s)
        //     .map(|(i, d)| d.is_finite().then(|| i))
        //     .collect_vec();
        // let mut unsorted_results = vec![None; sorted_results.len()];
        // sorted_query_indices
        //     .iter()
        //     .zip(sorted_results)
        //     .for_each(|(&q_idx, option_o_idx)| {
        //         unsorted_results[q_idx] = option_o_idx.map(|o_idx|
        // self.objects[o_idx])     });
        let elapsed = now.elapsed();
        //println!("\tpost process:\t{:.2?}", elapsed);

        Ok(original_order_nns)
    }
}

fn get_aabb(vecs: &[Vec3<f32>]) -> Aabb<f32> {
    let mut aabb = Aabb::new_empty(vecs[0]);
    for v in vecs {
        aabb.expand_to_contain_point(*v)
    }
    aabb
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
