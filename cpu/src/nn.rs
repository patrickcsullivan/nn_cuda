use crate::bvh::Bvh;
use cuda_std::vek::Vec3;
use cust::prelude::*;
use gpu::bvh::ObjectIndex;
use itertools::Itertools;
use std::error::Error;

static PTX: &str = include_str!("../../resources/gpu.ptx");

pub fn brute_force(
    bvh: &Bvh,
    queries: Vec<Vec3<f32>>,
) -> Result<Vec<(ObjectIndex, f32)>, Box<dyn Error>> {
    // Allocate memory on the CPU.
    let leaf_object_indices = bvh.leaf_nodes.iter().map(|n| n.0).collect_vec();
    let leaf_aabbs = bvh.leaf_nodes.iter().map(|n| n.1).collect_vec();
    let internal_left_child_indicies = bvh.internal_nodes.iter().map(|n| n.0).collect_vec();
    let internal_right_child_indicies = bvh.internal_nodes.iter().map(|n| n.1).collect_vec();
    let internal_aabbs = bvh.internal_nodes.iter().map(|n| n.2).collect_vec();
    let mut results_leaf_object_indices = vec![ObjectIndex(0); queries.len()];
    let mut results_distances = vec![0.0f32; queries.len()];

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let kernel = module.get_function("brute_force")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate memory on the GPU.
    let dev_leaf_object_indices = leaf_object_indices.as_slice().as_dbuf()?;
    let dev_leaf_aabbs = leaf_aabbs.as_slice().as_dbuf()?;
    let dev_internal_left_child_indices = internal_left_child_indicies.as_slice().as_dbuf()?;
    let dev_internal_right_child_indices = internal_right_child_indicies.as_slice().as_dbuf()?;
    let dev_internal_aabbs = internal_aabbs.as_slice().as_dbuf()?;
    let dev_queries = queries.as_slice().as_dbuf()?;
    let dev_results_leaf_object_indices = results_leaf_object_indices.as_slice().as_dbuf()?;
    let dev_results_distances = results_distances.as_slice().as_dbuf()?;

    let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (queries.len() as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(
            kernel<<<grid_size, block_size, 0, stream>>>(
                dev_leaf_object_indices.as_device_ptr(),
                dev_leaf_object_indices.len(),
                dev_leaf_aabbs.as_device_ptr(),
                dev_leaf_aabbs.len(),
                //-----
                dev_internal_left_child_indices.as_device_ptr(),
                dev_internal_left_child_indices.len(),
                dev_internal_right_child_indices.as_device_ptr(),
                dev_internal_right_child_indices.len(),
                dev_internal_aabbs.as_device_ptr(),
                dev_internal_aabbs.len(),
                //-----
                dev_queries.as_device_ptr(),
                dev_queries.len(),
                dev_results_leaf_object_indices.as_device_ptr(),
                dev_results_distances.as_device_ptr(),
            )
        )?;
    }
    stream.synchronize()?;

    // Copy results from GPU back to CPU.
    dev_results_leaf_object_indices.copy_to(&mut results_leaf_object_indices)?;
    dev_results_distances.copy_to(&mut results_distances)?;

    let results = results_leaf_object_indices
        .into_iter()
        .zip(results_distances)
        .collect_vec();
    Ok(results)
}
