use cuda_std::vek::Vec3;
use cust::prelude::*;
use gpu::aabb::DeviceCopyAabb;
use itertools::Itertools;
use std::error::Error;

static PTX: &str = include_str!("../../resources/gpu.ptx");

pub fn find_nn(
    objects: Vec<Vec3<f32>>,
    queries: Vec<Vec3<f32>>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    // Allocate memory on the CPU.
    let object_aabbs = objects
        .into_iter()
        .map(|o| DeviceCopyAabb::new_empty(o))
        .collect_vec();
    let mut results_distances = vec![0.0f32; queries.len()];

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let kernel = module.get_function("find_nn")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate memory on the GPU.
    let dev_object_aabbs = object_aabbs.as_slice().as_dbuf()?;
    let dev_queries = queries.as_slice().as_dbuf()?;
    let dev_results_distances = results_distances.as_slice().as_dbuf()?;

    let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (queries.len() as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(
            kernel<<<grid_size, block_size, 0, stream>>>(
                dev_object_aabbs.as_device_ptr(),
                dev_object_aabbs.len(),
                dev_queries.as_device_ptr(),
                dev_queries.len(),
                dev_results_distances.as_device_ptr(),
            )
        )?;
    }
    stream.synchronize()?;

    // Copy results from GPU back to CPU.
    dev_results_distances.copy_to(&mut results_distances)?;

    Ok(results_distances)
}
