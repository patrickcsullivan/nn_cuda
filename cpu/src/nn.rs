use cust::prelude::*;
use itertools::Itertools;
use std::error::Error;

static PTX: &str = include_str!("../../resources/gpu.ptx");

pub fn find_nn(
    leaf_node_object_indices: Vec<usize>,
    queries: Vec<usize>,
) -> Result<Vec<usize>, Box<dyn Error>> {
    // Allocate memory on the CPU.
    let mut c: Vec<usize> = vec![0; leaf_node_object_indices.len()];

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let kernel = module.get_function("find_nn")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate memory on the GPU.
    let dev_a = leaf_node_object_indices.as_slice().as_dbuf()?;
    let dev_b = queries.as_slice().as_dbuf()?;
    let dev_c = c.as_slice().as_dbuf()?;

    let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (leaf_node_object_indices.len() as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(
            kernel<<<grid_size, block_size, 0, stream>>>(
                dev_a.as_device_ptr(),
                dev_a.len(),
                dev_b.as_device_ptr(),
                dev_b.len(),
                dev_c.as_device_ptr(),
                leaf_node_object_indices.len()
            )
        )?;
    }
    stream.synchronize()?;

    // Copy c vector from GPU back to CPU.
    dev_c.copy_to(&mut c)?;

    Ok(c)
}

pub fn find_nn_2(queries: Vec<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
    // Allocate memory on the CPU.
    let mut results_distances = vec![0.0f32; queries.len()];

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let kernel = module.get_function("find_nn_2")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate memory on the GPU.
    let dev_queries = queries.as_slice().as_dbuf()?;
    let dev_results_distances = results_distances.as_slice().as_dbuf()?;

    let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (queries.len() as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(
            kernel<<<grid_size, block_size, 0, stream>>>(
                dev_queries.as_device_ptr(),
                dev_queries.len(),
                dev_results_distances.as_device_ptr(),
                dev_results_distances.len(),
            )
        )?;
    }
    stream.synchronize()?;

    // Copy results from GPU back to CPU.
    dev_results_distances.copy_to(&mut results_distances)?;

    Ok(results_distances)
}
