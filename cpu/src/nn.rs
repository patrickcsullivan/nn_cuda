use cust::prelude::*;
use itertools::Itertools;
use std::error::Error;

static PTX: &str = include_str!("../../resources/gpu.ptx");

pub fn add_vecs(a: Vec<i64>, b: Vec<i64>) -> Result<Vec<i64>, Box<dyn Error>> {
    // Allocate memory on the CPU.
    let mut c = vec![0i64; a.len()];

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let kernel = module.get_function("eg_01_add")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate memory on the GPU.
    let dev_a = a.as_slice().as_dbuf()?;
    let dev_b = b.as_slice().as_dbuf()?;
    let dev_c = c.as_slice().as_dbuf()?;

    let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (a.len() as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(
            kernel<<<grid_size, block_size, 0, stream>>>(
                dev_a.as_device_ptr(),
                dev_a.len(),
                dev_b.as_device_ptr(),
                dev_b.len(),
                dev_c.as_device_ptr(),
                a.len()
            )
        )?;
    }
    stream.synchronize()?;

    // Copy c vector from GPU back to CPU.
    dev_c.copy_to(&mut c)?;

    Ok(c)
}

pub fn find_nn(queries: Vec<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
    // Allocate memory on the CPU.
    let mut result_distances = vec![0.0f32; queries.len()];

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let kernel = module.get_function("find_nn")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate memory on the GPU.
    let dev_queries = queries.as_slice().as_dbuf()?;
    let dev_result_distances = result_distances.as_slice().as_dbuf()?;

    let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (queries.len() as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(
            kernel<<<grid_size, block_size, 0, stream>>>(
                dev_queries.as_device_ptr(),
                queries.len(),
                dev_result_distances.as_device_ptr(),
            )
        )?;
    }
    stream.synchronize()?;

    // Copy results from GPU back to CPU.
    dev_result_distances.copy_to(&mut result_distances)?;

    Ok(result_distances)
}
