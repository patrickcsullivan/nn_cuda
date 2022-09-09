//! Based on the dot product example from _Cuda by Example_, Section 5.3.1.

use cust::prelude::*;
use std::error::Error;

/// Length of vectors on which to perform dot product.
const N: usize = 33 * 1024;

const THREADS_PER_BLOCK: usize = 256;
const BLOCKS_PER_GRID: usize = 32;

static PTX: &str = include_str!("../../resources/gpu.ptx");

pub fn run() -> Result<(), Box<dyn Error>> {
    println!("Example 2, Dot Product:");

    // Allocate memory on the CPU.
    let mut a = vec![0f32; N];
    let mut b = vec![0f32; N];
    let mut partial_c = vec![0f32; BLOCKS_PER_GRID];

    (0..N).for_each(|i| {
        a[i] = ((i + 1) as f32) / 3.0;
        b[i] = 3.0 / ((i + 1) as f32);
    });

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let dot_func = module.get_function("eg_02_dot")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate memory on the GPU.
    let dev_a = a.as_slice().as_dbuf()?;
    let dev_b = b.as_slice().as_dbuf()?;
    let dev_partial_c = partial_c.as_slice().as_dbuf()?;

    unsafe {
        launch!(
            dot_func <<<BLOCKS_PER_GRID as u32, THREADS_PER_BLOCK as u32, 0, stream>>>(
                N,
                dev_a.as_device_ptr(),
                dev_a.len(),
                dev_b.as_device_ptr(),
                dev_b.len(),
                dev_partial_c.as_device_ptr(),
            )
        )?;
    }
    stream.synchronize()?;

    // Copy partial_c vector from GPU back to CPU.
    dev_partial_c.copy_to(&mut partial_c)?;

    // Finish sum on CPU.
    let c: f32 = partial_c.iter().sum();

    // Verify that GPU kernel worked.
    println!("Does GPU value {:.6} = {:.6}?", c, N as f32);

    Ok(())
}
