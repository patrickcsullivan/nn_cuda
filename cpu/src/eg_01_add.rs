//! Based on the vector sum example from _Cuda by Example_, Section 5.2.1.

use cust::prelude::*;
use std::error::Error;

/// How many numbers to generate and add together.
const NUMBERS_LEN: usize = 33 * 1024;

static PTX: &str = include_str!("../../resources/gpu.ptx");

pub fn run() -> Result<(), Box<dyn Error>> {
    println!("Example 1, Vector Sum");

    // Allocate memory on the CPU.
    let mut a = vec![0i64; NUMBERS_LEN];
    let mut b = vec![0i64; NUMBERS_LEN];
    let mut c = vec![0i64; NUMBERS_LEN];

    (0..NUMBERS_LEN).for_each(|i| {
        a[i] = i as i64;
        b[i] = (i * i) as i64;
    });

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let add_func = module.get_function("eg_01_add")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate memory on the GPU.
    let dev_a = a.as_slice().as_dbuf()?;
    let dev_b = b.as_slice().as_dbuf()?;
    let dev_c = c.as_slice().as_dbuf()?;

    // // Use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // // current CUDA device/architecture.
    // let (_, block_size) = add_func.suggested_launch_configuration(0, 0.into())?;
    // let grid_size = (NUMBERS_LEN as u32 + block_size - 1) / block_size;

    // println!(
    //     "using {} blocks and {} threads per block",
    //     grid_size, block_size
    // );

    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            add_func<<<128, 128, 0, stream>>>(
                dev_a.as_device_ptr(),
                dev_a.len(),
                dev_b.as_device_ptr(),
                dev_b.len(),
                dev_c.as_device_ptr(),
                NUMBERS_LEN
            )
        )?;
    }
    stream.synchronize()?;

    // Copy c vector from GPU back to CPU.
    dev_c.copy_to(&mut c)?;

    // Verify that GPU kernel worked.
    let mut success = true;
    (0..NUMBERS_LEN).for_each(|i| {
        //
        if a[i] + b[i] != c[i] {
            println!("Error: {} + {} != {}", a[i], b[i], c[i]);
            success = false;
        }
    });

    if success {
        println!("We did it!");
    }

    Ok(())
}
