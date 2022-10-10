use cust::prelude::*;
use kernel_tools::{brent_kung, kogge_stone, three_phase};
use std::error::Error;

static PTX: &str = include_str!("../../resources/kernel_tools.ptx");

/// Inclusive scan.
pub fn sequential_scan(xs: &[u32], ys: &mut [u32], max_i: usize) {
    let mut accumulator = xs[0];
    ys[0] = accumulator;

    for i in 1..max_i {
        accumulator += xs[i];
        ys[i] = accumulator;
    }
}

/// Inclusive scan.
pub fn kogge_stone_scan(stream: &Stream, xs: &[u32]) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut ys = vec![0u32; kogge_stone::SECTION_SIZE];

    let dev_xs = xs.as_dbuf()?;
    let dev_ys = ys.as_slice().as_dbuf()?;

    let module = Module::from_ptx(PTX, &[])?;
    let kernel = module.get_function("inclusive_kogge_stone_scan")?;

    unsafe {
        launch!(
            kernel<<<1, kogge_stone::BLOCK_SIZE as u32, 0, stream>>>(
                dev_xs.as_device_ptr(),
                dev_xs.len(),
                dev_ys.as_device_ptr(),
                kogge_stone::SECTION_SIZE
            )
        )?;
    }
    stream.synchronize()?;

    dev_ys.copy_to(&mut ys)?;
    Ok(ys)
}

/// Inclusive scan.
pub fn brent_kung_scan(stream: &Stream, xs: &[u32]) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut ys = vec![0u32; brent_kung::SECTION_SIZE];

    let dev_xs = xs.as_dbuf()?;
    let dev_ys = ys.as_slice().as_dbuf()?;

    let module = Module::from_ptx(PTX, &[])?;
    let kernel = module.get_function("inclusive_brent_kung_scan")?;

    unsafe {
        launch!(
            kernel<<<1, brent_kung::BLOCK_SIZE as u32, 0, stream>>>(
                dev_xs.as_device_ptr(),
                dev_xs.len(),
                dev_ys.as_device_ptr(),
                brent_kung::SECTION_SIZE
            )
        )?;
    }
    stream.synchronize()?;

    dev_ys.copy_to(&mut ys)?;
    Ok(ys)
}

/// Inclusive scan.
pub fn three_phase_scan(stream: &Stream, xs: &[u32]) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut ys = vec![0u32; three_phase::SECTION_SIZE];

    let dev_xs = xs.as_dbuf()?;
    let dev_ys = ys.as_slice().as_dbuf()?;

    let module = Module::from_ptx(PTX, &[])?;
    let kernel = module.get_function("inclusive_three_phase_scan")?;

    unsafe {
        launch!(
            kernel<<<1, three_phase::BLOCK_SIZE as u32, 0, stream>>>(
                dev_xs.as_device_ptr(),
                dev_xs.len(),
                dev_ys.as_device_ptr(),
                three_phase::SECTION_SIZE
            )
        )?;
    }
    stream.synchronize()?;

    dev_ys.copy_to(&mut ys)?;
    Ok(ys)
}
