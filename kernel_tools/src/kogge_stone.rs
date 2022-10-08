use crate::step::mult_step;
use cuda_std::{kernel, shared_array, thread};

/// The length of data to run the prefix operation on.
pub const SECTION_SIZE: usize = 64;

/// The kernel launch should use this as the block size so that the number of
/// threads is equal to the number of section elements.
pub const BLOCK_SIZE: usize = SECTION_SIZE;

/// Performs an inclusive prefix sum scan on the elements of a single 64-element
/// block.
///
/// The length of `xs`, `ys`, and the kernel launch block size should all be
/// equal to 64.
///
/// This uses the Kogge-Stone algorithm. The algorithm is step-efficient but not
/// work-efficent, therefore this algorithm is not the best choice when `xs` is
/// larger than the number of available execution units, but it is a good choice
/// when `xs` is small relative to the number of execution units available.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn inclusive_kogge_stone_scan(xs: &[u32], ys: *mut u32) {
    let ti = thread::thread_idx_x() as usize;
    let bi = thread::block_idx_x() as usize;
    let bd = thread::block_dim_x() as usize;

    // Width of the block and shared by the entire block.
    let xys = shared_array![u32; SECTION_SIZE];

    // Copy the contents of global memory xs into shared memory xys.
    let i = bi * bd + ti;
    if i < xs.len() {
        *(&mut *xys.add(ti)) = xs[i];
    }

    // Perform an iterative scan over xys.
    for stride in mult_step(1, 2).take_while(|&s| s < bd) {
        thread::sync_threads();

        if ti >= stride {
            *(&mut *xys.add(ti)) += *xys.add(ti - stride);
        }
    }

    // Copy the contents of shared memory xys into global memory ys.
    *(&mut *ys.add(i)) = *xys.add(ti);
}
