use cuda_std::{kernel, shared_array, thread};

use crate::step::{div_step, mult_step};

/// The length of data to run the prefix operation on.
pub const SECTION_SIZE: usize = 1024;

/// The kernel launch should use this as the block size so that the number of
/// threads is equal to half the number of section elements.
pub const BLOCK_SIZE: usize = SECTION_SIZE / 2;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn inclusive_brent_kung_scan(xs: &[u32], ys: *mut u32) {
    let t_idx = thread::thread_idx_x() as usize;
    let b_idx = thread::block_idx_x() as usize;
    let b_dim = thread::block_dim_x() as usize;
    let g_idx = 2 * b_idx * b_dim + t_idx; // global index

    // Copy the contents of global memory xs into shared memory xys. Since the
    // section size is twice the size of the single block, each thread must load two
    // elements.
    let xys = shared_array![u32; SECTION_SIZE];
    if g_idx < xs.len() {
        *(&mut *xys.add(t_idx)) = xs[g_idx];
    }
    if g_idx + b_dim < xs.len() {
        *(&mut *xys.add(t_idx + b_dim)) = xs[g_idx + b_dim];
    }

    // Use a decreasing number of contiguous threads to peform reduction to compute
    // partial sums.
    for stride in mult_step(1, 2).take_while(|&s| s < b_dim) {
        thread::sync_threads();
        let idx = (t_idx + 1) * 2 * stride - 1;
        if idx < SECTION_SIZE {
            *(&mut *xys.add(idx)) += *xys.add(idx - stride);
        }
    }

    for stride in div_step(SECTION_SIZE / 4, 2).take_while(|&s| s > 0) {
        thread::sync_threads();
        let idx = (t_idx + 1) * stride * 2 - 1;
        if idx + stride < SECTION_SIZE {
            *(&mut *xys.add(idx + stride)) += *xys.add(idx);
        }
    }

    // Copy the contents of shared memory xys into global memory ys.
    thread::sync_threads();
    if g_idx < xs.len() {
        *(&mut *ys.add(g_idx)) = *xys.add(t_idx);
    }
    if g_idx + b_dim < xs.len() {
        *(&mut *ys.add(g_idx + b_dim)) = *xys.add(t_idx + b_dim);
    }
}
