use crate::step::{div_step, mult_step};
use cuda_std::{kernel, shared_array, thread};

/// The length of data to run the prefix operation on.
///
/// The max section size of this algorithm is double the max block size.
pub const SECTION_SIZE: usize = 2048;

/// The kernel launch should use this as the block size so that the number of
/// threads is equal to half the number of section elements.
pub const BLOCK_SIZE: usize = SECTION_SIZE / 2;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn inclusive_brent_kung_scan(xs: &[u32], ys: *mut u32) {
    inclusive_brent_kung_scan_stride(xs.as_ptr(), 1, ys, 1, SECTION_SIZE)
}

pub unsafe fn inclusive_brent_kung_scan_stride(
    xs: *const u32,
    xs_stride: usize,
    ys: *mut u32,
    ys_stride: usize,
    input_size: usize,
) {
    let t_idx = thread::thread_idx_x() as usize;
    // let b_idx = thread::block_idx_x() as usize;
    let b_dim = thread::block_dim_x() as usize;

    let xs1_idx = xs_stride * t_idx;
    let xs2_idx = xs_stride * (b_dim + t_idx);

    let ys1_idx = ys_stride * t_idx;
    let ys2_idx = ys_stride * (b_dim + t_idx);

    // Copy the contents of global memory xs into shared memory xys. Since the
    // section size is twice the size of the single block, each thread must load two
    // elements.
    let xys = shared_array![u32; SECTION_SIZE];
    if xs1_idx < input_size {
        *(&mut *xys.add(t_idx)) = *xs.add(xs1_idx);
    }
    if xs2_idx < input_size {
        *(&mut *xys.add(t_idx + b_dim)) = *xs.add(xs2_idx);
    }

    // Use a decreasing number of contiguous threads to peform reduction to compute
    // partial sums.
    for stride in mult_step(1, 2).take_while(|&s| s <= b_dim) {
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
    if ys1_idx < input_size {
        *(&mut *ys.add(ys1_idx)) = *xys.add(t_idx);
    }
    if ys2_idx + b_dim < input_size {
        *(&mut *ys.add(ys2_idx)) = *xys.add(t_idx + b_dim);
    }
}
