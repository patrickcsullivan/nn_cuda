use cuda_std::{kernel, shared_array, thread};

/// Kernel launch should use this as the block size so that the number of
/// threads is equal to the number of section elements.
pub const SECTION_SIZE: usize = 64;

/// Inclusive prefix scan on the elements of a block. This assumes that the
/// `input_len`, `SECTION_SIZE`, and the kernel launch block size are all equal.
/// Uses the Kogge-Stone algorithm.
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn kogge_stone_scan(xs: &[u32], ys: *mut u32, input_len: usize) {
    let ti = thread::thread_idx_x() as usize;
    let bi = thread::block_idx_x() as usize;
    let bd = thread::block_dim_x() as usize;

    // Width of the block and shared by the entire block.
    let xys = shared_array![u32; SECTION_SIZE];

    // Copy the contents of global memory xs into shared memory xys.
    let i = bi * bd + ti;
    if i < input_len {
        *(&mut *xys.add(ti)) = xs[i];
    }

    // Perform an iterative scan over xys.
    let mut stride: usize = 1;
    while stride < bd {
        thread::sync_threads();

        if ti >= stride {
            *(&mut *xys.add(ti)) += *xys.add(ti - stride);
        }

        stride *= 2;
    }

    // Copy the contents of shared memory xys into global memory ys.
    *(&mut *ys.add(i)) = *xys.add(ti);
}

// #[inline(always)]
// unsafe fn get_at(array: *mut u32, i: usize) -> u32 {
//     *array.add(i)
// }

// #[inline(always)]
// unsafe fn set_at(array: *mut u32, i: usize, val: u32) {
//     *(&mut *array.add(i)) = val
// }

// #[inline(always)]
// fn thread_idx_x() -> usize {
//     cuda_std::thread::thread_idx_x as usize
// }

// #[inline(always)]
// fn block_dim_x() -> usize {
//     cuda_std::thread::block_dim_x as usize
// }

// #[inline(always)]
// fn block_idx_x() -> usize {
//     cuda_std::thread::block_idx_x as usize
// }
