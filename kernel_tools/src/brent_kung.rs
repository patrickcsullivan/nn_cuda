use cuda_std::{kernel, shared_array, thread};

/// The kernel launch should use this as the block size so that the number of
/// threads is equal to the number of section elements.
pub const BRENT_KUNG_SECTION_SIZE: usize = 1024;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn inclusive_brent_kung_scan(xs: &[u32], ys: *mut u32, input_len: usize) {
    let ti = thread::thread_idx_x() as usize;
    let bi = thread::block_idx_x() as usize;
    let bd = thread::block_dim_x() as usize;

    // Width of the block and shared by the entire block.
    let xys = shared_array![u32; BRENT_KUNG_SECTION_SIZE];

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
