use cuda_std::thread::{block_dim_x, sync_threads, thread_idx_x};

use crate::step::mult_step;

/// Performs an inclusive scan using minimum as the binary operator. The length
/// of `values` and `out` must be equal to the number of threads in a block.
///
/// Implemented using the the Kogge-Stone algorithm.
pub fn min_scan_block(values: *mut f32, out: *mut f32, shared_scratch: *mut f32) {
    let t_idx = thread_idx_x() as usize;

    // Copy the input values into shared memory.
    unsafe {
        *(&mut *shared_scratch.add(t_idx)) = *values.add(t_idx);
    }

    // Perform an iterative scan over the shared memory.
    for stride in mult_step(1, 2).take_while(|&s| s < block_dim_x() as usize) {
        sync_threads();

        if t_idx >= stride {
            unsafe {
                let min = (*shared_scratch.add(t_idx)).min(*shared_scratch.add(t_idx - stride));
                *(&mut *shared_scratch.add(t_idx)) = min;
            }
        }
    }

    // Copy the contents of shared memory into the output array.
    unsafe {
        *(&mut *out.add(t_idx)) = *shared_scratch.add(t_idx);
    }
}
