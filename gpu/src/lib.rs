#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_nn(a: &[usize], b: &[usize], c: *mut usize, n: usize) {
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while tid < n {
        let elem = &mut *c.add(tid);
        *elem = a[tid] + b[tid];
        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_nn_2(queries: &[f32], results_distances: *mut f32, results_count: usize) {
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while tid < results_count {
        let elem = &mut *results_distances.add(tid);
        *elem = queries[tid] * 10.0;
        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}
