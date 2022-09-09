#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn eg_01_add(a: &[i64], b: &[i64], c: *mut i64, n: usize) {
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while tid < n {
        let elem = &mut *c.add(tid);
        *elem = a[tid] + b[tid];
        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_nn(queries: &[f32], queries_count: usize, result_distances: *mut f32) {
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while tid < queries_count {
        let elem = &mut *result_distances.add(tid);
        *elem = queries[tid] * 10.0;
        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}
