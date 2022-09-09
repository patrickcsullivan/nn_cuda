#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::{prelude::*, shared_array};

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

pub const THREADS_PER_BLOCK: usize = 256;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn eg_02_dot(input_size: usize, a: &[f32], b: &[f32], c: *mut f32) {
    let cache = shared_array![f32; THREADS_PER_BLOCK];
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    let cache_index = thread::thread_idx_x() as usize;

    let mut temp = 0.0f32;
    while tid < input_size {
        temp += a[tid] * b[tid];
        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }

    *cache.add(cache_index) = temp;

    thread::sync_threads();

    // for reductions, THREADS_PER_BLOCK must be a power of 2 because of the
    // following code
    let mut i = (thread::block_dim_x() / 2) as usize;
    while i != 0 {
        if cache_index < i {
            *cache.add(cache_index) += *cache.add(cache_index + i);
        }
        thread::sync_threads();
        i /= 2;
    }

    if cache_index == 0 {
        *c.add(thread::block_idx_x() as usize) = *cache.add(0);
    }
}
