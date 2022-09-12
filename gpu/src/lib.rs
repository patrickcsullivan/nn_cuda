#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::{
    prelude::*,
    vek::{Aabb, Vec3},
};

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_nn(queries: &[Vec3<f32>], results_distances: *mut f32, results_count: usize) {
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while tid < results_count {
        let elem = &mut *results_distances.add(tid);
        *elem = queries[tid].distance_squared(Vec3::zero());
        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}
