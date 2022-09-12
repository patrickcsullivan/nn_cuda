use cuda_std::{prelude::*, vek::Vec3};

use crate::aabb::DeviceCopyAabb;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_nn(
    object_aabbs: &[DeviceCopyAabb<f32>],
    queries: &[Vec3<f32>],
    results_distances_squared: *mut f32,
) {
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while tid < queries.len() {
        let query = queries[tid];
        let mut nn_dist = f32::INFINITY;

        for aabb in object_aabbs {
            let dist = aabb.distance_squared_to_point(query);
            if dist < nn_dist {
                nn_dist = dist;
            }
        }

        let elem = &mut *results_distances_squared.add(tid);
        *elem = nn_dist;

        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}
