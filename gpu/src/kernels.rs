use cuda_std::{prelude::*, vek::Vec3};

use crate::{aabb::DeviceCopyAabb, bvh::ObjectIndex};

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_nn(
    leaf_object_indices: &[ObjectIndex],
    leaf_aabbs: &[DeviceCopyAabb<f32>],
    //------
    queries: &[Vec3<f32>],
    //------
    results_object_indices: *mut ObjectIndex,
    results_distances_squared: *mut f32,
) {
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while tid < queries.len() {
        let query = queries[tid];
        let mut nn_object_index = ObjectIndex(0);
        let mut nn_dist = f32::INFINITY;

        for leaf_index in 0..leaf_object_indices.len() {
            let aabb = leaf_aabbs[leaf_index];
            let dist = aabb.distance_squared_to_point(query);
            if dist < nn_dist {
                nn_object_index = leaf_object_indices[leaf_index];
                nn_dist = dist;
            }
        }

        let elem = &mut *results_object_indices.add(tid);
        *elem = nn_object_index;
        let elem = &mut *results_distances_squared.add(tid);
        *elem = nn_dist;

        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn brute_force(
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
