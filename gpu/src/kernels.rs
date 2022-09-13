use cuda_std::{prelude::*, vek::Vec3};

use crate::{
    aabb::DeviceCopyAabb,
    bvh::{NodeIndex, ObjectIndex},
};

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn brute_force(
    leaf_object_indices: &[ObjectIndex],
    leaf_aabbs: &[DeviceCopyAabb<f32>],
    //------
    internal_left_child_indicies: &[NodeIndex],
    internal_right_child_indicies: &[NodeIndex],
    internal_aabbs: &[DeviceCopyAabb<f32>],
    //------
    queries: &[Vec3<f32>],
    //------
    results_object_indices: *mut ObjectIndex,
    results_distances_squared: *mut f32,
) {
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while tid < queries.len() {
        let query = queries[tid];
        let (nn_object_index, nn_dist_squared) = brute_force_for_query(
            leaf_object_indices,
            leaf_aabbs,
            internal_left_child_indicies,
            internal_right_child_indicies,
            internal_aabbs,
            query,
        );

        let elem = &mut *results_object_indices.add(tid);
        *elem = nn_object_index;
        let elem = &mut *results_distances_squared.add(tid);
        *elem = nn_dist_squared;

        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}

pub fn brute_force_for_query(
    leaf_object_indices: &[ObjectIndex],
    leaf_aabbs: &[DeviceCopyAabb<f32>],
    //------
    internal_left_child_indicies: &[NodeIndex],
    internal_right_child_indicies: &[NodeIndex],
    internal_aabbs: &[DeviceCopyAabb<f32>],
    //------
    query: Vec3<f32>,
) -> (ObjectIndex, f32) {
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

    (nn_object_index, nn_dist)
}
