use crate::{dist2, rtree::RTree, stack::Stack};
use cuda_std::{prelude::*, shared_array, thread::thread_idx_x, vek::Vec3};

/// The number of elements in each node.
pub const M: usize = 4;

/// The height of the R-tree.
pub const H: usize = 8; // 4^8 = 65,536 leafs

// The number of threads per block.
pub const B: usize = 32;

// The number of elements to store in the objects caches that are used during
// brute force search of the leaf nodes.
pub const OBJECTS_CACHE_SIZE: usize = 256;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn bulk_find_neighbors(
    node_min_xs: &[f32],
    node_min_ys: &[f32],
    node_min_zs: &[f32],
    node_max_xs: &[f32],
    node_max_ys: &[f32],
    node_max_zs: &[f32],
    //------
    leaf_starts: &[usize],
    leaf_ends: &[usize],
    //------
    sorted_object_indices: &[usize],
    sorted_object_xs: &[f32],
    sorted_object_ys: &[f32],
    sorted_object_zs: &[f32],
    //------
    queries: &[Vec3<f32>],
    //------
    results_object_indices: *mut usize,
) {
    let mut i = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while i < queries.len() {
        let query = queries[i];

        let nn_obj_idx = find_neighbor(
            sorted_object_indices,
            sorted_object_xs,
            sorted_object_ys,
            sorted_object_zs,
            query,
        );

        *(&mut *results_object_indices.add(i)) = nn_obj_idx.unwrap();
        i += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}

unsafe fn find_neighbor(
    sorted_object_indices: &[usize],
    sorted_object_xs: &[f32],
    sorted_object_ys: &[f32],
    sorted_object_zs: &[f32],
    query: Vec3<f32>,
) -> Option<usize> {
    brute_force(
        sorted_object_indices,
        sorted_object_xs,
        sorted_object_ys,
        sorted_object_zs,
        query,
    )
}

unsafe fn brute_force(
    sorted_object_indices: &[usize],
    sorted_object_xs: &[f32],
    sorted_object_ys: &[f32],
    sorted_object_zs: &[f32],
    query: Vec3<f32>,
) -> Option<usize> {
    let mut min_dist2 = f32::INFINITY;
    let mut nn_object_idx = 0;

    for so_idx in 0..sorted_object_indices.len() {
        let x = sorted_object_xs[so_idx];
        let y = sorted_object_ys[so_idx];
        let z = sorted_object_zs[so_idx];
        let dist2 = dist2::to_point(query, x, y, z);

        if dist2 < min_dist2 {
            let o_idx = sorted_object_indices[so_idx];
            min_dist2 = dist2;
            nn_object_idx = o_idx;
        }
    }

    if nn_object_idx < usize::MAX {
        Some(nn_object_idx)
    } else {
        None
    }
}
