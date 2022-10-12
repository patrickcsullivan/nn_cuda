use cuda_std::{prelude::*, shared_array, vek::Vec3};

use crate::{priority_queue::PriorityQueue, rtree::RTree};

/// The number of elements in each node.
pub const M: usize = 4;

/// The height of the R-tree.
pub const H: usize = 8; // 4^8 = 65,536 leafs

// The number of threads per block.
pub const B: usize = 32;

// The number of elements to store in the objects caches that are used during
// brute force search of the leaf nodes.
pub const O: usize = 256;

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
    let rtree = RTree::new(
        H,
        M,
        node_min_xs,
        node_min_ys,
        node_min_zs,
        node_max_xs,
        node_max_ys,
        node_max_zs,
        leaf_starts,
        leaf_ends,
        sorted_object_indices,
        sorted_object_xs,
        sorted_object_ys,
        sorted_object_zs,
    );

    // Allocate shared memory.
    const QUEUE_SIZE: usize = M * H;
    // let queue: SharedPriorityQueue<QUEUE_SIZE> =
    // SharedPriorityQueue::new_empty(); let dist2s_to_qs = shared_array![f32; M
    // * B ]; let shared_sorted_object_xs = shared_array![f32; O];
    // let shared_sorted_object_ys = shared_array![f32; O];
    // let shared_sorted_object_zs = shared_array![f32; O];

    let mut i = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while i < queries.len() {
        let query = queries[i];

        let nn_obj_idx = find_neighbor(
            rtree,
            // queue,
            // dist2s_to_qs,
            // shared_sorted_object_xs,
            // shared_sorted_object_ys,
            // shared_sorted_object_zs,
            query,
        );

        *(&mut *results_object_indices.add(i)) = nn_obj_idx.unwrap();

        i += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}

unsafe fn find_neighbor(rtree: RTree, query: Vec3<f32>) -> Option<usize> {
    Some(42)
}
