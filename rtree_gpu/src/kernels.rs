use crate::{dist2, rtree::RTree, stack::SharedStack};
use cuda_std::{
    prelude::*,
    shared_array,
    thread::{sync_threads, thread_idx_x},
    vek::Vec3,
};

/// The number of elements in each node.
pub const M: usize = 4;

/// The height of the R-tree.
pub const H: usize = 8; // 4^8 = 65,536 leafs

// The number of threads per block.
pub const B: usize = 32;

// The number of elements to store in the objects caches that are used during
// brute force search of the leaf nodes.
pub const OBJECTS_CACHE_SIZE: usize = 256;

pub const QUEUE_SIZE: usize = M * H;

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
    // Package the data into an R-tree structure that we can more easily call
    // methods on.
    let rtree: RTree = RTree::new(
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

    // Allocate shared memory for queue.
    // let queue_mem = shared_array![usize; QUEUE_SIZE];

    let mut i = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while i < queries.len() {
        let query = queries[i];

        let nn_obj_idx = find_neighbor(
            &rtree,
            sorted_object_indices,
            sorted_object_xs,
            sorted_object_ys,
            sorted_object_zs,
            // queue_mem,
            query,
        );

        *(&mut *results_object_indices.add(i)) = nn_obj_idx.unwrap();
        i += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}

unsafe fn find_neighbor(
    rtree: &RTree,
    sorted_object_indices: &[usize],
    sorted_object_xs: &[f32],
    sorted_object_ys: &[f32],
    sorted_object_zs: &[f32],
    // queue_mem: *mut usize,
    query: Vec3<f32>,
) -> Option<usize> {
    // Initialize a traversal queue in shared memory.
    let queue_elements_mem = shared_array![usize; QUEUE_SIZE];
    let queue_size_mem = shared_array![usize; 1];
    let mut queue = SharedStack::new(queue_elements_mem, queue_size_mem, QUEUE_SIZE);

    let mut min_dist2 = f32::INFINITY;
    let mut nn_object_idx = 0;

    // Perform a depth-first traversal of the tree.
    queue.push(rtree.root());
    while let Some(_node_idx) = queue.top() {
        queue.pop();

        // Brute force search through each leaf.
        for leaf_idx in 0..rtree.leaf_starts.len() {
            let start = rtree.leaf_starts[leaf_idx];
            let end = rtree.leaf_ends[leaf_idx];

            for so_idx in start..=end {
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
        }
    }

    if nn_object_idx < usize::MAX {
        Some(nn_object_idx)
    } else {
        None
    }
}
