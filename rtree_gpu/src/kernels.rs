use crate::{
    dist2,
    rtree::{NodeContents, RTree},
    shared_aabbs::SharedAabbs,
    shared_stack::SharedStack,
};
use cuda_std::{
    prelude::*,
    shared_array,
    thread::{block_dim_x, block_idx_x, sync_threads, thread_idx_x},
    vek::{Aabb, Vec3},
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
    let t_idx = thread_idx_x() as usize;
    let b_idx = block_idx_x() as usize;
    let b_dim = block_dim_x() as usize;

    // Initialize a traversal queue in shared memory.
    let queue_elements_mem = shared_array![usize; QUEUE_SIZE];
    let queue_size_mem = shared_array![usize; 1];
    let mut queue = SharedStack::new(queue_elements_mem, queue_size_mem, QUEUE_SIZE);

    // Initialize shared memory for storing the bounding boxes of the current node's
    // children.
    let children_min_xs_mem = shared_array![f32; M];
    let children_min_ys_mem = shared_array![f32; M];
    let children_min_zs_mem = shared_array![f32; M];
    let children_max_xs_mem = shared_array![f32; M];
    let children_max_ys_mem = shared_array![f32; M];
    let children_max_zs_mem = shared_array![f32; M];

    // Initialize the shared memory for storing the squared distances between the
    // block's queries and the current node's children. The grid is row-major. Each
    // row contains the squared distances to an individual child node, and each
    // column contains the squared distances to an individual query.
    let dist2s_mem = shared_array![f32; B * M];

    let mut nn_dist2 = f32::INFINITY;
    let mut nn_object_idx = 0;

    // Perform a depth-first traversal of the tree.
    queue.push(rtree.root());
    while let Some(node_idx) = queue.top() {
        queue.pop();

        match rtree.get_contents(node_idx) {
            NodeContents::InteriorChildren {
                start: children_start_idx,
            } => {
                // Load the AABBs for the node's children into shared memory.
                if t_idx < rtree.children_per_node {
                    let child_idx = children_start_idx + t_idx;
                    *(&mut *children_min_xs_mem.add(t_idx)) = rtree.node_min_xs[child_idx];
                    *(&mut *children_min_ys_mem.add(t_idx)) = rtree.node_min_ys[child_idx];
                    *(&mut *children_min_zs_mem.add(t_idx)) = rtree.node_min_zs[child_idx];
                    *(&mut *children_max_xs_mem.add(t_idx)) = rtree.node_max_xs[child_idx];
                    *(&mut *children_max_ys_mem.add(t_idx)) = rtree.node_max_ys[child_idx];
                    *(&mut *children_max_zs_mem.add(t_idx)) = rtree.node_max_zs[child_idx];
                }
                sync_threads();

                // Find the squared distance of each child to each thread's query.
                for i in 0..rtree.children_per_node {
                    // Load the bounding box from shared memory.
                    let min_x = *children_min_xs_mem.add(i);
                    let min_y = *children_min_ys_mem.add(i);
                    let min_z = *children_min_zs_mem.add(i);
                    let max_x = *children_max_xs_mem.add(i);
                    let max_y = *children_max_ys_mem.add(i);
                    let max_z = *children_max_zs_mem.add(i);
                    let aabb = Aabb {
                        min: Vec3::new(min_x, min_y, min_z),
                        max: Vec3::new(max_x, max_y, max_z),
                    };

                    let dist2 = dist2::to_aabb(&query, &aabb);
                    let dist2 = if dist2 < nn_dist2 {
                        dist2
                    } else {
                        // Save the squared distance as infinity to indicate that we will have
                        // no need to visit the child node for the thread's query.
                        f32::INFINITY
                    };

                    let grid_idx = grid_index_at(b_dim, i, t_idx);
                    *(&mut *dist2s_mem.add(grid_idx)) = dist2;
                }
                sync_threads();

                // For each child node, find the minimum distance between the node and the
                // threads' queries.
                // TODO: Switch to efficient scan.
                if t_idx < rtree.children_per_node {
                    let mut min_dist2 = f32::INFINITY;
                    for col_idx in 0..b_dim {
                        let grid_idx = grid_index_at(b_dim, t_idx, col_idx);
                        let dist2 = *dist2s_mem.add(grid_idx);
                        if dist2 < min_dist2 {
                            min_dist2 = dist2;
                        }
                    }

                    let grid_idx = grid_index_at(b_dim, t_idx, b_dim - 1);
                    *(&mut *dist2s_mem.add(grid_idx)) = min_dist2;
                }

                // for i in 0..self.children_per_node {
                //     let shared_dist2s = shared_child_to_query_dist2s.get_row_ptr(i);
                //     // We write the results of the scan back onto the row in the shared grid,
                // so     // the last element of each row in the grid will
                // contain the minimum squared     // distance.
                //     min_scan_block(shared_dist2s, shared_dist2s, shared_scan_scratch);
                //     sync_threads();
                // }

                // Add new nodes to the traversal queue.
                for i in 0..rtree.children_per_node {
                    let child_idx = children_start_idx + i;
                    queue.push(child_idx);
                }
            }
            NodeContents::LeafObjects { start, end } => {
                // Brute force search through each leaf.
                // TODO: Use caching.
                for so_idx in start..=end {
                    let x = sorted_object_xs[so_idx];
                    let y = sorted_object_ys[so_idx];
                    let z = sorted_object_zs[so_idx];
                    let dist2 = dist2::to_point(query, x, y, z);

                    if dist2 < nn_dist2 {
                        let o_idx = sorted_object_indices[so_idx];
                        nn_dist2 = dist2;
                        nn_object_idx = o_idx;
                    }
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

fn grid_index_at(row_width: usize, row_idx: usize, col_idx: usize) -> usize {
    row_idx * row_width + col_idx
}
