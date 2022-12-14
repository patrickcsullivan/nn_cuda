use crate::{
    dist2, grid,
    rtree::{NodeContents, RTree},
    shared_aabbs::SharedAabbs,
    shared_stack::SharedStack,
};
use cuda_std::{
    prelude::*,
    print as cuda_print, println as cuda_println, shared_array,
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
pub const OBJECTS_CACHE_SIZE: usize = B;

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
        // cuda_println!("{}\t{}\t{}", block_idx_x(), thread_idx_x(), i);
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

    // Initialize the shared memory for storing the object positions when performing
    // a brute force search of objects in a leaf node.
    let xs_cache = shared_array![f32; OBJECTS_CACHE_SIZE];
    let ys_cache = shared_array![f32; OBJECTS_CACHE_SIZE];
    let zs_cache = shared_array![f32; OBJECTS_CACHE_SIZE];

    let mut nn_dist2 = f32::INFINITY;
    let mut nn_object_idx = 0;

    // Let the 0th thread initialize the queue size in shared memory before pushing
    // elements onto the queue.
    sync_threads();

    // Perform a depth-first traversal of the tree.
    queue.push(rtree.root());

    // Let the 0th thread push the root node onto the queue before all threads try
    // to read the queue top.
    sync_threads();

    while let Some(node_idx) = queue.top() {
        // Let all threads read the queue top before popping it.
        sync_threads();

        // debug(&format!("node {}", node_idx));

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

                // Let all threads finish writing the AABBs before reading the AABBs.
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

                    let grid_idx = grid_index_at(B, i, t_idx);
                    *(&mut *dist2s_mem.add(grid_idx)) = dist2;
                }

                // Let each thread finish writing the child distances to its query before trying
                // to find the minimum distance to each child node.
                sync_threads();

                // For each child node, find the minimum distance between the node and the
                // threads' queries.
                // TODO: Switch to efficient scan.
                if t_idx < rtree.children_per_node {
                    let mut min_dist2 = f32::INFINITY;
                    for col_idx in 0..B {
                        let grid_idx = grid_index_at(B, t_idx, col_idx);
                        let dist2 = *dist2s_mem.add(grid_idx);
                        if dist2 < min_dist2 {
                            min_dist2 = dist2;
                        }
                    }

                    // We write the results of the scan back onto the row in the shared grid, so
                    // the last element of each row in the grid will contain the minimum squared
                    // distance.
                    let grid_idx = grid_index_at(B, t_idx, B - 1);
                    *(&mut *dist2s_mem.add(grid_idx)) = min_dist2;
                }

                // Let all threads finish calculating and/or writing the minimum distance to
                // each child node before reading the minimum distances.
                sync_threads();

                // Add new nodes to the traversal queue.
                if t_idx == 0 {
                    for i in 0..rtree.children_per_node {
                        // Get the minimum distance for each child node from the last element of
                        // each row in the grid.
                        let grid_idx = grid_index_at(B, i, B - 1);
                        let dist2 = *dist2s_mem.add(grid_idx);

                        // A child node will have a finite minimum distance if it potentially
                        // contains a nearest neighbor for at least one thread's query.
                        if dist2 < f32::INFINITY {
                            let child_idx = children_start_idx + i;
                            queue.push(child_idx);
                        }
                    }
                }

                // Let the 0-th thread finish pushing nodes onto the queue before all threads
                // try to read the queue top.
                sync_threads();
            }
            NodeContents::LeafObjects { start, end } => {
                // Loop through each point in leaf, checking to see if it's the NN.
                for sorted_data_idx in start..=end {
                    // let cache_idx = (sorted_data_idx - start) % OBJECTS_CACHE_SIZE;

                    // // If we're trying to examine the first point in the cache, then scan forward
                    // // and load the cache.
                    // if cache_idx == 0 {
                    //     thread::sync_threads();
                    //     let mut look_ahead_idx = thread::thread_idx_x() as usize;
                    //     while look_ahead_idx < OBJECTS_CACHE_SIZE
                    //         && sorted_data_idx + look_ahead_idx <= end
                    //     {
                    //         *xs_cache.add(look_ahead_idx) =
                    //             sorted_object_xs[sorted_data_idx + look_ahead_idx];
                    //         *ys_cache.add(look_ahead_idx) =
                    //             sorted_object_ys[sorted_data_idx + look_ahead_idx];
                    //         *zs_cache.add(look_ahead_idx) =
                    //             sorted_object_zs[sorted_data_idx + look_ahead_idx];
                    //         look_ahead_idx += B;
                    //     }
                    //     thread::sync_threads();
                    // }

                    let x = sorted_object_xs[sorted_data_idx];
                    let y = sorted_object_ys[sorted_data_idx];
                    let z = sorted_object_zs[sorted_data_idx];
                    let dist2 = dist2::to_point(query, x, y, z);
                    if dist2 < nn_dist2 {
                        nn_object_idx = sorted_object_indices[sorted_data_idx];
                        nn_dist2 = dist2;
                    }
                }

                /*
                // Brute force search the objects in the leaf node.
                let end = end + 1; // make end exclusive

                let mut chunk_start = start;
                while chunk_start < end {
                    let chunk_end = (chunk_start + OBJECTS_CACHE_SIZE).min(end);
                    let chunk_size = chunk_end - chunk_start;

                    debug(&format!("chunk at [{}, {})", chunk_start, chunk_end));

                    // Let all threads finish reading data to the cache from the last iteration of
                    // the loop before all threads try to write to the cache again..
                    sync_threads();

                    debug(&format!("sync before"));

                    // Load the next chunk into the cache.
                    if t_idx < chunk_size {
                        let so_idx = chunk_start + t_idx;
                        *(&mut *xs_cache.add(t_idx)) = sorted_object_xs[so_idx];
                        *(&mut *ys_cache.add(t_idx)) = sorted_object_ys[so_idx];
                        *(&mut *zs_cache.add(t_idx)) = sorted_object_zs[so_idx];
                        debug(&format!("so_idx {}", so_idx));
                    }

                    // Let all threads finish writing data to the cache before all threads try to
                    // read from the cache.
                    sync_threads();

                    for i in 0..chunk_size {
                        let x = sorted_object_xs[chunk_start + i];
                        let y = sorted_object_ys[chunk_start + i];
                        let z = sorted_object_zs[chunk_start + i];
                        let dist2 = dist2::to_point(query, x, y, z);

                        if dist2 < nn_dist2 {
                            let o_idx = sorted_object_indices[chunk_start + i];
                            nn_dist2 = dist2;
                            nn_object_idx = o_idx;
                        }
                    }

                    // // Scan the loaded chunk.
                    // for i in 0..chunk_size {
                    //     let x = *xs_cache.add(i);
                    //     let y = *ys_cache.add(i);
                    //     let z = *zs_cache.add(i);
                    //     let dist2 = dist2::to_point(query, x, y, z);

                    //     if dist2 < nn_dist2 {
                    //         let o_idx = sorted_object_indices[chunk_start + i];
                    //         nn_dist2 = dist2;
                    //         nn_object_idx = o_idx;
                    //     }
                    // }

                    chunk_start += OBJECTS_CACHE_SIZE;
                }

                // Let the 0-th thread finish popping a node from the queue before all threads
                // try to read the queue top.
                sync_threads();
                 */
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

fn brute_force1(
    nn_object_idx: &mut usize,
    nn_dist2: &mut f32,
    sorted_object_xs: &[f32],
    sorted_object_ys: &[f32],
    sorted_object_zs: &[f32],
    sorted_object_indices: &[usize],
    query: Vec3<f32>,
) {
    for i in 0..sorted_object_indices.len() {
        let x = sorted_object_xs[i];
        let y = sorted_object_ys[i];
        let z = sorted_object_zs[i];
        let dist2 = dist2::to_point(query, x, y, z);

        if dist2 < *nn_dist2 {
            let o_idx = sorted_object_indices[i];
            *nn_dist2 = dist2;
            *nn_object_idx = o_idx;
        }
    }
}

fn brute_force2(
    nn_object_idx: &mut usize,
    nn_dist2: &mut f32,
    sorted_object_xs: &[f32],
    sorted_object_ys: &[f32],
    sorted_object_zs: &[f32],
    sorted_object_indices: &[usize],
    query: Vec3<f32>,
) {
    let xs_cache = shared_array![f32; OBJECTS_CACHE_SIZE];
    let ys_cache = shared_array![f32; OBJECTS_CACHE_SIZE];
    let zs_cache = shared_array![f32; OBJECTS_CACHE_SIZE];

    let mut chunk_start = 0;
    while chunk_start < sorted_object_indices.len() {
        let chunk_end = (chunk_start + OBJECTS_CACHE_SIZE).min(sorted_object_indices.len());
        let chunk_size = chunk_end - chunk_start;

        // Load the next chunk into the cache.
        sync_threads();
        let mut i = thread_idx_x() as usize;
        while i < chunk_size {
            let so_idx = chunk_start + i;
            unsafe {
                *(&mut *xs_cache.add(i)) = sorted_object_xs[so_idx];
                *(&mut *ys_cache.add(i)) = sorted_object_ys[so_idx];
                *(&mut *zs_cache.add(i)) = sorted_object_zs[so_idx];
            }
            i += block_dim_x() as usize;
        }

        // Scan the loaded chunk.
        sync_threads();
        for i in 0..chunk_size {
            let x = unsafe { *xs_cache.add(i) };
            let y = unsafe { *ys_cache.add(i) };
            let z = unsafe { *zs_cache.add(i) };
            let dist2 = dist2::to_point(query, x, y, z);

            if dist2 < *nn_dist2 {
                let o_idx = sorted_object_indices[chunk_start + i];
                *nn_dist2 = dist2;
                *nn_object_idx = o_idx;
            }
        }

        chunk_start += OBJECTS_CACHE_SIZE;
    }
}

fn debug(msg: &str) {
    if block_idx_x() == 0 {
        cuda_println!("{}\t{}", thread_idx_x(), msg);
    }
}
