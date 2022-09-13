use cuda_std::{prelude::*, vek::Vec3};

use crate::{
    aabb::DeviceCopyAabb,
    bvh::{NodeIndex, ObjectIndex},
    stack::Stack,
};

const TRAVERSAL_STACK_MAX_SIZE: usize = 64;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_nn(
    leaf_object_indices: &[ObjectIndex],
    leaf_aabbs: &[DeviceCopyAabb<f32>],
    //------
    internal_left_child_indicies: &[NodeIndex],
    internal_right_child_indicies: &[NodeIndex],
    internal_aabbs: &[DeviceCopyAabb<f32>],
    root_index: NodeIndex,
    //------
    queries: &[Vec3<f32>],
    //------
    results_object_indices: *mut ObjectIndex,
    results_distances_squared: *mut f32,
) {
    let mut tid = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while tid < queries.len() {
        let query = queries[tid];
        let (nn_object_index, nn_dist_squared) = find_nn_for_query(
            leaf_object_indices,
            leaf_aabbs,
            internal_left_child_indicies,
            internal_right_child_indicies,
            internal_aabbs,
            root_index,
            query,
        );

        let elem = &mut *results_object_indices.add(tid);
        *elem = nn_object_index;
        let elem = &mut *results_distances_squared.add(tid);
        *elem = nn_dist_squared;

        tid += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}

pub fn find_nn_for_query(
    leaf_object_indices: &[ObjectIndex],
    leaf_aabbs: &[DeviceCopyAabb<f32>],
    //------
    internal_left_child_indicies: &[NodeIndex],
    internal_right_child_indicies: &[NodeIndex],
    internal_aabbs: &[DeviceCopyAabb<f32>],
    root_index: NodeIndex,
    //------
    query: Vec3<f32>,
) -> (ObjectIndex, f32) {
    let mut nn_object_index = ObjectIndex(0);
    let mut nn_dist_squared = f32::INFINITY;

    // Each thread will maintain a stack of nodes to visit for a depth-first
    // traversal.
    let mut traversal_stack = Stack::<(NodeIndex, f32), TRAVERSAL_STACK_MAX_SIZE>::empty();
    let root_dist_squared = point_to_node(leaf_aabbs, internal_aabbs, root_index, query);
    traversal_stack.push((root_index, root_dist_squared));

    while let Some((node_index, dist_squared)) = traversal_stack.pop() {
        match node_index {
            NodeIndex::Leaf(leaf_index) => {
                if dist_squared < nn_dist_squared {
                    nn_object_index = leaf_object_indices[leaf_index];
                    nn_dist_squared = dist_squared;
                }
            }
            NodeIndex::Internal(internal_index) => {
                let left_child_index = internal_left_child_indicies[internal_index];
                let right_child_index = internal_right_child_indicies[internal_index];

                // Check the minimum possible distance to each child node.
                let left_dist_squared =
                    point_to_node(leaf_aabbs, internal_aabbs, left_child_index, query);
                let right_dist_squared =
                    point_to_node(leaf_aabbs, internal_aabbs, right_child_index, query);

                // Push children that need to be traversed onto the traversal stack. When both
                // children need to be traversed, traverse the left child first.
                if right_dist_squared < nn_dist_squared {
                    traversal_stack.push((right_child_index, right_dist_squared));
                }
                if left_dist_squared < nn_dist_squared {
                    traversal_stack.push((left_child_index, left_dist_squared));
                }
            }
        }
    }

    (nn_object_index, nn_dist_squared)
}

/// Returns the squared distance between the point and the bounding box of the
/// specified node.
fn point_to_node(
    leaf_aabbs: &[DeviceCopyAabb<f32>],
    internal_aabbs: &[DeviceCopyAabb<f32>],
    node_index: NodeIndex,
    point: Vec3<f32>,
) -> f32 {
    get_aabb(leaf_aabbs, internal_aabbs, node_index).distance_squared_to_point(point)
}

/// Returns the bounding box of the specified node.
fn get_aabb(
    leaf_aabbs: &[DeviceCopyAabb<f32>],
    internal_aabbs: &[DeviceCopyAabb<f32>],
    node_index: NodeIndex,
) -> DeviceCopyAabb<f32> {
    match node_index {
        NodeIndex::Leaf(leaf_index) => leaf_aabbs[leaf_index],
        NodeIndex::Internal(internal_index) => internal_aabbs[internal_index],
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn brute_force(
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
