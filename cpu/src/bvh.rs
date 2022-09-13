use std::fmt::Debug;

use crate::morton::morton_code;
use cuda_std::vek::{Aabb, Vec3};
use gpu::{
    aabb::DeviceCopyAabb,
    bvh::{NodeIndex, ObjectIndex},
};
use itertools::Itertools;

#[derive(Debug)]
pub struct Bvh {
    pub leaf_nodes: Vec<(ObjectIndex, DeviceCopyAabb<f32>)>,
    pub internal_nodes: Vec<(NodeIndex, NodeIndex, DeviceCopyAabb<f32>)>,
    pub root: NodeIndex,
}

impl Bvh {
    pub fn new_with_aabb(objects: &[Vec3<f32>], aabb: &Aabb<f32>) -> Self {
        let aabb = DeviceCopyAabb {
            min: aabb.min,
            max: aabb.max,
        };
        Self::new(objects, aabb)
    }

    pub fn new(objects: &[Vec3<f32>], aabb: DeviceCopyAabb<f32>) -> Self {
        let n = objects.len();
        let morton_codes = map_to_morton_codes(objects, &aabb);
        let mut sorted_object_indices = (0..n).map(ObjectIndex).collect_vec();

        // TODO: Radix sort on GPU.
        sorted_object_indices.sort_by_key(|&ObjectIndex(i)| morton_codes[i]);
        let sorted_morton_codes = sorted_object_indices
            .iter()
            .map(|&ObjectIndex(i)| morton_codes[i])
            .collect::<Vec<_>>();

        let root_node = top_down(
            objects,
            &sorted_object_indices,
            &sorted_morton_codes,
            0,
            n - 1,
        );

        let mut internal_nodes: Vec<(NodeIndex, NodeIndex, DeviceCopyAabb<f32>)> = vec![];
        let root_node_index = flatten(root_node, &mut internal_nodes);
        let leaf_nodes = sorted_object_indices
            .into_iter()
            .map(|oi| (oi, DeviceCopyAabb::new_empty(objects[oi.0])))
            .collect_vec();

        Self {
            leaf_nodes,
            internal_nodes,
            root: root_node_index,
        }
    }
}

#[derive(Debug)]
pub enum Node {
    Internal {
        left: Box<Node>,
        right: Box<Node>,
        aabb: DeviceCopyAabb<f32>,
    },
    Leaf {
        sorted_object_indices_index: usize,
        aabb: DeviceCopyAabb<f32>,
    },
}

impl Node {
    fn aabb(&self) -> DeviceCopyAabb<f32> {
        match self {
            Node::Internal {
                left: _,
                right: _,
                aabb,
            } => *aabb,
            Node::Leaf {
                sorted_object_indices_index: _,
                aabb,
            } => *aabb,
        }
    }
}

fn flatten(
    node: Node,
    internal_nodes: &mut Vec<(NodeIndex, NodeIndex, DeviceCopyAabb<f32>)>,
) -> NodeIndex {
    match node {
        Node::Internal { left, right, aabb } => {
            let left_index = flatten(*left, internal_nodes);
            let right_index = flatten(*right, internal_nodes);
            let index = internal_nodes.len();
            internal_nodes.push((left_index, right_index, aabb));
            NodeIndex::Internal(index)
        }
        Node::Leaf {
            sorted_object_indices_index,
            aabb: _,
        } => NodeIndex::Leaf(sorted_object_indices_index),
    }
}

fn top_down(
    objects: &[Vec3<f32>],
    sorted_object_indices: &[ObjectIndex],
    sorted_morton_codes: &[u32],
    first: usize,
    last: usize,
) -> Node {
    if first == last {
        let ObjectIndex(object_index) = sorted_object_indices[first];
        Node::Leaf {
            sorted_object_indices_index: first,
            aabb: DeviceCopyAabb::new_empty(objects[object_index]),
        }
    } else {
        let split = find_split(sorted_morton_codes, first, last);
        let left = top_down(
            objects,
            sorted_object_indices,
            sorted_morton_codes,
            first,
            split,
        );
        let right = top_down(
            objects,
            sorted_object_indices,
            sorted_morton_codes,
            split + 1,
            last,
        );
        let aabb = left.aabb().union(right.aabb());
        Node::Internal {
            left: Box::new(left),
            right: Box::new(right),
            aabb,
        }
    }
}

fn find_split(sorted_morton_codes: &[u32], first: usize, last: usize) -> usize {
    let first_code = sorted_morton_codes[first];
    let last_code = sorted_morton_codes[last];

    if first_code == last_code {
        // All Morton codes in the range are the same, so split down the middle.
        return (first + last) >> 1;
    }

    // Start the split position at the left-most index. Then move the split as far
    // forward as possible until the highest different bit is flipped. Try to
    // move forward at decreasing step sizes in a binary search fashion.
    let first_last_shared = (first_code ^ last_code).leading_zeros(); // TODO: CUDA __clz()
    let mut split = first;
    let mut step = last - first;

    loop {
        step = (step + 1) >> 1; // Cut search step in half, rounding up when step is odd.
        let new_split = split + step;

        if new_split < last {
            let split_code = sorted_morton_codes[new_split];
            let first_split_shared = (first_code ^ split_code).leading_zeros(); // TODO: CUDA __clz()
            if first_split_shared > first_last_shared {
                split = new_split;
            }
        }

        if step <= 1 {
            break;
        }
    }

    split
}

/// Maps each 3-dimensional point to a Morton code.
fn map_to_morton_codes(points: &[Vec3<f32>], aabb: &DeviceCopyAabb<f32>) -> Vec<u32> {
    let scale = aabb.size().recip();
    points
        .iter()
        .map(|p| {
            let p = (p - aabb.min) * scale;
            morton_code(p)
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::Bvh;
    use cuda_std::vek::Vec3;
    use gpu::aabb::DeviceCopyAabb;
    use itertools::Itertools;

    #[test]
    fn build_bvh() {
        let upper = 2;
        let mut points = (0..upper)
            .into_iter()
            .flat_map(|x| {
                (0..upper).flat_map(move |y| {
                    (0..upper).map(move |z| Vec3::new(x as f32, y as f32, z as f32))
                })
            })
            .collect_vec();
        points.reverse();
        let aabb = DeviceCopyAabb::new_empty(points[0])
            .union(DeviceCopyAabb::new_empty(points[points.len() - 1]));
        let _bvh = Bvh::new(&points, aabb);
        assert!(true);
    }
}
