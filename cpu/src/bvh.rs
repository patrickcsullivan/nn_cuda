use std::fmt::Debug;

use crate::morton::morton_code;
use cuda_std::vek::Vec3;
use gpu::aabb::DeviceCopyAabb;

pub enum Node {
    InternalNode {
        left: Box<Node>,
        right: Box<Node>,
        aabb: DeviceCopyAabb<f32>,
    },
    LeafNode {
        object_index: usize,
        aabb: DeviceCopyAabb<f32>,
    },
}

impl Node {
    fn aabb(&self) -> DeviceCopyAabb<f32> {
        match self {
            Node::InternalNode {
                left: _,
                right: _,
                aabb,
            } => *aabb,
            Node::LeafNode {
                object_index: _,
                aabb,
            } => *aabb,
        }
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalNode { left, right, aabb } => f
                .debug_struct("InternalNode")
                .field("left", left)
                .field("right", right)
                .field("aabb", aabb)
                .finish(),
            Self::LeafNode { object_index, aabb } => f
                .debug_struct("LeafNode")
                .field("object_index", object_index)
                .field("aabb", aabb)
                .finish(),
        }
    }
}

pub fn build(objects: &[Vec3<f32>], aabb: DeviceCopyAabb<f32>) -> Node {
    let n = objects.len();
    let morton_codes = map_to_morton_codes(objects, &aabb);
    let mut sorted_object_indices: Vec<usize> = (0..n).collect();

    // TODO: Radix sort on GPU.
    sorted_object_indices.sort_by_key(|&i| morton_codes[i]);
    let sorted_morton_codes = sorted_object_indices
        .iter()
        .map(|&i| morton_codes[i])
        .collect::<Vec<_>>();

    let root = top_down(
        objects,
        &sorted_object_indices,
        &sorted_morton_codes,
        0,
        n - 1,
    );

    root
}

fn top_down(
    objects: &[Vec3<f32>],
    sorted_object_indices: &[usize],
    sorted_morton_codes: &[u32],
    first: usize,
    last: usize,
) -> Node {
    if first == last {
        let object_index = sorted_object_indices[first];
        Node::LeafNode {
            object_index,
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
        Node::InternalNode {
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
    use super::build;
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
        let bvh = build(&points, aabb);
        println!("POINTS: {:#?}", points);
        println!("BVH: {:#?}", bvh);
        assert!(true);
    }
}
