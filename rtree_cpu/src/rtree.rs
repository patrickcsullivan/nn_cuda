use crate::morton::map_to_morton_codes;
use cuda_std::vek::Vec3;
use itertools::Itertools;
use rtree_gpu::{aabb::DeviceCopyAabb, bvh::ObjectIndex};

pub struct RTree {
    /// Contains the size of each level in the tree.
    ///
    /// The `i`-th element of `sizes` contains the number of nodes in level `i`.
    sizes: Vec<usize>,

    /// The `i`-th element of `starts` contains the first child node index of
    /// the `i`-th node.
    starts: Vec<usize>,

    /// The `i`-th element of `ends` contains the last child node index of the
    /// `i`-th node.
    ends: Vec<usize>,
}

impl RTree {
    pub fn new(
        objects: &[Vec3<f32>],
        min_entry_size: usize,
        max_entry_size: usize,
        aabb: DeviceCopyAabb<f32>,
    ) -> Self {
        // Sort objects by morton code to order them on a Z-order curve.
        let n = objects.len();
        let morton_codes = map_to_morton_codes(objects, &aabb);
        let mut sorted_object_indices = (0..n).map(ObjectIndex).collect_vec();

        let mut tree = Self {
            sizes: vec![],
            starts: vec![],
            ends: vec![],
        };

        // Determine size of each level.
        let mut last_level_size = n;
        while last_level_size > max_entry_size {
            let next_level_size = div_ceil(last_level_size, min_entry_size);
            tree.sizes.push(next_level_size);
            last_level_size = next_level_size;
        }

        for level_index in 0..tree.sizes.len() {
            let level_size = tree.sizes[level_index];
            let level_first_node_index = prefix_sum(&tree.sizes, level_index);

            // TODO: Continue here.
            // for node_index in
        }

        tree.build_level(0, n);

        todo!()
    }

    fn build_level(&mut self, level: usize, previous_level_size: usize) {}
}

fn div_ceil(numerator: usize, denominator: usize) -> usize {
    (numerator + denominator - 1) / denominator
}

fn prefix_sum(xs: &[usize], n: usize) -> usize {
    xs.iter().take(n).sum()
}
