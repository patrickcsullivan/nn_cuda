use crate::morton::map_to_morton_codes;
use cuda_std::vek::{Aabb, Vec3};
use itertools::Itertools;

/// Orders the given points along a Z-curve. Returns the ordered point indices
/// and the ordered points.
pub fn z_curve(vecs: &[Vec3<f32>], aabb: &Aabb<f32>) -> (Vec<usize>, Vec<Vec3<f32>>) {
    let morton_codes = map_to_morton_codes(&vecs, &aabb);
    let mut sorted_indices = (0..vecs.len()).collect_vec();
    sorted_indices.sort_by_key(|&i| morton_codes[i]);
    let sorted_vecs = sorted_indices.iter().map(|&i| vecs[i]).collect_vec();
    (sorted_indices, sorted_vecs)
}
