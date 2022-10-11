use super::geometry::{points_aabb, union_aabbs, vecs_aabb};
use super::split::split;
use crate::morton::map_to_morton_codes;
use crate::Point3;
use cuda_std::vek::{Aabb, Vec3};
use cust::prelude::DeviceBuffer;
use cust::util::SliceExt;
use itertools::Itertools;
use std::error::Error;

/// Contains data on the host that represents the R-tree.
pub struct HostData<const M: usize, const H: usize> {
    /// A heap that contains the bounding box of each node.
    node_aabbs: Vec<Aabb<f32>>,

    /// A vector containing the start and end indices of objects in each leaf
    /// node.
    leaf_ranges: Vec<(usize, usize)>,

    /// A vector containing the index of each object sorted on a Z-curve.
    sorted_object_indices: Vec<usize>,

    /// The position of each object sorted on a Z-curve.
    sorted_object_vecs: Vec<Vec3<f32>>,
}

/// Contains pointers to data on the device that represents the R-tree.
pub struct DeviceData {
    dev_node_min_xs: DeviceBuffer<f32>,
    dev_node_min_ys: DeviceBuffer<f32>,
    dev_node_min_zs: DeviceBuffer<f32>,
    dev_node_max_xs: DeviceBuffer<f32>,
    dev_node_max_ys: DeviceBuffer<f32>,
    dev_node_max_zs: DeviceBuffer<f32>,

    dev_sorted_object_indices: DeviceBuffer<usize>,
    dev_sorted_object_xs: DeviceBuffer<f32>,
    dev_sorted_object_ys: DeviceBuffer<f32>,
    dev_sorted_object_zs: DeviceBuffer<f32>,
    dev_leaf_starts: DeviceBuffer<usize>,
    dev_leaf_ends: DeviceBuffer<usize>,
}

impl<const M: usize, const H: usize> HostData<M, H> {
    pub fn new<T>(objects: &[T]) -> Self
    where
        T: Point3,
    {
        let (sorted_object_indices, sorted_object_vecs) = Self::build_zcurve(objects);
        let (node_aabbs, leaf_ranges) = Self::build_nodes(&sorted_object_vecs);

        Self {
            node_aabbs,
            leaf_ranges,
            sorted_object_indices,
            sorted_object_vecs,
        }
    }

    fn build_zcurve<T>(objects: &[T]) -> (Vec<usize>, Vec<Vec3<f32>>)
    where
        T: Point3,
    {
        let vecs = objects.iter().map(|o| o.into_vec3()).collect_vec();
        let aabb = points_aabb(objects);
        let morton_codes = map_to_morton_codes(&vecs, &aabb);
        let mut sorted_object_indices = (0..objects.len()).collect_vec();
        sorted_object_indices.sort_by_key(|&i| morton_codes[i]);
        let sorted_object_vecs = sorted_object_indices.iter().map(|&i| vecs[i]).collect_vec();
        (sorted_object_indices, sorted_object_vecs)
    }

    fn build_nodes(sorted_object_vecs: &[Vec3<f32>]) -> (Vec<Aabb<f32>>, Vec<(usize, usize)>) {
        let leaf_ranges = split(sorted_object_vecs, Self::leafs_count()).collect_vec();

        let mut node_aabbs = vec![]; // TODO: Pre-allocate to known size.
        let mut last_level_node_aabbs = leaf_ranges
            .iter()
            .map(|&(start, end)| vecs_aabb(&sorted_object_vecs[start..=end]))
            .collect_vec();

        while last_level_node_aabbs.len() > 1 {
            let next_level_node_aabbs = last_level_node_aabbs
                .iter()
                .chunks(M)
                .into_iter()
                .map(union_aabbs)
                .collect_vec();

            // Reverse the last level before appending it to nodes. When we're done adding
            // levels, we'll reverse the whole vector so the root is first and leaves are
            // last.
            last_level_node_aabbs.reverse();
            node_aabbs.append(&mut last_level_node_aabbs);

            last_level_node_aabbs = next_level_node_aabbs;
        }

        // The final level is the root node. Reverse so that the root is first.
        node_aabbs.append(&mut last_level_node_aabbs);
        node_aabbs.reverse();

        (node_aabbs, leaf_ranges)
    }

    fn node_min_xs(&self) -> Vec<f32> {
        self.node_aabbs.iter().map(|aabb| aabb.min.x).collect_vec()
    }

    fn node_min_ys(&self) -> Vec<f32> {
        self.node_aabbs.iter().map(|aabb| aabb.min.y).collect_vec()
    }

    fn node_min_zs(&self) -> Vec<f32> {
        self.node_aabbs.iter().map(|aabb| aabb.min.z).collect_vec()
    }

    fn node_max_xs(&self) -> Vec<f32> {
        self.node_aabbs.iter().map(|aabb| aabb.max.x).collect_vec()
    }

    fn node_max_ys(&self) -> Vec<f32> {
        self.node_aabbs.iter().map(|aabb| aabb.max.y).collect_vec()
    }

    fn node_max_zs(&self) -> Vec<f32> {
        self.node_aabbs.iter().map(|aabb| aabb.max.z).collect_vec()
    }

    fn leaf_starts_ends(&self) -> (Vec<usize>, Vec<usize>) {
        self.leaf_ranges.iter().map(|&range| range).unzip()
    }

    fn sorted_object_xs(&self) -> Vec<f32> {
        self.sorted_object_vecs.iter().map(|v| v.x).collect_vec()
    }

    fn sorted_object_ys(&self) -> Vec<f32> {
        self.sorted_object_vecs.iter().map(|v| v.y).collect_vec()
    }

    fn sorted_object_zs(&self) -> Vec<f32> {
        self.sorted_object_vecs.iter().map(|v| v.z).collect_vec()
    }

    fn leafs_count() -> usize {
        M.pow(H as u32)
    }

    pub fn copy_to_device(&self) -> Result<DeviceData, Box<dyn Error>> {
        let (leaf_starts, leaf_ends) = self.leaf_starts_ends();

        Ok(DeviceData {
            dev_node_min_xs: self.node_min_xs().as_slice().as_dbuf()?,
            dev_node_min_ys: self.node_min_ys().as_slice().as_dbuf()?,
            dev_node_min_zs: self.node_min_zs().as_slice().as_dbuf()?,
            dev_node_max_xs: self.node_max_xs().as_slice().as_dbuf()?,
            dev_node_max_ys: self.node_max_ys().as_slice().as_dbuf()?,
            dev_node_max_zs: self.node_max_zs().as_slice().as_dbuf()?,
            dev_sorted_object_indices: self.sorted_object_indices.as_slice().as_dbuf()?,
            dev_sorted_object_xs: self.sorted_object_xs().as_slice().as_dbuf()?,
            dev_sorted_object_ys: self.sorted_object_ys().as_slice().as_dbuf()?,
            dev_sorted_object_zs: self.sorted_object_zs().as_slice().as_dbuf()?,
            dev_leaf_starts: leaf_starts.as_slice().as_dbuf()?,
            dev_leaf_ends: leaf_ends.as_slice().as_dbuf()?,
        })
    }
}
