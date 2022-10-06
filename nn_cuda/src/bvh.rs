use crate::morton::map_to_morton_codes;
use cuda_std::vek::{Aabb, Vec3};
use cust::prelude::*;
use itertools::Itertools;
use std::{error::Error, time::Instant};

pub struct BitPartitionSearch {
    pub sorted_object_indices: Vec<usize>,
    pub sorted_object_xs: Vec<f32>,
    pub sorted_object_ys: Vec<f32>,
    pub sorted_object_zs: Vec<f32>,
    // pub module: Module,
}

impl BitPartitionSearch {
    pub fn new<'a, T>(objects: &'a [T], aabb: &Aabb<f32>) -> Result<Self, Box<dyn Error>>
    where
        &'a T: Into<Vec3<f32>>,
    {
        // Sort the objects along the Z-curve.
        let vecs: Vec<Vec3<f32>> = objects.iter().map(|o| o.into()).collect();
        let morton_codes = map_to_morton_codes(&vecs, &aabb);
        let mut sorted_object_indices = (0..objects.len()).collect_vec();
        sorted_object_indices.sort_by_key(|&i| morton_codes[i]);

        let sorted_object_vecs = sorted_object_indices.iter().map(|&i| vecs[i]).collect_vec();
        let sorted_object_xs = sorted_object_vecs.iter().map(|v| v.x).collect_vec();
        let sorted_object_ys = sorted_object_vecs.iter().map(|v| v.y).collect_vec();
        let sorted_object_zs = sorted_object_vecs.iter().map(|v| v.z).collect_vec();

        // Load the CUDA module.
        // let module = Module::from_ptx(PTX, &[])?;

        Ok(Self {
            sorted_object_indices,
            sorted_object_xs,
            sorted_object_ys,
            sorted_object_zs,
            // module,
        })
    }
}

fn find_split(morton_codes: &[u32], first: usize, last: usize) -> usize {
    let first_code = morton_codes[first];
    let last_code = morton_codes[last];

    if first_code == last_code {
        return ((first_code + last_code) >> 1) as usize;
    }

    let common_prefix_size = (first_code ^ last_code).leading_zeros();

    let mut split = first;
    let mut step = last - first;

    loop {
        step = (step + 1) >> 1;
        let new_split = split + step;

        if new_split < last {
            let split_code = morton_codes[new_split];
            let split_prefix_size = (first_code ^ split_code).leading_zeros();

            if split_prefix_size > common_prefix_size {
                split = new_split
            }
        }

        if step <= 1 {
            break;
        }
    }

    split
}
