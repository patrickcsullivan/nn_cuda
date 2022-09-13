mod dragon;

use cpu::bvh::Bvh;
use cuda_std::vek::{Aabb, Vec3};
use itertools::Itertools;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject};
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    dragon_test()?;
    Ok(())
}

fn dragon_test() -> Result<(), Box<dyn Error>> {
    let mut objects = dragon::ply_vertices("./app/data/dragon_vrip.ply");
    println!("{} objects", objects.len());

    // Shift the objects so that the bounding box is centered on the origin.
    let mut aabb = dragon::get_aabb(&objects);
    let center = aabb.center();
    objects.iter_mut().for_each(|o| {
        *o = *o - center;
    });
    aabb.min -= center;
    aabb.max -= center;
    // println!(
    //     "Objects AABB: ({}, {}, {}), ({}, {}, {})",
    //     aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z
    // );

    // Generate query sets.
    let mut queries = objects.iter().map(|&o| o * 0.25).collect_vec();
    let mut expanded_queries = objects.iter().map(|&o| o * 2.0).collect_vec();
    let mut shifted_queries = objects
        .iter()
        .map(|&o| o + Vec3::new(aabb.size().w / 2.0, 0.0, 0.0))
        .collect_vec();
    queries.append(&mut expanded_queries);
    queries.append(&mut shifted_queries);
    println!("{} queries", queries.len());

    // Build BVH.
    let bvh = Bvh::new_with_aabb(&objects, &aabb);

    // Test BVH CUDA.
    let now = Instant::now();
    let _bvh_results = cpu::nn::find_nn(&bvh, &queries)?;
    let elapsed = now.elapsed();
    println!("BVH CUDA:\t{:.2?}", elapsed);

    // Build RTree and RTree queries.
    let rtree_objects = objects
        .into_iter()
        .enumerate()
        .map(|(i, o)| IndexedPoint {
            index: i,
            point: [o.x, o.y, o.z],
        })
        .collect_vec();
    let rtree_queries = queries.iter().map(|q| [q.x, q.y, q.z]).collect_vec();
    let rtree = RTree::bulk_load(rtree_objects);

    // Test with R* single-threaded.
    let now = Instant::now();
    let _rtree_results_st = rtree_queries
        .iter()
        .map(|q| rtree.nearest_neighbor(q))
        .collect_vec();
    let elapsed = now.elapsed();
    println!("RTree (1-core):\t{:.2?}", elapsed);

    // Test with R* multi-threaded.
    let now = Instant::now();
    let _rtree_results_mt: Vec<_> = rtree_queries
        .par_iter()
        .map(|q| rtree.nearest_neighbor(q))
        .collect();
    let elapsed = now.elapsed();
    println!("RTree (8-core):\t{:.2?}", elapsed);

    Ok(())
}

struct IndexedPoint {
    index: usize,
    point: [f32; 3],
}

impl RTreeObject for IndexedPoint {
    type Envelope = rstar::AABB<[f32; 3]>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_point(self.point)
    }
}

impl rstar::PointDistance for IndexedPoint {
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        self.point.distance_2(point)
    }
}

fn simple_test() -> Result<(), Box<dyn Error>> {
    // let leaf_node_object_indices = (0..(3 * 1024)).collect_vec();
    // let queries = leaf_node_object_indices.iter().map(|n| n * n).collect_vec();
    let objects = vec![Vec3::new(-1.0, -1.0, -1.0), Vec3::new(10.0, 10.0, 10.0)];
    let aabb = Aabb::new_empty(objects[0]).union(Aabb::new_empty(objects[objects.len() - 1]));
    let bvh = Bvh::new_with_aabb(&objects, &aabb);

    let queries = (0..(3 * 1024))
        .map(|i| Vec3::new(i as f32, i as f32, i as f32))
        .collect_vec();
    let results = cpu::nn::find_nn(&bvh, &queries)?;
    println!("results: {:#?}", results.iter().take(15).collect_vec());

    Ok(())
}
