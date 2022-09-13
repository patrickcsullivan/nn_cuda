mod dragon;

use cpu::{bvh::Bvh, morton::map_to_morton_codes_tmp};
use cuda_std::vek::{Aabb, Vec3};
use itertools::Itertools;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject};
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    // simple_test()?;
    dragon_test()?;
    Ok(())
}

fn simple_test() -> Result<(), Box<dyn Error>> {
    let objects = vec![Vec3::new(-1.0, -1.0, -1.0), Vec3::new(10.0, 10.0, 10.0)];
    let aabb = Aabb::new_empty(objects[0]).union(Aabb::new_empty(objects[objects.len() - 1]));

    let queries = (0..(3 * 1024))
        .map(|i| Vec3::new(i as f32, i as f32, i as f32))
        .collect_vec();

    benchmarks(&objects, &aabb, &queries)
}

fn dragon_test() -> Result<(), Box<dyn Error>> {
    let mut objects = dragon::ply_vertices("./app/data/dragon_vrip.ply");

    // Shift the objects so that the bounding box is centered on the origin.
    let mut aabb = get_aabb(&objects);
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

    benchmarks(&objects, &aabb, &queries)
}

fn benchmarks(
    objects: &[Vec3<f32>],
    aabb: &Aabb<f32>,
    queries: &[Vec3<f32>],
) -> Result<(), Box<dyn Error>> {
    println!("{} objects", objects.len());
    println!("{} queries", queries.len());

    // Radix sort queries.
    let queries_aabb = get_aabb(queries);
    let morton_codes = map_to_morton_codes_tmp(queries, &queries_aabb);
    let mut sorted_query_indices = (0..queries.len()).collect_vec();
    sorted_query_indices.sort_by_key(|&i| morton_codes[i]);
    let sorted_queries = sorted_query_indices
        .into_iter()
        .map(|i| queries[i])
        .collect_vec();

    // Test brute force CUDA.
    let now = Instant::now();
    let bf_results = cpu::nn::brute_force(&objects, &queries)?;
    let elapsed = now.elapsed();
    println!("Brute Force CUDA:\t\t{:.2?}", elapsed);

    // Build BVH.
    let bvh = Bvh::new_with_aabb(&objects, &aabb);

    // Test BVH CUDA.
    let now = Instant::now();
    let bvh_results = cpu::nn::find_nn(&bvh, &queries)?;
    let elapsed = now.elapsed();
    println!("BVH CUDA:\t\t\t{:.2?}", elapsed);

    // Test BVH CUDA with sorted queries.
    let now = Instant::now();
    let _bvh_sorted_results = cpu::nn::find_nn(&bvh, &sorted_queries)?;
    let elapsed = now.elapsed();
    println!("BVH CUDA (sorted queries):\t{:.2?}", elapsed);

    // Build RTree and RTree queries.
    let rtree_objects = objects
        .iter()
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
    println!("RTree (1-core):\t\t\t{:.2?}", elapsed);

    // Test with R* multi-threaded.
    let now = Instant::now();
    let rtree_results_mt: Vec<_> = rtree_queries
        .par_iter()
        .map(|q| rtree.nearest_neighbor(q))
        .collect();
    let elapsed = now.elapsed();
    println!("RTree (8-core):\t\t\t{:.2?}", elapsed);

    let fails = (0..queries.len())
        .filter(|&i| bvh_results[i].unwrap().0 != bf_results[i].unwrap().0)
        .collect_vec();
    println!(
        "BVH CUDA and Brute Force CUDA find different NNs on {} queries",
        fails.len()
    );

    let fails = (0..queries.len())
        .filter(|&i| {
            (bvh_results[i].unwrap().1.sqrt() - bf_results[i].unwrap().1.sqrt()).abs()
                > f32::EPSILON
        })
        .collect_vec();
    println!(
        "BVH CUDA and Brute Force CUDA find different NN dists on {} queries",
        fails.len()
    );

    let fails = (0..queries.len())
        .filter(|&i| bvh_results[i].unwrap().0 != rtree_results_mt[i].unwrap().index)
        .collect_vec();
    println!(
        "BVH CUDA and RTree (8-core) find different NNs on {} queries",
        fails.len()
    );

    let fails = (0..queries.len())
        .filter(|&i| {
            let query = queries[i];
            let rtree_nn_idx = rtree_results_mt[i].unwrap().index;
            let rtree_nn = objects[rtree_nn_idx];
            let rtree_dist = query.distance(rtree_nn);

            let bvh_dist = bvh_results[i].unwrap().1.sqrt();

            (bvh_dist - rtree_dist).abs() > f32::EPSILON
        })
        .collect_vec();
    println!(
        "BVH CUDA and RTree (8-core) find different NN dists on {} queries",
        fails.len()
    );

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

fn get_aabb(vs: &[Vec3<f32>]) -> Aabb<f32> {
    let mut aabb = Aabb::new_empty(vs[0]);
    for v in vs {
        aabb.expand_to_contain_point(*v);
    }
    aabb
}
