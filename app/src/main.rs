mod dragon;

use bit_partition_search_cpu::partition::BitPartitionSearch;
use bvh_cpu::{bvh::Bvh, morton::map_to_morton_codes_tmp};
use cuda_std::vek::{Aabb, Vec3};
use itertools::Itertools;
use kiddo::KdTree;
use partition_search_cpu::partition::PartitionSearch;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject};
use std::{env, error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    // simple_test()?;
    let args: Vec<String> = env::args().collect();
    let default_path = "./app/data/dragon_vrip.ply".to_string();
    let path = args.get(1).unwrap_or(&default_path);
    dragon_test(path)?;
    Ok(())
}

fn dragon_test(path: &str) -> Result<(), Box<dyn Error>> {
    let mut objects = dragon::ply_vertices(path);

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

// - Thread divergence
// - Bad memory accesses vs coallesced
// - Algorthim
// - Launch parameters
// - SM underutilization -...

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
    let bf_results = bvh_cpu::nn::brute_force(&objects, &sorted_queries)?;
    let elapsed = now.elapsed();
    println!("Brute Force CUDA:\t\t{:.2?}", elapsed);

    // Build BVH.
    let bvh = Bvh::new_with_aabb(&objects, &aabb);

    // Test BVH CUDA.
    let now = Instant::now();
    let bvh_results = bvh_cpu::nn::find_nn(&bvh, &sorted_queries)?;
    let elapsed = now.elapsed();
    println!("BVH CUDA:\t\t\t{:.2?}", elapsed);

    // Test fixed width partition search CUDA.
    let partition_objs = objects.iter().map(|&v| PartitionObj(v)).collect_vec();
    let partitions = PartitionSearch::new(&partition_objs, aabb);
    let now = Instant::now();
    let partition_results = partitions.find_nns(&sorted_queries)?;
    let elapsed = now.elapsed();
    println!("Fixed Partition Search CUDA:\t{:.2?}", elapsed);

    // Test bit partition search CUDA.
    let bit_partition_objs = objects.iter().map(|&v| PartitionObj(v)).collect_vec();
    let partitions = BitPartitionSearch::new(&bit_partition_objs, aabb);
    let now = Instant::now();
    let bit_partition_results = partitions.find_nns(&sorted_queries)?;
    let elapsed = now.elapsed();
    println!("Bit Partition Search CUDA:\t{:.2?}", elapsed);

    // // Test BVH CUDA with sorted queries.
    // let now = Instant::now();
    // let _bvh_sorted_results = bvh_cpu::nn::find_nn(&bvh, &sorted_queries)?;
    // let elapsed = now.elapsed();
    // println!("BVH CUDA (sorted queries):\t{:.2?}", elapsed);

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

    // // Test with RTree single-threaded.
    // let now = Instant::now();
    // let _rtree_results_st = rtree_queries
    //     .iter()
    //     .map(|q| rtree.nearest_neighbor(q))
    //     .collect_vec();
    // let elapsed = now.elapsed();
    // println!("rstar (1-core):\t\t\t{:.2?}", elapsed);

    // Test with RTree multi-threaded.
    let now = Instant::now();
    let rtree_results_mt: Vec<_> = rtree_queries
        .par_iter()
        .map(|q| rtree.nearest_neighbor(q))
        .collect();
    let elapsed = now.elapsed();
    println!("rstar (8-core):\t\t\t{:.2?}", elapsed);

    // Build KDTree and KDTree queries.
    let kdtree_queries = queries.iter().map(|q| [q.x, q.y, q.z]).collect_vec();
    let mut kdtree = KdTree::new();
    for (i, o) in objects.iter().enumerate() {
        kdtree.add(&[o.x, o.y, o.z], i)?;
    }

    // // Test with KDTree single-threaded.
    // let now = Instant::now();
    // let _kdtree_results_st = kdtree_queries
    //     .iter()
    //     .map(|q| kdtree.nearest_one(q, &kiddo::distance::squared_euclidean))
    //     .collect_vec();
    // let elapsed = now.elapsed();
    // println!("kiddo (1-core):\t\t\t{:.2?}", elapsed);

    // Test with KDTree multi-threaded.
    let now = Instant::now();
    let _kdtree_results_mt: Vec<_> = kdtree_queries
        .par_iter()
        .map(|q| kdtree.nearest_one(q, &kiddo::distance::squared_euclidean))
        .collect();
    let elapsed = now.elapsed();
    println!("kiddo (8-core):\t\t\t{:.2?}", elapsed);

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
        .filter(|&i| partition_results[i].map(|r| r.0) != bf_results[i].map(|r| r.0))
        .collect_vec();
    println!(
        "Fixed Partitions CUDA and Brute Force CUDA find different NNs on {} queries",
        fails.len()
    );

    let fails = (0..queries.len())
        .filter(|&i| (partition_results[i].map(|r| r.1) != bf_results[i].map(|r| r.1)))
        .collect_vec();
    println!(
        "Fixed Partitions CUDA and Brute Force CUDA find different NN dists on {} queries",
        fails.len()
    );

    let no_nn = partition_results
        .iter()
        .filter(|r| r.is_none())
        .collect_vec();
    println!(
        "Fixed Partitions CUDA does not find NN for {} queries",
        no_nn.len()
    );

    let fails = (0..queries.len())
        .filter(|&i| bit_partition_results[i].map(|r| r.0) != bf_results[i].map(|r| r.0))
        .collect_vec();
    println!(
        "Bit Partitions CUDA and Brute Force CUDA find different NNs on {} queries",
        fails.len()
    );

    let fails = (0..queries.len())
        .map(|i| {
            ((
                bit_partition_results[i].map(|r| r.1),
                bf_results[i].map(|r| r.1),
            ))
        })
        .filter(|(r1, r2)| r1 != r2)
        .collect_vec();
    println!(
        "Bit Partitions CUDA and Brute Force CUDA find different NN dists on {} queries",
        fails.len()
    );
    // println!("Different NN dists: {:?}", fails);

    // let fails = (0..queries.len())
    //     .filter(|&i| bvh_results[i].unwrap().0 !=
    // rtree_results_mt[i].unwrap().index)     .collect_vec();
    // println!(
    //     "BVH CUDA and rstar (8-core) find different NNs on {} queries",
    //     fails.len()
    // );

    // let fails = (0..queries.len())
    //     .filter(|&i| {
    //         let query = queries[i];
    //         let rtree_nn_idx = rtree_results_mt[i].unwrap().index;
    //         let rtree_nn = objects[rtree_nn_idx];
    //         let rtree_dist = query.distance(rtree_nn);

    //         let bvh_dist = bvh_results[i].unwrap().1.sqrt();

    //         (bvh_dist - rtree_dist).abs() > f32::EPSILON
    //     })
    //     .collect_vec();
    // println!(
    //     "BVH CUDA and rstar (8-core) find different NN dists on {} queries",
    //     fails.len()
    // );

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

struct PartitionObj(Vec3<f32>);

impl partition_search_cpu::partition::HasVec3 for PartitionObj {
    fn vec3(&self) -> Vec3<f32> {
        self.0
    }
}

impl bit_partition_search_cpu::partition::HasVec3 for PartitionObj {
    fn vec3(&self) -> Vec3<f32> {
        self.0
    }
}
