mod dragon;

use cuda_std::vek::{Aabb, Vec3};
use itertools::Itertools;
use kiddo::KdTree;
use nn_cuda::morton::map_to_morton_codes_tmp;
use nn_cuda::partition::BitPartitionSearch;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject};
use std::{env, error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
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

    // Test bit partition search CUDA.
    let bit_partition_objs = objects.iter().map(|&v| PartitionObj(v)).collect_vec();
    let partitions = BitPartitionSearch::new(&bit_partition_objs, aabb);
    let now = Instant::now();
    let bit_partition_results = partitions.find_nns(&sorted_queries)?;
    let elapsed = now.elapsed();
    println!("Bit Partition Search CUDA:\t{:.2?}", elapsed);

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

    // Test with RTree multi-threaded.
    let now = Instant::now();
    let rtree_results_mt: Vec<_> = rtree_queries
        .par_iter()
        .map(|q| rtree.nearest_neighbor(q))
        .collect();
    let elapsed = now.elapsed();
    println!("rstar (8-core):\t\t\t{:.2?}", elapsed);

    // -------------------------------------------------------------------------

    let fails = (0..queries.len())
        .filter(|&i| bit_partition_results[i].unwrap().0 != rtree_results_mt[i].unwrap().index)
        .collect_vec();
    println!(
        "Bit Partitions CUDA and rstar (8-core) find different NNs on {} queries",
        fails.len()
    );

    let fails = (0..queries.len())
        .filter(|&i| {
            let query = queries[i];
            let rtree_nn_idx = rtree_results_mt[i].unwrap().index;
            let rtree_nn = objects[rtree_nn_idx];
            let rtree_dist = query.distance(rtree_nn);

            let bvh_dist = bit_partition_results[i].unwrap().1.sqrt();

            (bvh_dist - rtree_dist).abs() > f32::EPSILON
        })
        .collect_vec();
    println!(
        "Bit Partitions CUDA and rstar (8-core) find different NN dists on {} queries",
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

struct PartitionObj(Vec3<f32>);

impl nn_cuda::partition::HasVec3 for PartitionObj {
    fn vec3(&self) -> Vec3<f32> {
        self.0
    }
}
