use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use cuda_std::vek::{Aabb, Vec3};
use cust::stream::{Stream, StreamFlags};
use itertools::Itertools;
use nn_cuda::partition::BitPartitionSearch;
use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use rstar::RTree;

const SEED: &[u8; 32] = b"LVXn6sWNasjDReRS2OZ9a0eY1aprVNYX";

struct PartitionObj([f32; 3]);

impl nn_cuda::partition::HasVec3 for PartitionObj {
    fn vec3(&self) -> Vec3<f32> {
        new_vec3(&self.0)
    }
}

/// Find the axis-aligned bounding box of the given points.
fn get_aabb(points: &[[f32; 3]]) -> Aabb<f32> {
    let mut aabb = Aabb::new_empty(new_vec3(&points[0]));
    for v in points {
        aabb.expand_to_contain_point(new_vec3(v));
    }
    aabb
}

/// Create a new `Vec3` from the given point.
fn new_vec3(point: &[f32; 3]) -> Vec3<f32> {
    Vec3::new(point[0], point[1], point[2])
}

/// Creates the specified number of random points.
fn create_random_points(points_count: usize, rng: &mut impl Rng) -> Vec<[f32; 3]> {
    let mut result = Vec::with_capacity(points_count);
    for _ in 0..points_count {
        result.push(rng.gen());
    }
    result
}

pub fn nn_comparison(c: &mut Criterion) {
    let mut rng = Hc128Rng::from_seed(*SEED);
    let points = create_random_points(1_000_000, &mut rng);
    let queries = create_random_points(10_000_000, &mut rng);
    let points_aabb = get_aabb(&points);

    // Prepare bit partition CUDA search.
    let _ctx = cust::quick_init().unwrap();
    let partition_points = points.iter().map(|&p| PartitionObj(p)).collect_vec();
    let partition_queries = queries.iter().map(new_vec3).collect_vec();
    let partitions = BitPartitionSearch::new(&partition_points, &points_aabb).unwrap();

    // Prepare R* search.
    let rtree = RTree::bulk_load(points);

    let mut group = c.benchmark_group("BitPartitions and RTree comparison");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    group.bench_function("BitPartitions", |b| {
        b.iter(|| {
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
            let _ = partitions
                .find_nns(stream, &partition_queries, None)
                .unwrap();
        })
    });

    group.bench_function("RTree", |b| {
        b.iter(|| {
            let _: Vec<_> = queries
                .par_iter()
                .map(|q| rtree.nearest_neighbor(q))
                .collect();
        })
    });

    group.finish();
}

criterion_group!(benches, nn_comparison);
criterion_main!(benches);
