use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use cuda_std::vek::{Aabb, Vec3};
use cust::stream::{Stream, StreamFlags};
use nn_cuda::{BitPartitions, Point3, RTree};
use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use rstar::RTree as RStar;

const SEED: &[u8; 32] = b"LVXn6sWNasjDReRS2OZ9a0eY1aprVNYX";

#[derive(Debug, Clone, Copy)]
struct PointObject([f32; 3]);

impl nn_cuda::Point3 for PointObject {
    fn xyz(&self) -> [f32; 3] {
        self.0
    }
}

impl nn_cuda::Point3 for &PointObject {
    fn xyz(&self) -> [f32; 3] {
        self.0
    }
}

impl rstar::RTreeObject for PointObject {
    type Envelope = rstar::AABB<[f32; 3]>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_point(self.0)
    }
}

impl rstar::PointDistance for PointObject {
    fn distance_2(&self, point: &[f32; 3]) -> f32 {
        self.xyz().distance_2(point)
    }
}

/// Find the axis-aligned bounding box of the given points.
fn get_aabb(points: &[PointObject]) -> Aabb<f32> {
    let mut aabb = Aabb::new_empty(new_vec3(&points[0]));
    for v in points {
        aabb.expand_to_contain_point(new_vec3(v));
    }
    aabb
}

/// Create a new `Vec3` from the given point.
fn new_vec3(point: &PointObject) -> Vec3<f32> {
    Vec3::new(point.0[0], point.0[1], point.0[2])
}

/// Creates the specified number of random points.
fn create_random_points(points_count: usize, rng: &mut impl Rng) -> Vec<PointObject> {
    let mut result = Vec::with_capacity(points_count);
    for _ in 0..points_count {
        result.push(PointObject(rng.gen()));
    }
    result
}

pub fn nn_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Algorithm comparison");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for points_count in [500_000u64, 1_000_000u64, 5_000_000u64, 10_000_000u64] {
        let _ctx = cust::quick_init().unwrap();
        let mut rng = Hc128Rng::from_seed(*SEED);
        let points = create_random_points(points_count as usize, &mut rng);
        let queries = create_random_points(1_000_000, &mut rng);
        let points_aabb = get_aabb(&points);

        // Prepare R* search.
        let rstar = RStar::bulk_load(points.clone());

        // Prepare bit partition CUDA search.
        let partitions = BitPartitions::new(&points, &points_aabb).unwrap();

        // Prepare R-tree GPU search.
        let rtree = RTree::new(&points).unwrap();

        group.bench_with_input(
            BenchmarkId::new("BitPartitions", points_count),
            &points_count,
            |b, _| {
                b.iter(|| {
                    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
                    let _ = partitions.find_nns(stream, &queries, None).unwrap();
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("RTree", points_count),
            &points_count,
            |b, _| {
                b.iter(|| {
                    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
                    let results = rtree.batch_find_neighbors(&stream, &queries).unwrap();
                    println!("Results: {:?}", results.iter().take(10));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("RStar", points_count),
            &points_count,
            |b, _| {
                b.iter(|| {
                    let _: Vec<_> = queries
                        .par_iter()
                        .map(|q| rstar.nearest_neighbor(&q.0))
                        .collect();
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, nn_comparison);
criterion_main!(benches);
