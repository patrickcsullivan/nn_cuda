use cust::stream::{Stream, StreamFlags};
use kernel_tools::{brent_kung, kogge_stone, three_phase};
use nn_cuda::{scan, Point3, RTree};
use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;

pub fn main() {
    let _ctx = cust::quick_init().unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    let mut rng = Hc128Rng::from_seed(*SEED);
    let points = create_random_points(1_000_000 as usize, &mut rng);
    let queries = create_random_points(1_000, &mut rng);
    let rtree = RTree::new(&points).unwrap();
    let nns = rtree.batch_find_neighbors(&stream, &queries).unwrap();

    println!("Results: {:?}", &nns[0..10]);

    // let xs = [1u32; kogge_stone::SECTION_SIZE];

    // let mut ys = [0u32; kogge_stone::SECTION_SIZE];
    // scan::sequential_scan(&xs, &mut ys, xs.len());
    // println!("xs: {:?}", xs);
    // println!("ys: {:?}", ys);

    // let ys = scan::kogge_stone_scan(&stream, &xs);
    // println!("xs: {:?}", xs);
    // println!("ys: {:?}", ys);

    // let xs = [1u32; brent_kung::SECTION_SIZE];
    // let ys = scan::brent_kung_scan(&stream, &xs);
    // println!("xs: {:?}", xs);
    // println!("ys: {:?}", ys);

    // let xs = [1u32; three_phase::SECTION_SIZE];
    // let ys = scan::three_phase_scan(&stream, &xs);
    // println!("xs: {:?}", xs);
    // println!("ys: {:?}", ys);
}

const SEED: &[u8; 32] = b"LVXn6sWNasjDReRS2OZ9a0eY1aprVNYX";

#[derive(Debug, Clone, Copy, PartialEq)]
struct PointObject([f32; 3]);

impl crate::Point3 for PointObject {
    fn xyz(&self) -> [f32; 3] {
        self.0
    }
}

fn create_random_points(points_count: usize, rng: &mut impl Rng) -> Vec<PointObject> {
    let mut result = Vec::with_capacity(points_count);
    for _ in 0..points_count {
        result.push(PointObject(rng.gen()));
    }
    result
}

fn find_neighbor_brute_force<'a, T, Q>(points: &'a [T], query: &Q) -> &'a T
where
    T: Point3,
    Q: Point3,
{
    let mut min_dist2 = f32::INFINITY;
    let mut nn = None;
    for p in points {
        let dist2 = query.into_vec3().distance_squared(p.into_vec3());
        if dist2 < min_dist2 {
            min_dist2 = dist2;
            nn = Some(p)
        }
    }
    nn.unwrap()
}
