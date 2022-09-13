use cpu::bvh::Bvh;
use cuda_std::vek::{Aabb, Vec3};
use itertools::Itertools;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello from app");

    // let leaf_node_object_indices = (0..(3 * 1024)).collect_vec();
    // let queries = leaf_node_object_indices.iter().map(|n| n * n).collect_vec();
    let objects = vec![Vec3::new(-1.0, -1.0, -1.0), Vec3::new(10.0, 10.0, 10.0)];
    let aabb = Aabb::new_empty(objects[0]).union(Aabb::new_empty(objects[objects.len() - 1]));
    let bvh = Bvh::new_with_aabb(&objects, aabb);

    let queries = (0..(3 * 1024))
        .map(|i| Vec3::new(i as f32, i as f32, i as f32))
        .collect_vec();
    let results = cpu::nn::find_nn(&bvh, queries)?;
    println!("results: {:#?}", results.iter().take(15).collect_vec());

    Ok(())
}
