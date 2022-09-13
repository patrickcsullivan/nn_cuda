mod dragon;

use cpu::bvh::Bvh;
use cuda_std::vek::{Aabb, Vec3};
use itertools::Itertools;
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

    // Test with BVH CUDA.
    let bvh = Bvh::new_with_aabb(&objects, &aabb);
    let now = Instant::now();
    let _results = cpu::nn::find_nn(&bvh, queries)?;
    let elapsed = now.elapsed();
    println!("BVH CUDA, shrunk:\t{:.2?}", elapsed);

    Ok(())
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
    let results = cpu::nn::find_nn(&bvh, queries)?;
    println!("results: {:#?}", results.iter().take(15).collect_vec());

    Ok(())
}
