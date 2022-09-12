use cuda_std::vek::Vec3;
use itertools::Itertools;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello from app");

    // let leaf_node_object_indices = (0..(3 * 1024)).collect_vec();
    // let queries = leaf_node_object_indices.iter().map(|n| n * n).collect_vec();

    let queries = (0..(3 * 1024))
        .map(|i| Vec3::new(i as f32, i as f32, i as f32))
        .collect_vec();
    let results = cpu::nn::find_nn(queries)?;
    println!("results: {:#?}", results.iter().take(10).collect_vec());

    Ok(())
}
