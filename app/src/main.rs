use itertools::Itertools;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello from app");

    let numbers_len: i64 = 3 * 1024;
    let a = (0..numbers_len).collect_vec();
    let b = a.iter().map(|n| n * n).collect_vec();
    let results = cpu::nn::add_vecs(a, b)?;
    println!("results: {:#?}", results.iter().take(10).collect_vec());

    let queries = (0..1000).map(|i| i as f32).collect_vec();
    let results = cpu::nn::find_nn(queries)?;
    println!("results: {:#?}", results);

    Ok(())
}
