use cust::stream::{Stream, StreamFlags};
use kernel_tools::{brent_kung, kogge_stone};
use nn_cuda::scan;

pub fn main() {
    let _ctx = cust::quick_init().unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let xs = [1u32; kogge_stone::SECTION_SIZE];

    let mut ys = [0u32; kogge_stone::SECTION_SIZE];
    scan::sequential_scan(&xs, &mut ys, xs.len());
    println!("xs: {:?}", xs);
    println!("ys: {:?}", ys);

    let ys = scan::kogge_stone_scan(&stream, &xs);
    println!("xs: {:?}", xs);
    println!("ys: {:?}", ys);

    let xs = [1u32; brent_kung::SECTION_SIZE];
    let mut ys = [0u32; brent_kung::SECTION_SIZE];
    let ys = scan::brent_kung_scan(&stream, &xs);
    println!("xs: {:?}", xs);
    println!("ys: {:?}", ys);
}
