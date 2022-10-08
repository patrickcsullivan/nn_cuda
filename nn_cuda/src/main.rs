use cust::stream::{Stream, StreamFlags};
use nn_cuda::scan;
use scan_gpu::kogge_stone::KOGGE_STONE_SECTION_SIZE;

pub fn main() {
    let _ctx = cust::quick_init().unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let xs = [1u32; KOGGE_STONE_SECTION_SIZE];
    let mut ys = [0u32; KOGGE_STONE_SECTION_SIZE];
    scan::sequential_scan(&xs, &mut ys, xs.len());
    println!("xs: {:?}", xs);
    println!("ys: {:?}", ys);

    let ys = scan::parallel_scan(stream, &xs);
    println!("xs: {:?}", xs);
    println!("ys: {:?}", ys);
}
