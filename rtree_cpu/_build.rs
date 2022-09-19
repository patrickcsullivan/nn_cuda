use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../rtree_gpu")
        .copy_to("../resources/rtree_gpu.ptx")
        .build()
        .unwrap();
}
