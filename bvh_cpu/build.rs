use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../bvh_gpu")
        .copy_to("../resources/bvh_gpu.ptx")
        .build()
        .unwrap();
}
