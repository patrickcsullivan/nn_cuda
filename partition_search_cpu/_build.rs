use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../partition_search_gpu")
        .copy_to("../resources/partition_search_gpu.ptx")
        .build()
        .unwrap();
}
