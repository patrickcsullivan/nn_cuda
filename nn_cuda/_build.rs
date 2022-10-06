use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../bit_partition_gpu")
        .copy_to("../resources/bit_partition_gpu.ptx")
        .build()
        .unwrap();
}
