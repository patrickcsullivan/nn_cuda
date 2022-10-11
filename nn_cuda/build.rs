use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../bit_partition_gpu")
        .copy_to("../resources/bit_partition_gpu.ptx")
        .build()
        .unwrap();

    CudaBuilder::new("../kernel_tools")
        .copy_to("../resources/kernel_tools.ptx")
        .build()
        .unwrap();

    CudaBuilder::new("../rtree_gpu")
        .copy_to("../resources/rtree_gpu.ptx")
        .build()
        .unwrap();
}
