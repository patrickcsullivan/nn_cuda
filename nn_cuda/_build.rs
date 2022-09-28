use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../nn_cuda_gpu")
        .copy_to("../resources/nn_cuda_gpu.ptx")
        .build()
        .unwrap();
}
