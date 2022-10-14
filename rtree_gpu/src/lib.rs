#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

// Used for printing from CUDA.
extern crate alloc;

mod dist2;
mod grid;
pub mod kernels;
mod mem;
mod priority_queue;
mod rtree;
mod scan;
mod shared_aabbs;
mod shared_stack;
mod step;
mod vec3s;
