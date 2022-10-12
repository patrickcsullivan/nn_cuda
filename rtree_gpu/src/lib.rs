#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

mod aabbs;
mod dist2;
mod grid;
pub mod kernels;
mod mem;
mod priority_queue;
mod rtree;
mod scan;
mod step;
mod vec3s;
