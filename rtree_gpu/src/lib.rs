#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

mod dist2;
mod grid;
pub mod kernels;
mod priority_queue;
mod rtree;
