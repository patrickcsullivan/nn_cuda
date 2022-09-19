#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

pub mod aabb;
pub mod bvh;
pub mod kernels;
mod stack;