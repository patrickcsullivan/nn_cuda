#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

pub mod brent_kung;
pub mod kogge_stone;
mod step;
pub mod three_phase;
