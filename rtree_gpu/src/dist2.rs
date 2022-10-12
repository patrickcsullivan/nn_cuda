use cuda_std::vek::{Aabb, Vec3};

pub fn to_point(p1: Vec3<f32>, p2_x: f32, p2_y: f32, p2_z: f32) -> f32 {
    let x = p2_x - p1.x;
    let y = p2_y - p1.y;
    let z = p2_z - p1.z;
    x * x + y * y + z * z
}

pub fn to_aabb(p: &Vec3<f32>, aabb: &Aabb<f32>) -> f32 {
    let x = to_range(p.x, aabb.min.x, aabb.max.x);
    let y = to_range(p.y, aabb.min.y, aabb.min.z);
    let z = to_range(p.z, aabb.min.z, aabb.max.z);
    x * x + y * y + z * z
}

fn to_range(val: f32, min: f32, max: f32) -> f32 {
    if val < min {
        min - val
    } else if val > max {
        val - max
    } else {
        0.0
    }
}
