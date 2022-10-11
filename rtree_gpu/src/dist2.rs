use cuda_std::vek::Vec3;

pub fn dist2_to_point(p1: Vec3<f32>, p2_x: f32, p2_y: f32, p2_z: f32) -> f32 {
    let x = p2_x - p1.x;
    let y = p2_y - p1.y;
    let z = p2_z - p1.z;
    x * x + y * y + z * z
}

pub fn dist2_to_aabb(
    p: Vec3<f32>,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
) -> f32 {
    let x = dist2_to_range(p.x, min_x, max_x);
    let y = dist2_to_range(p.y, min_y, max_y);
    let z = dist2_to_range(p.z, min_z, max_z);
    x * x + y * y + z * z
}

fn dist2_to_range(val: f32, min: f32, max: f32) -> f32 {
    if val < min {
        min - val
    } else if val > max {
        val - max
    } else {
        0.0
    }
}
