use cuda_std::vek::{Aabb, Vec3};
use gpu::aabb::DeviceCopyAabb;

pub fn map_to_morton_codes_tmp(points: &[Vec3<f32>], aabb: &Aabb<f32>) -> Vec<u32> {
    let aabb = DeviceCopyAabb {
        min: aabb.min,
        max: aabb.max,
    };
    map_to_morton_codes(points, &aabb)
}

/// Maps each 3-dimensional point to a Morton code.
pub fn map_to_morton_codes(points: &[Vec3<f32>], aabb: &DeviceCopyAabb<f32>) -> Vec<u32> {
    let scale = aabb.size().recip();
    points
        .iter()
        .map(|p| {
            let p = (p - aabb.min) * scale;
            morton_code(p)
        })
        .collect::<Vec<_>>()
}

/// Returns the 3-dimensional 30-bit Morton code for the given 3D point
/// inside the unit cube [(0, 0, 0), (1, 1, 1)].
pub fn morton_code(v: Vec3<f32>) -> u32 {
    let x = fixed_point_u10(v.x);
    let y = fixed_point_u10(v.y);
    let z = fixed_point_u10(v.z);
    morton_u10(x, y, z)
}

/// Converts a floating point number in the range [0, 1] into a fixed-point
/// value with 10 fractional bits.
fn fixed_point_u10(x: f32) -> u32 {
    partial_min(partial_max(x * 1024.0, 0.0), 1023.0) as u32
}

/// Returns the 3-dimensional 30-bit Morton code for three 10-bit integers.
fn morton_u10(x: u32, y: u32, z: u32) -> u32 {
    // https://stackoverflow.com/a/1024889
    let x = expand_bits_u10(x);
    let y = expand_bits_u10(y);
    let z = expand_bits_u10(z);
    x | (y << 1) | (z << 2)
}

/// Expands a 10-bit integer into 28 bits by inserting two zeros between
/// each bit.
fn expand_bits_u10(x: u32) -> u32 {
    // let x = x.wrapping_mul(0x00010001u32) & 0xFF0000FFu32;
    // let x = x.wrapping_mul(0x00000101u32) & 0x0F00F00Fu32;
    // let x = x.wrapping_mul(0x00000011u32) & 0xC30C30C3u32;
    // x.wrapping_mul(0x00000005u32) & 0x49249249u32

    // https://stackoverflow.com/a/1024889
    let x = (x | (x << 16)) & 0x030000FF;
    let x = (x | (x << 8)) & 0x0300F00F;
    let x = (x | (x << 4)) & 0x030C30C3;
    (x | (x << 2)) & 0x09249249
}

#[inline(always)]
fn partial_min<T>(v1: T, v2: T) -> T
where
    T: PartialOrd,
{
    if v1.le(&v2) {
        v1
    } else {
        v2
    }
}

#[inline(always)]
fn partial_max<T>(v1: T, v2: T) -> T
where
    T: PartialOrd,
{
    if v1.ge(&v2) {
        v1
    } else {
        v2
    }
}

#[cfg(test)]
mod tests {
    use super::{expand_bits_u10, fixed_point_u10, morton_code};
    use cuda_std::vek::Vec3;

    #[test]
    fn morton_code_test() {
        let actual = morton_code(Vec3::new(0.0, 0.0, 0.0));
        let expected = 0b000000000000000000000000000000u32;
        assert_eq!(actual, expected);

        let actual = morton_code(Vec3::new(0.5, 0.0, 0.5));
        let expected = 0b101000000000000000000000000000u32;
        assert_eq!(actual, expected);
    }

    #[test]
    fn fixed_point_u10_test() {
        let actual = fixed_point_u10(0.0);
        let expected = 0b0000000000u32;
        assert_eq!(actual, expected);

        let actual = fixed_point_u10(0.5);
        let expected = 0b1000000000u32;
        assert_eq!(actual, expected);

        let actual = fixed_point_u10(0.25);
        let expected = 0b0100000000u32;
        assert_eq!(actual, expected);
    }

    #[test]
    fn expand_bits_u10_test() {
        let actual = expand_bits_u10(0b1111111111u32);
        let expected = 0b1001001001001001001001001001u32;
        assert_eq!(actual, expected);
    }
}
