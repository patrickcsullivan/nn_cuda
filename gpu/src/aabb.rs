use cuda_std::vek::{approx::RelativeEq, num_traits::Float, Clamp, Vec3};
use cust_core::DeviceCopy;

// use crate::f32::{partial_max, partial_min};

#[derive(Clone, Copy, Debug, DeviceCopy)]
#[repr(C)]
pub struct DeviceCopyAabb<T>
where
    T: Copy + PartialOrd,
{
    pub min: Vec3<T>,
    pub max: Vec3<T>,
}

impl<T> DeviceCopyAabb<T>
where
    T: Copy + PartialOrd,
{
    pub fn new_empty(p: Vec3<T>) -> Self {
        Self { min: p, max: p }
    }
}

impl<T> DeviceCopyAabb<T>
where
    T: Clamp + Copy + PartialOrd + RelativeEq + Float,
{
    pub fn distance_squared_to_point(self, p: Vec3<T>) -> T {
        let x = Self::distance_to_range(p.x, self.min.x, self.max.x);
        let y = Self::distance_to_range(p.y, self.min.y, self.max.y);
        let z = Self::distance_to_range(p.z, self.min.z, self.max.z);
        x + y + z
    }

    fn distance_to_range(val: T, min: T, max: T) -> T {
        if val < min {
            min - val
        } else if val > max {
            val - max
        } else {
            T::zero()
        }
    }
}
