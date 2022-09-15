use core::fmt::Debug;

use cuda_std::vek::{approx::RelativeEq, num_traits::Float, Clamp, Extent3, Vec3};
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

    pub fn union(self, other: Self) -> Self {
        let min = Vec3::<T>::partial_min(self.min, other.min);
        let max = Vec3::<T>::partial_max(self.max, other.max);
        Self { min, max }
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
        x * x + y * y + z * z
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

    pub fn size(self) -> Extent3<T> {
        let diff = self.max - self.min;
        Extent3::new(diff.x, diff.y, diff.z)
    }
}
