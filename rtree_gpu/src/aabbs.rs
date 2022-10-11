use cuda_std::vek::{Aabb, Vec3};

/// Contains axis-aligned bounding boxes organized as a struct of arrays.
pub struct Aabbs {
    pub min_xs: *mut f32,
    pub min_ys: *mut f32,
    pub min_zs: *mut f32,
    pub max_xs: *mut f32,
    pub max_ys: *mut f32,
    pub max_zs: *mut f32,
}

impl Aabbs {
    pub fn new(
        min_xs: *mut f32,
        min_ys: *mut f32,
        min_zs: *mut f32,
        max_xs: *mut f32,
        max_ys: *mut f32,
        max_zs: *mut f32,
    ) -> Self {
        Self {
            min_xs,
            min_ys,
            min_zs,
            max_xs,
            max_ys,
            max_zs,
        }
    }

    pub fn get_at(&self, i: usize) -> Aabb<f32> {
        unsafe {
            let min = Vec3::new(
                *self.min_xs.add(i),
                *self.min_ys.add(i),
                *self.min_zs.add(i),
            );
            let max = Vec3::new(
                *self.max_xs.add(i),
                *self.max_ys.add(i),
                *self.max_zs.add(i),
            );
            Aabb { min, max }
        }
    }
}
