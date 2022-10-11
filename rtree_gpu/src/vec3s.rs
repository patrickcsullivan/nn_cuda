use cuda_std::vek::Vec3;

/// Contains 3-dimensional vectors organized as a struct of arrays.
pub struct Vec3s {
    pub xs: *mut f32,
    pub ys: *mut f32,
    pub zs: *mut f32,
}

impl Vec3s {
    pub fn new(xs: *mut f32, ys: *mut f32, zs: *mut f32) -> Self {
        Self { xs, ys, zs }
    }

    pub fn get_at(&self, i: usize) -> Vec3<f32> {
        unsafe { Vec3::new(*self.xs.add(i), *self.ys.add(i), *self.zs.add(i)) }
    }
}
