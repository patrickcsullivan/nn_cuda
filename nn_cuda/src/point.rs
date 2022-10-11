use cuda_std::vek::Vec3;

pub trait Point3 {
    fn xyz(&self) -> [f32; 3];

    fn into_vec3(&self) -> Vec3<f32> {
        let xyz = self.xyz();
        Vec3::new(xyz[0], xyz[1], xyz[2])
    }
}
