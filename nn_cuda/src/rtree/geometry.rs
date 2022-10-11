use crate::Point3;
use cuda_std::vek::{Aabb, Vec3};
use itertools::Itertools;

pub fn union_aabbs<'a, I>(mut aabbs: I) -> Aabb<f32>
where
    I: Iterator<Item = &'a Aabb<f32>>,
{
    let mut aabb = aabbs.next().unwrap().clone();
    for a in aabbs {
        aabb.expand_to_contain(*a)
    }
    aabb
}

pub fn points_aabb<T>(points: &[T]) -> Aabb<f32>
where
    T: Point3,
{
    let mut vecs = points.iter().map(|p| p.into_vec3()).collect_vec();
    vecs_aabb(&vecs)
}

pub fn vecs_aabb(vecs: &[Vec3<f32>]) -> Aabb<f32> {
    let mut vecs = vecs.iter();
    let mut aabb = Aabb::new_empty(*vecs.next().unwrap());
    for v in vecs {
        aabb.expand_to_contain_point(*v)
    }
    aabb
}
