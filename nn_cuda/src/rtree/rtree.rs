use super::{
    data::{DeviceData, HostData},
    geometry::{points_aabb, vecs_aabb},
};
use crate::{morton::map_to_morton_codes, point::Point3, z_curve::z_curve};
use cuda_std::vek::Vec3;
use cust::prelude::*;
use itertools::Itertools;
use std::{error::Error, time::Instant};

static PTX: &str = include_str!("../../../resources/rtree_gpu.ptx");

// These should match the constants in the kernel for now.
pub const M: usize = 4;
pub const H: usize = 8;
pub const B: usize = 32;

pub struct RTree<'a, T> {
    objects: &'a [T],
    module: Module,
    device_data: DeviceData,
}

impl<'a, T> RTree<'a, T>
where
    T: Point3,
{
    pub fn new(objects: &'a [T]) -> Result<Self, Box<dyn Error>> {
        let module = Module::from_ptx(PTX, &[])?;
        let host_data: HostData<M, H> = HostData::new(objects);
        let device_data = host_data.copy_to_device()?;
        Ok(Self {
            objects,
            module,
            device_data,
        })
    }

    pub fn batch_find_neighbors<Q>(
        &self,
        stream: &Stream,
        queries: &[Q],
    ) -> Result<Vec<&'a T>, Box<dyn Error>>
    where
        Q: Point3,
    {
        // Order queries along Z-curve.
        let vecs = queries.iter().map(|q| q.into_vec3()).collect_vec();
        let aabb = points_aabb(&queries);
        let (sorted_query_indices, sorted_queries) = z_curve(&vecs, &aabb);

        // Allocate host memory for results.
        let mut result_object_indices = vec![0usize; queries.len()];

        // Allocate device memory for queries and results.
        let dev_sorted_queries = sorted_queries.as_slice().as_dbuf()?;
        let dev_result_object_indices = result_object_indices.as_slice().as_dbuf()?;

        self.launch_kernel(stream, &dev_sorted_queries, &dev_result_object_indices)?;
        stream.synchronize()?;

        // Copy results from the device back to the host.
        dev_result_object_indices.copy_to(&mut result_object_indices)?;

        // Reorder results so they are in the same order as the original queries.
        let mut original_order_nn_object_indices = vec![0; queries.len()];
        sorted_query_indices
            .into_iter()
            .zip(result_object_indices)
            .for_each(|(qi, oi)| original_order_nn_object_indices[qi] = oi);
        let original_order_nns = original_order_nn_object_indices
            .into_iter()
            .map(|oi| &self.objects[oi])
            .collect_vec();

        Ok(original_order_nns)
    }

    fn launch_kernel(
        &self,
        stream: &Stream,
        dev_queries: &DeviceBuffer<Vec3<f32>>,
        dev_result_object_indices: &DeviceBuffer<usize>,
    ) -> Result<(), Box<dyn Error>> {
        let kernel = self.module.get_function("bulk_find_neighbors")?;
        let grid_size = (dev_queries.len() + B - 1) / B;
        unsafe {
            launch!(
                kernel<<<grid_size as u32, B as u32, 0, stream>>>(
                    self.device_data.node_min_xs.as_device_ptr(),
                    self.device_data.node_min_xs.len(),
                    self.device_data.node_min_ys.as_device_ptr(),
                    self.device_data.node_min_ys.len(),
                    self.device_data.node_min_zs.as_device_ptr(),
                    self.device_data.node_min_zs.len(),
                    self.device_data.node_max_xs.as_device_ptr(),
                    self.device_data.node_max_xs.len(),
                    self.device_data.node_max_ys.as_device_ptr(),
                    self.device_data.node_max_ys.len(),
                    self.device_data.node_max_zs.as_device_ptr(),
                    self.device_data.node_max_zs.len(),
                    //-----
                    self.device_data.leaf_starts.as_device_ptr(),
                    self.device_data.leaf_starts.len(),
                    self.device_data.leaf_ends.as_device_ptr(),
                    self.device_data.leaf_ends.len(),
                    //-----
                    self.device_data.sorted_object_indices.as_device_ptr(),
                    self.device_data.sorted_object_indices.len(),
                    self.device_data.sorted_object_xs.as_device_ptr(),
                    self.device_data.sorted_object_xs.len(),
                    self.device_data.sorted_object_ys.as_device_ptr(),
                    self.device_data.sorted_object_ys.len(),
                    self.device_data.sorted_object_zs.as_device_ptr(),
                    self.device_data.sorted_object_zs.len(),
                    //-----
                    dev_queries.as_device_ptr(),
                    dev_queries.len(),
                    dev_result_object_indices.as_device_ptr(),
                )
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::RTree;
    use crate::Point3;
    use cust::stream::{Stream, StreamFlags};
    use rand::{seq::IteratorRandom, Rng, SeedableRng};
    use rand_hc::Hc128Rng;

    const SEED: &[u8; 32] = b"LVXn6sWNasjDReRS2OZ9a0eY1aprVNYX";

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct PointObject([f32; 3]);

    impl crate::Point3 for PointObject {
        fn xyz(&self) -> [f32; 3] {
            self.0
        }
    }

    #[test]
    fn finds_neighbors() {
        let _ctx = cust::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let mut rng = Hc128Rng::from_seed(*SEED);
        let points = create_random_points(1_000_000 as usize, &mut rng);
        let queries = create_random_points(1_000, &mut rng);
        let rtree = RTree::new(&points).unwrap();
        let nns = rtree.batch_find_neighbors(&stream, &queries).unwrap();

        for (q, nn) in queries.iter().zip(nns).choose_multiple(&mut rng, 100) {
            let expected = find_neighbor_brute_force(&points, q).into_vec3();
            assert_eq!(
                nn.into_vec3().distance_squared(q.into_vec3()),
                expected.distance_squared(q.into_vec3())
            );
        }
    }

    fn create_random_points(points_count: usize, rng: &mut impl Rng) -> Vec<PointObject> {
        let mut result = Vec::with_capacity(points_count);
        for _ in 0..points_count {
            result.push(PointObject(rng.gen()));
        }
        result
    }

    fn find_neighbor_brute_force<'a, T, Q>(points: &'a [T], query: &Q) -> &'a T
    where
        T: Point3,
        Q: Point3,
    {
        let mut min_dist2 = f32::INFINITY;
        let mut nn = None;
        for p in points {
            let dist2 = query.into_vec3().distance_squared(p.into_vec3());
            if dist2 < min_dist2 {
                min_dist2 = dist2;
                nn = Some(p)
            }
        }
        nn.unwrap()
    }
}
