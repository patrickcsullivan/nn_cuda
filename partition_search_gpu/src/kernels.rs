use cuda_std::{prelude::*, shared_array, vek::Vec3};

pub const PARTITIONS_COUNT: usize = 400; // 16 bytes per partition
const OBJECTS_CACHE_SIZE: usize = 256; // 12 bytes per object

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn partition_search_for_queries(
    sorted_object_indices: &[usize],
    sorted_object_xs: &[f32],
    sorted_object_ys: &[f32],
    sorted_object_zs: &[f32],
    //------
    partition_size: usize,
    partition_min_xs: &[f32],
    partition_min_ys: &[f32],
    partition_min_zs: &[f32],
    partition_max_xs: &[f32],
    partition_max_ys: &[f32],
    partition_max_zs: &[f32],
    //------
    queries: &[Vec3<f32>],
    //------
    results_object_indices: *mut usize,
    results_dists2: *mut f32,
) {
    // Allocate shared memory for partition voting.
    let partition_votes = shared_array![usize; PARTITIONS_COUNT];
    let next_partition_idx = shared_array![usize; 1];

    // Allocate shared memory for partition AABBs.
    let shared_partition_min_xs = shared_array![f32; PARTITIONS_COUNT];
    let shared_partition_min_ys = shared_array![f32; PARTITIONS_COUNT];
    let shared_partition_min_zs = shared_array![f32; PARTITIONS_COUNT];
    let shared_partition_max_xs = shared_array![f32; PARTITIONS_COUNT];
    let shared_partition_max_ys = shared_array![f32; PARTITIONS_COUNT];
    let shared_partition_max_zs = shared_array![f32; PARTITIONS_COUNT];

    // Allocate shared memory for objects cache.
    let shared_sorted_object_xs = shared_array![f32; OBJECTS_CACHE_SIZE];
    let shared_sorted_object_ys = shared_array![f32; OBJECTS_CACHE_SIZE];
    let shared_sorted_object_zs = shared_array![f32; OBJECTS_CACHE_SIZE];

    // Initialize shared memory for partition AABBs.
    let mut block_thread_idx = thread::thread_idx_x() as usize;
    while block_thread_idx < PARTITIONS_COUNT {
        *shared_partition_min_xs.add(block_thread_idx) = partition_min_xs[block_thread_idx];
        *shared_partition_min_ys.add(block_thread_idx) = partition_min_ys[block_thread_idx];
        *shared_partition_min_zs.add(block_thread_idx) = partition_min_zs[block_thread_idx];
        *shared_partition_max_xs.add(block_thread_idx) = partition_max_xs[block_thread_idx];
        *shared_partition_max_ys.add(block_thread_idx) = partition_max_ys[block_thread_idx];
        *shared_partition_max_zs.add(block_thread_idx) = partition_max_zs[block_thread_idx];
        block_thread_idx += thread::block_dim_x() as usize;
    }

    // Wait until the partition AABBs have been copied into shared memory.
    thread::sync_threads();

    let mut grid_thread_idx =
        (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while grid_thread_idx < queries.len() {
        let query = queries[grid_thread_idx];

        let (nn_obj_idx, nn_dist2) = partition_search(
            sorted_object_indices,
            sorted_object_xs,
            sorted_object_ys,
            sorted_object_zs,
            shared_sorted_object_xs,
            shared_sorted_object_ys,
            shared_sorted_object_zs,
            partition_size,
            shared_partition_min_xs,
            shared_partition_min_ys,
            shared_partition_min_zs,
            shared_partition_max_xs,
            shared_partition_max_ys,
            shared_partition_max_zs,
            partition_votes,
            next_partition_idx,
            query,
        );

        *(&mut *results_object_indices.add(grid_thread_idx)) = nn_obj_idx;
        *(&mut *results_dists2.add(grid_thread_idx)) = nn_dist2;

        grid_thread_idx += (thread::block_dim_x() * thread::grid_dim_x()) as usize;
    }
}

unsafe fn partition_search(
    sorted_object_indices: &[usize],
    sorted_object_xs: &[f32],
    sorted_object_ys: &[f32],
    sorted_object_zs: &[f32],
    //-----
    shared_sorted_object_xs: *mut f32,
    shared_sorted_object_ys: *mut f32,
    shared_sorted_object_zs: *mut f32,
    //------
    partition_size: usize,
    partition_min_xs: *mut f32,
    partition_min_ys: *mut f32,
    partition_min_zs: *mut f32,
    partition_max_xs: *mut f32,
    partition_max_ys: *mut f32,
    partition_max_zs: *mut f32,
    partition_votes: *mut usize,
    next_partition_idx: *mut usize,
    //------
    query: Vec3<f32>,
) -> (usize, f32) {
    // Wait until all threads are ready to reset partition votes.
    thread::sync_threads();

    // Initialize shared memory for partition voting.
    let mut block_thread_idx = thread::thread_idx_x() as usize;
    if block_thread_idx == 0 {
        *next_partition_idx = 0;
    }
    while block_thread_idx < PARTITIONS_COUNT {
        *partition_votes.add(block_thread_idx) = 0;
        block_thread_idx += thread::block_dim_x() as usize;
    }

    // Wait until the votes for all partitions have been reset.
    thread::sync_threads();

    let mut nn_object_idx = usize::MAX;
    let mut nn_dist2 = f32::INFINITY;

    while *next_partition_idx < PARTITIONS_COUNT {
        // Wait until all threads are ready to start voting.
        thread::sync_threads();

        // Find partitions that could contain the NN and vote on them.
        for p_idx in 0..PARTITIONS_COUNT {
            // When partition_votes[p_idx] is usize::MAX, the partition has already been
            // searched, so we don't want to check it again.
            if *partition_votes.add(p_idx) < usize::MAX {
                let dist2 = dist2_to_aabb(
                    query,
                    *partition_min_xs.add(p_idx),
                    *partition_min_ys.add(p_idx),
                    *partition_min_zs.add(p_idx),
                    *partition_max_xs.add(p_idx),
                    *partition_max_ys.add(p_idx),
                    *partition_max_zs.add(p_idx),
                );
                if dist2 < nn_dist2 {
                    // Unsynchronized reads and writes to partition_votes could result in race
                    // conditions and inaccurate vote counts, but this is ok as since the vote
                    // counts don't need to be exact.
                    *partition_votes.add(p_idx) = *partition_votes.add(p_idx) + 1;
                }
            }
        }

        // Wait until all threads have finished voting on the next partition to search.
        thread::sync_threads();

        // Use the 0-th thread in the block to find the partition with the most votes
        // and to reset vote counts for the next iteration of the main loop.
        if thread::thread_idx_x() as usize == 0 {
            let mut max_votes_idx = 0;
            let mut max_votes_count = 0;
            for p_idx in 0..PARTITIONS_COUNT {
                let votes_count = *partition_votes.add(p_idx);

                // If the partition hasn't been visited yet...
                if votes_count < usize::MAX {
                    // Save the partition if it has the most votes so far.
                    if votes_count > max_votes_count {
                        max_votes_idx = p_idx;
                        max_votes_count = votes_count;
                    }

                    // Reset the partition's votes count.
                    *partition_votes.add(p_idx) = 0;
                }
            }

            // Mark the winning partition as visited so that we don't check it in future
            // iterations of the main loop.
            *partition_votes.add(max_votes_idx) = usize::MAX;

            // Record the index of the winning partition to shared memory. If no votes were
            // case, use usize::MAX to indicate that there is no next partition to search.
            *next_partition_idx = if max_votes_count > 0 {
                max_votes_idx
            } else {
                usize::MAX
            };
        }

        // Wait until the next partition has been selected..
        thread::sync_threads();

        let partition_idx = *next_partition_idx;

        // Every thread in the block searches the partition with the most votes.
        if partition_idx < PARTITIONS_COUNT {
            // Calculate the partition's start and end indices in the sorted_object arrays.
            let partition_from = partition_idx * partition_size;
            let partition_to =
                ((partition_idx + 1) * partition_size).min(sorted_object_indices.len());

            // Loop through each point in partition, checking to see if it's the NN.
            for sorted_data_idx in partition_from..partition_to {
                let cache_idx = (sorted_data_idx - partition_from) % OBJECTS_CACHE_SIZE;

                // If we're trying to examine the first point in the cache, then scan forward
                // and load the cache.
                if cache_idx == 0 {
                    thread::sync_threads();
                    let mut look_ahead_idx = thread::thread_idx_x() as usize;
                    while look_ahead_idx < OBJECTS_CACHE_SIZE
                        && sorted_data_idx + look_ahead_idx < partition_to
                    {
                        *shared_sorted_object_xs.add(look_ahead_idx) =
                            sorted_object_xs[sorted_data_idx + look_ahead_idx];
                        *shared_sorted_object_ys.add(look_ahead_idx) =
                            sorted_object_ys[sorted_data_idx + look_ahead_idx];
                        *shared_sorted_object_zs.add(look_ahead_idx) =
                            sorted_object_zs[sorted_data_idx + look_ahead_idx];
                        look_ahead_idx += thread::block_dim_x() as usize;
                    }
                    thread::sync_threads();
                }

                let x = *shared_sorted_object_xs.add(cache_idx);
                let y = *shared_sorted_object_ys.add(cache_idx);
                let z = *shared_sorted_object_zs.add(cache_idx);
                let dist2 = dist2_to_point(query, x, y, z);
                if dist2 < nn_dist2 {
                    nn_object_idx = sorted_object_indices[sorted_data_idx];
                    nn_dist2 = dist2;
                }
            }
        }
    }

    (nn_object_idx, nn_dist2)
}

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

fn div_ceil(numerator: usize, denominator: usize) -> usize {
    (numerator + denominator - 1) / denominator
}

// Calculate the partition's start and end indices in the sorted_object arrays.
/*
let partition_from = partition_idx * partition_size;
let partition_to =
    ((partition_idx + 1) * partition_size).min(sorted_object_indices.len());
let partition_size = partition_to - partition_from;

let cache_count = div_ceil(partition_size, OBJECTS_CACHE_SIZE);
for cache_idx in 0..cache_count {
    // Calculate the cache's start and end indicies in the sorted_object arrays.
    let cache_from = partition_from + cache_idx * OBJECTS_CACHE_SIZE;
    let cache_to =
        (partition_from + (cache_idx + 1) * OBJECTS_CACHE_SIZE).min(partition_to);
    let cache_size = cache_to - cache_from;

    // Load the cache.
    thread::sync_threads();
    let mut look_ahead_idx = thread::thread_idx_x() as usize;
    while look_ahead_idx < cache_size {
        *shared_sorted_object_xs.add(look_ahead_idx) =
            sorted_object_xs[cache_from + look_ahead_idx];
        *shared_sorted_object_ys.add(look_ahead_idx) =
            sorted_object_ys[cache_from + look_ahead_idx];
        *shared_sorted_object_zs.add(look_ahead_idx) =
            sorted_object_zs[cache_from + look_ahead_idx];
        look_ahead_idx += thread::block_dim_x() as usize;
    }
    thread::sync_threads();

    // Search the loaded cache.
    for look_ahead_idx in 0..cache_size {
        let x = *shared_sorted_object_xs.add(look_ahead_idx);
        let y = *shared_sorted_object_ys.add(look_ahead_idx);
        let z = *shared_sorted_object_zs.add(look_ahead_idx);
        let dist2 = dist2_to_point(query, x, y, z);
        if dist2 < nn_dist2 {
            nn_object_idx = sorted_object_indices[cache_from + look_ahead_idx];
            nn_dist2 = dist2;
        }
    }
}
*/
