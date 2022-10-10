use cuda_std::{kernel, shared_array, thread};

/// The length of data to run the prefix operation on.
///
/// The max section size of this algorithm is the max shared memory size for a
/// block.
pub const SECTION_SIZE: usize = 12 * 1024;

/// The kernel launch should use this as the block size.
pub const BLOCK_SIZE: usize = 1024;

pub const SUBSECTION_SIZE: usize = SECTION_SIZE / BLOCK_SIZE;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn inclusive_three_phase_scan(xs: &[u32], ys: *mut u32) {
    let t_idx = thread::thread_idx_x() as usize;
    // let b_idx = thread::block_idx_x() as usize;
    // let b_dim = thread::block_dim_x() as usize;
    // let g_idx = b_idx * b_dim + t_idx; // global index
    let xys = shared_array![u32; SECTION_SIZE];

    // Copy the contents of global memory xs into shared memory xys. The threads
    // cooperatively load each subsection from global memory in a coalesced manner
    // before moving on to the next subsection.
    let mut i = t_idx;
    while i < SECTION_SIZE {
        *(&mut *xys.add(i)) = xs[i];
        i += BLOCK_SIZE;
    }

    thread::sync_threads();

    // Each thread gets its own subsection of xys.
    let subsection_start = t_idx * SUBSECTION_SIZE;

    // Each thread performs a sequential scan on its subsection.
    inclusive_sequential_scan(xys, subsection_start, subsection_start + SUBSECTION_SIZE);

    thread::sync_threads();

    // Copy the contents of sharesd memory xys into global memory ys. The threads
    // cooperatively load each subsection from shared memory in a coalesced manner
    // before moving on to the next subsection.
    let mut i = t_idx;
    while i < SECTION_SIZE {
        *(&mut *ys.add(i)) = *xys.add(i);
        i += BLOCK_SIZE;
    }
}

/// In-place inclusive sequential scan.
unsafe fn inclusive_sequential_scan(xs: *mut u32, start: usize, end: usize) -> u32 {
    let mut accumulator = *xs.add(start);

    for i in start + 1..end {
        accumulator += *xs.add(i);
        *(&mut *xs.add(i)) = accumulator;
    }

    accumulator
}
