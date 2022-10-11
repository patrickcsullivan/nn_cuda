use cuda_std::shared_array;

pub struct SharedPriorityQueue {
    elements: *mut (usize, f32),
    max_size: usize,
    size: usize,
}

impl SharedPriorityQueue {
    pub fn new(start_pointer: *mut (usize, f32), max_size: usize) -> Self {
        Self {
            elements: start_pointer,
            max_size,
            size: 0,
        }
    }
}
