use cuda_std::shared_array;

pub struct PriorityQueue<T> {
    elements: *mut (T, f32),
    max_size: usize,
    size: usize,
}

impl<T> PriorityQueue<T> {
    pub fn new(start_pointer: *mut (T, f32), max_size: usize) -> Self {
        Self {
            elements: start_pointer,
            max_size,
            size: 0,
        }
    }

    pub fn push(&mut self, item: T, cost: f32) {
        todo!()
    }

    pub fn pop(&mut self) -> Option<T> {
        todo!()
    }
}
