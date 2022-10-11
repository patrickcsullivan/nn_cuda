pub struct Grid {
    elements: *mut f32,
    width: usize,
}

impl Grid {
    pub fn new(start_pointer: *mut f32, width: usize) -> Self {
        Self {
            elements: start_pointer,
            width,
        }
    }

    pub fn get_at(&self, row_idx: usize, col_idx: usize) -> f32 {
        unsafe { *self.elements.add(self.index_at(row_idx, col_idx)) }
    }

    pub fn set_at(&mut self, row_idx: usize, col_idx: usize, val: f32) {
        let idx = self.index_at(row_idx, col_idx);
        unsafe {
            *(&mut *self.elements.add(idx)) = val;
        }
    }

    fn index_at(&self, row_idx: usize, col_idx: usize) -> usize {
        col_idx * self.width + row_idx
    }
}
