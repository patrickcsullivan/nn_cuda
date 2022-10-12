use cuda_std::thread::{sync_threads, thread_idx_x};

pub struct SharedStack<T> {
    elements: *mut T,
    size: *mut usize,
    max_size: usize,
}

impl<T> SharedStack<T>
where
    T: Copy,
{
    pub fn new(elements_ptr: *mut T, size_ptr: *mut usize, max_size: usize) -> Self {
        if thread_idx_x() as usize == 0 {
            unsafe {
                *(&mut *size_ptr) = 0;
            }
        }
        sync_threads();

        Self {
            elements: elements_ptr,
            size: size_ptr,
            max_size,
        }
    }

    pub fn push(&mut self, item: T) {
        if thread_idx_x() as usize == 0 {
            unsafe {
                let size = *self.size;
                if size < self.max_size {
                    *(&mut *self.elements.add(size)) = item;
                    *(&mut *self.size) = size + 1;
                }
            }
        }
        sync_threads();
    }

    pub fn pop(&mut self) {
        if thread_idx_x() as usize == 0 {
            unsafe {
                let size = *self.size;
                if size > 0 {
                    *(&mut *self.size) = size - 1;
                }
            }
        }
        sync_threads();
    }

    pub fn top(&self) -> Option<T> {
        unsafe {
            let size = *self.size;
            if size > 0 {
                let e = *self.elements.add(size - 1);
                Some(e)
            } else {
                None
            }
        }
    }
}
