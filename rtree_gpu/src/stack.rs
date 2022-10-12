pub struct Stack<T> {
    elements: *mut T,
    max_size: usize,
    size: usize,
}

impl<T> Stack<T>
where
    T: Copy,
{
    pub fn new(start_pointer: *mut T, max_size: usize) -> Self {
        Self {
            elements: start_pointer,
            max_size,
            size: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.size < self.max_size {
            unsafe {
                *(&mut *self.elements.add(self.size)) = item;
            }
            self.size += 1;
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.size > 0 {
            unsafe {
                let e = *self.elements.add(self.size - 1);
                self.size -= 1;
                Some(e)
            }
        } else {
            None
        }
    }

    pub fn top(&self) -> Option<T> {
        if self.size > 0 {
            unsafe {
                let e = *self.elements.add(self.size - 1);
                Some(e)
            }
        } else {
            None
        }
    }
}
