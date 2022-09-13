pub struct Stack<T, const N: usize> {
    values: [T; N],
    pointer: usize,
}

impl<T, const N: usize> Stack<T, N>
where
    T: Default,
    T: Copy,
{
    pub fn empty() -> Self {
        Self {
            values: [T::default(); N],
            pointer: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        if self.pointer < N {
            self.values[self.pointer] = value;
            self.pointer += 1;
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.pointer > 0 {
            self.pointer -= 1;
            Some(self.values[self.pointer])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Stack;

    #[test]
    fn stack_test() {
        let mut stack = Stack::<i32, 4>::empty();
        assert_eq!(stack.pop(), None);

        stack.push(1);
        stack.push(2);
        stack.push(3);

        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn full_stack_test() {
        let mut stack = Stack::<i32, 2>::empty();

        stack.push(1);
        stack.push(2);
        stack.push(3);
        stack.push(4);

        // 3 and 4 aren't added, since the stack is full.
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
    }
}
