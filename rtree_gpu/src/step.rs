pub struct MultStep {
    factor: usize,
    next: usize,
}

impl Iterator for MultStep {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next;
        self.next *= self.factor;
        Some(next)
    }
}

// Returns an iterator that generates numbers by multiplying by the given
// factor.
pub fn mult_step(init: usize, factor: usize) -> MultStep {
    MultStep { factor, next: init }
}
