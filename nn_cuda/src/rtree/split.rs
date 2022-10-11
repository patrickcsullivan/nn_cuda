struct Split {
    /// The number of chunks left.
    chunks_left: usize,

    /// The minimum size of chunks.
    chunk_min_size: usize,

    /// The number of items left that won't fit into evenly divided chunks.
    uneven_items_left: usize,

    /// The start index of the next chunk.
    next_chunk_start: usize,
}

impl Iterator for Split {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.chunks_left == 0 {
            return None;
        }

        let mut next_chunk_size = self.chunk_min_size;
        if self.uneven_items_left > 0 {
            next_chunk_size += 1;
            self.uneven_items_left -= 1;
        }

        let next_chunk_range = (
            self.next_chunk_start,
            self.next_chunk_start + next_chunk_size - 1,
        );
        self.next_chunk_start += next_chunk_size;
        self.chunks_left -= 1;

        Some(next_chunk_range)
    }
}

/// Returns an iterator that
pub fn split<T>(slice: &[T], n: usize) -> impl Iterator<Item = (usize, usize)> {
    // If we divide slice into n chunks then each chunk will have at least
    // chunk_size items.
    let chunk_min_size = slice.len() / n;

    // If we divide slice into n chunks of length chunk_min_size, then there will be
    // rem items left over.
    let uneven_items_left = slice.len() % n;

    Split {
        chunks_left: n,
        chunk_min_size,
        uneven_items_left,
        next_chunk_start: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::split;
    use itertools::Itertools;

    #[test]
    fn splits_approx_even() {
        let xs = vec![0; 14];
        let ranges = split(&xs, 4).collect_vec();
        assert_eq!(ranges, vec![(0, 3), (4, 7), (8, 10), (11, 13)]);
    }
}
