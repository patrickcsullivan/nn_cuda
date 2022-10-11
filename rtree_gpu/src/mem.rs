use cuda_std::thread;

/// Use the block of threads to copy the array of data at source memory into the
/// destination memory.
pub fn copy_at_ptr<T>(src: *const T, src_offset: usize, dst: *mut T, dst_offset: usize, len: usize)
where
    T: Copy,
{
    let mut i = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while i < len {
        unsafe {
            *(&mut *dst.add(i + dst_offset)) = *src.add(i + src_offset);
        }
        i += thread::block_idx_x() as usize;
    }
}

/// Use the block of threads to copy the array of data at source memory into the
/// destination memory.
pub fn copy_slice<T>(src: &[T], src_offset: usize, dst: *mut T, dst_offset: usize, len: usize)
where
    T: Copy,
{
    let mut i = (thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x()) as usize;
    while i < len {
        unsafe {
            *(&mut *dst.add(i + dst_offset)) = src[i + src_offset];
        }
        i += thread::block_idx_x() as usize;
    }
}
