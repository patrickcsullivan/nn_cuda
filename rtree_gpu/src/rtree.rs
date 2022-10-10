pub struct RTree<'a, const M: usize, const H: usize> {
    /// A heap containing the minimum x coordinate each node's bounding box.
    node_min_xs: &'a [f32],

    /// A heap containing the minimum y coordinate each node's bounding box.
    node_min_ys: &'a [f32],

    /// A heap containing the minimum z coordinate each node's bounding box.
    node_min_zs: &'a [f32],

    /// A heap containing the maximum x coordinate each node's bounding box.
    node_max_xs: &'a [f32],

    /// A heap containing the maximum y coordinate each node's bounding box.
    node_max_ys: &'a [f32],

    /// A heap containing the maximum z coordinate each node's bounding box.
    node_max_zs: &'a [f32],

    /// Contains the start index of the objects inside by each leaf node.
    leaf_starts: &'a [usize],

    /// Contains the end index of the objects inside by each leaf node.
    leaf_ends: &'a [usize],

    /// Contins the x coordinates of the objects inside the tree.
    object_xs: &'a [usize],

    object_ys: &'a [usize],

    object_zs: &'a [usize],
}

impl<'a, const M: usize, const H: usize> RTree<'a, M, H> {
    /// Returns the index of the `i`-th child of the node at `node_idx`.
    pub fn get_child(self, node_idx: usize, i: usize) -> usize {
        todo!()
    }

    /// Returns the child node start and end indices if the node is interior or
    /// the object start and end indices if the node is a leaf.
    pub fn get_children(self, node_idx: usize) -> NodeData {
        todo!()
    }
}

pub enum NodeData {
    InteriorChildren { start: usize, end: usize },
    LeafObjects { start: usize, end: usize },
}
