use cuda_std::vek::Vec3;

#[derive(Clone, Copy)]
pub struct RTree<'a> {
    /// The fixed height of the R-tree.
    pub height: usize,

    /// The number of children per node.
    pub children_per_node: usize,

    /// The number of interior nodes.
    pub interior_count: usize,

    /// A heap containing the minimum x coordinate each node's bounding box.
    pub node_min_xs: &'a [f32],

    /// A heap containing the minimum y coordinate each node's bounding box.
    pub node_min_ys: &'a [f32],

    /// A heap containing the minimum z coordinate each node's bounding box.
    pub node_min_zs: &'a [f32],

    /// A heap containing the maximum x coordinate each node's bounding box.
    pub node_max_xs: &'a [f32],

    /// A heap containing the maximum y coordinate each node's bounding box.
    pub node_max_ys: &'a [f32],

    /// A heap containing the maximum z coordinate each node's bounding box.
    pub node_max_zs: &'a [f32],

    /// Contains the start index of the objects inside by each leaf node.
    pub leaf_starts: &'a [usize],

    /// Contains the end index of the objects inside by each leaf node.
    pub leaf_ends: &'a [usize],

    /// Contains the original indices of the objects inside the tree.
    pub sorted_object_indices: &'a [usize],

    /// Contins the x coordinates of the objects inside the tree.
    pub sorted_object_xs: &'a [f32],

    /// Contins the y coordinates of the objects inside the tree.
    pub sorted_object_ys: &'a [f32],

    /// Contins the z coordinates of the objects inside the tree.
    pub sorted_object_zs: &'a [f32],
}

/// Represents the data contained at either an interior node or a leaf node.
pub enum NodeContents {
    /// The data contained inside an interior node.
    InteriorChildren {
        /// An index into the R-tree's heap of nodes that points to the interior
        /// node's first child.
        start: usize,
    },

    /// The data contained inside a leaf node.
    LeafObjects {
        /// An index into the R-tree's sorted objects that points to the first
        /// object contained in the leaf node.
        start: usize,

        /// An index into the R-tree's sorted objects that points to the last
        /// object contained in the leaf node.
        end: usize,
    },
}

impl<'a> RTree<'a> {
    pub fn new(
        height: usize,
        children_per_node: usize,
        node_min_xs: &'a [f32],
        node_min_ys: &'a [f32],
        node_min_zs: &'a [f32],
        node_max_xs: &'a [f32],
        node_max_ys: &'a [f32],
        node_max_zs: &'a [f32],
        leaf_starts: &'a [usize],
        leaf_ends: &'a [usize],
        sorted_object_indices: &'a [usize],
        sorted_object_xs: &'a [f32],
        sorted_object_ys: &'a [f32],
        sorted_object_zs: &'a [f32],
    ) -> Self {
        let interior_count = (children_per_node.pow(height as u32) - 1) / (children_per_node - 1);

        Self {
            height,
            children_per_node,
            interior_count,

            node_min_xs,
            node_min_ys,
            node_min_zs,
            node_max_xs,
            node_max_ys,
            node_max_zs,
            leaf_starts,
            leaf_ends,
            sorted_object_indices,
            sorted_object_xs,
            sorted_object_ys,
            sorted_object_zs,
        }
    }

    pub fn find_neighbor(&self, query: Vec3<f32>) -> Option<usize> {
        Some(42)
    }

    /// Returns the index of the root node.
    fn root(&self) -> usize {
        0
    }

    /// Returns the child node start and end indices if the node is interior or
    /// the object start and end indices if the node is a leaf.
    fn get_contents(self, node_idx: usize) -> NodeContents {
        if self.is_interior(node_idx) {
            NodeContents::InteriorChildren {
                start: self.first_child_index(node_idx),
            }
        } else {
            let leaf_offset = node_idx - self.interior_count;
            NodeContents::LeafObjects {
                start: self.leaf_starts[leaf_offset],
                end: self.leaf_ends[leaf_offset],
            }
        }
    }

    /// Returns true if the given index refers to an interior node and false if
    /// it refers to a leaf node.
    fn is_interior(&self, node_idx: usize) -> bool {
        node_idx < self.interior_count
    }

    /// Returns the index of the first child of the specified node.
    fn first_child_index(&self, node_idx: usize) -> usize {
        self.children_per_node * node_idx + 1
    }
}
