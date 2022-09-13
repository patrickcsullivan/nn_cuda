use cuda_std::vek::Aabb;
use cuda_std::vek::Vec3;
use itertools::Itertools;
use ply_rs::parser;
use ply_rs::ply;

#[derive(Debug)]
struct Vertex {
    v: Vec3<f32>,
}

impl Vertex {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Vertex {
            v: Vec3::new(x, y, z),
        }
    }
}

impl ply::PropertyAccess for Vertex {
    fn new() -> Self {
        Vertex { v: Vec3::zero() }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.v.x = v,
            ("y", ply::Property::Float(v)) => self.v.y = v,
            ("z", ply::Property::Float(v)) => self.v.z = v,
            (k, _) => panic!("Vertex: Unexpected key/value combination: key: {}", k),
        }
    }
}

pub fn ply_vertices(ply_path: &str) -> Vec<Vec3<f32>> {
    let file = std::fs::File::open(ply_path).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let vertex_parser = parser::Parser::<Vertex>::new();

    let header = vertex_parser.read_header(&mut reader).unwrap();
    let mut vertices = Vec::new();

    for (_, element) in &header.elements {
        if element.name == "vertex" {
            vertices = vertex_parser
                .read_payload_for_element(&mut reader, element, &header)
                .unwrap();
        }
    }

    vertices.iter().map(|v| v.v).collect_vec()
}

pub fn get_aabb(vs: &[Vec3<f32>]) -> Aabb<f32> {
    let mut aabb = Aabb::new_empty(vs[0]);
    for v in vs {
        aabb.expand_to_contain_point(*v);
    }
    aabb
}
