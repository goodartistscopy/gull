extern crate nalgebra as na;
extern crate nalgebra_glm as glm;
pub type Vector3 = na::Vector3<f32>;
pub type Point3 = na::Point3<f32>;
pub type Matrix3 = na::Matrix3<f32>;
pub type Isometry3 = na::Isometry3<f32>;
pub use na::{vector, point};

pub struct Mesh {
    pub positions: Vec<Point3>,
    pub normals: Vec<Vector3>,
    pub indices: Vec<[u32; 3]>
}

//use gltf::*;
use gltf::buffer::Target;
use gltf::accessor::{DataType, Dimensions};
use gltf::mesh::Mode;
use gltf::Semantic;

use std::mem::size_of;
use std::slice;

impl Mesh {
    pub fn num_vertices(&self) -> u32 {
        self.positions.len() as u32
    }

    pub fn num_triangles(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn new_icosphere(radius: f32, num_subdivisions: u8) -> Self
    {
        use std::f32::consts::PI;

        let mut sphere = Mesh { positions: Vec::new(), normals: Vec::new(), indices: Vec::new() };
        const LONG_ANGLE: f32 = 2.0 * PI / 5.0;
        const LAT_ANGLE: f32 = PI / 3.0;
        // 22 vertices instead of 12 because poles and one "meridian" are split for texture coordinates
        // north pole (5 times)
        sphere.positions.extend_from_slice(&[point![0.0, 0.0, radius]; 5]);
        sphere.normals.extend_from_slice(&[vector![0.0, 0.0, 1.0]; 5]);
        // 2 middle layers 
        for i in 0..2 {
            let theta = (i + 1) as f32 * LAT_ANGLE;
            let (stheta, ctheta) = theta.sin_cos();
            for j in 0..=5 {
                let phi = (j as f32) * LONG_ANGLE + i as f32 * LONG_ANGLE / 2.0;
                let normal = vector![stheta * phi.cos(), stheta * phi.sin(), ctheta];
                sphere.positions.push((radius * normal).into());
                sphere.normals.push(normal);
            }
        }
        // south pole (5 times)
        sphere.positions.extend_from_slice(&[point![0.0, 0.0, -radius]; 5]);
        sphere.normals.extend_from_slice(&[vector![0.0, 0.0, -1.0]; 5]);

        sphere.indices = vec![
            // north segment
            [0, 5, 6],
            [1, 6, 7],
            [2, 7, 8],
            [3, 8, 9],
            [4, 9, 10],
            // equatorian segment
            [6, 5, 11],
            [6, 11, 12],
            [7, 6, 12], 
            [7, 12, 13],
            [8, 7, 13],
            [8, 13, 14],
            [9, 8, 14],
            [9, 14, 15],
            [10, 9, 15],
            [10, 15, 16],
            // south segment
            [11, 17, 12],
            [12, 18, 13],
            [13, 19, 14],
            [14, 20, 15],
            [15, 21, 16],
            ];

            for _ in 0..num_subdivisions {
                let mut new_triangles = Vec::new();
                while let Some(tri_indices) =  sphere.indices.pop() {
                    let idx = sphere.positions.len() as u32;
                    let new_indices = [ idx, idx + 1, idx + 2]; 
                    new_triangles.push([tri_indices[0], new_indices[2], new_indices[1]]);
                    new_triangles.push([new_indices[2], tri_indices[1], new_indices[0]]);
                    new_triangles.push([new_indices[2], new_indices[0], new_indices[1]]);
                    new_triangles.push([new_indices[1], new_indices[0], tri_indices[2]]);

                    let v0 = sphere.positions[tri_indices[0] as usize];
                    let v1 = sphere.positions[tri_indices[1] as usize];
                    let v2 = sphere.positions[tri_indices[2] as usize];

                    let n0 = (0.5 * (v1.coords + v2.coords)).normalize();
                    let p0 = Point3::from(radius * n0);

                    let n1 = (0.5 * (v2.coords + v0.coords)).normalize();
                    let p1 = Point3::from(radius * n1);

                    let n2 = (0.5 * (v0.coords + v1.coords)).normalize();
                    let p2 = Point3::from(radius * n2);

                    sphere.positions.push(p0.into());
                    sphere.positions.push(p1.into());
                    sphere.positions.push(p2.into());

                    sphere.normals.push(n0.into());
                    sphere.normals.push(n1.into());
                    sphere.normals.push(n2.into());
                }
                sphere.indices = new_triangles;
            }

            println!("Sphere has {} vertices", sphere.positions.len());

            sphere
    }

    pub fn from_gltf(document: &gltf::Gltf, bin_data: &[u8], mesh_name: &str) -> Option<Vec<Self>>
    {
        if let Some(mesh) = document.meshes()
            .filter(|mesh| mesh.name().unwrap_or("") == mesh_name)
            .next()
        {
            let primitives = mesh.primitives();
            let mut meshes = Vec::with_capacity(primitives.len());

            println!("found object {} ({} primitives)", mesh_name, primitives.len());
            for primitive in primitives {
                assert!(primitive.mode() == Mode::Triangles);

                let indices_accessor = primitive.indices().unwrap();
                assert!((indices_accessor.data_type() == DataType::U16) ||
                        (indices_accessor.data_type() == DataType::U32));
                assert!(indices_accessor.dimensions() == Dimensions::Scalar);
                assert!(indices_accessor.view().unwrap().target() == Some(Target::ElementArrayBuffer));
                assert!((indices_accessor.view().unwrap().stride() == None) ||
                        (indices_accessor.view().unwrap().stride().unwrap() == 3_usize * size_of::<u32>()));
                assert!(indices_accessor.count() % 3 == 0);

                let view = indices_accessor.view().unwrap();
                let mut indices = Vec::new();
                if indices_accessor.data_type() == DataType::U16 {
                    indices.reserve(indices_accessor.count() / 3);
                    let offset = view.offset() + indices_accessor.offset();
                    let size = indices_accessor.count() * size_of::<u16>();
                    let ptr = bin_data[offset..offset + size].as_ptr() as *const [u16; 3];
                    let data = unsafe { slice::from_raw_parts(ptr, indices_accessor.count() / 3) };
                    for idx16 in data {
                        indices.push([idx16[0] as u32, idx16[1] as u32, idx16[2] as u32]);
                    }
                } else {
                    let offset = view.offset() + indices_accessor.offset();
                    let size = indices_accessor.count() * size_of::<u32>();
                    let ptr = bin_data[offset..offset + size].as_ptr() as *const [u32; 3];
                    indices = unsafe { slice::from_raw_parts(ptr, indices_accessor.count() / 3) }.to_vec();
                }

                let mut positions = Vec::new();
                let mut normals = Vec::new();

                for attribute in primitive.attributes() {
                    match attribute.0 {
                        Semantic::Positions => {
                            let accessor = attribute.1;
                            assert!(accessor.data_type() == DataType::F32);
                            assert!(accessor.dimensions() == Dimensions::Vec3);
                            assert!((accessor.view().unwrap().stride() == None) ||
                                    (accessor.view().unwrap().stride().unwrap() == size_of::<Point3>()));
                            
                            let view = accessor.view().unwrap();
                            let offset = view.offset() + accessor.offset();
                            let size = accessor.count() * size_of::<Point3>();
                            let ptr = bin_data[offset..offset + size].as_ptr() as *const Point3;
                            positions = unsafe { slice::from_raw_parts(ptr, accessor.count()) }.to_vec();
                        },
                        Semantic::Normals => {
                            let accessor = attribute.1;
                            assert!(accessor.data_type() == DataType::F32);
                            assert!(accessor.dimensions() == Dimensions::Vec3);
                            assert!((accessor.view().unwrap().stride() == None) ||
                                    (accessor.view().unwrap().stride().unwrap() == size_of::<Vector3>()));
                            
                            let view = accessor.view().unwrap();
                            let offset = view.offset() + accessor.offset();
                            let size = accessor.count() * size_of::<Vector3>();
                            let ptr = bin_data[offset..offset + size].as_ptr() as *const Vector3;
                            normals = unsafe { slice::from_raw_parts(ptr, accessor.count()) }.to_vec();
                        },
                        _ => ()
                    }
                }
                
                meshes.push(Mesh { positions, normals, indices });
            }
        
            return Some(meshes);
        }

        None
    }
}


