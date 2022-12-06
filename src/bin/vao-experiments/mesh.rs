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

impl Mesh {
    pub fn num_vertices(&self) -> u32 {
        self.positions.len() as u32
    }

    pub fn num_triangles(&self) -> u32 {
        self.indices.len() as u32
    }
}

pub fn create_icosphere(radius: f32, num_subdivisions: u8) -> Mesh
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
