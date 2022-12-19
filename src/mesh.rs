//extern crate nalgebra as na;
//extern crate nalgebra_glm as glm;
//pub type Vector3 = na::Vector3<f32>;
//pub type Point3 = na::Point3<f32>;
//pub type Matrix3 = na::Matrix3<f32>;
//pub type Isometry3 = na::Isometry3<f32>;
//pub use na::{vector, point};

use std::mem;
use std::mem::size_of;
use std::collections::HashMap;
use std::ops::Rem;
use std::marker::PhantomData;

use memoffset::offset_of;

pub type Vector3 = nalgebra::Vector3<f32>;
pub type Point3 = nalgebra::Point3<f32>;
pub use nalgebra::{vector, point};

use self::attribute::*;
use self::vertex_buffer::*;

pub mod attribute {
    #[derive(Copy, Clone)]
    #[derive(Debug)]
    pub enum BaseType {
        HalfFloat,
        Float,
        Double,
        Byte,
        Short,
        Int,
        UnsignedByte,
        UnsignedShort,
        UnsignedInt
    }

    impl BaseType {
        pub fn size(&self) -> u32 {
            match self {
                BaseType::Double => 8,
                BaseType::Float | BaseType::Int | BaseType::UnsignedInt => 4,
                BaseType::HalfFloat | BaseType::Short | BaseType::UnsignedShort => 2,
                BaseType::Byte | BaseType::UnsignedByte => 1
            }
        }
    }

    #[derive(Copy, Clone)]
    #[derive(PartialEq, Debug)]
    pub enum Semantic {
        Position,
        Normal,
        Tangent,
        Bitangent,
        Color(u8),
        TexCoord(u8),
        Generic(u8),
    }

    #[derive(Clone)]
    #[derive(Debug)]
    pub struct Attribute {
        pub semantic: Semantic,
        pub dimension: u8,
        pub base_type: BaseType,
        pub normalized: bool,
        pub offset: u16,
    }

    impl Attribute {
        pub fn size(&self) -> u32 {
            self.dimension as u32 * self.base_type.size()
        }
    }
}

pub mod vertex_buffer {
    use super::attribute::*;

    #[derive(Clone)]
    pub struct VertexLayout(pub(super) Vec<Attribute>);

    impl VertexLayout {
        pub fn attributes(&self) -> &Vec<Attribute> {
            &self.0
        }

        pub fn attributes_mut(&mut self) -> &mut Vec<Attribute> {
            &mut self.0
        }
        /// Size of all atributes. Note that this can differ from the vertex buffer stride
        pub fn size(&self) -> u16 {
            self.0.iter().fold(0, |size, attribute| size + attribute.size() as u16)
        }
    
        /// Returns the Attribute of the attribute having the given semantic,
        /// if the vertex buffer contains such attribute
        pub fn attribute(&self, semantic: Semantic) -> Option<&Attribute> {
           self.0.iter().find(|attr| attr.semantic == semantic)
        }
    }

    /// Builder for VertexLayout, taking care of attribute semantic unicity and 
    /// computing offsets
    pub struct VertexLayoutBuilder {
        layout: Vec<Attribute>,
        size: u16,
    }

    impl VertexLayoutBuilder  {
        pub fn new() -> VertexLayoutBuilder {
            VertexLayoutBuilder { layout: Vec::new(), size: 0 }
        }

        // TODO check unicity of semantic
        pub fn add(mut self, mut attribute: Attribute) -> Self {
            attribute.offset = self.size; 
            self.size += attribute.size() as u16;
            self.layout.push(attribute);
            self
        }

        pub fn build(self) -> VertexLayout {
            VertexLayout(self.layout)
        }
    }

    /// The trait for type storing vertex attributes data
    pub trait VertexBuffer {
        fn stride(&self) -> u16;

        fn ptr(&self) -> *const u8;

        fn layout(&self) -> &VertexLayout;

        fn num_vertices(&self) -> u32;

        fn size(&self) -> u32 {
            self.num_vertices() * self.stride() as u32
        }

        /// Returns the Attribute of the attribute having the given semantic,
        /// if the vertex buffer contains such attribute
        fn attribute(&self, semantic: Semantic) -> Option<&Attribute> {
           self.layout().0.iter().find(|attr| attr.semantic == semantic)
        }
    }

    #[derive(Clone)]
    pub struct IndexLayout {
        pub base_type: BaseType,
        pub primitive_type: PrimitiveType,
    }

    #[derive(Clone)]
    pub enum PrimitiveType {
        Triangles,
        TriangleStrip,
        TriangleFan,
        Points,
        Lines,
        LineStrip
    }

    impl PrimitiveType {
        pub fn num_indices_for_items(&self, num_items: u32) -> u32 {
            match self {
                PrimitiveType::Triangles => 3 * num_items,
                _ => unimplemented!()
            }
        }
    }
    
    /// Trait for types storing index data for meshes
    ///
    pub trait IndexBuffer {
        fn ptr(&self) -> *const u8;

        fn base_type(&self) -> BaseType;

        fn num_indices(&self) -> u32;

        fn size(&self) -> u32 {
            self.num_indices() * self.base_type().size()
        }
    }
}

pub struct Mesh {
    buffers: Vec<Box<dyn VertexBuffer>>,
    indices: Option<Box<dyn IndexBuffer>>
}

impl Mesh {
    pub fn vertex_buffers(&self) -> &Vec<Box<dyn VertexBuffer>> {
        &self.buffers
    }

    pub fn add_vertex_buffer(&mut self, buffer: Box<dyn VertexBuffer>) {
        self.buffers.push(buffer);
    }

    pub fn add_vec_vertex_buffer<T: VertexType + 'static>(&mut self) {
        self.buffers.push(Box::new(VecVertexBuffer::<T>::with_capacity(self.num_vertices() as usize)));
    }

    pub fn attribute(&self, semantic: Semantic) -> Option<(&Attribute, &Box<dyn VertexBuffer>)> {
        self.vertex_buffers().iter().filter_map(|buffer| {
            if let Some(attribute) = buffer.attribute(semantic) {
                Some((attribute, buffer))
            } else {
                None
            }
        }).nth(0)
    }

    pub fn num_vertices(&self) -> u32 {
        self.buffers[0].num_vertices()
    }

    pub fn iter_over<'a, T>(&'a self, semantic: Semantic) -> Option<AttributeIter<T>> {
        if let Some((attribute, buffer)) = self.attribute(semantic) {
            unsafe {
                let ptr = buffer.ptr().offset(attribute.offset as isize);
                let stride = buffer.stride() as usize;
                let size = stride as u32 * self.num_vertices();
                Some(AttributeIter { next: &*ptr, end: ptr.offset(size as isize), stride, _phantom: PhantomData::<T> })
            }
        } else {
            None
        }
    }

    pub fn positions<'a>(&'a self) -> Option<Point3Iter> {
        self.iter_over::<Point3>(Semantic::Position)
    }

    pub fn normals<'a>(&'a self) -> Option<Vector3Iter> {
        self.iter_over::<Vector3>(Semantic::Normal)
    }

    pub fn index_buffer(&self) -> Option<&Box<dyn IndexBuffer>> {
        self.indices.as_ref()
    }

    pub fn num_indices(&self) -> u32 {
        self.indices.as_ref().map_or(0, |ib| ib.num_indices())
    }

    //pub fn primitive_type(&sel) -> PrimitiveType;
}

/// Trait for integral types (u8, u16, u32) used in index buffers
trait IntegralType {
    const BASE_TYPE: BaseType;
}

impl IntegralType for u8 { const BASE_TYPE: BaseType = BaseType::UnsignedByte; }
impl IntegralType for u16 { const BASE_TYPE: BaseType = BaseType::UnsignedShort; }
impl IntegralType for u32 { const BASE_TYPE: BaseType = BaseType::UnsignedInt; }

pub struct AttributeIter<'a, T> {
    next: &'a u8,
    end: *const u8,
    stride: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: 'a> Iterator for AttributeIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let mut ret = None;
        if (self.next as *const u8) < self.end {
            unsafe {
                (ret, self.next) = (Some(&*mem::transmute::<&u8, &T>(self.next)), &*(self.next as *const u8).offset(self.stride as isize));
            }
        }
        ret
    }
}

pub type Point3Iter<'a> = AttributeIter<'a, Point3>;
pub type Vector3Iter<'a> = AttributeIter<'a, Vector3>;

/// Trait for types that can be stored in a Vertex buffer
/// It needs to be able to describe its layout as a VertexLayout
pub trait VertexType {
    fn layout() -> &'static [Attribute];
}

/// VertexBuffer that owns its data. The attributes are stored into a vector.
struct VecVertexBuffer<T: VertexType> {
    data: Vec<T>,
    layout: VertexLayout,
}

impl<T: VertexType> VecVertexBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        VecVertexBuffer::<T> { layout: VertexLayout(T::layout().to_vec()), data: Vec::with_capacity(capacity) }
    }
}

impl<T: VertexType> VertexBuffer for VecVertexBuffer<T> {
    fn stride(&self) -> u16 {
        size_of::<T>().try_into().unwrap()
    }

    fn ptr(&self) -> *const u8 {
        unsafe { mem::transmute(self.data.as_ptr()) }
    }
    
    fn layout(&self) -> &VertexLayout {
        &self.layout
    }

    fn num_vertices(&self) -> u32 {
        self.data.len() as u32
    }
}

struct VecTriBuffer<T: IntegralType> {
    data: Vec<[T; 3]>
}

impl<T: IntegralType> VecTriBuffer<T> {
    fn with_capacity(capacity: usize) -> VecTriBuffer<T> {
        VecTriBuffer { data: Vec::with_capacity(capacity) }
    }
}

impl<T: IntegralType> IndexBuffer for VecTriBuffer<T> {
    fn ptr(&self) -> *const u8 {
        unsafe { mem::transmute(self.data.as_ptr()) }
    }

    fn base_type(&self) -> BaseType {
        T::BASE_TYPE
    }

    fn num_indices(&self) -> u32 {
        3 * self.data.len() as u32
    }
}

/// VertexBuffer that does not own its data
struct SliceVertexBuffer<'a> {
    data: &'a [u8],
    layout: VertexLayout,
    stride: u16,
    num_vertices: u32,
}

impl<'a>  SliceVertexBuffer<'a> {
    pub fn new(data: &'a [u8], layout: &VertexLayout, stride: u16, num_vertices: u32) -> Self {
        SliceVertexBuffer { data, layout: layout.clone(), stride, num_vertices }
    }
}

impl<'a> VertexBuffer for SliceVertexBuffer<'a> {
    fn stride(&self) -> u16 {
        self.stride
    }

    fn ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn layout(&self) -> &VertexLayout {
        &self.layout
    }

    fn num_vertices(&self) -> u32 {
        self.num_vertices
    }
}




#[derive(Copy, Clone)]
struct PositionNormal {
    position: Point3,
    normal: Vector3,
}

impl VertexType for PositionNormal {
    fn layout() -> &'static [Attribute] {
        static mut LAYOUT: [Attribute; 2] = [
            Attribute { semantic: Semantic::Position, offset: 0, dimension: 3, base_type: BaseType::Float, normalized: false },
            Attribute { semantic: Semantic::Normal, offset: 0, dimension: 3, base_type: BaseType::Float, normalized: false },
        ];
        // we must resort to this contorsions because offset_of! is not callable in static context
        unsafe {
            LAYOUT[0].offset = offset_of!(PositionNormal, position) as u16;
            LAYOUT[1].offset = offset_of!(PositionNormal, normal) as u16;
            &LAYOUT
        }
    }
}

macro_rules! impl_icosphere {
    ($type:ty, $fn_name:ident, $max_subdiv:expr) => { 
        pub fn $fn_name(radius: f32, num_subdivisions: u8) -> Mesh {
            assert!(num_subdivisions <= $max_subdiv);

            use std::f32::consts::PI;

            // TODO compute capacity (vertices only)
            let mut vertices = VecVertexBuffer::<PositionNormal>::with_capacity(0);
            let mut indices = VecTriBuffer::<$type>::with_capacity(0);

            const LONG_ANGLE: f32 = 2.0 * PI / 5.0;
            const LAT_ANGLE: f32 = PI / 3.0;
            // Base dodecahedron has 22 vertices instead of 12 because poles and one "meridian" are split (XXX for future texture coordinates)
            // north pole (5 times — UVs are split)
            let north_pole = PositionNormal { position: point![0.0, 0.0, radius], normal: vector![0.0, 0.0, 1.0] };
            vertices.data.extend_from_slice(&[north_pole; 5]);
            // 2 middle layers 
            for i in 0..2 {
                let theta = (i + 1) as f32 * LAT_ANGLE;
                let (stheta, ctheta) = theta.sin_cos();
                for j in 0..=5 {
                    let phi = (j as f32) * LONG_ANGLE + i as f32 * LONG_ANGLE / 2.0;
                    let normal = vector![stheta * phi.cos(), stheta * phi.sin(), ctheta];
                    vertices.data.push(PositionNormal { position: (radius * normal).into(), normal });
                }
            }
            // south pole (5 times)
            let south_pole = PositionNormal { position: point![0.0, 0.0, -radius], normal: vector![0.0, 0.0, -1.0] };
            vertices.data.extend_from_slice(&[south_pole; 5]);

            indices.data = vec![
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
                #[derive(Eq, PartialEq, Hash)]
                struct Edge($type, $type);
                
                // order agnostic edge from tuple
                impl<'a> From<($type, $type)> for Edge {
                    fn from(value: ($type, $type)) -> Edge {
                        if value.0 > value.1 {
                            Edge(value.1, value.0)
                        } else {
                            Edge(value.0, value.1)
                        }
                    }
                }
                let mut split_edges = HashMap::<Edge, $type>::new();
                let mut new_triangles = Vec::new();
                while let Some(tri_indices) = indices.data.pop() {
                    let mut new_indices = [0 as $type; 3];

                    for i in 0..3 {
                        let edge = (tri_indices[i], tri_indices[(i + 1).rem(3)]);
                        new_indices[(i + 2).rem(3)] = *split_edges.entry(edge.into())
                            .or_insert_with(|| {
                                let v0 = vertices.data[edge.0 as usize].position;
                                let v1 = vertices.data[edge.1 as usize].position;
                                let normal = (0.5 * (v0.coords + v1.coords)).normalize();
                                let position = Point3::from(radius * normal);
                                vertices.data.push(PositionNormal { position, normal });
                                vertices.data.len() as $type - 1
                            });
                    }
                    new_triangles.push([tri_indices[0], new_indices[2], new_indices[1]]);
                    new_triangles.push([new_indices[2], tri_indices[1], new_indices[0]]);
                    new_triangles.push([new_indices[2], new_indices[0], new_indices[1]]);
                    new_triangles.push([new_indices[1], new_indices[0], tri_indices[2]]);
                }
                indices.data = new_triangles;

                split_edges.clear();
            }

            println!("Sphere has {} vertices / {} indices", vertices.data.len(), 3 * indices.data.len());

            Mesh { buffers: vec![Box::new(vertices)], indices: Some(Box::new(indices)) }
        }
    };
}

pub mod geometries {
    use super::PositionNormal;
    use crate::mesh::*;

    impl_icosphere!(u8, icosphere_coarse, 2);
    impl_icosphere!(u16, icosphere_medium, 6);
    impl_icosphere!(u32, icosphere_fine, 10);
}

mod gltf_helpers {
    use super::attribute::*;
    use super::vertex_buffer::*;

    use std::collections::HashMap;

    use gltf::Gltf;
    use gltf::buffer::View;
    use gltf::mesh::Mode;
    use gltf::accessor::{DataType, Dimensions};

    impl From<gltf::Semantic> for Semantic {
        fn from(value: gltf::Semantic) -> Self {
            match value {
                Positions => Semantic::Position,
                _ => unimplemented!()
            }
            // Normals,
            // Tangents,
            // Colors(u32),
            // TexCoords(u32),
            // Joints(u32),
            // Weights(u32),
        }
    }
    
    impl From<DataType> for BaseType {
        fn from(value: DataType) -> Self {
            match value {
                DataType::I8 => BaseType::Byte,
                DataType::U8 => BaseType::UnsignedByte,
                DataType::I16 => BaseType::Short,
                DataType::U16 => BaseType::UnsignedShort,
                DataType::U32 => BaseType::Int,
                DataType::F32 => BaseType::UnsignedInt
            }
        }
    }
    
    
    fn dimensions_val(dimensions: Dimensions) -> u8 {
        match dimensions {
            Dimensions::Scalar => 1,
            Dimensions::Vec2 => 2,
            Dimensions::Vec3 => 3,
            Dimensions::Vec4 => 4,
            Dimensions::Mat2 => 4,
            Dimensions::Mat3 => 9,
            Dimensions::Mat4 => 16,
        }
    }

    impl From<gltf::Accessor::<'_>> for Attribute {
        fn from(value: gltf::Accessor) -> Self {
            Attribute {
                semantic: Semantic::Generic(0),
                dimension: dimensions_val(value.dimensions()),
                base_type: (value.data_type() as DataType).into(),
                normalized: value.normalized(),
                offset: value.offset() as u16,
            }
        }
    }

    fn primitive_layouts<'a>(doc: &'a Gltf, primitive: &'a gltf::Primitive) -> Vec<(usize, VertexLayout)> {
        let mut layouts = HashMap::new();
        for attribute in primitive.attributes() {
            // skip sparse attributes for now
            if let Some(view) = attribute.1.view() {
                let semantic = attribute.0.into();
                let attr = Attribute { semantic, ..attribute.1.into() }; 
                layouts.entry(view.index()).or_insert(VertexLayout(Vec::new())).attributes_mut().push(attr);
            }
        }

        layouts.iter().map(|(view_id, layout)| (*view_id, layout.clone())).collect()
    }

    // struct GltfData {
    //     doc: gltf::Gltf,
    //     buffers: Vec<Vec<u8>>
    // }

    // impl GltfData {
    //     fn from_file(filepath: &std::fs::Path) -> GltfData {
    //         unimplemented!()
    //     }

    //     fn gltf(&self) -> &gltf::Gltf {
    //         unimplemented!()
    //     }

    //     fn buffer_ptr(&self, id: usize) -> Option<&[u8]> {
    //         unimplemented!()
    //     }
    // }
}
