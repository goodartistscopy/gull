#![allow(dead_code)]
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
use std::slice::from_raw_parts;

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
        pub fn size(&self) -> usize {
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
        pub fn size(&self) -> usize {
            self.dimension as usize * self.base_type.size()
        }
    }

}

pub mod vertex_buffer {
    use super::attribute::*;

    #[derive(Clone)]
    pub struct VertexLayout(pub(super) Vec<Attribute>);

    /// VertexLayout is the layout of a row in a vertex_buffer
    /// Attribute are sorted by offset
    impl VertexLayout {
        pub fn attributes(&self) -> &Vec<Attribute> {
            &self.0
        }

        pub fn attributes_mut(&mut self) -> &mut Vec<Attribute> {
            &mut self.0
        }

        ///
        pub fn size(&self) -> u16 {
            self.0.last().map(|attr| attr.offset + attr.size() as u16).unwrap_or(0)
        }
    
        /// Returns the Attribute having the given semantic, if it exists in the vertex buffer
        pub fn attribute(&self, semantic: Semantic) -> Option<&Attribute> {
           self.0.iter().find(|attr| attr.semantic == semantic)
        }

        /// 
        pub fn overlaps(&self, other: &VertexLayout) -> bool {
            for attrib_2 in other.0.iter() {
                if self.attribute(attrib_2.semantic).is_some() {
                    return true;
                }
            }
            return false;
        }
    }

    /// Builder for VertexLayout, taking care of attribute semantic unicity and 
    /// computing offsets
    pub struct VertexLayoutBuilder {
        layout: Vec<Attribute>,
        size: usize,
    }

    impl VertexLayoutBuilder  {
        pub fn new() -> VertexLayoutBuilder {
            VertexLayoutBuilder { layout: Vec::new(), size: 0 }
        }

        pub fn pad(mut self, padding: usize) -> Self {
            self.size += padding;
            self
        }

        // Does nothing if the attribute's semantic is already present in the builder
        pub fn add(mut self, mut attribute: Attribute) -> Self {
            if self.layout.iter().find(|attr| attr.semantic == attribute.semantic).is_some() {
                println!("Ignoring duplicate attribute ({:?})", attribute.semantic);
                return self;
            }
            attribute.offset = self.size as u16; 
            self.size += attribute.size();
            self.layout.push(attribute);
            self
        }

        pub fn build(self) -> VertexLayout {
            VertexLayout(self.layout)
        }
    }

    pub struct VertexBuffer {
        data: Vec<u8>,
        stride: usize,
        layout: VertexLayout,
    }

    impl VertexBuffer {
        ///
        pub fn new(num_vertices: u32, layout: VertexLayout) -> VertexBuffer {
            let size = (num_vertices * layout.size() as u32) as usize;
            let mut data = Vec::with_capacity(size);
            data.resize(size, 0);
            VertexBuffer { data, stride: 0, layout } 
        }

        pub fn with_buffer(size: usize, stride: usize) -> VertexBuffer {
            assert_eq!(size % stride, 0, "size must be a multiple of stride");

            let mut data = Vec::with_capacity(size);
            data.resize(size, 0);
            VertexBuffer { data, stride, layout: VertexLayout(Vec::new()) }
        }

        pub fn data(&self) -> &[u8] {
            &self.data
        }

        pub fn layout(&self) -> &VertexLayout {
            &self.layout
        }

        pub fn stride(&self) -> usize {
            self.stride
        }

        pub fn actual_stride(&self) -> usize {
            if self.stride != 0 { self.stride } else { self.layout.size().into() }
        }

        pub fn num_vertices(&self) -> u32 {
            (self.data.len() / self.actual_stride()) as u32
        }

        /// Returns the Attribute of the attribute having the given semantic,
        /// if the vertex buffer contains such attribute
        pub fn attribute(&self, semantic: Semantic) -> Option<&Attribute> {
           self.layout().0.iter().find(|attr| attr.semantic == semantic)
        }
    }
/*
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
    */
}

#[cfg(test)]
mod tests {
    use super::vertex_buffer::*;
    use super::attribute::*;

    fn float3_attribute(semantic: Semantic) -> Attribute {
        Attribute { semantic, dimension: 3, base_type: BaseType::Float, normalized: false, offset: 0 }
    }

    #[test]
    fn build_vertex_layout() {
        let layout = VertexLayoutBuilder::new().build();
        assert_eq!(layout.size(), 0);

        let layout = VertexLayoutBuilder::new()
            .add(float3_attribute(Semantic::Position))
            .add(float3_attribute(Semantic::Normal))
            .build();
        assert_eq!(layout.size(), 24);
        let offset: Vec<u16> = layout.0.iter().map(|attr| attr.offset).collect();
        assert_eq!(offset, [0, 12]);
    
        let layout = VertexLayoutBuilder::new()
            .add(float3_attribute(Semantic::Position))
            .pad(4)
            .add(float3_attribute(Semantic::Normal))
            .build();
        assert_eq!(layout.size(), 28);
        let offset: Vec<u16> = layout.0.iter().map(|attr| attr.offset).collect();
        assert_eq!(offset, [0, 16]);

        // padding at the end is ignored
        let layout = VertexLayoutBuilder::new()
            .add(float3_attribute(Semantic::Position))
            .add(float3_attribute(Semantic::Normal))
            .pad(4)
            .build();
        assert_eq!(layout.size(), 24);
        let offset: Vec<u16> = layout.0.iter().map(|attr| attr.offset).collect();
        assert_eq!(offset, [0, 12]);

        // second Position is ignored
        let layout = VertexLayoutBuilder::new()
            .add(float3_attribute(Semantic::Position))
            .add(float3_attribute(Semantic::Normal))
            .add(float3_attribute(Semantic::Position))
            .add(float3_attribute(Semantic::Generic(0)))
            .build();
        
        assert_eq!(layout.size(), 36);
        let offset: Vec<u16> = layout.0.iter().map(|attr| attr.offset).collect();
        assert_eq!(offset, [0, 12, 24]);
    }

    #[test]
    fn vertex_layout_overlap() {
        let layout = VertexLayoutBuilder::new()
            .add(float3_attribute(Semantic::Position))
            .pad(4)
            .add(float3_attribute(Semantic::Normal))
            .build();

        let layout2 = VertexLayoutBuilder::new()
            .add(float3_attribute(Semantic::Position))
            .build();
        assert!(layout2.overlaps(&layout));

        let layout2 = VertexLayoutBuilder::new()
            .add(float3_attribute(Semantic::TexCoord(0)))
            .build();
        assert!(!layout2.overlaps(&layout));
    }

    #[test]
    fn build_vertex_buffer() {
        let vb = VertexBuffer::with_buffer(120, 12);

        assert_eq!(vb.stride(), 12);
        assert_eq!(vb.num_vertices(), 10);

        let layout = VertexLayoutBuilder::new()
            .add(float3_attribute(Semantic::Position))
            .pad(4)
            .add(float3_attribute(Semantic::Normal))
            .build();
        let vb = VertexBuffer::new(5, layout.clone());

        assert_eq!(vb.stride(), 0);
        assert_eq!(vb.num_vertices(), 5);
        assert_eq!(vb.actual_stride(), layout.size() as usize);
    }

    #[test]
    #[should_panic]
    fn build_vertex_buffer_wrong() {
        let _vb = VertexBuffer::with_buffer(120, 16);
    }

    #[test]
    fn get_vertex_buffer_attribute() {
        let layout = VertexLayoutBuilder::new()
            .add(float3_attribute(Semantic::Position))
            .add(float3_attribute(Semantic::Normal))
            .pad(1)
            .add(Attribute { semantic: Semantic::Generic(0), dimension: 2, base_type: BaseType::Float, normalized: false, offset: 0 })
            .build();

        let vb = VertexBuffer::new(2, layout);
        
        let normal = vb.attribute(Semantic::Normal).unwrap();
        assert_eq!(normal.offset, 12);

        let gen0 = vb.attribute(Semantic::Generic(0)).unwrap();
        assert_eq!(gen0.offset, 25);

        assert!(vb.attribute(Semantic::TexCoord(0)).is_none());
    }
}

/*

/// A Mesh is made of a vector of VertexBuffer's and a single IndexBuffer
pub struct Mesh {
    buffers: Vec<Box<dyn VertexBuffer>>,
    indices: Option<Box<dyn IndexBuffer>>
}

impl Mesh {
    pub fn new() -> Mesh {
        Mesh { buffers: Vec::new(), indices: None }
    }

    pub fn vertex_buffers(&self) -> &Vec<Box<dyn VertexBuffer>> {
        &self.buffers
    }

    pub fn add_vertex_buffer(&mut self, buffer: Box<dyn VertexBuffer>) {
        self.buffers.push(buffer);
    }

    pub fn add_vertex_buffer_of<T: VertexType + 'static>(&mut self) {
        self.buffers.push(Box::new(VertexBufferOf::<T>::with_capacity(self.num_vertices() as usize)));
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

    pub fn iter_over<T>(&self, semantic: Semantic) -> Option<AttributeIter<T>> {
        if let Some((attribute, buffer)) = self.attribute(semantic) {
            let stride = buffer.stride() as usize;
            let data = unsafe { from_raw_parts(buffer.ptr().offset(attribute.offset as isize), stride * self.num_vertices() as usize) };
            Some(AttributeIter { data, stride, _phantom: PhantomData::<T> })
        } else {
            None
        }
    }

    pub fn positions(&self) -> Option<Point3Iter> {
        self.iter_over::<Point3>(Semantic::Position)
    }

    pub fn normals(&self) -> Option<Vector3Iter> {
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
    data: &'a [u8],
    // next: &'a u8,
    // end: *const u8,
    stride: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: 'a> Iterator for AttributeIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let stride = if self.data.len() >= self.stride {
            Some(self.stride)
        } else if self.data.len() >= size_of::<T>() {
            Some(size_of::<T>())
        } else {
            None
        };

        if let Some(stride) = stride {
            let (val, data) = self.data.split_at(stride);
            self.data = data;
            unsafe { Some(&*mem::transmute::<*const u8, *const T>(val.as_ptr())) }
        } else {
            None
        }
    }
}

pub type Point3Iter<'a> = AttributeIter<'a, Point3>;
pub type Vector3Iter<'a> = AttributeIter<'a, Vector3>;

/// Trait for types that can be stored in a VertexBufferOf<T>
/// It needs to be able to describe its layout as a VertexLayout
pub trait VertexType {
    fn layout() -> &'static [Attribute];
}

/// VertexBuffer that owns its data. Individual vertices are of type VertexType.
struct VertexBufferOf<T: VertexType> {
    data: Vec<T>,
    layout: VertexLayout,
}

impl<T: VertexType> VertexBufferOf<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        VertexBufferOf::<T> { layout: VertexLayout(T::layout().to_vec()), data: Vec::with_capacity(capacity) }
    }
}

impl<T: VertexType> VertexBuffer for VertexBufferOf<T> {
    fn origin(&self) -> *const u8 {
        self.data.as_ptr() as *const u8
    }

    fn stride(&self) -> usize {
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
struct VertexBufferView<'a> {
    data: &'a [u8],
    //ptr: *const u8,
    //size: usize,
    layout: VertexLayout,
    stride: usize,
    num_vertices: u32,
 //   _phantom: PhantomData<&'a u8>
}

impl<'a> VertexBufferView<'a> {
    pub fn new(data: &'a [u8], layout: &VertexLayout, stride: usize, num_vertices: u32) -> Self {
        VertexBufferView { data, layout: layout.clone(), stride, num_vertices } //, _phantom: PhantomData}
    }

    // pub fn new2(ptr: *const u8, size: usize, layout: &VertexLayout, stride: usize, num_vertices: u32) -> Self {
    //     VertexBufferView { ptr, size, layout: layout.clone(), stride, num_vertices, _phantom: PhantomData}
    // }
}

impl<'a> VertexBuffer for VertexBufferView<'a> {
    fn origin(&self) -> *const u8 {
         self.data.as_ptr()
        //self.ptr
    }

    fn stride(&self) -> usize {
        self.stride
    }

    fn ptr(&self) -> *const u8 {
        self.data.as_ptr()
        //self.ptr
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

pub mod geometries {
    use super::PositionNormal;
    use crate::mesh::*;

    macro_rules! impl_icosphere {
        ($type:ty, $fn_name:ident, $max_subdiv:expr) => { 
            pub fn $fn_name<'a>(radius: f32, num_subdivisions: u8) -> Mesh {
                assert!(num_subdivisions <= $max_subdiv);

                use std::f32::consts::PI;

                // TODO compute capacity (vertices only)
                let mut vertices = VertexBufferOf::<PositionNormal>::with_capacity(0);
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
                    // equatorial segment
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

    impl_icosphere!(u8, icosphere_coarse, 2);
    impl_icosphere!(u16, icosphere_medium, 6);
    impl_icosphere!(u32, icosphere_fine, 10);
}

mod gltf_helpers {
    use super::attribute::*;
    use super::vertex_buffer::*;
    use super::*;

    use std::collections::HashMap;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::convert::AsRef;
    use std::fs::read;
    use std::cell::RefCell;

    use gltf::Gltf;
    use gltf::Error;
    use gltf::buffer::{View, Source};
    use gltf::mesh::Mode;
    use gltf::accessor::{DataType, Dimensions};

    impl From<gltf::Semantic> for Semantic {
        fn from(value: gltf::Semantic) -> Self {
            match value {
                gltf::Semantic::Positions => Semantic::Position,
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
    
    impl From<&gltf::Accessor::<'_>> for Attribute {
        fn from(value: &gltf::Accessor) -> Self {
            Attribute {
                semantic: Semantic::Generic(0),
                dimension: value.dimensions().multiplicity() as u8,
                base_type: (value.data_type() as DataType).into(),
                normalized: value.normalized(),
                offset: value.offset() as u16,
            }
        }
    }

    // Construct the VertexLayout for each view (along with the number of vertices). VertexLayout is built from individual attributes
    fn primitive_layouts<'a>(_doc: &'a Gltf, primitive: &'a gltf::Primitive) -> Vec<(usize, usize, VertexLayout)> {
        let mut layouts = HashMap::new();
        for attribute in primitive.attributes() {
            if let Some(view) = attribute.1.view() {
                let semantic = attribute.0.into();
                let attr = Attribute { semantic, ..(&attribute.1).into() }; 
                layouts.entry(view.index()).or_insert((attribute.1.count(), VertexLayout(Vec::new()))).1.attributes_mut().push(attr);
            } else {
                unimplemented!()
            }
        }

        layouts.iter().map(|(view_id, vb_desc)| (*view_id, vb_desc.0, vb_desc.1.clone())).collect()
    }

    struct GltfData {
        doc: gltf::Gltf,
        base_dir: PathBuf,
        buffers: RefCell<Vec<Option<Vec<u8>>>>
    }

    #[derive(Debug)]
    enum BufferError {
        IOError(io::Error),
        NoSuchBuffer,
        InvalidMeshIndex,
    }

    impl From<io::Error> for BufferError {
        fn from(err: io::Error) -> BufferError {
            BufferError::IOError(err)
        }
    }

    impl GltfData {
        fn from_file<P: AsRef<Path>>(filepath: P) -> Result<GltfData, gltf::Error> {
            let doc = gltf::Gltf::open(filepath.as_ref())?;

            let count = doc.buffers().count();
            let mut buffers = Vec::with_capacity(count);
            buffers.resize(count, None);

            let gtlf_data = GltfData { doc, base_dir: filepath.as_ref().parent().unwrap().to_path_buf(), buffers: RefCell::new(buffers) };
            
            Ok(gtlf_data)
        }

        fn gltf(&self) -> &gltf::Gltf {
            &self.doc
        }

        fn buffer_ptr(&self, index: usize) -> Result<&[u8], BufferError> {
            if index >= self.buffers.borrow().len() {
                return Err(BufferError::NoSuchBuffer)
            }

            if self.buffers.borrow()[index].is_none() {
                let source = self.doc.buffers().nth(index).ok_or(BufferError::NoSuchBuffer)?.source();
                match source {
                    Source::Bin => unimplemented!(),
                    Source::Uri(uri) => {
                        let content = read(self.base_dir.join(uri))?;
                        self.buffers.borrow_mut()[index] = Some(content);
                    }
                }
            }
            let buffer: &Option<Vec<u8>> = &self.buffers.borrow_mut()[index];
            //.unwrap();
            return Ok(buffer.as_ref().unwrap()) //self.buffers.borrow_mut()[index].as_ref().unwrap())
        }

        fn create_mesh(&mut self, index: usize) -> Result<Vec<Mesh>, BufferError> {
            let mut meshes = Vec::new();
            let gltf_mesh = self.doc.meshes().nth(0).ok_or(BufferError::NoSuchBuffer)?;
            let prim = gltf_mesh.primitives();
            for primitive in prim { //gltf_mesh.primitives() {
                let mut mesh = Mesh::new();
                let layouts = primitive_layouts(&self.doc, &primitive);
                for (view_index, num_vertices, layout) in layouts.iter() {
                    let view = self.doc.views().nth(*view_index).unwrap();
                    let ptr = self.buffer_ptr(view.buffer().index()).unwrap();
                    let offset = view.offset();
                    let length = view.length();
                    let data = &ptr[offset..offset + length];
                    let stride = view.stride().or(Some(layout.size().into())).unwrap();
                    let vb = VertexBufferView::new(data, layout, stride, *num_vertices as u32);
                    //let vb = VertexBufferView::new2(ptr, length, layout, stride, *num_vertices as u32);
                    mesh.add_vertex_buffer(Box::new(vb));
                    //mesh.vertex_buffers().push(Box::new(vb));
                    //mesh.vertex_buffers().push(Box::new(VertexBufferView { data: attrib_data, layout: layout.clone(), stride, num_vertices: *num_vertices as u32})); 
                }
                meshes.push(mesh);
            }
            Ok(meshes)
            // let ptr = self.buffer_ptr(index).unwrap();
            // let vbuffer = Box::new(VertexBufferView::new(ptr, &VertexLayout(Vec::new()), 10, 10));
            // Ok(vec![Mesh { buffers: vec![vbuffer], indices: None }])
        }
    }
}


*/

