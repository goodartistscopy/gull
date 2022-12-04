#![allow(dead_code)]

use gl::types::*;

use nalgebra::RawStorage;

use crate::mesh::*;
use crate::shader::VertexShaderInput;

#[derive(Copy,Clone,PartialEq,Debug)]
pub enum AttributeSemantic {
    Position,
    Normal,
    Tangent,
    Bitangent,
    Color(u8),
    TexCoord(u8),
    Generic(u8),
    Padding
}

#[derive(Copy,Clone,Debug)]
pub enum AttributeType {
    Float16,
    Float32,
    Float64,
    Int8,
    UInt8,
    Int16,
    UInt16,
    /* Padcked formats
     UInt32_A2B10G10R10,
    UInt32_A2B10G10R10,
    UInt32_R10FG11FR11F,*/
}

impl AttributeType {
    fn size(&self) -> u32 {
        match self {
            AttributeType::Int8 | AttributeType::UInt8 => 1,
            AttributeType::Int16 | AttributeType::UInt16 => 2,
            AttributeType::Float16 => 2,
            AttributeType::Float32 => 4,
            AttributeType::Float64 => 8,
        }
    }
}

impl Into<GLenum> for AttributeType {
    fn into(self) -> GLenum {
        match self {
            AttributeType::Float16=> gl::HALF_FLOAT,
            AttributeType::Float32 => gl::FLOAT,
            AttributeType::Float64 => gl::DOUBLE,
            AttributeType::Int8 => gl::BYTE,
            AttributeType::UInt8 => gl::UNSIGNED_BYTE,
            AttributeType::Int16 => gl::SHORT,
            AttributeType::UInt16 => gl::UNSIGNED_SHORT,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Attribute {
    pub semantic: AttributeSemantic,
    pub base_type: AttributeType,
    pub len: u8,
    pub normalized: bool
}

impl Attribute {
    fn to_gl_type(&self) -> GLenum {
        self.base_type.into()
    }

    // size in bytes
    fn size(&self) -> u32 {
        match self.semantic {
            AttributeSemantic::Padding => self.len.into(),
            _ => self.base_type.size() * self.len as u32
        }
    }
}

#[derive(Debug)]
pub struct BufferView {
    pub buffer_id : u32,
    pub offset: u32
}

#[derive(Clone, Debug)]
pub struct VertexLayout {
    pub attributes: Vec<Attribute>
}

impl VertexLayout {
    pub fn size(&self) -> u32 {
        let mut size = 0;
        for attribute in &self.attributes {
            size += attribute.size();
        }
        size
    }

    fn contains(&self, any_semantics: &[AttributeSemantic]) -> bool {
        for attribute in &self.attributes {
            if any_semantics.contains(&attribute.semantic) {
                return true;
            }
        }
        false
    }

    pub fn offset(&self, semantic: AttributeSemantic) -> Option<u32> {
        let mut offset = 0;
        for attribute in &self.attributes {
            if attribute.semantic == semantic {
                return Some(offset)
            }
            offset += attribute.size();
        }
        None
    }

    pub fn attribute<'a>(&'a self, semantic: AttributeSemantic) -> Option<(u32, &'a Attribute)> {
        let mut offset = 0;
        for attribute in &self.attributes {
            if attribute.semantic == semantic {
                return Some((offset, attribute));
            }
            offset += attribute.size();
        }
        None
    }
}


pub struct DrawData {
    pub vertex_buffers: Vec<BufferView>,
    pub layouts: Vec<VertexLayout>, // FIXME not the right place or just reference ?
    pub index_buffer: Option<BufferView>,
    pub num_vertices: i32,
    pub num_elems: i32,
}

pub fn allocate_vertex_buffers(stream_layouts: &Vec<VertexLayout>, num_vertices: u32) -> Vec<BufferView> {
    let mut buffers = Vec::<BufferView>::with_capacity(stream_layouts.len());
    for layout in stream_layouts {
        let mut buffer = BufferView { buffer_id: 0, offset: 0 };
        unsafe {
            gl::CreateBuffers(1, &mut buffer.buffer_id);
            gl::NamedBufferStorage(buffer.buffer_id, (layout.size() * num_vertices) as isize, std::ptr::null(), gl::MAP_WRITE_BIT);
            // #[cfg(debug_assertions)]
            // gl::ObjectLabel(gl::BUFFER, buffer.buffer_id, 8, "VBuffer".as_ptr() as *const i8);
        }
        buffers.push(buffer);
    }

    buffers
}

pub fn allocate_index_buffer(num_triangles: u32) -> BufferView {
    let mut buffer = BufferView { buffer_id: 0, offset: 0 };
    unsafe {
        gl::CreateBuffers(1, &mut buffer.buffer_id);
        let size = (num_triangles * std::mem::size_of::<[u32; 3]>() as u32) as isize;
        gl::NamedBufferStorage(buffer.buffer_id, size, std::ptr::null(), gl::MAP_WRITE_BIT | gl::DYNAMIC_STORAGE_BIT);
    }
    buffer
}

pub fn fill_with_mesh_data(draw_data: &mut DrawData, mesh: &Mesh) {
    unsafe {
        for (layout, buffer) in draw_data.layouts.iter().zip(draw_data.vertex_buffers.iter()) {
            let stride = layout.size() as usize;
            let position_offset = layout.offset(AttributeSemantic::Position);
            let normal_offset = layout.offset(AttributeSemantic::Normal);
            
            let buffer_addr = if position_offset.is_some() || normal_offset.is_some() {
                gl::MapNamedBufferRange(buffer.buffer_id, buffer.offset as isize, (layout.size() * mesh.num_vertices()) as isize, gl::MAP_WRITE_BIT)
            } else {
                std::ptr::null()
            };
            
            if let Some(rel_offset) = position_offset {
                let mut dst = buffer_addr.add(rel_offset as usize);
                for pos in &mesh.positions {
                    // TODO type conversion !
                    let src = pos.coords.data.get_address_unchecked_linear(0);
                    src.copy_to_nonoverlapping(dst as *mut f32, 3);
                    dst = dst.add(stride);
                }
            }

            if let Some(rel_offset) = normal_offset {
                let mut dst = buffer_addr.add(rel_offset as usize);
                for normal in &mesh.normals {
                    // TODO type conversion !
                    let src = normal.data.get_address_unchecked_linear(0);
                    src.copy_to_nonoverlapping(dst as *mut f32, 3);
                    dst = dst.add(stride);
                }
            }

            if buffer_addr != std::ptr::null() {
                gl::UnmapNamedBuffer(buffer.buffer_id);
            }
        }
        
        draw_data.num_vertices = mesh.num_vertices() as i32;

        if let Some(index_buffer) = &draw_data.index_buffer {
            let offset = index_buffer.offset as isize;
            let size = (mesh.num_triangles() * std::mem::size_of::<[u32; 3]>() as u32) as isize;
            let src = mesh.indices.as_ptr() as *const GLvoid;
            gl::NamedBufferSubData(index_buffer.buffer_id, offset, size, src);

            draw_data.num_elems = 3 * mesh.num_triangles() as i32;
        }
    }
}

impl DrawData {
    pub fn with_mesh(stream_layouts: Vec<VertexLayout>, mesh: &Mesh) -> Self {
        let vbuffers = allocate_vertex_buffers(&stream_layouts, mesh.num_vertices());
        let ibuffer = allocate_index_buffer(mesh.num_triangles());
        let mut draw_data = DrawData { vertex_buffers: vbuffers, layouts: stream_layouts, index_buffer: Some(ibuffer), num_vertices: 0, num_elems: 0 };
        fill_with_mesh_data(&mut draw_data, mesh);
        draw_data
    }
}

#[derive(Default)]
pub struct InputAssembly {
    vao: u32
}

impl InputAssembly {
    pub fn new() -> Self {
        let mut ia = InputAssembly { vao: 0 };
        unsafe {
            gl::CreateVertexArrays(1, &mut ia.vao);
        }
        ia
    }

    pub fn configure_and_bind(&self, program_inputs: &Vec::<VertexShaderInput>, draw_data: &DrawData) {
        unsafe {
            let mut binding = 0;
            for (buffer, layout) in draw_data.vertex_buffers.iter().zip(draw_data.layouts.iter()) {
                let mut offset = 0;
                for attribute in &layout.attributes {
                    let matching_input = program_inputs.iter().find(|&input| input.semantic == attribute.semantic);
                    if let Some(input) = matching_input {
                        gl::EnableVertexArrayAttrib(self.vao, input.location);
                        gl::VertexArrayAttribBinding(self.vao, input.location, binding);
                        gl::VertexArrayVertexBuffer(self.vao, binding, buffer.buffer_id, buffer.offset as isize, layout.size() as i32);
                        match input.base_type {
                            gl::FLOAT | gl::FLOAT_VEC2 | gl::FLOAT_VEC3 | gl::FLOAT_VEC4 => {
                                gl::VertexArrayAttribFormat(self.vao, input.location, attribute.len as i32, attribute.to_gl_type(), attribute.normalized as GLboolean, offset);
                            },
                            gl::INT | gl::INT_VEC2 | gl::INT_VEC3 | gl::INT_VEC4 |
                            gl::UNSIGNED_INT | gl::UNSIGNED_INT_VEC2 | gl::UNSIGNED_INT_VEC3 | gl::UNSIGNED_INT_VEC4 => {
                                gl::VertexArrayAttribIFormat(self.vao, input.location, attribute.len as i32, attribute.to_gl_type(), offset);
                            },
                            gl::DOUBLE | gl::DOUBLE_VEC2 | gl::DOUBLE_VEC3 | gl::DOUBLE_VEC4 => {
                                gl::VertexArrayAttribLFormat(self.vao, input.location, attribute.len as i32, attribute.to_gl_type(), offset);
                            }
                            _ => ()
                        }

                        binding += 1;
                    } 

                    offset += attribute.size();
                }
            }

            if let Some(ib) = &draw_data.index_buffer {
                gl::VertexArrayElementBuffer(self.vao, ib.buffer_id);
            }
        }
    }

    pub fn bind(&self, draw_data: &DrawData) {
        unsafe {
            let mut binding = 0;
            for (buffer, layout) in draw_data.vertex_buffers.iter().zip(draw_data.layouts.iter()) {
                gl::VertexArrayVertexBuffer(self.vao, binding, buffer.buffer_id, buffer.offset as isize, layout.size() as i32);
                binding += 1;
            }

            if let Some(ib) = &draw_data.index_buffer {
                gl::VertexArrayElementBuffer(self.vao, ib.buffer_id);
            }}
    }

    pub fn activate(&self) {
        unsafe {
            gl::BindVertexArray(self.vao);
        }
    }
}

