#![allow(dead_code)]

use gl::types::*;

use nalgebra::RawStorage;

use crate::mesh::*;
use crate::shader::VertexShaderInput;

pub use crate::mesh::attribute::*;
use crate::mesh::vertex_buffer::*;

impl Into<GLenum> for BaseType {
    fn into(self) -> GLenum {
        match self {
            BaseType::HalfFloat=> gl::HALF_FLOAT,
            BaseType::Float => gl::FLOAT,
            BaseType::Double => gl::DOUBLE,
            BaseType::Byte => gl::BYTE,
            BaseType::UnsignedByte => gl::UNSIGNED_BYTE,
            BaseType::Short => gl::SHORT,
            BaseType::UnsignedShort => gl::UNSIGNED_SHORT,
            BaseType::Int => gl::INT,
            BaseType::UnsignedInt => gl::UNSIGNED_INT,
        }
    }
}

#[derive(Debug)]
pub struct BufferView {
    pub buffer_id : u32,
    pub offset: u32
}

pub struct GPUVertexBuffer {
    pub buffer_view: BufferView,
    pub layout: VertexLayout
}

pub struct GPUIndexBuffer {
    pub buffer_view: BufferView,
    pub layout: IndexLayout
}

pub struct DrawData {
    pub vertex_buffers: Vec<GPUVertexBuffer>,
    pub index_buffer: Option<GPUIndexBuffer>,
    pub num_vertices: u32,
    pub num_indices: u32,
}

impl DrawData { 
    pub fn new(vertex_layouts: &Vec<VertexLayout>, num_vertices: u32, index_layout: Option<IndexLayout>, num_indices: u32) -> DrawData {
        let mut vertex_buffers = Vec::with_capacity(vertex_layouts.len());
        for layout in vertex_layouts {
            let mut buffer = BufferView { buffer_id: 0, offset: 0 };
            unsafe {
                gl::CreateBuffers(1, &mut buffer.buffer_id);
                gl::NamedBufferStorage(buffer.buffer_id, (layout.size() as u32 * num_vertices) as isize, std::ptr::null(), gl::MAP_WRITE_BIT);
            }
            vertex_buffers.push(GPUVertexBuffer { buffer_view: buffer, layout: layout.clone() });
        }

        let mut index_buffer = None;
        if let Some(layout) = index_layout {
            let mut buffer = BufferView { buffer_id: 0, offset: 0 };
            unsafe {
                gl::CreateBuffers(1, &mut buffer.buffer_id);
                gl::NamedBufferStorage(buffer.buffer_id, (layout.base_type.size() * num_indices) as isize , std::ptr::null(), gl::MAP_WRITE_BIT | gl::DYNAMIC_STORAGE_BIT);
            }
            index_buffer = Some(GPUIndexBuffer { buffer_view: buffer, layout: layout.clone() });
        }

        DrawData { vertex_buffers, index_buffer, num_vertices, num_indices }
    }

    pub fn with_mesh(mesh: &Mesh) -> Self {
        let draw_data = DrawData::new(
            &mesh.vertex_buffers().iter().map(|buffer| buffer.layout().clone()).collect(),
            mesh.num_vertices(),
            mesh.index_buffer().and_then(|buffer| Some(IndexLayout { base_type: buffer.base_type(), primitive_type: PrimitiveType::Triangles })),
            mesh.num_indices()
        );

        unsafe {
            for (gpu_buffer, cpu_buffer) in draw_data.vertex_buffers.iter().zip(mesh.vertex_buffers().iter()) {
                let size = (gpu_buffer.layout.size() as u32 * mesh.num_vertices()) as isize;
                let dst = gl::MapNamedBufferRange(gpu_buffer.buffer_view.buffer_id, gpu_buffer.buffer_view.offset as isize, size, gl::MAP_WRITE_BIT);
                let src = cpu_buffer.ptr();
                src.copy_to_nonoverlapping(dst.cast(), size as usize);
                gl::UnmapNamedBuffer(gpu_buffer.buffer_view.buffer_id);
            }

            if let Some(gpu_buffer) = &draw_data.index_buffer {
                let indices = mesh.index_buffer().unwrap();
                let size = (gpu_buffer.layout.base_type.size() as u32 * mesh.num_indices()) as isize;
                let dst = gl::MapNamedBufferRange(gpu_buffer.buffer_view.buffer_id, gpu_buffer.buffer_view.offset as isize, size, gl::MAP_WRITE_BIT);
                let src = indices.ptr();
                src.copy_to_nonoverlapping(dst.cast(), size as usize);
                gl::UnmapNamedBuffer(gpu_buffer.buffer_view.buffer_id);
            }
        }
        draw_data
    }

    /// Fill the draw data buffers with corresponding data read from the mesh,
    /// performing the necessary conversions.
    pub fn fill_with_mesh(&mut self, mesh: &Mesh) {
        unsafe {
            for buffer in &self.vertex_buffers {
                let stride = buffer.layout.size() as u32;
                let position = buffer.layout.attribute(Semantic::Position);
                let normal = buffer.layout.attribute(Semantic::Normal);
                
                let buffer_addr = if position.is_some() || normal.is_some() {
                    gl::MapNamedBufferRange(buffer.buffer_view.buffer_id, buffer.buffer_view.offset as isize, (stride * mesh.num_vertices()) as isize, gl::MAP_WRITE_BIT)
                } else {
                    std::ptr::null()
                };
                
                if let Some(position) = position {
                    let mut dst = buffer_addr.add(position.offset as usize);
                    for pos in mesh.positions().unwrap() {
                        // TODO type conversion !
                        let src = pos.coords.data.get_address_unchecked_linear(0);
                        src.copy_to_nonoverlapping(dst as *mut f32, 3);
                        dst = dst.add(stride as usize);
                    }
                }

                if let Some(normal) = normal {
                    let mut dst = buffer_addr.add(normal.offset as usize);
                    for normal in mesh.normals().unwrap() {
                        // TODO type conversion !
                        let src = normal.data.get_address_unchecked_linear(0);
                        src.copy_to_nonoverlapping(dst as *mut f32, 3);
                        dst = dst.add(stride as usize);
                    }
                }

                if buffer_addr != std::ptr::null() {
                    gl::UnmapNamedBuffer(buffer.buffer_view.buffer_id);
                }
            }
            
            self.num_vertices = mesh.num_vertices();

            if let Some(index_buffer) = &self.index_buffer {
                let indices = mesh.index_buffer().unwrap();
                let offset = index_buffer.buffer_view.offset as isize;
                let size = indices.size() as isize;
                let src =  indices.ptr() as *const GLvoid;
                gl::NamedBufferSubData(index_buffer.buffer_view.buffer_id, offset, size, src);

                self.num_indices = mesh.num_indices();
            }
        }
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
            for buffer in draw_data.vertex_buffers.iter() {
                for attribute in buffer.layout.attributes() {
                    let matching_input = program_inputs.iter().find(|&input| input.semantic == attribute.semantic);
                    if let Some(input) = matching_input {
                        gl::EnableVertexArrayAttrib(self.vao, input.location);
                        gl::VertexArrayAttribBinding(self.vao, input.location, binding);
                        gl::VertexArrayVertexBuffer(self.vao, binding, buffer.buffer_view.buffer_id, buffer.buffer_view.offset as isize, buffer.layout.size() as i32);
                        match input.base_type {
                            gl::FLOAT | gl::FLOAT_VEC2 | gl::FLOAT_VEC3 | gl::FLOAT_VEC4 => {
                                gl::VertexArrayAttribFormat(self.vao, input.location, attribute.dimension as i32, attribute.base_type.into(), attribute.normalized as GLboolean, attribute.offset as u32);
                            },
                            gl::INT | gl::INT_VEC2 | gl::INT_VEC3 | gl::INT_VEC4 |
                            gl::UNSIGNED_INT | gl::UNSIGNED_INT_VEC2 | gl::UNSIGNED_INT_VEC3 | gl::UNSIGNED_INT_VEC4 => {
                                gl::VertexArrayAttribIFormat(self.vao, input.location, attribute.dimension as i32, attribute.base_type.into(), attribute.offset as u32);
                            },
                            gl::DOUBLE | gl::DOUBLE_VEC2 | gl::DOUBLE_VEC3 | gl::DOUBLE_VEC4 => {
                                gl::VertexArrayAttribLFormat(self.vao, input.location, attribute.dimension as i32, attribute.base_type.into(), attribute.offset as u32);
                            }
                            _ => ()
                        }

                        binding += 1;
                    } 
                }
            }

            if let Some(ib) = &draw_data.index_buffer {
                gl::VertexArrayElementBuffer(self.vao, ib.buffer_view.buffer_id);
            }
        }
    }

    pub fn bind(&self, draw_data: &DrawData) {
        unsafe {
            let mut binding = 0;
            for buffer in draw_data.vertex_buffers.iter() {
                gl::VertexArrayVertexBuffer(self.vao, binding, buffer.buffer_view.buffer_id, buffer.buffer_view.offset as isize, buffer.layout.size() as i32);
                binding += 1;
            }

            if let Some(ib) = &draw_data.index_buffer {
                gl::VertexArrayElementBuffer(self.vao, ib.buffer_view.buffer_id);
            }}
    }

    pub fn activate(&self) {
        unsafe {
            gl::BindVertexArray(self.vao);
        }
    }
}

