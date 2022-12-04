use crate::vertex_layout::AttributeSemantic;
use std::{slice,str};

pub struct VertexShaderInput
{
    pub semantic: AttributeSemantic,
    pub base_type: u32,
    pub location: u32
}

#[derive(Default)]
pub struct ShaderProgram {
    pub id: u32,
    pub inputs: Vec<VertexShaderInput>
}

impl ShaderProgram {
    pub unsafe fn new(vertex_source: &str, fragment_source: &str) -> Self {
        let vs = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(vs, 1, &vertex_source.as_ptr().cast() , &(vertex_source.len().try_into().unwrap()));
        gl::CompileShader(vs);
        
        let mut compiled = 0;
        gl::GetShaderiv(vs, gl::COMPILE_STATUS, &mut compiled);
        if compiled == 0 {
            let mut log: [u8;1024] = [0; 1024];
            let mut log_len = 0;
            gl::GetShaderInfoLog(vs, 1024, &mut log_len, log.as_mut_ptr().cast());

            let log = std::str::from_utf8_unchecked(std::slice::from_raw_parts(log.as_ptr(), log_len as usize));
            println!("Compilation log:\n{}\n---", log);
        }

        let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(fs, 1, &fragment_source.as_ptr().cast(), &(fragment_source.len().try_into().unwrap()));
        gl::CompileShader(fs);

        gl::GetShaderiv(fs, gl::COMPILE_STATUS, &mut compiled);
        if compiled == 0 {
            let mut log: [u8;1024] = [0; 1024];
            let mut log_len = 0;
            gl::GetShaderInfoLog(fs, 1024, &mut log_len, log.as_mut_ptr().cast());

            let log = str::from_utf8_unchecked(slice::from_raw_parts(log.as_ptr(), log_len as usize));
            println!("Compilation log:\n{}\n---", log);
        }

        let program = ShaderProgram { id: gl::CreateProgram(), inputs: Vec::new() };
        gl::AttachShader(program.id, vs);
        gl::AttachShader(program.id, fs);
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
        gl::LinkProgram(program.id);
        let mut linked: i32 = 0;
        gl::GetProgramiv(program.id, gl::LINK_STATUS, &mut linked as *mut i32); 
        if linked == 0 {
            let mut log: [u8;1024] = [0; 1024];
            let mut log_len: i32 = 0;
            gl::GetProgramInfoLog(program.id, 1024, &mut log_len, log.as_mut_ptr().cast());

            let log = str::from_utf8_unchecked(slice::from_raw_parts(log.as_ptr(), log_len as usize));
            println!("Link log:\n{}\n---", log);
        }

        program
    }

    pub unsafe fn activate(&self) {
        gl::UseProgram(self.id);
    }

    fn guess_semantic(variable_name: &str) -> Option<AttributeSemantic> {
        if variable_name.starts_with("position") {
            Some(AttributeSemantic::Position)
        } else if variable_name.starts_with("normal") {
            Some(AttributeSemantic::Normal)
        } else if variable_name.starts_with("tangent") {
            Some(AttributeSemantic::Tangent)
        } else if variable_name.starts_with("bitangent") {
            Some(AttributeSemantic::Bitangent)
        } else if variable_name.starts_with("texcoords") {
            if variable_name.len() <= "texcoords".len() {
                None
            } else if let Ok(n) = variable_name[9..10].parse::<u8>() {
                Some(AttributeSemantic::TexCoord(n))
            } else {
                None
            }
        } else if variable_name.starts_with("uv") {
            if variable_name.len() <= "uv".len() {
                None
            } else if let Ok(n) = variable_name[2..3].parse::<u8>() {
                Some(AttributeSemantic::TexCoord(n))
            } else {
                None
            }
        } else if variable_name.starts_with("color") {
            if variable_name.len() <= "color".len() {
                None
            } else if let Ok(n) = variable_name[5..6].parse::<u8>() {
                Some(AttributeSemantic::Color(n))
            } else {
                None
            }
        } else if variable_name.starts_with("generic") {
            if variable_name.len() <= "generic".len() {
                None
            } else if let Ok(n) = variable_name[7..8].parse::<u8>() {
                Some(AttributeSemantic::Generic(n))
            } else {
                None
            }
        } else {
            None
        }
    }

    pub unsafe fn get_vertex_shader_inputs(&self)  -> Vec<VertexShaderInput> {
        let mut num_inputs = 0;
        gl::GetProgramInterfaceiv(self.id, gl::PROGRAM_INPUT, gl::ACTIVE_RESOURCES, &mut num_inputs);

        let mut max_name_len = 0;
        gl::GetProgramInterfaceiv(self.id, gl::PROGRAM_INPUT, gl::MAX_NAME_LENGTH, &mut max_name_len);
        let mut name_store = Vec::<u8>::with_capacity(max_name_len as usize);
        name_store.set_len(max_name_len as usize);

        let mut inputs = Vec::new();
        for input_id in 0..num_inputs as u32 { 
            let mut name_len = 0;
            gl::GetProgramResourceName(self.id, gl::PROGRAM_INPUT, input_id, max_name_len, &mut name_len, name_store.as_mut_ptr() as *mut i8);
            let name = str::from_utf8_unchecked(slice::from_raw_parts(name_store.as_ptr(), name_len as usize));

            if let Some(semantic) = Self::guess_semantic(name) {
                let mut params: [i32; 2] = [0; 2];
                gl::GetProgramResourceiv(self.id, gl::PROGRAM_INPUT, input_id, 2, [gl::TYPE, gl::LOCATION].as_mut_ptr(), 2, std::ptr::null_mut(), params.as_mut_ptr());

                println!("attribute: {} {:?}, type: {}, location: {}", name, semantic, params[0], params[1]);

                inputs.push(VertexShaderInput { semantic, location: params[1] as u32, base_type: params[0] as u32 });
            }
        }

        inputs
    }
}
