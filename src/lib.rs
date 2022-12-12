pub mod vertex_layout;
pub mod shader;
pub mod mesh;

pub mod utils {
    use gl::types::*;
    use colored::Colorize;

    pub extern "system" fn debug_callback(source: GLenum, gltype: GLenum, id: GLuint, severity: GLenum, length: GLsizei, message: *const GLchar, _user_param: *mut GLvoid) {
        let source_str = match source {
            gl::DEBUG_SOURCE_API => "API",
            gl::DEBUG_SOURCE_WINDOW_SYSTEM => "Window System",
            gl::DEBUG_SOURCE_SHADER_COMPILER => "Shader Compiler",
            gl::DEBUG_SOURCE_THIRD_PARTY => "Third Party",
            gl::DEBUG_SOURCE_APPLICATION => "Application",
            gl::DEBUG_SOURCE_OTHER => "Other",
            _ => "Unknown"
        };

        let type_str = match gltype {
            gl::DEBUG_TYPE_ERROR => "Error",
            gl::DEBUG_TYPE_DEPRECATED_BEHAVIOR => "Deprecated",
            gl::DEBUG_TYPE_UNDEFINED_BEHAVIOR => "Undefined Behavior",
            gl::DEBUG_TYPE_PORTABILITY => "Portabiltity",
            gl::DEBUG_TYPE_PERFORMANCE => "Performance",
            gl::DEBUG_TYPE_MARKER => "Marker",
            gl::DEBUG_TYPE_PUSH_GROUP => "Push Group",
            gl::DEBUG_TYPE_POP_GROUP => "Pop Group",
            gl::DEBUG_TYPE_OTHER => "Other",
            _ => "Unknown"
        };

        let tag = format!("[OpenGL ({}) {}|{}]", id, source_str, type_str);
        let error_str = match severity {
            gl::DEBUG_SEVERITY_HIGH => tag.red(),
            gl::DEBUG_SEVERITY_MEDIUM => tag.yellow(),
            gl::DEBUG_SEVERITY_LOW => tag.green(),
            gl::DEBUG_SEVERITY_NOTIFICATION => tag.purple(),
            _ => tag.normal()
        };

        // if severity == gl::DEBUG_SEVERITY_NOTIFICATION {
        //     return;
        // }

        let message = unsafe {
            String::from_utf8_unchecked(std::slice::from_raw_parts(message as *const u8, length as usize).to_owned())
        };

        println!("{} {}", error_str, message);
    }
}
