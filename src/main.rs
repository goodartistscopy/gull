
use std::{
    ffi::{CStr, c_char},
    cell::RefCell,
    rc::Rc,
};

use gtk::{gdk, glib, prelude::*};
use gtk::glib::clone;

use gl::types::*;

const APP_ID: &str = "goodartistscopy.Gull";

//#[derive(Copy,Clone,Default)]
#[derive(Default)]
struct DrawData {
    context: Option<gdk::GLContext>,
    vao: u32,
    buffers: [u32; 3],
    program: u32,
    color: [f32; 4],
    xform: [f32; 6],
    timer_query: u32,
    fps_label: Option<gtk::Label>
}

fn main() {
    // Initialize GL function pointers
    #[cfg(target_os = "macos")]
    let library = unsafe { libloading::os::unix::Library::new("libepoxy.0.dylib") }.unwrap();
    #[cfg(all(unix, not(target_os = "macos")))]
    let library = unsafe { libloading::os::unix::Library::new("libepoxy.so.0") }.unwrap();
    #[cfg(windows)]
    let library = libloading::os::windows::Library::open_already_loaded("epoxy-0.dll").unwrap();

    epoxy::load_with(|name| {
        unsafe { library.get::<_>(name.as_bytes()) }
        .map(|symbol| *symbol)
            .unwrap_or(std::ptr::null())
    });
    gl::load_with(|s| epoxy::get_proc_addr(s));

    let app = gtk::Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_ui);
    app.run();
}

fn build_ui(app: &gtk::Application) {
    let window = gtk::ApplicationWindow::builder()
        .application(app)
        .title("Gull")
        .build();

    let gl_canvas = gtk::GLArea::builder()
        .auto_render(true)
        .valign(gtk::Align::Fill)
        .vexpand(true)
        .width_request(400)
        .height_request(400)
        .build();

    let fps_label = gtk::Label::builder()
        .halign(gtk::Align::Start)
        .valign(gtk::Align::Start)
        .margin_top(5)
        .margin_start(5)
        .opacity(0.5)
        .visible(true)
        .label("fps")
        .build();

    let data = Rc::new(RefCell::new(DrawData::default()));

    data.borrow_mut().fps_label = Some(fps_label.clone());

    gl_canvas.connect_create_context(clone!(@strong data =>
        move |canvas| {
            // When using EGL (default on Linux)
            let context = gdk::Display::default().and_then(|display| { display.create_gl_context().ok() })?;
            // GLX crashees when the context is surfaceless (as of gtk 4.9.2). Use this instead
            // let context = canvas.native()?.surface().create_gl_context().ok()?;

            context.set_required_version(4, 6);
            context.set_forward_compatible(true);
            context.set_debug_enabled(true);

            if let Err(error) = context.realize()
            {
                canvas.set_error(Some(&glib::Error::new(gdk::GLError::NotAvailable, error.message())));
                return None;
            }

            context.make_current();
            let (major, minor) = context.version();
            let vendor = unsafe { CStr::from_ptr(gl::GetString(gl::VENDOR) as *const c_char) };
            let renderer = unsafe { CStr::from_ptr(gl::GetString(gl::RENDERER) as *const c_char) }; 
            println!("OpenGL version: {}.{} (forward compatible:{}, debug:{})", major, minor, context.is_forward_compatible(), context.is_debug_enabled());
            println!("Vendor: {:?}\nRenderer: {:?}", vendor, renderer);

            data.borrow_mut().context = Some(context.clone());

            Some(context)
    }));


    gl_canvas.connect_realize(clone!(@strong data =>
        move |_canvas: &gtk::GLArea| {
            initialize(&mut data.borrow_mut());
    }));

    gl_canvas.connect_render(clone!(@strong data =>
        move |_canvas: &gtk::GLArea, _context: &gdk::GLContext| {
            render(&data.borrow());

            gtk::Inhibit(true)
    }));

    let rotation_slider = gtk::Scale::builder()
        .orientation(gtk::Orientation::Horizontal)
        .draw_value(true)
        .value_pos(gtk::PositionType::Left)
        .round_digits(2)
        .build();

    rotation_slider.set_range(0.0, 360.0);

    rotation_slider.connect_change_value(clone!(@strong data, @strong gl_canvas =>
        move |_scale, _scroll_type, value| {
            let c = value.to_radians().cos() as f32;
            let s = value.to_radians().sin() as f32;
            let g = (0.5, 3_f32.sqrt() / 6.0);
            let tx = -(c * g.0 - s * g.1) + g.0 - 0.5;
            let ty = -(s * g.0 + c * g.1) + g.1 - 0.5;
            let mut data = data.borrow_mut();
            data.xform = [c, s, -s, c, tx, ty];

            data.context.as_ref().unwrap().make_current();

            unsafe {
                gl::UseProgram(data.program);
                let loc = gl::GetUniformLocation(data.program, "xform\0".as_ptr().cast());
                gl::UniformMatrix3x2fv(loc, 1, false as GLboolean, data.xform.as_ptr().cast());
            }

            gl_canvas.queue_render();
            gtk::Inhibit(false)
    }));

    let container = gtk::Box::builder()
        .orientation(gtk::Orientation::Vertical)
        .build();

    let overlay = gtk::Overlay::new();

    overlay.set_child(Some(&gl_canvas));

    overlay.add_overlay(&fps_label);

    container.append(&overlay);
    container.append(&rotation_slider);

    window.set_child(Some(&container));
    window.present();
}

const VERTEX_SHADER: &str = r#"
    #version 450

    layout(location = 0) in vec2 position;
    layout(location = 1) in vec3 color;

    layout(location = 7) uniform mat3x2 xform = mat3x2(vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(-0.5, -0.5));

    out vec3 vcolor;

    void main() {
        vcolor = color;
        vec2 p = xform * vec3(position, 1.0);
        gl_Position = vec4(p, 0.5, 1.0);
    }
"#;

const FRAGMENT_SHADER: &str = r#"
    #version 450

    layout(location = 0) out vec4 color;
    in vec3 vcolor;

    void main() {
        color = vec4(vcolor, 1.0);
    }
"#;

fn initialize(data: &mut DrawData) {
    data.color = [0.3, 0.1, 0.5, 1.0];

    #[repr(C)]
    struct VertexData {
        pos: [f32; 2],
        color: [u8; 3]
    }
    let vertex_data : [VertexData; 3] = [
        VertexData { pos: [0.0, 0.0], color: [255, 0, 0] },
        VertexData { pos: [1.0, 0.0], color: [0, 255, 0] },
        VertexData { pos: [0.5, 3.0_f32.sqrt()/2.0], color: [0, 0, 255] },
    ];
    let indices = [0, 1, 2];
    unsafe {
        gl::CreateVertexArrays(1, &mut data.vao as *mut GLuint);
        gl::BindVertexArray(data.vao);

        gl::GenBuffers(3, data.buffers.as_mut_ptr());

        gl::BindBuffer(gl::ARRAY_BUFFER, data.buffers[0]);
        let vbuffer_size = std::mem::size_of_val(&vertex_data) as isize;
        gl::BufferStorage(gl::ARRAY_BUFFER, vbuffer_size, vertex_data.as_ptr().cast(), 0);

        gl::EnableVertexAttribArray(0);
        gl::EnableVertexAttribArray(1);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, false as GLboolean, std::mem::size_of::<VertexData>() as i32, std::ptr::null());
        gl::VertexAttribPointer(1, 3, gl::UNSIGNED_BYTE, true as GLboolean, std::mem::size_of::<VertexData>() as i32, std::mem::size_of::<[f32; 2]>() as *const GLvoid);
    
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, data.buffers[2]);
        gl::BufferStorage(gl::ELEMENT_ARRAY_BUFFER, std::mem::size_of_val(&indices) as isize, indices.as_ptr().cast(), 0);
    
        let vs = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(vs, 1, &VERTEX_SHADER.as_ptr().cast() , &(VERTEX_SHADER.len().try_into().unwrap()));
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
        gl::ShaderSource(fs, 1, &FRAGMENT_SHADER.as_ptr().cast(), &(FRAGMENT_SHADER.len().try_into().unwrap()));
        gl::CompileShader(fs);

        gl::GetShaderiv(fs, gl::COMPILE_STATUS, &mut compiled);
        if compiled == 0 {
            let mut log: [u8;1024] = [0; 1024];
            let mut log_len = 0;
            gl::GetShaderInfoLog(fs, 1024, &mut log_len, log.as_mut_ptr().cast());

            let log = std::str::from_utf8_unchecked(std::slice::from_raw_parts(log.as_ptr(), log_len as usize));
            println!("Compilation log:\n{}\n---", log);
        }

        data.program = gl::CreateProgram();
        gl::AttachShader(data.program, vs);
        gl::AttachShader(data.program, fs);
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
        gl::LinkProgram(data.program);
        let mut linked: i32 = 0;
        gl::GetProgramiv(data.program, gl::LINK_STATUS, &mut linked as *mut i32); 
        if linked == 0 {
            let mut log: [u8;1024] = [0; 1024];
            let mut log_len: i32 = 0;
            gl::GetProgramInfoLog(data.program, 1024, &mut log_len, log.as_mut_ptr().cast());

            let log = std::str::from_utf8_unchecked(std::slice::from_raw_parts(log.as_ptr(), log_len as usize));
            println!("Link log:\n{}\n---", log);
        }

        gl::GenQueries(1, &mut data.timer_query);
        let mut size: i32 = 0;
        gl::GetQueryiv(gl::TIME_ELAPSED, gl::QUERY_COUNTER_BITS, &mut size);

        println!("Timer query precision: {} bits", size);
    }
}

fn render(data: &DrawData) {
    unsafe {
        gl::BeginQuery(gl::TIME_ELAPSED, data.timer_query);

        let color = data.color;
        gl::ClearColor(color[0], color[1], color[2], color[3]);
        gl::Clear(gl::COLOR_BUFFER_BIT);
        gl::UseProgram(data.program);
        gl::BindVertexArray(data.vao);
        
        gl::DrawElements(gl::TRIANGLES, 3, gl::UNSIGNED_INT, std::ptr::null());

        gl::EndQuery(gl::TIME_ELAPSED);

        let mut elapsed: u64 = 0;
        gl::GetQueryObjectui64v(data.timer_query, gl::QUERY_RESULT, &mut elapsed);

        glib::idle_add_local_once(glib::clone!(@strong data.fps_label as fps_label =>
            move  || {
                let last_ms = ((elapsed as f64) * 1e-6) as f32;
                let last_fps = (last_ms * 1e-3).recip();
                if fps_label.as_ref().unwrap().is_realized()
                {
                    fps_label.as_ref().unwrap().set_label(std::format!("{} ms, {} fps", last_ms, last_fps).as_str());
                }
        }));
    }
}
