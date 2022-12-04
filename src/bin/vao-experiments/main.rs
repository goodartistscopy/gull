mod vertex_layout;
mod mesh;
mod shader;

use std::{
    ffi::{CStr, c_char},
    cell::RefCell,
    rc::Rc,
    mem::size_of
};

use gtk::{
    gdk,
    glib,
    glib::{clone, source::Continue},
    prelude::*,
    pango
};

use nalgebra as na;

use na::{
    RawStorage,
};
type Matrix4 = na::Matrix4::<f32>;
use std::f32::consts::PI;

use gl::types::*;

use mesh::*;
use shader::*;
use vertex_layout::*;

use gull::utils::*;

const APP_ID: &str = "vao-experiments.Gull";

struct ObjectData {
    draw_data: DrawData,
    xform: Matrix4,
    xform_buffer: BufferView,
    inputs: InputAssembly
}

//#[derive(Copy,Clone,Default)]
#[derive(Default)]
struct AppData {
    context: Option<gdk::GLContext>,
    global_input_assembly: InputAssembly,
    program: ShaderProgram,
    color: [f32; 4],
    timer_query: u32,
    fps_label: Option<gtk::Label>,
    objects: Vec::<ObjectData>,
    orig_view_point: (f32, f32), // theta, phi
    current_view_point: (f32, f32), // theta, phi
    view_point_distance: f32,
    view_matrices: [Matrix4; 2],
    view_data_buffer: u32,
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
        .has_depth_buffer(true)
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
        .opacity(1.0)
        .visible(true)
        .label("fps")
        .build();

    let label_attrs = pango::AttrList::new();
    label_attrs.insert(pango::AttrColor::new_foreground(0, 0, 0));
    fps_label.set_attributes(Some(&label_attrs));

    let data = Rc::new(RefCell::new(AppData::default()));

    data.borrow_mut().fps_label = Some(fps_label.clone());

    gl_canvas.connect_create_context(clone!(@strong data =>
        move |canvas| {
            // When using EGL (default on Linux)
            let context = gdk::Display::default().and_then(|display| { display.create_gl_context().ok() })?;
            //GLX crashees when the context is surfaceless (as of gtk 4.9.2). Use this instead
            //let context = canvas.native()?.surface().create_gl_context().ok()?;

            context.set_required_version(4, 6);
            context.set_forward_compatible(true);
            context.set_debug_enabled(std::cfg!(debug_assertions));

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

            #[cfg(debug_assertions)]
            unsafe{
                gl::Enable(gl::DEBUG_OUTPUT);
                gl::DebugMessageCallback(Some(debug_callback), std::ptr::null());
            }

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

    let mouse_ctrl = gtk::GestureDrag::new();
    
    mouse_ctrl.connect_drag_begin(clone!(@strong data =>
        move |_canvas, _x, _y| {
            let mut data = data.borrow_mut();
            data.orig_view_point = data.current_view_point;
    }));

    mouse_ctrl.connect_drag_update(clone!(@strong data, @strong gl_canvas =>
        move |ctrl, _x, _y| {
            let mut data = data.borrow_mut();
            if let Some((dx,dy)) = ctrl.offset() {
                data.current_view_point = (
                    (data.orig_view_point.0 + -0.01 * dy as f32).clamp(0.01, PI-0.01),
                    (data.orig_view_point.1 + -0.01 * dx as f32).rem_euclid(2.0 * PI)
                );
                let theta = data.current_view_point.0;
                let phi = data.current_view_point.1;
                let (ctheta, stheta) = (theta.cos(), theta.sin());
                let (cphi, sphi) = (phi.cos(), phi.sin());
                let p = data.view_point_distance * Point3::new(stheta * cphi, stheta * sphi, ctheta);
                data.view_matrices[0] = Matrix4::look_at_rh(&p, &Point3::origin(), &Vector3::z_axis());

                unsafe {
                    gl::NamedBufferSubData(data.view_data_buffer, 0, size_of::<Matrix4>() as isize, data.view_matrices[0].data.ptr().cast());
                }

                gl_canvas.queue_render();
            }
    }));

    gl_canvas.add_controller(&mouse_ctrl);

    let container = gtk::Box::builder()
        .orientation(gtk::Orientation::Vertical)
        .build();

    let overlay = gtk::Overlay::new();

    overlay.set_child(Some(&gl_canvas));

    overlay.add_overlay(&fps_label);

    container.append(&overlay);

    window.set_child(Some(&container));
    window.present();
}

const VERTEX_SHADER: &str = r#"
    #version 450

    layout(location = 0) in vec4 position;
    layout(location = 1) in vec3 normal;

    layout(std140)
    uniform ViewMatrices {
        mat4 viewMat;
        mat4 projMat;
    };

    layout(std140)
    uniform ObjectMatrix {
        mat4 modelMat;
    };

    out VertexData
    {
        vec3 worldPosition;
        vec3 worldNormal;
    };

    void main() {
        worldPosition = (modelMat * position).xyz;
        worldNormal = inverse(transpose(mat3(modelMat))) * normal;

        gl_Position = projMat * viewMat * vec4(worldPosition, 1.0);
    }
"#;

const FRAGMENT_SHADER: &str = r#"
    #version 450

    in VertexData
    {
        vec3 worldPosition;
        vec3 worldNormal;
    };

    layout(location = 0) out vec4 color;

    const vec3 LIGHT = vec3(0.0, 0.0, 10.0);

    void main() {
        vec3 l = normalize(LIGHT - worldPosition);
        float lambert = clamp(dot(l, worldNormal), 0.0, 1.0);
        color = vec4(vec3(lambert), 1.0);
    }
"#;

// fn update_object_grid() {

// }

fn initialize(data: &mut AppData) {
    data.color = [0.9, 0.8, 0.85, 1.0];
    
    let sphere = create_icosphere(0.8, 2);

    unsafe {
        let stream_layouts = vec![
            VertexLayout {
                attributes: vec![
                    Attribute {semantic: AttributeSemantic::Padding, base_type: AttributeType::UInt8, len: 1, normalized: false },
                    Attribute {semantic: AttributeSemantic::Position, base_type: AttributeType::Float32, len: 3, normalized: false },
                ]
            },
            VertexLayout {
                attributes: vec![
                   Attribute {semantic: AttributeSemantic::Padding, base_type: AttributeType::UInt8, len: 2, normalized: false },
                   Attribute {semantic: AttributeSemantic::Normal, base_type: AttributeType::Float32, len: 3, normalized: false },
                ]
            }
        ];

        data.program = ShaderProgram::new(VERTEX_SHADER, FRAGMENT_SHADER);

        let vs_inputs = data.program.get_vertex_shader_inputs();
    
        // first object
        let draw_data = DrawData::with_mesh(stream_layouts.clone(), &sphere);
        let inputs = InputAssembly::new();
        inputs.configure_and_bind(&vs_inputs, &draw_data);

        let mut matrices_buffer = 0;
        gl::CreateBuffers(1, &mut matrices_buffer as *mut _);
        let mat = Matrix4::new_translation(&Vector3::new(-1.0, 0.0, 0.0));
        gl::NamedBufferStorage(matrices_buffer, size_of::<Matrix4>() as isize, mat.data.ptr().cast(), gl::DYNAMIC_STORAGE_BIT);
        data.objects.push(ObjectData { draw_data, xform: Matrix4::identity(), xform_buffer: BufferView { buffer_id: matrices_buffer, offset: 0}, inputs });

        // another object
        let draw_data = DrawData::with_mesh(stream_layouts, &sphere);
        let inputs = InputAssembly::new();
        inputs.configure_and_bind(&vs_inputs, &draw_data);
        let mut matrices_buffer = 0;
        gl::CreateBuffers(1, &mut matrices_buffer as *mut _);
        let mat = Matrix4::new_translation(&Vector3::new(1.0, 0.0, 0.0));
        gl::NamedBufferStorage(matrices_buffer, size_of::<Matrix4>() as isize, mat.data.ptr().cast(), gl::DYNAMIC_STORAGE_BIT);
        data.objects.push(ObjectData { draw_data, xform: Matrix4::identity(), xform_buffer: BufferView { buffer_id: matrices_buffer, offset: 0}, inputs });

        gl::CreateBuffers(1, &mut data.view_data_buffer as *mut _);
        data.view_point_distance = 5.0;
        data.current_view_point = (PI/2.0, 0.0);
        data.view_matrices[0] = Matrix4::look_at_rh(&Point3::new(data.view_point_distance, 0.0, 0.0), &Point3::origin(), &Vector3::z_axis());
        data.view_matrices[1] = Matrix4::new_perspective(1.0, 45.0_f32.to_radians(), 0.1, 1e3);
        gl::NamedBufferStorage(data.view_data_buffer, 2 * size_of::<Matrix4>() as isize, data.view_matrices.as_ptr().cast(), gl::DYNAMIC_STORAGE_BIT);

        let index = gl::GetUniformBlockIndex(data.program.id, "ViewMatrices\0".as_ptr().cast());
        println!("Bindex block index {} to binding 0", index);
        gl::UniformBlockBinding(data.program.id, index, 0);
        let index = gl::GetUniformBlockIndex(data.program.id, "ObjectMatrix\0".as_ptr().cast());
        println!("Bindex block index {} to binding 1", index);
        gl::UniformBlockBinding(data.program.id, index, 1);

        gl::GenQueries(1, &mut data.timer_query);
        let mut size: i32 = 0;
        gl::GetQueryiv(gl::TIME_ELAPSED, gl::QUERY_COUNTER_BITS, &mut size);

        println!("Timer query precision: {} bits", size);
    }
}

fn render(data: &AppData) {
    unsafe {
        gl::BeginQuery(gl::TIME_ELAPSED, data.timer_query);

        let color = data.color;
        gl::ClearColor(color[0], color[1], color[2], color[3]);
        gl::ClearDepthf(1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
      
        data.program.activate();

        gl::Enable(gl::CULL_FACE);
        gl::Enable(gl::DEPTH_TEST);
        gl::DepthFunc(gl::LESS);

        gl::BindBufferBase(gl::UNIFORM_BUFFER, 0, data.view_data_buffer);
        for object in &data.objects {
            object.inputs.activate();

            gl::BindBufferRange(gl::UNIFORM_BUFFER, 1, object.xform_buffer.buffer_id, object.xform_buffer.offset as isize, size_of::<[Matrix4; 2]>() as isize);
            gl::DrawElements(gl::TRIANGLES, object.draw_data.num_elems, gl::UNSIGNED_INT, std::ptr::null());
        }

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

