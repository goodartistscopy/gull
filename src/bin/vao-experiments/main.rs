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
    uniform_buffer_alignment: i32,
    global_input_assembly: InputAssembly,
    program: ShaderProgram,
    color: [f32; 4],
    timer_query: u32,
    fps_label: Option<gtk::Label>,
    objects: Vec::<ObjectData>,
    object_xforms_buffer: u32,
    orig_view_point: (f32, f32, f32), // theta, phi, distance
    current_view_point: (f32, f32, f32), // theta, phi, distance
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

fn update_view_matrix(data: &mut AppData)
{
    let theta = data.current_view_point.0;
    let phi = data.current_view_point.1;
    let (stheta, ctheta) = theta.sin_cos();
    let (sphi, cphi) = phi.sin_cos();
    let p = data.current_view_point.2 * Point3::new(stheta * cphi, stheta * sphi, ctheta);
    data.view_matrices[0] = Matrix4::look_at_rh(&p, &Point3::origin(), &Vector3::z_axis());

    unsafe {
        gl::NamedBufferSubData(data.view_data_buffer, 0, size_of::<Matrix4>() as isize, data.view_matrices[0].data.ptr().cast());
    }
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
                    (data.orig_view_point.1 + -0.01 * dx as f32).rem_euclid(2.0 * PI),
                    data.orig_view_point.2
                );
                update_view_matrix(&mut data);

                gl_canvas.queue_render();
            }
    }));
    gl_canvas.add_controller(&mouse_ctrl);

    let scroll_ctrl = gtk::EventControllerScroll::new(gtk::EventControllerScrollFlags::VERTICAL);
    scroll_ctrl.connect_scroll(clone!(@strong data, @strong gl_canvas =>
        move |_canvas, _, dy| {
            let mut data = data.borrow_mut();
            data.current_view_point.2 += 0.6 * (dy as f32);

            update_view_matrix(&mut data);

            gl_canvas.queue_render();

            gtk::Inhibit(true)
    }));
    gl_canvas.add_controller(&scroll_ctrl);

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
        float lambert = clamp(dot(l, normalize(worldNormal)), 0.0, 1.0);
        color = vec4(vec3(lambert), 1.0);
    }
"#;

fn update_object_grid(data: &mut AppData, mesh: &Mesh, vs_inputs: &Vec::<VertexShaderInput>, grid_dim: (u32, u32, u32)) {
    unsafe {
        let num_objects = (grid_dim.0 * grid_dim.1 * grid_dim.2) as usize;
        let per_object_size = ((size_of::<Matrix4>() as i32 / data.uniform_buffer_alignment) + 1) * data.uniform_buffer_alignment; 
        if num_objects > data.objects.len() {
            let mut buffer = data.object_xforms_buffer;
            if gl::IsBuffer(buffer) == gl::TRUE {
               gl::DeleteBuffers(1, &buffer as *const _);
            }
            gl::CreateBuffers(1, &mut buffer as *mut _);
            // Note: this might cause significant over-allocation (e.g. on an RTX 3080, alignment is 256)
            gl::NamedBufferStorage(buffer, (num_objects as i32 * per_object_size) as isize, std::ptr::null(), gl::DYNAMIC_STORAGE_BIT);
            data.object_xforms_buffer = buffer;
        }

        const GRID_SPACING : f32 = 2.0;
        let back = -((grid_dim.0 - 1) as f32 * GRID_SPACING / 2.0);
        let left = -((grid_dim.1 - 1) as f32 * GRID_SPACING / 2.0);
        let bottom = -((grid_dim.2 - 1) as f32 * GRID_SPACING / 2.0);
        for i in 0..grid_dim.0 {
            for j in 0..grid_dim.1 {
                for k in 0..grid_dim.2 {
                    let linear_idx = (i * (grid_dim.0 * grid_dim.1) + j * grid_dim.0 + k) as usize;
                    if linear_idx >= data.objects.len() {
                        let stream_layouts = vec![
                            VertexLayout {
                                attributes: vec![
                                    Attribute {semantic: AttributeSemantic::Position, base_type: AttributeType::Float32, len: 3, normalized: false },
                                ]
                            },
                            VertexLayout {
                                attributes: vec![
                                    Attribute {semantic: AttributeSemantic::Normal, base_type: AttributeType::Float32, len: 3, normalized: false },
                                ]
                            }
                        ];
                        let draw_data = DrawData::with_mesh(stream_layouts, mesh);
                        let inputs = InputAssembly::new();
                        inputs.configure_and_bind(vs_inputs, &draw_data);

                        let t = Vector3::new(back, left, bottom) + GRID_SPACING * Vector3::new(i as f32, j as f32, k as f32);
                        let mat = Matrix4::new_translation(&t);

                        let offset = (linear_idx as i32 * per_object_size) as isize;
                        gl::NamedBufferSubData(data.object_xforms_buffer, offset, size_of::<Matrix4>() as isize, mat.data.ptr().cast());
                        let xform_buffer = BufferView {
                            buffer_id: data.object_xforms_buffer,
                            offset: offset as u32
                        };
                        data.objects.push(ObjectData { draw_data, xform: mat, xform_buffer, inputs });
                    }
                }
            }
        }
    }
}

fn initialize(data: &mut AppData) {
    unsafe {
        gl::GenQueries(1, &mut data.timer_query);
        let mut size: i32 = 0;
        gl::GetQueryiv(gl::TIME_ELAPSED, gl::QUERY_COUNTER_BITS, &mut size);
        println!("Timer query precision: {} bits", size);

        gl::GetIntegerv(gl::UNIFORM_BUFFER_OFFSET_ALIGNMENT, &mut data.uniform_buffer_alignment as *mut _);
        println!("Uniform buffer alignment: {}", data.uniform_buffer_alignment);

        data.color = [0.85, 0.85, 0.85, 1.0];
        let sphere = create_icosphere(0.8, 1);

        data.program = ShaderProgram::new(VERTEX_SHADER, FRAGMENT_SHADER);

        let vs_inputs = data.program.get_vertex_shader_inputs();
    
        let grid_size = (5, 5, 5);
        update_object_grid(data, &sphere, &vs_inputs, grid_size);

        gl::CreateBuffers(1, &mut data.view_data_buffer as *mut _);
        data.current_view_point = (PI/2.0, 0.0, 4.0 * grid_size.0.max(grid_size.1).max(grid_size.2) as f32);
        data.view_matrices[0] = Matrix4::look_at_rh(&Point3::new(data.current_view_point.2, 0.0, 0.0), &Point3::origin(), &Vector3::z_axis());
        data.view_matrices[1] = Matrix4::new_perspective(1.0, 45.0_f32.to_radians(), 0.1, 1e3);
        gl::NamedBufferStorage(data.view_data_buffer, 2 * size_of::<Matrix4>() as isize, data.view_matrices.as_ptr().cast(), gl::DYNAMIC_STORAGE_BIT);

        let index = gl::GetUniformBlockIndex(data.program.id, "ViewMatrices\0".as_ptr().cast());
        println!("Bindex block index {} to binding 0", index);
        gl::UniformBlockBinding(data.program.id, index, 0);
        let index = gl::GetUniformBlockIndex(data.program.id, "ObjectMatrix\0".as_ptr().cast());
        println!("Bindex block index {} to binding 1", index);
        gl::UniformBlockBinding(data.program.id, index, 1);

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

