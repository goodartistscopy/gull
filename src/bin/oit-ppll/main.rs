use std::{
    ffi::{CStr, c_char},
    cell::RefCell,
    rc::Rc,
    mem::size_of,
    f32::consts::PI,
    collections::VecDeque,
    fs::File,
    io::Read,
};

use gtk::{
    gdk,
    glib,
    glib::clone,
    prelude::*,
};

use nalgebra as na;

use na::{
    RawStorage,
};

type Matrix4 = na::Matrix4::<f32>;
type Vector4 = na::Vector4::<f32>;

use gltf::Gltf;

use gull::mesh::*;
use gull::shader::*;
use gull::vertex_layout::*;

use gull::utils::*;

const APP_ID: &str = "vao-experiments.Gull";

const UI: &str = r#"
<interface>
    <object class='GtkApplicationWindow' id='main_window'>
        <property name='title'>Per-pixel linked-list OIT</property>
        <child>
            <object class='GtkBox'>
                <property name="orientation">GTK_ORIENTATION_VERTICAL</property>
                <child>
                    <object class='GtkOverlay'>
                        <property name='valign'>GTK_ALIGN_FILL</property>
                        <property name='vexpand'>TRUE</property>
                        <child>
                            <object class='GtkGLArea' id='canvas'>
                                <property name='width-request'>512</property>
                                <property name='height-request'>512</property>
                                <property name='valign'>GTK_ALIGN_FILL</property>
                                <property name='vexpand'>TRUE</property>
                                <property name='auto-render'>TRUE</property>
                                <property name='has-depth-buffer'>TRUE</property>
                            </object>
                        </child>child
                        <child type='overlay'>
                            <object class='GtkLabel' id='fps_label'>
                                <property name='halign'>GTK_ALIGN_START</property>
                                <property name='valign'>GTK_ALIGN_START</property>
                                <property name='margin-top'>5</property>
                                <property name='margin-start'>5</property>
                                <attributes>
                                    <attribute name='foreground' value='#000000'/>
                                </attributes>
                            </object>
                        </child>
                    </object>
                </child>
            </object>
        </child>
    </object>
</interface>
"#;

struct ObjectData {
    draw_data: DrawData,
    #[allow(dead_code)] // read via pointer
    xform: Matrix4,
    xform_buffer: BufferView,
    inputs: InputAssembly
}

//#[derive(Copy,Clone,Default)]
#[derive(Default)]
struct AppData {
    context: Option<gdk::GLContext>,
    uniform_buffer_alignment: i32,
    //global_input_assembly: InputAssembly,
    program: ShaderProgram,
    color: [f32; 4],
    timer_query: u32,
    frame_time_history: VecDeque::<f32>,
    fps_label: gtk::Label,
    objects: Vec::<ObjectData>,
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

fn build_ui(app: &gtk::Application) {
    let builder = gtk::Builder::from_string(UI);
    let gl_canvas: gtk::GLArea = builder.object("canvas").unwrap();
    let fps_label: gtk::Label = builder.object("fps_label").unwrap();
    let window: gtk::ApplicationWindow = builder.object("main_window").unwrap();
    window.set_application(Some(app));

    let data = Rc::new(RefCell::new(AppData::default()));

    data.borrow_mut().fps_label = fps_label;

    gl_canvas.connect_create_context(clone!(@strong data =>
        move |canvas| {
            // When using EGL (default on Linux)
            let context = gdk::Display::default().and_then(|display| { display.create_gl_context().ok() })?;
            //GLX crashes when the context is surfaceless (as of gtk 4.9.2). Use this instead
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
            render(data.clone());

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
            data.current_view_point.2 *= 0.1 * (dy as f32 + 1.0) + 0.9;

            update_view_matrix(&mut data);

            gl_canvas.queue_render();

            gtk::Inhibit(true)
    }));
    gl_canvas.add_controller(&scroll_ctrl);

    window.present();
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

fn create_object(mesh: &Mesh, vs_inputs: &Vec::<VertexShaderInput>) -> ObjectData {
    let stream_layouts = vec![
        VertexLayout {
            attributes: vec![
                Attribute {semantic: AttributeSemantic::Position, base_type: AttributeType::Float32, len: 3, normalized: false },
                Attribute {semantic: AttributeSemantic::Normal, base_type: AttributeType::Float32, len: 3, normalized: false },
            ]
        },
    ];
    let draw_data = DrawData::with_mesh(stream_layouts, mesh);
    let inputs = InputAssembly::new();
    inputs.configure_and_bind(vs_inputs, &draw_data);

    #[allow(dead_code)]
    struct ObjectUniformData {
        mat: Matrix4,
        color: Vector4,
    }
    let object_data = ObjectUniformData { mat: Matrix4::from_axis_angle(&Vector3::x_axis(), PI / 2.0), color: Vector4::new(1.0, 0.0, 0.0, 0.5) };
    let mut buffer_id = 0;
    unsafe {
        gl::CreateBuffers(1, &mut buffer_id);
        gl::NamedBufferStorage(buffer_id, size_of::<ObjectUniformData>() as isize, (&object_data as *const ObjectUniformData).cast(), 0);
    }
    let xform_buffer = BufferView { buffer_id, offset: 0 };
    ObjectData { draw_data, xform: object_data.mat, xform_buffer, inputs }
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
        gl::ClearColor(data.color[0], data.color[1], data.color[2], data.color[3]);
        gl::ClearDepthf(1.0);
        gl::Disable(gl::CULL_FACE);
        gl::Disable(gl::DEPTH_TEST);
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

        data.program = ShaderProgram::from_files("src/bin/oit-ppll/basic.vs.glsl", "src/bin/oit-ppll/basic.fs.glsl").unwrap();

        let gltf_doc = Gltf::open("glTF/Duck.gltf").expect("Could not load gltf file");
        let mut gltf_data = Vec::new();
        File::open("glTF/Duck0.bin").expect("Could not load gltf bin file").read_to_end(&mut gltf_data).expect("read error"); 
        if let Some(objects) = Mesh::from_gltf(&gltf_doc, &gltf_data, "LOD3spShape") {
            for object in &objects {
                println!("adding object: {} vert.", object.positions.len());
                data.objects.push(create_object(object, &data.program.get_vertex_shader_inputs()));
            }
        }

 //       let vs_inputs = data.program.get_vertex_shader_inputs();
    
        gl::CreateBuffers(1, &mut data.view_data_buffer as *mut _);
        let scene_size = 50.0;
        data.current_view_point = (PI/2.0, 0.0, 4.0 * scene_size);
        data.view_matrices[0] = Matrix4::look_at_rh(&Point3::new(data.current_view_point.2, 0.0, 0.0), &Point3::origin(), &Vector3::z_axis());
        data.view_matrices[1] = Matrix4::new_perspective(1.0, 45.0_f32.to_radians(), 0.1, 1e3);
        gl::NamedBufferStorage(data.view_data_buffer, 2 * size_of::<Matrix4>() as isize, data.view_matrices.as_ptr().cast(), gl::DYNAMIC_STORAGE_BIT);

        let index = gl::GetUniformBlockIndex(data.program.id, "ViewMatrices\0".as_ptr().cast());
        gl::UniformBlockBinding(data.program.id, index, 0);
        let index = gl::GetUniformBlockIndex(data.program.id, "ObjectMatrix\0".as_ptr().cast());
        gl::UniformBlockBinding(data.program.id, index, 1);
    }
}

fn render(data_rc: Rc::<RefCell::<AppData>>) {
    unsafe {
        let data = data_rc.borrow();

        gl::BeginQuery(gl::TIME_ELAPSED, data.timer_query);

        gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
      
        // gtk seems to interact with this setting
        gl::Disable(gl::DEPTH_TEST);

        data.program.activate();

        gl::BindBufferBase(gl::UNIFORM_BUFFER, 0, data.view_data_buffer);

        for object in &data.objects {
            object.inputs.activate();
            gl::BindBufferRange(gl::UNIFORM_BUFFER, 1, object.xform_buffer.buffer_id, object.xform_buffer.offset as isize, size_of::<[Matrix4; 2]>() as isize);
            gl::DrawElements(gl::TRIANGLES, object.draw_data.num_elems, gl::UNSIGNED_INT, std::ptr::null());
        }

        gl::EndQuery(gl::TIME_ELAPSED);

        let mut elapsed: u64 = 0;
        gl::GetQueryObjectui64v(data.timer_query, gl::QUERY_RESULT, &mut elapsed);

        glib::idle_add_local_once(glib::clone!(@strong data_rc =>
            move  || {
                let mut data = data_rc.borrow_mut();
                if !data.fps_label.is_realized() {
                    return;
                }
                
                let last_ms = ((elapsed as f64) * 1e-6) as f32;
                let history = &mut data.frame_time_history;
                history.push_front(last_ms);
                if history.len() > 30 {
                    history.pop_back();
                }
                let mean_ms = history.iter().fold(0.0, |s, x| { s + x }) / history.len() as f32;
                let mean_fps = (mean_ms * 1e-3).recip();
                data.fps_label.set_label(std::format!("{:.3} ms, {:.0} fps", mean_ms, mean_fps).as_str());
        }));
    }
}

