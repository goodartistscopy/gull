use gtk::prelude::*;
use gtk::{Application, ApplicationWindow, GLArea, Inhibit};
use gtk::gdk::GLContext;

const APP_ID: &str = "org.gtk_rs.HelloWorld1";

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

    let app = Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_ui);
    app.run();
}

fn build_ui(app: &Application) {
    let window = ApplicationWindow::builder()
        .application(app)
        .title("Gull")
        .build();

    let gl_canvas = GLArea::builder()
        .auto_render(false)
        .build();
    gl_canvas.set_required_version(4, 6);
    gl_canvas.connect_render(render);

    window.set_child(Some(&gl_canvas));
    window.present();
}

fn render(_canvas: &GLArea, context: &GLContext) -> Inhibit {
    unsafe {
        gl::ClearColor(1.0, 0.0, 0.0, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);
    }
    let (major, minor) = context.version();
    println!("OpenGL version: {}.{}", major, minor);
    Inhibit(true)
}
