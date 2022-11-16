use gtk::prelude::*;
use gtk::{Application, ApplicationWindow, GLArea, Inhibit, Widget, Window};
use gtk::gdk::{GLContext, GLError};
use gtk::gdk;
use gtk::glib::error::Error;
use std::ffi::{CStr, c_char};

const APP_ID: &str = "goodartistscopy.Gull";

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

    gl_canvas.connect_create_context(|canvas| {
        let context = gdk::Display::default().and_then(|display| { display.create_gl_context().ok() })?;
        
        context.set_required_version(4, 6);
        context.set_forward_compatible(true);
        context.set_debug_enabled(true);

        if let Err(error) = context.realize()
        {
            canvas.set_error(Some(&Error::new(GLError::NotAvailable, error.message())));
            return None;
        }

        context.make_current();
        let (major, minor) = context.version();
        let vendor = unsafe { CStr::from_ptr(gl::GetString(gl::VENDOR) as *const c_char) };
        let renderer = unsafe { CStr::from_ptr(gl::GetString(gl::RENDERER) as *const c_char) }; 
        println!("OpenGL version: {}.{} (forward compatible:{}, debug:{})", major, minor, context.is_forward_compatible(), context.is_debug_enabled());
        println!("Vendor: {:?}\nRenderer: {:?}", vendor, renderer);

        Some(context)
    });

    gl_canvas.connect_render(render);

    window.set_child(Some(&gl_canvas));
    window.present();
}

fn render(_canvas: &GLArea, _context: &GLContext) -> Inhibit {
    unsafe {
        gl::ClearColor(1.0, 0.0, 0.0, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);
    }

    Inhibit(true)
}
