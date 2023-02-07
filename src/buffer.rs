use std::collections::HashMap;
use std::rc::Rc;
use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::borrow::Borrow;

use gl::types::GLenum;

struct BufferManager {
    bakings: HashMap<*const u8, BufferId> 
}

impl BufferManager {
    fn new() -> BufferManager {
        BufferManager { bakings: HashMap::new() }
    }

    pub fn r#for(&mut self, addr: &u8) -> Buffer {
        let id = self.bakings.entry(addr).or_insert_with(|| {
            BufferId::new()
        });
        Buffer { id: Rc::new(RefCell::new(id.clone())), offset: 0 }
    }
}

#[derive(Clone, Default)]
struct BufferId(u32);

impl BufferId {
    fn new() -> BufferId {
        unsafe {
            let mut id = 0;
            gl::CreateBuffers(1, &mut id);
            println!("Creating buffer: {}", id);
            BufferId(id)
        }
    }
}

impl Drop for BufferId {
    fn drop(&mut self) {
        unsafe {
            println!("Dropping buffer: {}", self.0);
            assert!(self.0 == 0 || gl::IsBuffer(self.0) == gl::TRUE);
            gl::DeleteBuffers(1, &self.0);
        }
    }
}

impl Into<u32> for &BufferId {
    fn into(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Default)]
pub struct Buffer {
    id: Rc<RefCell<BufferId>>,
    offset: isize,
}

impl Buffer {
    pub fn new(size: usize) -> Buffer {
        let id = BufferId::new();
        unsafe {
            gl::NamedBufferStorage(id.0, size as isize, std::ptr::null(), gl::DYNAMIC_STORAGE_BIT | gl::MAP_WRITE_BIT);
        }
        Buffer { id: Rc::new(RefCell::new(id)), offset: 0 }
    }

    pub fn with_data(data: &[u8]) -> Buffer {
        let id = BufferId::new();
        unsafe {
            gl::NamedBufferStorage(id.0, data.len() as isize, data.as_ptr().cast(), gl::DYNAMIC_STORAGE_BIT | gl::MAP_WRITE_BIT);
        }
        Buffer { id: Rc::new(RefCell::new(id)), offset: 0 }
    }
    // pub fn allocate(&self, size: isize) {
    //     unsafe {
    //         gl::NamedBufferStorage(self.id(), size, std::ptr::null(), gl::DYNAMIC_STORAGE_BIT);
    //     }
    // }

    pub fn id(&self) -> u32 {
        (&(*self.id).borrow() as &BufferId).into()
    }

    pub fn offset(&self) -> isize {
        self.offset
    }

    pub fn bind(&self, target: GLenum) {
        unsafe {
            gl::BindBuffer(target, self.id());
        }
    }

    pub fn view(&self, offset: isize) -> Buffer {
        Buffer { id: self.id.clone(), offset }
    }
}
