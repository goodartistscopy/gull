#version 450

const vec4 g_screen_triangle_verts[3] = {
    vec4(-1.0, -1.0, 0.0, 1.0),
    vec4(3.0, -1.0, 0.0, 1.0),
    vec4(-1.0, 3.0, 0.0, 1.0)
};

void main() {
    gl_Position = g_screen_triangle_verts[gl_VertexID];
}
