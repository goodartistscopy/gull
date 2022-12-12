#version 450

const vec4 g_ScreenTriangleVertices[3] = {
    vec4(-1.0, -1.0, 0.0, 1.0),
    vec4(3.0, -1.0, 0.0, 1.0),
    vec4(-1.0, 3.0, 0.0, 1.0)
};

void main() {
    gl_Position = g_ScreenTriangleVertices[gl_VertexID];
}
