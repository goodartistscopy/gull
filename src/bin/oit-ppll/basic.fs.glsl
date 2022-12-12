#version 450

in VertexData
{
    vec3 worldPosition;
    vec3 worldNormal;
};

layout(std140, binding=1)
uniform ObjectData {
    mat4 modelMat;
    vec4 color;
} object;

layout(location = 0) out vec4 color;

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

void main() {
    vec3 l = LIGHT; //normalize(LIGHT - worldPosition);
    float lambert = clamp(dot(l, normalize(worldNormal)), 0.0, 1.0);
    color = vec4(lambert * object.color.rgb, object.color.a);
}
