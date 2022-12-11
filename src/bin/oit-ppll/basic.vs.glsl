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
    vec4 color;
} object;

out VertexData
{
    vec3 worldPosition;
    vec3 worldNormal;
};

void main() {
    worldPosition = (object.modelMat * position).xyz;
    worldNormal = inverse(transpose(mat3(object.modelMat))) * normal;

    gl_Position = projMat * viewMat * vec4(worldPosition, 1.0);
}
