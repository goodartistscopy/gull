#version 450

in VertexData
{
    vec3 worldPosition;
    vec3 worldNormal;
};

layout(std140, binding = 1)
uniform ObjectData {
    mat4 modelMat;
    vec4 color;
} object;

layout(r32ui, location = 0, binding = 0) uniform uimage2D list_heads;

layout(binding = 0) uniform atomic_uint next_fragment;

struct Fragment {
    vec4 color;
    float depth;
    uint next;
};

layout(std430, binding = 0) buffer FragmentStore {
    uint max_num_fragments;
    Fragment fragments[];
};

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

//layout(location = 0) out vec4 color;

void main() {
    vec3 l = LIGHT;
    float lambert = clamp(dot(l, normalize(worldNormal)), 0.0, 1.0);
    vec4 frag_color = vec4(lambert * object.color.rgb, object.color.a);

    ivec2 coord = ivec2(gl_FragCoord.xy);

    uint new_fragment_addr = atomicCounterIncrement(next_fragment);

    if (new_fragment_addr < max_num_fragments) {
#if 0
        uint current_head = imageLoad(list_heads, coord);
        fragments[new_fragment_addr]= Fragment(frag_color, gl_FragCoord.z, current_head);
        imageStore(list_heads, coord, new_fragment_addr);
#else
        uint current_head = imageAtomicExchange(list_heads, coord, new_fragment_addr);
        fragments[new_fragment_addr] = Fragment(frag_color, gl_FragCoord.z, current_head);
#endif
    }
}
