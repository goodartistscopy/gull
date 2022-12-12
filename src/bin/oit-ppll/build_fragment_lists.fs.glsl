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

layout(r32ui, location = 0, binding = 0) uniform uimage2D listHeads;

layout(binding = 0) uniform atomic_uint nextFragmentAddr;

struct Fragment {
    vec4 color;
    float depth;
    uint next;
};

layout(std430, binding = 0) buffer FragmentStore {
    uint maxNumFragments;
    Fragment fragments[];
};

const vec3 g_LightDirection = vec3(0.0, 0.0, 1.0);

void main() {
    float lambert = clamp(dot(g_LightDirection, normalize(worldNormal)), 0.0, 1.0);
    vec4 fragColor = vec4(lambert * object.color.rgb, object.color.a);

    ivec2 coord = ivec2(gl_FragCoord.xy);

    uint newFragmentAddr = atomicCounterIncrement(nextFragmentAddr);

    if (newFragmentAddr < maxNumFragments) {
#if 0
        // /!\ incorrect critical section
        uint headAddr = imageLoad(listHeads, coord).x;
        fragments[newFragmentAddr]= Fragment(fragColor, gl_FragCoord.z, headAddr);
        imageStore(listHeads, coord, uvec4(newFragmentAddr));
#else
        uint headAddr = imageAtomicExchange(listHeads, coord, newFragmentAddr);
        fragments[newFragmentAddr] = Fragment(fragColor, gl_FragCoord.z, headAddr);
#endif
    }
}
