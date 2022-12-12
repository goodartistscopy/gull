#version 450

layout(r32ui, location=0, binding=0) uniform uimage2D listHeads;

struct Fragment {
    vec4 color;
    float depth;
    uint next;
};

layout(std430, binding = 0) readonly buffer FragmentStore {
    uint maxNumFragments;
    Fragment fragments[];
};

layout(location = 0) out vec4 color;

const int MAX_NUM_LAYERS = 5;

void insertSorted(inout Fragment array[MAX_NUM_LAYERS], in Fragment frag) {
    int i = 0;
    while (i < array.length() && array[i].depth < frag.depth) {
        i++;
    }

    if (i < array.length()) {
        for (int j = array.length() - 1; j > i; j--) {
            array[j] = array[j - 1];
        }

        array[i] = frag;
        array[i].next = 1; // mark the layer as present
    }
}

vec4 compositeBackToFront(in Fragment array[MAX_NUM_LAYERS]) {
    vec4 composite = vec4(0.0);

    for (int i = array.length() - 1; i >= 0; i--) {
        if (array[i].next == 0) {
            continue;
        }
        vec4 color = array[i].color;
        composite.rgb = color.a * color.rgb + (1.0 - color.a) * composite.a * composite.rgb;
        composite.a = color.a + composite.a - color.a * composite.a;
    }

    return composite;
}

vec4 compositeFrontToBack(in Fragment array[MAX_NUM_LAYERS]) {
    vec4 composite = vec4(0.0);

    for (int i = 0; i < array.length(); i++) {
        if (array[i].next == 0) {
            break;
        }
        vec4 color = array[i].color;
        composite.rgb = color.a * color.rgb + (1.0 - color.a) * composite.a * composite.rgb;
        composite.a = color.a + (1.0 - color.a) * composite.a;
    }

    return composite;
}

Fragment sortedFragments[MAX_NUM_LAYERS];

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    uint head = imageLoad(listHeads, coord).x;
    if (head == 0) {
        discard;
    }

    for (int i = 0; i < MAX_NUM_LAYERS; ++i) {
        sortedFragments[i].color = vec4(0.0);
        sortedFragments[i].depth = 1.0;
        sortedFragments[i].next = 0; // used as marker here
    }

    uint itemAddr = head;
    while (itemAddr != 0) {
        Fragment frag = fragments[itemAddr];
        insertSorted(sortedFragments, frag);
        itemAddr = frag.next;
    }

    color = compositeBackToFront(sortedFragments);
//    color = compositeFrontToBack(sortedFragments);
}

