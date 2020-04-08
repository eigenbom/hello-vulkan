#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D tex_sampler;

layout(location = 0) in vec3 frag_colour;
layout(location = 1) in vec2 frag_tex_coord;
layout(location = 2) in float frag_height;

layout(location = 0) out vec4 out_colour;

const vec3 colour_sky = vec3 (0.537, 0.671, 0.847);

void main() {
    vec4 colour = vec4(frag_colour * texture(tex_sampler, frag_tex_coord).rgb, 1.0);
    float depth = clamp(0.1 * gl_FragCoord.z / gl_FragCoord.w, 0, 1);
    float fog = 0.5 * mix(0, clamp(mix(-1.0, 2.0, clamp(1.0 - exp(-depth * 2), 0, 1)), 0, 1), clamp(1 - frag_height, 0, 1));
    colour = mix(colour, vec4(colour_sky, 1), fog);
    out_colour = colour;
}
