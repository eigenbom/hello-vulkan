#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float time;
} ubo;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_colour;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_tex_coord;

layout(location = 0) out vec3 frag_colour;
layout(location = 1) out vec2 frag_tex_coord;
layout(location = 2) out float frag_height;

const vec3 colour_sun = vec3 (1, 0.894, 0.518);
const vec3 colour_sky = vec3 (0.537, 0.671, 0.847);
const int num_lights = 5; 
struct Light {
    vec3 position;
    vec3 colour;
};

const Light lights[num_lights] = {    
    { vec3(90, 100, 90), colour_sun * 2.0 },
    { vec3(-100, 100, 0), colour_sky * 0.1 },
    { vec3(100, 100, 0), colour_sky * 0.1 },
    { vec3(0, 100, 100), colour_sky * 0.1 },
    { vec3(0, 100, -100), colour_sky * 0.1},
};

mat4 rotate_around_y( in float angle ) {
	return mat4(cos(angle), 0, sin(angle), 0, 0, 1, 0, 0, -sin(angle), 0, cos(angle), 0, 0, 0, 0, 1);
}

void main() {
    vec3 mv_position = (ubo.view * ubo.model * vec4(in_position, 1.0)).xyz;
    vec3 mv_normal = normalize((transpose(inverse(ubo.view * ubo.model)) * vec4(in_normal, 1)).xyz);
    const vec3 ambient = colour_sky * 0.05;
    vec3 diffuse = vec3(0, 0, 0);
    for (int i=0; i<num_lights; ++i){
        vec4 w_light_coord = vec4(lights[i].position, 1);
        if (i == 0) w_light_coord = rotate_around_y(ubo.time * 1) * w_light_coord;
        vec4 light_coord = (ubo.view * ubo.model * w_light_coord);
        vec3 light_dir = normalize(vec3(light_coord) - mv_position);
        diffuse += lights[i].colour * 0.5 * max(dot(mv_normal, light_dir), 0.0);
    }
    vec3 colour = ambient * in_colour + diffuse * in_colour;
    vec4 clip_space = ubo.proj * ubo.view * ubo.model * vec4(in_position, 1);
    frag_colour = colour;
    frag_tex_coord = in_tex_coord;
    frag_height = ((ubo.model * vec4(in_position, 1.0)).y + 1) / 2;
    gl_Position = clip_space;
}
