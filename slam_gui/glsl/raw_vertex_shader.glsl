#version 330 

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;

out vec4 fragment_colour;

void main() {
	
	gl_Position = vec4(vertex_position, 1.0);
	fragment_colour = vec4(vertex_colour, 1.0);
} 