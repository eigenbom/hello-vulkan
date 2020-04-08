# Hello Vulkan

A simple Hello World application in Vulkan.

## Description

This code demonstrates Vulkan initialisation and displays a 3D object with simple lighting. It is based heavily on the [Vulkan Tutorial](https://vulkan-tutorial.com/) and most of it I  copied verbatim. 

![alt text](https://github.com/eigenbom/hello-vulkan/raw/master/www/screenshot.png "A screenshot")

## Dependencies

The code was written in C++20 using Visual Studio 2019.

A project file is provided for Visual Studio 2019. You'll need to download the [Vulkan SDK](https://vulkan.lunarg.com/) and set the correct path within Visual Studio.

Some other third party dependencies are used, but already included in the repository. These are [GLFW](https://www.glfw.org/), [GLM](https://glm.g-truc.net/), [GSL](https://github.com/microsoft/GSL), [fmtlib](https://github.com/fmtlib/fmt), [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader), and the [stb libraries](https://github.com/nothings/stb).

## Acknowledgments

* [Vulkan Tutorial](https://vulkan-tutorial.com/) 
* [Lighthouse Model](https://sketchfab.com/3d-models/the-lighthouse-1a85945dd2a840f594bf6cb003176a54) by Cotman Sam (Licensed under CC BY 4.0).
