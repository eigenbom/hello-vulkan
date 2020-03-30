#include <cstdlib>
#include <iostream>
#include <string_view>
#include <fmt/core.h>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>


template<class... Args>
void logError(std::string_view formatString, Args&&... args)
{
	std::cerr << "[Error] " << fmt::format(formatString, std::forward<Args>(args)...) << std::endl;
}

template<class... Args>
void logInfo(std::string_view formatString, Args&&... args)
{
	std::cout << "[Info] " << fmt::format(formatString, std::forward<Args>(args)...) << std::endl;
}

static void errorCallback(int error, const char* description)
{
	logError(description);
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}

int main(int argc, char* argv[]) 
{
	logInfo("Initialising GLFW version \"{}\"", glfwGetVersionString());
	if (!glfwInit())
	{
		logError("Couldn't initialise GLFW");
		return EXIT_FAILURE;
	}
	logInfo("Initialised GLFW");
	glfwSetErrorCallback(errorCallback);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* window = glfwCreateWindow(640, 480, "Hello Vulkan", nullptr, nullptr);
	if (!window)
	{
		logError("Window or OpenGL context creation failed");
		return EXIT_FAILURE;
	}
	glfwSetKeyCallback(window, keyCallback);
	logInfo("Created window");

	HWND windowHandle = glfwGetWin32Window(window);
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}

	glfwDestroyWindow(window);

	logInfo("Success");
	glfwTerminate();
	return EXIT_SUCCESS;
}
