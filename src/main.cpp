#include <cstdlib>
#include <iostream>
#include <string_view>
#include <fmt/core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

template<class... Args>
void log_error(std::string_view formatString, Args&&... args)
{
	std::cerr << "[Error] " << fmt::format(formatString, std::forward<Args>(args)...) << std::endl;
}

template<class... Args>
void log_info(std::string_view formatString, Args&&... args)
{
	std::cout << "[Info] " << fmt::format(formatString, std::forward<Args>(args)...) << std::endl;
}

static void errorCallback(int error, const char* description)
{
	log_error(description);
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}

class Application {
private:
	const int width_ = 800;
	const int height_ = 600;
	GLFWwindow* window_ = nullptr;

public:
	void run() 
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	void initWindow()
	{
		log_info("Initialising GLFW version \"{}\"", glfwGetVersionString());
		if (!glfwInit())
		{
			throw std::runtime_error("Couldn't initialise GLFW");
		}
		log_info("Initialised GLFW");
		glfwSetErrorCallback(errorCallback);

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window_ = glfwCreateWindow(640, 480, "Hello Vulkan", nullptr, nullptr);
		if (!window_)
		{
			throw std::runtime_error("Window or OpenGL context creation failed");
		}
		glfwSetKeyCallback(window_, keyCallback);
		log_info("Created window");
	}

	void initVulkan()
	{
		// TODO:
	}

	void mainLoop()
	{
		// HWND windowHandle = glfwGetWin32Window(window_);
		while (!glfwWindowShouldClose(window_))
		{
			glfwPollEvents();
		}
	}

	void cleanup()
	{
		glfwDestroyWindow(window_);
		glfwTerminate();
	}
};

int main(int argc, char* argv[])
{
	Application application;
	try {
		application.run();
	}
	catch (const std::exception& e) {
		log_error(e.what());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
