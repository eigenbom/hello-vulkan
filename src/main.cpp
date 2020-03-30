#include <fmt/core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

template <class... Args> void log_error(std::string_view format_string, Args &&... args)
{
    std::cerr << "[Error] " << fmt::format(format_string, std::forward<Args>(args)...) << std::endl;
}

template <class... Args> void log_info(std::string_view format_string, Args &&... args)
{
    std::cout << "[Info] " << fmt::format(format_string, std::forward<Args>(args)...) << std::endl;
}

static void error_callback(int error, const char *description)
{
    log_error(description);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

class Application
{
  private:
    const int width_     = 800;
    const int height_    = 600;
    GLFWwindow *window_  = nullptr;
    VkInstance instance_ = nullptr;

  public:
    void run()
    {
        init_window();
        init_vulkan();
        main_loop();
        cleanup();
    }

  private:
    void init_window()
    {
        log_info("Initialising GLFW version \"{}\"", glfwGetVersionString());
        if (!glfwInit())
        {
            throw std::runtime_error("Couldn't initialise GLFW");
        }
        log_info("Initialised GLFW");
        glfwSetErrorCallback(error_callback);

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ = glfwCreateWindow(width_, height_, "Hello Vulkan", nullptr, nullptr);
        if (!window_)
        {
            throw std::runtime_error("Window or OpenGL context creation failed");
        }
        glfwSetKeyCallback(window_, key_callback);
        log_info("Created window");
    }

    void init_vulkan()
    {
        createInstance();
    }

    void main_loop()
    {
        // HWND windowHandle = glfwGetWin32Window(window_);
        while (!glfwWindowShouldClose(window_))
        {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        vkDestroyInstance(instance_, nullptr);
        instance_ = nullptr;
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
    }

    void createInstance()
    {
        VkApplicationInfo app_info{};
        app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName   = "Hello Vulkan";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName        = "No Engine";
        app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion         = VK_API_VERSION_1_0;

        uint32_t required_extensions_count = 0;
        const char **required_extensions =
            glfwGetRequiredInstanceExtensions(&required_extensions_count);

        VkInstanceCreateInfo create_info{};
        create_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo        = &app_info;
        create_info.enabledExtensionCount   = required_extensions_count;
        create_info.ppEnabledExtensionNames = required_extensions;
        create_info.enabledLayerCount       = 0;

        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions (extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
        log_info("Found extensions.");
        for (const auto &extension : extensions)
        {
            log_info("    {}", extension.extensionName);
        }

        if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }
        log_info("Created vulkan instance");
    }
};

int main(int argc, char *argv[])
{
    Application application;
    try
    {
        application.run();
    }
    catch (const std::exception &e)
    {
        log_error(e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
