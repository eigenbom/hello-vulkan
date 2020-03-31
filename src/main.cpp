#include <fmt/core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

enum class BuildMode
{
    Release,
    Debug
};

struct BuildConfig
{
    BuildMode mode;
};

#ifdef NDEBUG
const BuildConfig gBuildConfig = {BuildMode::Release};
#else
const BuildConfig gBuildConfig = {BuildMode::Debug};
#endif

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
    const int width_                  = 800;
    const int height_                 = 600;
    GLFWwindow *window_               = nullptr;
    VkInstance instance_              = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = nullptr;
    VkDevice device_                  = nullptr;
    VkQueue graphics_queue_           = nullptr;
    VkQueue present_queue_            = nullptr;
    VkSurfaceKHR surface_             = VK_NULL_HANDLE;

    const bool enable_validation_                      = (gBuildConfig.mode == BuildMode::Debug);
    const std::vector<const char *> validation_layers_ = {"VK_LAYER_KHRONOS_validation"};

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
        create_instance();
        setup_debug_messenger();
        create_surface();
        pick_physical_device();
        create_logical_device();
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
        vkDestroyDevice(device_, nullptr);
        device_ = nullptr;
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
        vkDestroyInstance(instance_, nullptr);
        instance_ = nullptr;
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
    }

    void create_instance()
    {
        if (enable_validation_ && !check_validation_layer_support())
        {
            throw std::runtime_error("Validation layers not available!");
        }

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
        create_info.enabledLayerCount =
            enable_validation_ ? static_cast<uint32_t>(validation_layers_.size()) : 0;
        create_info.ppEnabledLayerNames = enable_validation_ ? validation_layers_.data() : nullptr;

        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
        log_info("Found extensions");
        for (const auto &extension : extensions)
        {
            log_info("    {}", extension.extensionName);
        }

        if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create instance");
        }
        log_info("Created Vulkan instance");
    }

    bool check_validation_layer_support() const
    {
        uint32_t layer_count;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> available_layers(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
        return std::find_if(validation_layers_.begin(), validation_layers_.end(),
                            [available_layers](auto validation_layer) {
                                return std::find_if(
                                           available_layers.begin(), available_layers.end(),
                                           [validation_layer](auto available_layer) {
                                               return strcmp(validation_layer,
                                                             available_layer.layerName) == 0;
                                           }) != available_layers.end();
                            }) != validation_layers_.end();
    }

    void setup_debug_messenger()
    {
        // TODO: This
    }

    void create_surface()
    {
        if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create window surface");
        }
    }

    void pick_physical_device()
    {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
        if (device_count == 0)
        {
            throw std::runtime_error("No Vulkan-compatible devices found");
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
        const auto it = std::find_if(devices.begin(), devices.end(),
                                     [this](auto d) { return is_device_suitable(d); });
        if (it == devices.end())
        {
            throw std::runtime_error("No suitable Vulkan-compatible devices found");
        };
        physical_device_ = *it;
    }

    void create_logical_device()
    {
        QueueFamilyIndices indices = find_queue_families(physical_device_);

        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
        std::set<uint32_t> unique_queue_families = {
            indices.graphics_family.value(),
            indices.present_family.value(),
        };

        const float queue_priority = 1.0f; // Needs lifetime beyond this loop
        for (uint32_t family : unique_queue_families)
        {
            VkDeviceQueueCreateInfo info = {};
            info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            info.queueFamilyIndex        = family;
            info.queueCount              = 1;
            info.pQueuePriorities        = &queue_priority;
            queue_create_infos.push_back(info);
        }

        VkPhysicalDeviceFeatures device_features = {};
        VkDeviceCreateInfo create_info           = {};
        create_info.sType                        = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.pQueueCreateInfos            = queue_create_infos.data();
        create_info.queueCreateInfoCount         = static_cast<uint32_t>(queue_create_infos.size());
        create_info.pEnabledFeatures             = &device_features;
        create_info.enabledExtensionCount        = 0;
        create_info.enabledLayerCount =
            enable_validation_ ? static_cast<uint32_t>(validation_layers_.size()) : 0;
        create_info.ppEnabledLayerNames = enable_validation_ ? validation_layers_.data() : nullptr;

        if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create logical device");
        }
        log_info("Created device");

        vkGetDeviceQueue(device_, indices.graphics_family.value(), 0, &graphics_queue_);
        vkGetDeviceQueue(device_, indices.present_family.value(), 0, &present_queue_);
    }

    bool is_device_suitable(VkPhysicalDevice device) const
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        QueueFamilyIndices indices = find_queue_families(device);
        return indices.is_complete() &&
               properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    }

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphics_family;
        std::optional<uint32_t> present_family;

        bool is_complete() const
        {
            return graphics_family.has_value() && present_family.has_value();
        }
    };

    QueueFamilyIndices find_queue_families(VkPhysicalDevice device) const
    {
        QueueFamilyIndices indices;

        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);

        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());

        int i = 0;
        for (const auto &family : families)
        {
            if (family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphics_family = i;
            }

            VkBool32 present_support = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_support);
            if (present_support)
            {
                indices.present_family = i;
            }

            if (indices.is_complete())
            {
                break;
            }
            i++;
        }

        return indices;
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
