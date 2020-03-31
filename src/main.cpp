#include <fmt/core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define NOMINMAX
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <cassert>
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
    VkSwapchainKHR swap_chain_        = VK_NULL_HANDLE;
    VkFormat swap_chain_image_format_ = {};
    VkExtent2D swap_chain_extent_     = {};
    std::vector<VkImage> swap_chain_images_;
    std::vector<VkImageView> swap_chain_image_views_;

    const bool enable_validation_                      = (gBuildConfig.mode == BuildMode::Debug);
    const std::vector<const char *> validation_layers_ = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char *> device_extensions_ = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

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
        create_swap_chain();
        create_image_views();
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
        for (auto view : swap_chain_image_views_)
        {        
            vkDestroyImageView(device_, view, nullptr);
        }
        swap_chain_image_views_.clear();
        vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
        swap_chain_ = VK_NULL_HANDLE;
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
        create_info.enabledExtensionCount        = static_cast<uint32_t>(device_extensions_.size());
        create_info.ppEnabledExtensionNames      = device_extensions_.data();
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

    void create_swap_chain()
    {
        SwapChainSupportDetails swap_chain_support = query_swap_chain_support(physical_device_);

        VkSurfaceFormatKHR surface_format = choose_swap_surface_format(swap_chain_support.formats);
        VkPresentModeKHR present_mode = choose_swap_present_mode(swap_chain_support.present_modes);
        VkExtent2D extent             = choose_swap_extent(swap_chain_support.capabilities);

        uint32_t min_images = swap_chain_support.capabilities.minImageCount;
        uint32_t max_images = swap_chain_support.capabilities.maxImageCount;
        uint32_t image_count =
            max_images == 0 ? (min_images + 1) : std::min(min_images + 1, max_images);

        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface          = surface_;
        create_info.minImageCount    = min_images;
        create_info.imageFormat      = surface_format.format;
        create_info.imageColorSpace  = surface_format.colorSpace;
        create_info.imageExtent      = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices      = find_queue_families(physical_device_);
        uint32_t queue_family_indices[] = {indices.graphics_family.value(),
                                           indices.present_family.value()};
        if (indices.graphics_family != indices.present_family)
        {
            create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices   = queue_family_indices;
        }
        else
        {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        create_info.preTransform   = swap_chain_support.capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode    = present_mode;
        create_info.clipped        = VK_TRUE;
        create_info.oldSwapchain   = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swap chain");
        }

        uint32_t image_count;
        vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, nullptr);
        swap_chain_images_.resize(image_count);
        vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, swap_chain_images_.data());

        swap_chain_image_format_ = surface_format.format;
        swap_chain_extent_       = extent;
    }

    void create_image_views()
    {
        swap_chain_image_views_.resize(swap_chain_images_.size());
        for (std::size_t i = 0; i < swap_chain_images_.size(); ++i)
        {
            VkImageViewCreateInfo create_info{};
            create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            create_info.image                           = swap_chain_images_[i];
            create_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            create_info.format                          = swap_chain_image_format_;
            create_info.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            create_info.subresourceRange.baseMipLevel   = 0;
            create_info.subresourceRange.levelCount     = 1;
            create_info.subresourceRange.baseArrayLayer = 0;
            create_info.subresourceRange.layerCount     = 1;

            if (vkCreateImageView(device_, &create_info, nullptr, &swap_chain_image_views_[i]) !=
                VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image views");            
            }
        }
    }

    bool is_device_suitable(VkPhysicalDevice device) const
    {
        // VkPhysicalDeviceProperties properties;
        // vkGetPhysicalDeviceProperties(device, &properties);
        const bool extensions_supported = check_device_extension_support(device);
        const bool swap_chain_adequate  = [this, device, extensions_supported]() {
            if (!extensions_supported)
                return false;
            const auto swap_details = query_swap_chain_support(device);
            return !swap_details.formats.empty() && !swap_details.present_modes.empty();
        }();
        QueueFamilyIndices indices = find_queue_families(device);
        return indices.is_complete() && extensions_supported && swap_chain_adequate;
        // properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    }

    bool check_device_extension_support(VkPhysicalDevice device) const
    {
        uint32_t extension_count;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> available_extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                             available_extensions.data());
        std::set<std::string> required_extensions{device_extensions_.begin(),
                                                  device_extensions_.end()};
        for (const auto &ext : available_extensions)
        {
            required_extensions.erase(ext.extensionName);
        }
        return required_extensions.empty();
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

    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> present_modes;
    };

    SwapChainSupportDetails query_swap_chain_support(VkPhysicalDevice device) const
    {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);
        uint32_t format_count;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, nullptr);
        if (format_count != 0)
        {
            details.formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count,
                                                 details.formats.data());
        }

        uint32_t present_mode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, nullptr);
        if (present_mode_count != 0)
        {
            details.present_modes.resize(present_mode_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count,
                                                      details.present_modes.data());
        }
        return details;
    }

    VkSurfaceFormatKHR choose_swap_surface_format(
        const std::vector<VkSurfaceFormatKHR> &available_formats)
    {
        assert(!available_formats.empty());

        for (const auto &format : available_formats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }
        return available_formats[0];
    }

    VkPresentModeKHR choose_swap_present_mode(
        const std::vector<VkPresentModeKHR> &available_present_modes) const
    {
        for (const auto &mode : available_present_modes)
        {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return mode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR &capabilities) const
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        else
        {
            VkExtent2D actual_extent = {static_cast<uint32_t>(width_),
                                        static_cast<uint32_t>(height_)};
            actual_extent.width =
                std::max(capabilities.minImageExtent.width,
                         std::min(capabilities.maxImageExtent.width, actual_extent.width));
            actual_extent.height =
                std::max(capabilities.minImageExtent.height,
                         std::min(capabilities.maxImageExtent.height, actual_extent.height));
            return actual_extent;
        }
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
