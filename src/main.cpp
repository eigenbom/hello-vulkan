// Hello Vulkan
// Benjamin Porter, 2020
//
// Code adapted from vulkan-tutorial.com

#define NOMINMAX
#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#define GLM_FORCE_LEFT_HANDED

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <fmt/core.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gsl/gsl>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using glm::vec2, glm::vec3, glm::vec4, glm::mat4;

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
constexpr BuildConfig gBuildConfig = {BuildMode::Release};
#else
constexpr BuildConfig gBuildConfig = {BuildMode::Debug};
#endif

template <class... Args>
void log_error(std::string_view format_string, Args &&... args)
{
    std::cerr << "[Error] "
              << fmt::format(format_string, std::forward<Args>(args)...)
              << std::endl;
}

template <class... Args>
void log(std::string_view header, std::string_view format_string,
         Args &&... args)
{
    std::cout << "[" << header << "] "
              << fmt::format(format_string, std::forward<Args>(args)...)
              << std::endl;
}

template <class... Args>
void log_info(std::string_view format_string, Args &&... args)
{
    log("Info", format_string, std::forward<Args>(args)...);
}

template <class... Args>
void log_warn(std::string_view format_string, Args &&... args)
{
    log("Warning", format_string, std::forward<Args>(args)...);
}

// Explicitly loaded extension
[[gsl::suppress(26490)]] // Don't warn about the reinterpret_cast below
static VkResult
CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) noexcept
{
    const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// Explicitly loaded extension
[[gsl::suppress(26490)]] // Don't warn about the reinterpret_cast below
static void
DestroyDebugUtilsMessengerEXT(VkInstance instance,
                              VkDebugUtilsMessengerEXT debugMessenger,
                              const VkAllocationCallbacks *pAllocator) noexcept
{
    const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

struct Vertex
{
    vec3 pos;
    vec3 colour;

    static constexpr VkVertexInputBindingDescription
    get_binding_description() noexcept
    {
        return {.binding   = 0,
                .stride    = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    }

    static constexpr std::array<VkVertexInputAttributeDescription, 2>
    get_attribute_descriptions() noexcept
    {
        return {{
            {.location = 0,
             .binding  = 0,
             .format   = VK_FORMAT_R32G32B32_SFLOAT,
             .offset   = offsetof(Vertex, pos)},

            {.location = 1,
             .binding  = 0,
             .format   = VK_FORMAT_R32G32B32_SFLOAT,
             .offset   = offsetof(Vertex, colour)},
        }};
    }
};

struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

static constexpr vec4 rgba_to_vec4(uint32_t rgba) noexcept
{
    return {
        ((rgba & 0xff000000) >> 24) / 255.0f,
        ((rgba & 0x00ff0000) >> 16) / 255.0f,
        ((rgba & 0x0000ff00) >> 8) / 255.0f,
        ((rgba & 0x000000ff) >> 0) / 255.0f,
    };
}

namespace Colours
{

static constexpr vec4 red    = rgba_to_vec4(0xfe4a49ff);
static constexpr vec4 blue   = rgba_to_vec4(0x2ab7caff);
static constexpr vec4 yellow = rgba_to_vec4(0xfed766ff);

static constexpr vec4 light   = rgba_to_vec4(0xe6e6eaff);
static constexpr vec4 lighter = rgba_to_vec4(0xf4f4f8ff);

} // namespace Colours

class Application
{
  private:
    static constexpr int initial_width_        = 800;
    static constexpr int initial_height_       = 800;
    static constexpr int max_frames_in_flight_ = 2;
    static constexpr bool enable_validation_layers_ =
        (gBuildConfig.mode == BuildMode::Debug);
    static constexpr std::array<const char *, 1> validation_layers_ = {
        "VK_LAYER_KHRONOS_validation"};
    static constexpr std::array<const char *, 1> device_extensions_ = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // Two equilateral triangles, the first on top of the second
    static constexpr std::array<Vertex, 6> vertices_ = {{
        {{1.0f, 0.0f, 0.25f}, Colours::red},
        {{-0.5f, 0.86f, 0.25f}, Colours::red},
        {{-0.5f, -0.86f, 0.25f}, Colours::red},

        {{-1.0f, 0.0f, 0.5f}, Colours::blue},
        {{0.5f, -0.86f, 0.5f}, Colours::blue},
        {{0.5f, 0.86f, 0.5f}, Colours::blue},
    }};

    static constexpr std::array<uint16_t, 6> indices_ = {{0, 1, 2, 3, 4, 5}};

    GLFWwindow *window_                                  = nullptr;
    VkInstance instance_                                 = nullptr;
    VkPhysicalDevice physical_device_                    = nullptr;
    VkDevice device_                                     = nullptr;
    VkQueue graphics_queue_                              = nullptr;
    VkQueue present_queue_                               = nullptr;
    VkSurfaceKHR surface_                                = VK_NULL_HANDLE;
    VkSwapchainKHR swap_chain_                           = VK_NULL_HANDLE;
    VkFormat swap_chain_image_format_                    = {};
    VkExtent2D swap_chain_extent_                        = {};
    std::vector<VkImage> swap_chain_images_              = {};
    std::vector<VkImageView> swap_chain_image_views_     = {};
    VkRenderPass render_pass_                            = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_         = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_                    = VK_NULL_HANDLE;
    VkPipeline graphics_pipeline_                        = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> swap_chain_framebuffers_  = {};
    VkCommandPool command_pool_                          = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_        = {};
    std::vector<VkSemaphore> image_available_semaphores_ = {};
    std::vector<VkSemaphore> render_finished_semaphores_ = {};
    std::vector<VkFence> in_flight_fences_               = {};
    std::vector<VkFence> images_in_flight_               = {};
    int current_frame_                                   = 0;
    bool framebuffer_resized_                            = false;
    VkDebugUtilsMessengerEXT debug_messenger_            = VK_NULL_HANDLE;
    VkBuffer vertex_buffer_                              = VK_NULL_HANDLE;
    VkDeviceMemory vertex_buffer_memory_                 = VK_NULL_HANDLE;
    VkBuffer index_buffer_                               = VK_NULL_HANDLE;
    VkDeviceMemory index_buffer_memory_                  = VK_NULL_HANDLE;
    std::vector<VkBuffer> uniform_buffers_               = {};
    std::vector<VkDeviceMemory> uniform_buffers_memory_  = {};
    VkDescriptorPool descriptor_pool_                    = {};
    std::vector<VkDescriptorSet> descriptor_sets_        = {};
    VkSampleCountFlagBits msaa_samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkImage colour_image_               = VK_NULL_HANDLE;
    VkDeviceMemory colour_image_memory_ = VK_NULL_HANDLE;
    VkImageView colour_image_view_      = VK_NULL_HANDLE;
    VkImage depth_image_                = VK_NULL_HANDLE;
    VkDeviceMemory depth_image_memory_  = VK_NULL_HANDLE;
    VkImageView depth_image_view_       = VK_NULL_HANDLE;

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
            throw std::runtime_error("Couldn't initialise GLFW!");
        }
        log_info("Initialised GLFW");
        glfwSetErrorCallback(error_callback);

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(initial_width_, initial_height_,
                                   "Hello Vulkan", nullptr, nullptr);
        if (!window_)
        {
            throw std::runtime_error(
                "Window or OpenGL context creation failed!");
        }
        glfwSetWindowUserPointer(window_, this);
        glfwSetKeyCallback(window_, key_callback);
        glfwSetFramebufferSizeCallback(window_, framebuffer_resize_callback);
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
        create_render_pass();
        create_descriptor_set_layout();
        create_graphics_pipeline();
        create_command_pool();
        create_colour_resources();
        create_depth_resources();
        create_framebuffers();
        create_vertex_buffer();
        create_index_buffer();
        create_uniform_buffers();
        create_descriptor_pool();
        create_descriptor_sets();
        create_command_buffers();
        create_sync_objects();
    }

    void main_loop()
    {
        auto time_start      = std::chrono::steady_clock::now();
        uint32_t frame_count = 0;
        while (!glfwWindowShouldClose(window_))
        {
            glfwPollEvents();
            draw_frame();

            ++frame_count;
            if (frame_count == 10000)
            {
                const auto time_end = std::chrono::steady_clock::now();
                log_info(
                    "Frame time: {} FPS",
                    1000000.0 /
                        (std::chrono::duration_cast<std::chrono::microseconds>(
                             time_end - time_start)
                             .count() /
                         10000));
                time_start  = time_end;
                frame_count = 0;
            }
        }
        vkDeviceWaitIdle(device_);
    }

    void cleanup() noexcept
    {
        cleanup_swap_chain();
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
        vkDestroyBuffer(device_, index_buffer_, nullptr);
        index_buffer_ = VK_NULL_HANDLE;
        vkFreeMemory(device_, index_buffer_memory_, nullptr);
        index_buffer_memory_ = VK_NULL_HANDLE;
        vkDestroyBuffer(device_, vertex_buffer_, nullptr);
        vertex_buffer_ = VK_NULL_HANDLE;
        vkFreeMemory(device_, vertex_buffer_memory_, nullptr);
        vertex_buffer_memory_ = VK_NULL_HANDLE;
        for (auto semaphore : render_finished_semaphores_)
        {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
        render_finished_semaphores_.clear();
        for (auto semaphore : image_available_semaphores_)
        {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
        image_available_semaphores_.clear();
        for (auto fence : in_flight_fences_)
        {
            vkDestroyFence(device_, fence, nullptr);
        }
        in_flight_fences_.clear();
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        device_ = nullptr;
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
        if (enable_validation_layers_)
        {
            DestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
            debug_messenger_ = VK_NULL_HANDLE;
        }
        vkDestroyInstance(instance_, nullptr);
        instance_ = nullptr;
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
    }

    void create_instance()
    {
        if (enable_validation_layers_ && !check_validation_layer_support())
        {
            throw std::runtime_error("Validation layers not available!!");
        }

        const VkApplicationInfo app_info {
            .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName   = "Hello Vulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = VK_API_VERSION_1_0,
        };

        const auto required_extensions = get_required_extensions();
        const auto debug_utils_messenger_info =
            get_debug_utils_messenger_info();
        const VkInstanceCreateInfo create_info {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = enable_validation_layers_ ? &debug_utils_messenger_info
                                               : nullptr,
            .pApplicationInfo = &app_info,
            .enabledLayerCount =
                enable_validation_layers_
                    ? gsl::narrow_cast<uint32_t>(validation_layers_.size())
                    : 0u,
            .ppEnabledLayerNames =
                enable_validation_layers_ ? validation_layers_.data() : nullptr,
            .enabledExtensionCount =
                gsl::narrow_cast<uint32_t>(required_extensions.size()),
            .ppEnabledExtensionNames = required_extensions.data(),
        };

        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                               nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                               extensions.data());
        log_info("Found extensions");
        for (const auto &extension : extensions)
        {
            log_info("    {}", &extension.extensionName[0]);
        }

        if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create instance!");
        }
        log_info("Created Vulkan instance");
    }

    bool check_validation_layer_support() const
    {
        uint32_t layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> available_layers(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count,
                                           available_layers.data());
        return std::find_if(
                   validation_layers_.begin(), validation_layers_.end(),
                   [available_layers](auto validation_layer) {
                       return std::find_if(
                                  available_layers.begin(),
                                  available_layers.end(),
                                  [validation_layer](
                                      auto available_layer) noexcept {
                                      return strncmp(
                                                 validation_layer,
                                                 &available_layer.layerName[0],
                                                 VK_MAX_EXTENSION_NAME_SIZE) ==
                                             0;
                                  }) != available_layers.end();
                   }) != validation_layers_.end();
    }

    void setup_debug_messenger()
    {
        if (!enable_validation_layers_)
            return;
        const auto debug_utils_messenger_info =
            get_debug_utils_messenger_info();
        if (CreateDebugUtilsMessengerEXT(instance_, &debug_utils_messenger_info,
                                         nullptr,
                                         &debug_messenger_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failes to set up debug messenger!");
        }
    }

    void create_surface()
    {
        if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create window surface!");
        }
        log_info("Created surface");
    }

    void pick_physical_device()
    {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
        if (device_count == 0)
        {
            throw std::runtime_error("No Vulkan-compatible devices found!");
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

        const auto surface = surface_;
        const auto it =
            std::partition(devices.begin(), devices.end(), [surface](auto d) {
                return is_device_suitable(d, surface);
            });

        if (it == devices.begin())
        {
            throw std::runtime_error(
                "No suitable Vulkan-compatible devices found!");
        }

        for (auto jt = devices.begin(); jt != it; ++jt)
        {
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(*jt, &properties);
            log_info("Found physical device \"{}\"", properties.deviceName);
        }

        physical_device_ = devices.at(0);
        msaa_samples_    = get_max_usable_sample_count(physical_device_);

        log_info("Selected physical device with {} bit multisampling",
                 static_cast<uint32_t>(msaa_samples_));
    }

    void create_logical_device()
    {
        const QueueFamilyIndices indices =
            find_queue_families(physical_device_, surface_);

        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
        std::set<uint32_t> unique_queue_families = {
            indices.graphics_family.value(),
            indices.present_family.value(),
        };

        // NB: queue_priorities lifetime needs to match queue_create_infos
        constexpr float queue_priorities[] = {1.0f};
        for (const uint32_t family : unique_queue_families)
        {
            const VkDeviceQueueCreateInfo info = {
                .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = family,
                .queueCount =
                    gsl::narrow_cast<uint32_t>(std::size(queue_priorities)),
                .pQueuePriorities = &queue_priorities[0],
            };
            queue_create_infos.push_back(info);
        }

        VkPhysicalDeviceFeatures device_features = {};
        VkDeviceCreateInfo create_info           = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount =
                gsl::narrow_cast<uint32_t>(queue_create_infos.size()),
            .pQueueCreateInfos = queue_create_infos.data(),
            .enabledLayerCount =
                enable_validation_layers_
                    ? gsl::narrow_cast<uint32_t>(validation_layers_.size())
                    : 0u,
            .ppEnabledLayerNames =
                enable_validation_layers_ ? validation_layers_.data() : nullptr,
            .enabledExtensionCount =
                gsl::narrow_cast<uint32_t>(device_extensions_.size()),
            .ppEnabledExtensionNames = device_extensions_.data(),
            .pEnabledFeatures        = &device_features,
        };

        if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create logical device!");
        }
        vkGetDeviceQueue(device_, indices.graphics_family.value(), 0,
                         &graphics_queue_);
        vkGetDeviceQueue(device_, indices.present_family.value(), 0,
                         &present_queue_);

        log_info("Created logical device");
    }

    void create_swap_chain()
    {
        const SwapChainSupportDetails swap_chain_support =
            query_swap_chain_support(physical_device_, surface_);

        const VkSurfaceFormatKHR surface_format =
            choose_swap_surface_format(swap_chain_support.formats);
        const VkPresentModeKHR present_mode =
            choose_swap_present_mode(swap_chain_support.present_modes);

        int width  = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        Expects(width >= 0 && height >= 0);

        const VkExtent2D extent =
            choose_swap_extent(swap_chain_support.capabilities, width, height);

        const uint32_t min_images =
            swap_chain_support.capabilities.minImageCount;
        const uint32_t max_images =
            swap_chain_support.capabilities.maxImageCount;
        const uint32_t image_count = max_images == 0
                                         ? (min_images + 1)
                                         : std::min(min_images + 1, max_images);

        VkSwapchainCreateInfoKHR create_info = {
            .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface          = surface_,
            .minImageCount    = min_images,
            .imageFormat      = surface_format.format,
            .imageColorSpace  = surface_format.colorSpace,
            .imageExtent      = extent,
            .imageArrayLayers = 1,
            .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform   = swap_chain_support.capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode    = present_mode,
            .clipped        = VK_TRUE,
            .oldSwapchain   = VK_NULL_HANDLE,
        };

        const QueueFamilyIndices indices =
            find_queue_families(physical_device_, surface_);
        const uint32_t queue_family_indices[] = {
            indices.graphics_family.value(), indices.present_family.value()};
        if (indices.graphics_family != indices.present_family)
        {
            create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices   = &queue_family_indices[0];
        }
        else
        {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        if (vkCreateSwapchainKHR(device_, &create_info, nullptr,
                                 &swap_chain_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swap chain!");
        }

        uint32_t swap_chain_image_count = 0;
        vkGetSwapchainImagesKHR(device_, swap_chain_, &swap_chain_image_count,
                                nullptr);
        swap_chain_images_.resize(swap_chain_image_count);
        vkGetSwapchainImagesKHR(device_, swap_chain_, &swap_chain_image_count,
                                swap_chain_images_.data());

        swap_chain_image_format_ = surface_format.format;
        swap_chain_extent_       = extent;

        log_info("Created swap chain");
    }

    void create_image_views()
    {
        swap_chain_image_views_.resize(swap_chain_images_.size());
        for (gsl::index i = 0; i < std::ssize(swap_chain_images_); ++i)
        {
            const VkImageViewCreateInfo create_info = {
                .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image    = swap_chain_images_.at(i),
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format   = swap_chain_image_format_,
                .components =
                    {
                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
                .subresourceRange =
                    {
                        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel   = 0,
                        .levelCount     = 1,
                        .baseArrayLayer = 0,
                        .layerCount     = 1,
                    },
            };

            if (vkCreateImageView(device_, &create_info, nullptr,
                                  &swap_chain_image_views_.at(i)) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image views!");
            }
        }

        log_info("Created image views");
    }

    void create_render_pass()
    {
        const VkAttachmentDescription colour_attachment = {
            .format         = swap_chain_image_format_,
            .samples        = msaa_samples_,
            .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        const VkAttachmentReference colour_attachment_ref = {
            .attachment = 0,
            .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        const VkAttachmentDescription colour_attachment_resolve = {
            .format         = swap_chain_image_format_,
            .samples        = VK_SAMPLE_COUNT_1_BIT,
            .loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        const VkAttachmentReference colour_attachment_resolve_ref = {
            .attachment = 1,
            .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        const VkAttachmentDescription depth_attachment = {
            .format         = find_depth_format(physical_device_),
            .samples        = msaa_samples_,
            .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        const VkAttachmentReference depth_attachment_ref = {
            .attachment = 2,
            .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        const VkSubpassDescription subpass = {
            .pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount    = 1,
            .pColorAttachments       = &colour_attachment_ref,
            .pResolveAttachments     = &colour_attachment_resolve_ref,
            .pDepthStencilAttachment = &depth_attachment_ref,
        };

        const VkSubpassDependency dependency = {
            .srcSubpass    = VK_SUBPASS_EXTERNAL,
            .dstSubpass    = 0,
            .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};

        const std::array<VkAttachmentDescription, 3> attachments = {
            colour_attachment,
            colour_attachment_resolve,
            depth_attachment,
        };

        const VkRenderPassCreateInfo render_pass_info = {
            .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = gsl::narrow_cast<uint32_t>(attachments.size()),
            .pAttachments    = attachments.data(),
            .subpassCount    = 1,
            .pSubpasses      = &subpass,
            .dependencyCount = 1,
            .pDependencies   = &dependency,
        };

        if (vkCreateRenderPass(device_, &render_pass_info, nullptr,
                               &render_pass_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create render pass!");
        }

        log_info("Created render pass");
    }

    void create_descriptor_set_layout()
    {
        const VkDescriptorSetLayoutBinding ubo_layout_binding = {
            .binding            = 0,
            .descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount    = 1,
            .stageFlags         = VK_SHADER_STAGE_VERTEX_BIT,
            .pImmutableSamplers = nullptr,
        };

        const VkDescriptorSetLayoutCreateInfo layout_info = {
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings    = &ubo_layout_binding,
        };

        if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr,
                                        &descriptor_set_layout_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
    }

    void create_graphics_pipeline()
    {
        // Shader modules

        const auto vert_shader_code = read_bytes("shaders\\vert.spv");
        const auto frag_shader_code = read_bytes("shaders\\frag.spv");
        const VkShaderModule vert_shader_module =
            create_shader_module(device_, vert_shader_code);
        const VkShaderModule frag_shader_module =
            create_shader_module(device_, frag_shader_code);

        const VkPipelineShaderStageCreateInfo shader_stages[2] = {
            {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vert_shader_module,
                .pName  = "main",
            },

            {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = frag_shader_module,
                .pName  = "main",
            }};

        // Vertex input

        const auto binding_description = Vertex::get_binding_description();
        const auto attribute_descriptions =
            Vertex::get_attribute_descriptions();
        const VkPipelineVertexInputStateCreateInfo vertex_input_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions    = &binding_description,
            .vertexAttributeDescriptionCount =
                gsl::narrow_cast<uint32_t>(std::size(attribute_descriptions)),
            .pVertexAttributeDescriptions = attribute_descriptions.data(),
        };

        // Input assembly

        const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
            .sType =
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        // Viewport

        const VkViewport viewport = {
            .x        = 0.0f,
            .y        = 0.0f,
            .width    = static_cast<float>(swap_chain_extent_.width),
            .height   = static_cast<float>(swap_chain_extent_.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        const VkRect2D scissor = {
            .offset = {0, 0},
            .extent = swap_chain_extent_,
        };

        const VkPipelineViewportStateCreateInfo viewport_state = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports    = &viewport,
            .scissorCount  = 1,
            .pScissors     = &scissor,
        };

        // Rasterizer

        const VkPipelineRasterizationStateCreateInfo rasterizer = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable        = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode             = VK_POLYGON_MODE_FILL,
            .cullMode                = VK_CULL_MODE_BACK_BIT,
            .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable         = VK_FALSE,
            .lineWidth               = 1.0f,
        };

        // Multisampling

        const VkPipelineMultisampleStateCreateInfo multisampling = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = msaa_samples_,
            .sampleShadingEnable  = VK_FALSE,
        };

        // Colour blending

        const VkPipelineColorBlendAttachmentState colour_blend_attachment = {
            .blendEnable = VK_FALSE,
            .colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};

        const VkPipelineColorBlendStateCreateInfo colour_blending = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable   = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments    = &colour_blend_attachment,
        };

        // Depth testing

        const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable       = VK_TRUE,
            .depthWriteEnable      = VK_TRUE,
            .depthCompareOp        = VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = VK_FALSE,
        };

        // Create pipeline layout

        const VkPipelineLayoutCreateInfo pipeline_layout_info = {
            .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts    = &descriptor_set_layout_,
        };

        if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr,
                                   &pipeline_layout_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        // Create graphics pipeline

        const VkGraphicsPipelineCreateInfo pipeline_info = {
            .sType      = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages    = &shader_stages[0],
            .pVertexInputState   = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState      = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState   = &multisampling,
            .pDepthStencilState  = &depth_stencil,
            .pColorBlendState    = &colour_blending,
            .layout              = pipeline_layout_,
            .renderPass          = render_pass_,
            .subpass             = 0,
        };

        if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1,
                                      &pipeline_info, nullptr,
                                      &graphics_pipeline_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create graphics pipeline!");
        }

        // Cleanup
        vkDestroyShaderModule(device_, frag_shader_module, nullptr);
        vkDestroyShaderModule(device_, vert_shader_module, nullptr);

        log_info("Created graphics pipeline");
    }

    void create_colour_resources()
    {
        const VkFormat colour_format = swap_chain_image_format_;
        std::tie(colour_image_, colour_image_memory_) =
            create_image(physical_device_, device_, swap_chain_extent_.width,
                         swap_chain_extent_.height, msaa_samples_,
                         colour_format, VK_IMAGE_TILING_OPTIMAL,
                         VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        colour_image_view_ =
            create_image_view(device_, colour_image_, colour_format,
                              VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    void create_depth_resources()
    {
        const VkFormat depth_format = find_depth_format(physical_device_);

        std::tie(depth_image_, depth_image_memory_) =
            create_image(physical_device_, device_, swap_chain_extent_.width,
                         swap_chain_extent_.height, msaa_samples_, depth_format,
                         VK_IMAGE_TILING_OPTIMAL,
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        depth_image_view_ = create_image_view(
            device_, depth_image_, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

        transition_image_layout(
            depth_image_, depth_format, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    void create_framebuffers()
    {
        swap_chain_framebuffers_.resize(swap_chain_image_views_.size());
        for (gsl::index i = 0; i < std::ssize(swap_chain_image_views_); ++i)
        {
            const std::array<VkImageView, 3> attachments = {
                colour_image_view_,
                swap_chain_image_views_.at(i),
                depth_image_view_,
            };
            const VkFramebufferCreateInfo framebuffer_info = {
                .sType      = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = render_pass_,
                .attachmentCount =
                    gsl::narrow_cast<uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .width        = swap_chain_extent_.width,
                .height       = swap_chain_extent_.height,
                .layers       = 1,
            };

            if (vkCreateFramebuffer(device_, &framebuffer_info, nullptr,
                                    &swap_chain_framebuffers_.at(i)) !=
                VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create framebuffer!");
            }
        }

        log_info("Created framebuffers");
    }

    void create_command_pool()
    {
        const QueueFamilyIndices queue_family_indices =
            find_queue_families(physical_device_, surface_);

        const VkCommandPoolCreateInfo pool_info = {
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags            = 0,
            .queueFamilyIndex = queue_family_indices.graphics_family.value(),
        };

        if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create command pool!");
        }

        log_info("Created command pool");
    }

    void create_vertex_buffer()
    {
        constexpr VkDeviceSize buffer_size =
            sizeof(Vertex) * gsl::narrow_cast<VkDeviceSize>(vertices_.size());

        const auto [staging_buffer, staging_buffer_memory] =
            create_buffer(physical_device_, device_, buffer_size,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        void *data = nullptr;
        vkMapMemory(device_, staging_buffer_memory, 0, buffer_size, 0, &data);
        std::memcpy(data, vertices_.data(),
                    gsl::narrow_cast<std::size_t>(buffer_size));
        vkUnmapMemory(device_, staging_buffer_memory);

        std::tie(vertex_buffer_, vertex_buffer_memory_) =
            create_buffer(physical_device_, device_, buffer_size,
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        copy_buffer(staging_buffer, vertex_buffer_, buffer_size);
        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_buffer_memory, nullptr);
    }

    void create_index_buffer()
    {
        const VkDeviceSize buffer_size =
            sizeof(indices_[0]) *
            gsl::narrow_cast<VkDeviceSize>(indices_.size());

        const auto [staging_buffer, staging_buffer_memory] =
            create_buffer(physical_device_, device_, buffer_size,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        void *data = nullptr;
        vkMapMemory(device_, staging_buffer_memory, 0, buffer_size, 0, &data);
        std::memcpy(data, indices_.data(),
                    gsl::narrow_cast<std::size_t>(buffer_size));
        vkUnmapMemory(device_, staging_buffer_memory);

        std::tie(index_buffer_, index_buffer_memory_) = create_buffer(
            physical_device_, device_, buffer_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        copy_buffer(staging_buffer, index_buffer_, buffer_size);

        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_buffer_memory, nullptr);
    }

    void create_uniform_buffers()
    {
        constexpr VkDeviceSize buffer_size = sizeof(UniformBufferObject);
        uniform_buffers_.resize(swap_chain_images_.size());
        uniform_buffers_memory_.resize(swap_chain_images_.size());
        for (gsl::index i = 0; i < std::ssize(swap_chain_images_); ++i)
        {
            std::tie(uniform_buffers_.at(i), uniform_buffers_memory_.at(i)) =
                create_buffer(physical_device_, device_, buffer_size,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
    }

    void create_descriptor_pool()
    {
        const VkDescriptorPoolSize pool_size = {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount =
                gsl::narrow_cast<uint32_t>(swap_chain_images_.size()),
        };

        const VkDescriptorPoolCreateInfo pool_info = {
            .sType   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = gsl::narrow_cast<uint32_t>(swap_chain_images_.size()),
            .poolSizeCount = 1,
            .pPoolSizes    = &pool_size,
        };

        if (vkCreateDescriptorPool(device_, &pool_info, nullptr,
                                   &descriptor_pool_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create descriptor pool!");
        }
    }

    void create_descriptor_sets()
    {
        descriptor_sets_.resize(swap_chain_images_.size());

        std::vector<VkDescriptorSetLayout> layouts(swap_chain_images_.size(),
                                                   descriptor_set_layout_);

        const VkDescriptorSetAllocateInfo alloc_info = {
            .sType          = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount =
                gsl::narrow_cast<uint32_t>(swap_chain_images_.size()),
            .pSetLayouts = layouts.data()};

        if (vkAllocateDescriptorSets(device_, &alloc_info,
                                     descriptor_sets_.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate descriptor sets!");
        }

        for (gsl::index i = 0; i < std::ssize(swap_chain_images_); ++i)
        {
            const VkDescriptorBufferInfo buffer_info = {
                .buffer = uniform_buffers_.at(i),
                .offset = 0,
                .range  = sizeof(UniformBufferObject),
            };

            const VkWriteDescriptorSet descriptor_write = {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = descriptor_sets_.at(i),
                .dstBinding      = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo     = &buffer_info,
            };

            vkUpdateDescriptorSets(device_, 1, &descriptor_write, 0, nullptr);
        }
    }

    void create_command_buffers()
    {
        command_buffers_.resize(swap_chain_framebuffers_.size());

        const VkCommandBufferAllocateInfo alloc_info = {
            .sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool_,
            .level       = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount =
                gsl::narrow_cast<uint32_t>(command_buffers_.size()),
        };

        if (vkAllocateCommandBuffers(device_, &alloc_info,
                                     command_buffers_.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate command buffers!");
        }

        // Record

        for (gsl::index i = 0; i < std::ssize(command_buffers_); ++i)
        {
            const VkCommandBufferBeginInfo begin_info = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            };

            if (vkBeginCommandBuffer(command_buffers_.at(i), &begin_info) !=
                VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to begin recording command buffer!");
            }

            constexpr auto bg = Colours::lighter;
            const std::array<VkClearValue, 3> clear_values = {{
                {bg.r, bg.g, bg.b, bg.a},
                {bg.r, bg.g, bg.b, bg.a},
                {1.0f},
            }};

            const VkRenderPassBeginInfo render_pass_info = {
                .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass  = render_pass_,
                .framebuffer = swap_chain_framebuffers_.at(i),
                .renderArea  = {.offset = {0, 0}, .extent = swap_chain_extent_},
                .clearValueCount =
                    gsl::narrow_cast<uint32_t>(clear_values.size()),
                .pClearValues = clear_values.data(),
            };

            vkCmdBeginRenderPass(command_buffers_.at(i), &render_pass_info,
                                 VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(command_buffers_.at(i),
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              graphics_pipeline_);
            const VkBuffer vertex_buffers[] = {vertex_buffer_};
            const VkDeviceSize offsets[]    = {0};
            vkCmdBindVertexBuffers(command_buffers_.at(i), 0, 1,
                                   &vertex_buffers[0], &offsets[0]);
            vkCmdBindIndexBuffer(command_buffers_.at(i), index_buffer_, 0,
                                 VK_INDEX_TYPE_UINT16);

            vkCmdBindDescriptorSets(
                command_buffers_.at(i), VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout_, 0, 1, &descriptor_sets_.at(i), 0, nullptr);

            vkCmdDrawIndexed(command_buffers_.at(i),
                             gsl::narrow_cast<uint32_t>(indices_.size()), 1, 0,
                             0, 0);
            vkCmdEndRenderPass(command_buffers_.at(i));
            if (vkEndCommandBuffer(command_buffers_.at(i)) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to record command buffer!");
            }
        }

        log_info("Created command buffers");
    }

    void create_sync_objects()
    {
        image_available_semaphores_.resize(max_frames_in_flight_);
        render_finished_semaphores_.resize(max_frames_in_flight_);
        in_flight_fences_.resize(max_frames_in_flight_);
        images_in_flight_.resize(swap_chain_images_.size(), VK_NULL_HANDLE);

        const VkSemaphoreCreateInfo semaphore_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        const VkFenceCreateInfo fence_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };

        for (gsl::index i = 0; i < max_frames_in_flight_; ++i)
        {
            if (vkCreateSemaphore(device_, &semaphore_info, nullptr,
                                  &image_available_semaphores_.at(i)) !=
                VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create semaphore!");
            }

            if (vkCreateSemaphore(device_, &semaphore_info, nullptr,
                                  &render_finished_semaphores_.at(i)) !=
                VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create semaphore!");
            }

            if (vkCreateFence(device_, &fence_info, nullptr,
                              &in_flight_fences_.at(i)) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create fence!");
            }
        }

        log_info("Created sync objects");
    }

    void draw_frame()
    {
        vkWaitForFences(device_, 1, &in_flight_fences_.at(current_frame_),
                        VK_TRUE, UINT64_MAX);

        uint32_t image_index      = 0;
        const auto acquire_result = vkAcquireNextImageKHR(
            device_, swap_chain_, UINT64_MAX,
            image_available_semaphores_.at(current_frame_), VK_NULL_HANDLE,
            &image_index);

        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR ||
            acquire_result == VK_SUBOPTIMAL_KHR || framebuffer_resized_)
        {
            framebuffer_resized_ = false;
            recreate_swap_chain();
            return;
        }
        else if (acquire_result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }

        if (images_in_flight_.at(image_index) != VK_NULL_HANDLE)
        {
            vkWaitForFences(device_, 1, &images_in_flight_.at(image_index),
                            VK_TRUE, UINT64_MAX);
        }
        images_in_flight_.at(image_index) =
            in_flight_fences_.at(current_frame_);

        const VkSemaphore wait_semaphores[] = {
            image_available_semaphores_.at(current_frame_),
        };
        const VkPipelineStageFlags wait_stages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        };
        const VkSemaphore signal_semaphores[] = {
            render_finished_semaphores_.at(current_frame_),
        };

        update_uniform_buffer(image_index);

        const VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount =
                gsl::narrow_cast<uint32_t>(std::size(wait_semaphores)),
            .pWaitSemaphores    = &wait_semaphores[0],
            .pWaitDstStageMask  = &wait_stages[0],
            .commandBufferCount = 1,
            .pCommandBuffers    = &command_buffers_.at(image_index),
            .signalSemaphoreCount =
                gsl::narrow_cast<uint32_t>(std::size(signal_semaphores)),
            .pSignalSemaphores = &signal_semaphores[0],
        };

        vkResetFences(device_, 1, &in_flight_fences_.at(current_frame_));
        if (vkQueueSubmit(graphics_queue_, 1, &submit_info,
                          in_flight_fences_.at(current_frame_)) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        const VkSwapchainKHR swap_chains[] = {
            swap_chain_,
        };

        const VkPresentInfoKHR present_info = {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount =
                gsl::narrow_cast<uint32_t>(std::size(signal_semaphores)),
            .pWaitSemaphores = &signal_semaphores[0],
            .swapchainCount =
                gsl::narrow_cast<uint32_t>(std::size(swap_chains)),
            .pSwapchains   = &swap_chains[0],
            .pImageIndices = &image_index,
        };

        vkQueuePresentKHR(present_queue_, &present_info);

        current_frame_ = (current_frame_ + 1) % max_frames_in_flight_;
    }

    void cleanup_swap_chain() noexcept
    {
        vkDestroyImageView(device_, colour_image_view_, nullptr);
        colour_image_view_ = VK_NULL_HANDLE;
        vkDestroyImage(device_, colour_image_, nullptr);
        colour_image_ = VK_NULL_HANDLE;
        vkFreeMemory(device_, colour_image_memory_, nullptr);
        colour_image_memory_ = VK_NULL_HANDLE;

        vkDestroyImageView(device_, depth_image_view_, nullptr);
        depth_image_view_ = VK_NULL_HANDLE;
        vkDestroyImage(device_, depth_image_, nullptr);
        depth_image_ = VK_NULL_HANDLE;
        vkFreeMemory(device_, depth_image_memory_, nullptr);
        depth_image_memory_ = VK_NULL_HANDLE;

        for (auto framebuffer : swap_chain_framebuffers_)
        {
            vkDestroyFramebuffer(device_, framebuffer, nullptr);
        }
        swap_chain_framebuffers_.clear();
        for (auto buffer : uniform_buffers_)
        {
            vkDestroyBuffer(device_, buffer, nullptr);
        }
        uniform_buffers_.clear();
        for (auto buffer : uniform_buffers_memory_)
        {
            vkFreeMemory(device_, buffer, nullptr);
        }
        uniform_buffers_memory_.clear();
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        vkFreeCommandBuffers(
            device_, command_pool_,
            gsl::narrow_cast<uint32_t>(command_buffers_.size()),
            command_buffers_.data());
        vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        vkDestroyRenderPass(device_, render_pass_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
        for (auto view : swap_chain_image_views_)
        {
            vkDestroyImageView(device_, view, nullptr);
        }
        swap_chain_image_views_.clear();
        vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
        swap_chain_ = VK_NULL_HANDLE;
    }

    void recreate_swap_chain()
    {
        // Wait until not-minimised
        int width  = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device_);

        // NB: Could use vkWaitSemaphore instead of recreating the semaphore
        for (auto &semaphore : image_available_semaphores_)
        {
            vkDestroySemaphore(device_, semaphore, nullptr);
            const VkSemaphoreCreateInfo semaphore_info = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            };
            if (vkCreateSemaphore(device_, &semaphore_info, nullptr,
                                  &semaphore) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create semaphore!");
            }
        }

        cleanup_swap_chain();

        create_swap_chain();
        create_image_views();
        create_render_pass();
        create_graphics_pipeline();
        create_colour_resources();
        create_depth_resources();
        create_framebuffers();
        create_uniform_buffers();
        create_descriptor_pool();
        create_descriptor_sets();
        create_command_buffers();

        log_info("Recreated swap chain");
    }

    // Helpers

    static bool is_device_suitable(VkPhysicalDevice device,
                                   VkSurfaceKHR surface)
    {
        const bool extensions_supported =
            check_device_extension_support(device);
        const bool swap_chain_adequate = [surface, device,
                                          extensions_supported]() {
            if (!extensions_supported)
            {
                return false;
            }
            const auto swap_details = query_swap_chain_support(device, surface);
            return !swap_details.formats.empty() &&
                   !swap_details.present_modes.empty();
        }();
        const QueueFamilyIndices indices = find_queue_families(device, surface);
        return indices.is_complete() && extensions_supported &&
               swap_chain_adequate;
    }

    static bool check_device_extension_support(VkPhysicalDevice device)
    {
        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                             nullptr);
        std::vector<VkExtensionProperties> available_extensions(
            extension_count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                             available_extensions.data());

        std::set<std::string> required_extensions {device_extensions_.begin(),
                                                   device_extensions_.end()};
        for (const auto &ext : available_extensions)
        {
            required_extensions.erase(std::string(&ext.extensionName[0]));
        }
        return required_extensions.empty();
    }

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphics_family;
        std::optional<uint32_t> present_family;

        bool is_complete() const noexcept
        {
            return graphics_family.has_value() && present_family.has_value();
        }
    };

    static QueueFamilyIndices find_queue_families(VkPhysicalDevice device,
                                                  VkSurfaceKHR surface)
    {
        QueueFamilyIndices indices = {};

        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count,
                                                 nullptr);

        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count,
                                                 families.data());

        uint32_t i = 0;
        for (const auto &family : families)
        {
            if (family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphics_family = i;
            }

            VkBool32 present_support = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
                                                 &present_support);
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
        VkSurfaceCapabilitiesKHR capabilities       = {};
        std::vector<VkSurfaceFormatKHR> formats     = {};
        std::vector<VkPresentModeKHR> present_modes = {};
    };

    static SwapChainSupportDetails query_swap_chain_support(
        VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        SwapChainSupportDetails details = {};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                                  &details.capabilities);
        uint32_t format_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count,
                                             nullptr);
        if (format_count != 0)
        {
            details.formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count,
                                                 details.formats.data());
        }

        uint32_t present_mode_count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                                  &present_mode_count, nullptr);
        if (present_mode_count != 0)
        {
            details.present_modes.resize(present_mode_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                device, surface, &present_mode_count,
                details.present_modes.data());
        }
        return details;
    }

    static VkSurfaceFormatKHR choose_swap_surface_format(
        const std::vector<VkSurfaceFormatKHR> &available_formats)
    {
        Expects(!available_formats.empty());

        for (const auto &format : available_formats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }
        return available_formats.at(0);
    }

    static VkPresentModeKHR choose_swap_present_mode(
        const std::vector<VkPresentModeKHR> &available_present_modes)
    {
        Expects(!available_present_modes.empty());

        if (std::find(
                available_present_modes.begin(), available_present_modes.end(),
                VK_PRESENT_MODE_MAILBOX_KHR) != available_present_modes.end())
        {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
        else
        {
            return VK_PRESENT_MODE_FIFO_KHR;
        }
    }

    static VkExtent2D choose_swap_extent(
        const VkSurfaceCapabilitiesKHR &capabilities, uint32_t width,
        uint32_t height) noexcept
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        else
        {
            const auto &minExtent = capabilities.minImageExtent;
            const auto &maxExtent = capabilities.maxImageExtent;
            return {std::clamp(width, minExtent.width, maxExtent.width),
                    std::clamp(height, minExtent.height, maxExtent.height)};
        }
    }

    [[gsl::suppress(26490)]] // Don't warn about the reinterpret_cast below
    static VkShaderModule
    create_shader_module(VkDevice device, const std::vector<char> &code)
    {
        VkShaderModuleCreateInfo create_info = {
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = gsl::narrow_cast<uint32_t>(code.size()),
            .pCode    = reinterpret_cast<const uint32_t *>(code.data()),
        };

        VkShaderModule shader_module;
        if (vkCreateShaderModule(device, &create_info, nullptr,
                                 &shader_module) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create shader module!");
        }
        return shader_module;
    }

    static std::vector<const char *> get_required_extensions()
    {
        uint32_t glfw_extension_count = 0;
        const char *const *glfw_extensions =
            glfwGetRequiredInstanceExtensions(&glfw_extension_count);
        std::vector<const char *> extensions(
            glfw_extensions, std::next(glfw_extensions, glfw_extension_count));
        if (enable_validation_layers_)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT message_type,
        const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
        [[maybe_unused]] void *user_data)
    {
        Expects(callback_data != nullptr);

        switch (message_severity)
        {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: {
            log_info("[Vulkan] {}", callback_data->pMessage);
            break;
        }
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: {
            log_warn("[Vulkan] {}", callback_data->pMessage);
            break;
        }
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: {
            log_error("[Vulkan] {}", callback_data->pMessage);
            break;
        }
        }

        return VK_FALSE;
    }

    static VkDebugUtilsMessengerCreateInfoEXT
    get_debug_utils_messenger_info() noexcept
    {
        return {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debug_callback,
        };
    }

    static std::vector<char> read_bytes(const std::string &filename)
    {
        std::ifstream file {filename, std::ios::ate | std::ios::binary};
        if (!file.is_open())
        {
            throw std::runtime_error(
                fmt::format("Failed to open {}! Reason: Read error.", filename)
                    .c_str());
        }
        const std::size_t size = static_cast<std::size_t>(file.tellg());
        std::vector<char> buffer(size);
        file.seekg(0);
        file.read(buffer.data(), size);
        file.close();
        return buffer;
    }

    static void error_callback(int error, const char *description)
    {
        log_error(description);
    }

    static void key_callback(GLFWwindow *window, int key, int scancode,
                             int action, int mods) noexcept
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }

    static void framebuffer_resize_callback(GLFWwindow *window, int width,
                                            int height) noexcept
    {
        auto *app =
            static_cast<Application *>(glfwGetWindowUserPointer(window));
        app->framebuffer_resized_ = true;
    }

    static uint32_t find_memory_type(VkPhysicalDevice physical_device,
                                     uint32_t type_filter,
                                     VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memory_properties = {};
        vkGetPhysicalDeviceMemoryProperties(physical_device,
                                            &memory_properties);

        [[gsl::suppress(bounds .2)]] [[gsl::suppress(
            bounds .4)]] for (uint32_t i = 0;
                              i < memory_properties.memoryTypeCount; ++i)
        {
            const bool matches_type_filter = (type_filter & (1 << i)) != 0;
            const bool matches_properties =
                (memory_properties.memoryTypes[i].propertyFlags & properties) ==
                properties;
            if (matches_type_filter && matches_properties)
            {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    }

    static std::pair<VkBuffer, VkDeviceMemory> create_buffer(
        VkPhysicalDevice physical_device, VkDevice device, VkDeviceSize size,
        VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
    {
        const VkBufferCreateInfo buffer_info = {
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = size,
            .usage       = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        VkBuffer buffer;
        if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("failed to create vertex buffer!");
        }

        VkMemoryRequirements memory_requirements = {};
        vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

        const auto memory_type = find_memory_type(
            physical_device, memory_requirements.memoryTypeBits, properties);
        const VkMemoryAllocateInfo alloc_info = {
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize  = memory_requirements.size,
            .memoryTypeIndex = memory_type,
        };

        VkDeviceMemory buffer_memory;
        if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory) !=
            VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to allocate vertex buffer memory!");
        }

        vkBindBufferMemory(device, buffer, buffer_memory, 0);

        return {buffer, buffer_memory};
    }

    void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer,
                     VkDeviceSize size) noexcept
    {
        const VkCommandBuffer command_buffer =
            begin_single_time_commands(device_, command_pool_);

        const VkBufferCopy copy_region = {
            .size = size,
        };
        vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1,
                        &copy_region);

        end_single_time_commands(device_, command_pool_, graphics_queue_,
                                 command_buffer);
    }

    void copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width,
                              uint32_t height) noexcept
    {
        VkCommandBuffer command_buffer =
            begin_single_time_commands(device_, command_pool_);

        const VkBufferImageCopy region = {
            .bufferOffset      = 0,
            .bufferRowLength   = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent =
                {
                    width,
                    height,
                    1,
                },
        };
        vkCmdCopyBufferToImage(command_buffer, buffer, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);

        end_single_time_commands(device_, command_pool_, graphics_queue_,
                                 command_buffer);
    }

    void update_uniform_buffer(uint32_t current_image)
    {
        const float time = []() noexcept {
            static const auto start_time =
                std::chrono::high_resolution_clock::now();
            const auto current_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<float, std::chrono::seconds::period>(
                       current_time - start_time)
                .count();
        }();

        const float aspect_ratio =
            gsl::narrow_cast<float>(swap_chain_extent_.width) /
            swap_chain_extent_.height;

        const auto elastic_turn = [](float p) -> float {
            if (p < 0.5f)
            {
                const float f = 2.0f * p;
                return 0.5f *
                       (f * f * f - f * sin(f * std::numbers::pi_v<float>));
            }
            else
            {
                const float f = (1.0f - (2.0f * p - 1.0f));
                return 0.5f *
                           (1.0f - (f * f * f -
                                    f * sin(f * std::numbers::pi_v<float>))) +
                       0.5f;
            }
        };

        constexpr float parts = 6.0f;
        constexpr float speed = 2.0f;
        const float direction =
            (std::fmod(time * speed, 2 * parts) <= parts) ? 1.0f : -1.0f;
        const float part  = std::fmod(time * speed, parts);
        const float ipart = std::floor(part);
        const float dpart = std::clamp(
            0.5f + 1.8f * (std::fmod(part, 1.0f) - 0.5f), 0.0f, 1.0f);

        const float angle =
            direction * (ipart + std::lerp(dpart, elastic_turn(dpart), 0.25f)) *
            glm::radians(360.0f / parts);
        const float scale =
            0.75f - 0.035f * std::sin(dpart * std::numbers::pi_v<float>);

        const UniformBufferObject ubo = {
            .model = glm::scale(
                glm::rotate(mat4(1.0f), angle, vec3(0.0f, 0.0f, 1.0f)),
                vec3(scale, scale, 1.0f)),
            .view = glm::identity<mat4>(),
            .proj = glm::ortho(-aspect_ratio, aspect_ratio, 1.0f, -1.0f, 0.01f,
                               1.1f),
        };

        void *data = nullptr;
        vkMapMemory(device_, uniform_buffers_memory_.at(current_image), 0,
                    sizeof(ubo), 0, &data);
        std::memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device_, uniform_buffers_memory_.at(current_image));
    }

    static VkSampleCountFlagBits get_max_usable_sample_count(
        VkPhysicalDevice physical_device) noexcept
    {
        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(physical_device, &properties);
        const VkSampleCountFlags counts =
            properties.limits.framebufferColorSampleCounts &
            properties.limits.framebufferDepthSampleCounts;
        const VkSampleCountFlagBits preferred_bits[] = {
            VK_SAMPLE_COUNT_64_BIT, VK_SAMPLE_COUNT_32_BIT,
            VK_SAMPLE_COUNT_16_BIT, VK_SAMPLE_COUNT_8_BIT,
            VK_SAMPLE_COUNT_4_BIT,  VK_SAMPLE_COUNT_2_BIT,
        };
        for (const auto preferred_bit : preferred_bits)
        {
            if (counts & preferred_bit)
            {
                return preferred_bit;
            }
        }

        return VK_SAMPLE_COUNT_1_BIT;
    }

    static std::pair<VkImage, VkDeviceMemory> create_image(
        VkPhysicalDevice physical_device, VkDevice device, uint32_t width,
        uint32_t height, VkSampleCountFlagBits num_samples, VkFormat format,
        VkImageTiling tiling, VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties)
    {
        VkImage image               = VK_NULL_HANDLE;
        VkDeviceMemory image_memory = VK_NULL_HANDLE;

        const VkImageCreateInfo image_info = {
            .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType     = VK_IMAGE_TYPE_2D,
            .format        = format,
            .extent        = {.width = width, .height = height, .depth = 1},
            .mipLevels     = 1,
            .arrayLayers   = 1,
            .samples       = num_samples,
            .tiling        = tiling,
            .usage         = usage,
            .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create image!");
        }

        VkMemoryRequirements requirements;
        vkGetImageMemoryRequirements(device, image, &requirements);

        const VkMemoryAllocateInfo alloc_info = {
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize  = requirements.size,
            .memoryTypeIndex = find_memory_type(
                physical_device, requirements.memoryTypeBits, properties)};

        if (vkAllocateMemory(device, &alloc_info, nullptr, &image_memory) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, image_memory, 0);

        return {image, image_memory};
    }

    static VkImageView create_image_view(VkDevice device, VkImage image,
                                         VkFormat format,
                                         VkImageAspectFlags aspect_flags,
                                         uint32_t mip_levels)
    {
        const VkImageViewCreateInfo view_info = {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image            = image,
            .viewType         = VK_IMAGE_VIEW_TYPE_2D,
            .format           = format,
            .subresourceRange = {
                .aspectMask     = aspect_flags,
                .baseMipLevel   = 0,
                .levelCount     = mip_levels,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            }};

        VkImageView image_view = VK_NULL_HANDLE;
        if (vkCreateImageView(device, &view_info, nullptr, &image_view) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create texture image view!");
        }

        return image_view;
    }

    static VkFormat find_supported_format(
        VkPhysicalDevice physical_device,
        const std::vector<VkFormat> &candidates, VkImageTiling tiling,
        VkFormatFeatureFlags features)
    {
        for (const VkFormat format : candidates)
        {
            VkFormatProperties props = {};
            vkGetPhysicalDeviceFormatProperties(physical_device, format,
                                                &props);

            if (tiling == VK_IMAGE_TILING_LINEAR &&
                (props.linearTilingFeatures & features) == features)
            {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                     (props.optimalTilingFeatures & features) == features)
            {
                return format;
            }
        }

        throw std::runtime_error("Failed to find supported format!");
    }

    static VkFormat find_depth_format(VkPhysicalDevice physical_device)
    {
        return find_supported_format(
            physical_device,
            {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
             VK_FORMAT_D24_UNORM_S8_UINT},
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    static constexpr bool has_stencil_component(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
               format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    // Supported layout transitions are:
    //     VK_IMAGE_LAYOUT_UNDEFINED ->
    //         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    //     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ->
    //         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    //     VK_IMAGE_LAYOUT_UNDEFINED ->
    //         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    void transition_image_layout(VkImage image, VkFormat format,
                                 VkImageLayout old_layout,
                                 VkImageLayout new_layout)
    {
        Expects(
            (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
             new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) ||
            (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ||
            (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
             new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL));

        VkCommandBuffer command_buffer =
            begin_single_time_commands(device_, command_pool_);

        const VkImageAspectFlags aspect_mask =
            new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                ? (VK_IMAGE_ASPECT_DEPTH_BIT |
                   (has_stencil_component(format) ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                  : 0))
                : VK_IMAGE_ASPECT_COLOR_BIT;

        VkImageMemoryBarrier barrier = {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout           = old_layout,
            .newLayout           = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = image,
            .subresourceRange =
                {
                    .aspectMask     = aspect_mask,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
        };

        VkPipelineStageFlags source_stage      = {};
        VkPipelineStageFlags destination_stage = {};
        if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
            new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            source_stage          = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage     = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                 new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            source_stage          = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destination_stage     = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
                 new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            source_stage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else
        {
            throw std::invalid_argument("Unsupported layout transition!");
        }

        vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);

        end_single_time_commands(device_, command_pool_, graphics_queue_,
                                 command_buffer);
    }

    static VkCommandBuffer begin_single_time_commands(
        VkDevice device, VkCommandPool command_pool) noexcept
    {
        const VkCommandBufferAllocateInfo alloc_info = {
            .sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool,
            .level       = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        VkCommandBuffer command_buffer = {};
        vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

        const VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        vkBeginCommandBuffer(command_buffer, &begin_info);

        return command_buffer;
    }

    static void end_single_time_commands(
        VkDevice device, VkCommandPool command_pool, VkQueue graphics_queue,
        VkCommandBuffer command_buffer) noexcept
    {
        vkEndCommandBuffer(command_buffer);

        const VkSubmitInfo submit_info = {
            .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers    = &command_buffer,
        };
        vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphics_queue);

        vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    }
};

int main(int argc, char **argv)
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
