// Hello Vulkan
// Benjamin Porter, 2020
//
// Code adapted from vulkan-tutorial.com

#include <fmt/core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define NOMINMAX
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <gsl/gsl>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
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

static std::vector<char> read_bytes(const std::string &filename)
{
    std::ifstream file {filename, std::ios::ate | std::ios::binary};
    if (!file.is_open())
    {
        throw std::runtime_error(
            fmt::format("Failed to open {}!", filename).c_str());
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

static void key_callback(GLFWwindow *window, int key, int scancode, int action,
                         int mods) noexcept
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

class Application
{
  private:
    static constexpr int max_frames_in_flight_ = 2;
    static constexpr bool enable_validation_ =
        (gBuildConfig.mode == BuildMode::Debug);
    static constexpr std::array<const char *, 1> validation_layers_ {
        "VK_LAYER_KHRONOS_validation"};
    static constexpr std::array<const char *, 1> device_extensions_ {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    int width_          = 800;
    int height_         = 600;
    GLFWwindow *window_ = nullptr;

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
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ =
            glfwCreateWindow(width_, height_, "Hello Vulkan", nullptr, nullptr);
        if (!window_)
        {
            throw std::runtime_error(
                "Window or OpenGL context creation failed!");
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
        create_render_pass();
        create_graphics_pipeline();
        create_framebuffers();
        create_command_pool();
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

    void cleanup()
    {
        for (gsl::index i = 0; i < max_frames_in_flight_; ++i)
        {
            vkDestroySemaphore(device_, render_finished_semaphores_.at(i),
                               nullptr);
            vkDestroySemaphore(device_, image_available_semaphores_.at(i),
                               nullptr);
            vkDestroyFence(device_, in_flight_fences_.at(i), nullptr);
        }
        render_finished_semaphores_.clear();
        image_available_semaphores_.clear();
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        for (auto framebuffer : swap_chain_framebuffers_)
        {
            vkDestroyFramebuffer(device_, framebuffer, nullptr);
        }
        swap_chain_framebuffers_.clear();
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
            throw std::runtime_error("Validation layers not available!!");
        }

        const VkApplicationInfo app_info {
            .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName   = "Hello Vulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = VK_API_VERSION_1_0};

        uint32_t required_extensions_count = 0;
        const char *const *required_extensions =
            glfwGetRequiredInstanceExtensions(&required_extensions_count);

        VkInstanceCreateInfo create_info {
            .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledLayerCount =
                enable_validation_ ? validation_layers_.size() : 0u,
            .ppEnabledLayerNames =
                enable_validation_ ? validation_layers_.data() : nullptr,
            .enabledExtensionCount   = required_extensions_count,
            .ppEnabledExtensionNames = required_extensions,
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
        // TODO: This
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

        auto it =
            std::partition(devices.begin(), devices.end(),
                           [this](auto d) { return is_device_suitable(d); });

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

        log_info("Selected physical device");
    }

    void create_logical_device()
    {
        QueueFamilyIndices indices = find_queue_families(physical_device_);

        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
        std::set<uint32_t> unique_queue_families = {
            indices.graphics_family.value(),
            indices.present_family.value(),
        };

        // NB: queue_priorities lifetime needs to match queue_create_infos
        constexpr float queue_priorities[] = {1.0f};
        for (const uint32_t family : unique_queue_families)
        {
            VkDeviceQueueCreateInfo info = {};
            info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            info.queueFamilyIndex = family;
            info.queueCount       = std::size(queue_priorities);
            info.pQueuePriorities = &queue_priorities[0];
            queue_create_infos.push_back(info);
        }

        VkPhysicalDeviceFeatures device_features = {};
        VkDeviceCreateInfo create_info           = {
            .sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = queue_create_infos.size(),
            .pQueueCreateInfos    = queue_create_infos.data(),
            .enabledLayerCount =
                enable_validation_ ? validation_layers_.size() : 0u,
            .ppEnabledLayerNames =
                enable_validation_ ? validation_layers_.data() : nullptr,
            .enabledExtensionCount   = device_extensions_.size(),
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
            query_swap_chain_support(physical_device_);

        const VkSurfaceFormatKHR surface_format =
            choose_swap_surface_format(swap_chain_support.formats);
        const VkPresentModeKHR present_mode =
            choose_swap_present_mode(swap_chain_support.present_modes);
        const VkExtent2D extent =
            choose_swap_extent(swap_chain_support.capabilities);

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
            find_queue_families(physical_device_);
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
                .subresourceRange = {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                }};

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
        const VkAttachmentDescription color_attachment = {
            .format         = swap_chain_image_format_,
            .samples        = VK_SAMPLE_COUNT_1_BIT,
            .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        const VkAttachmentReference color_attachment_ref = {
            .attachment = 0,
            .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        const VkSubpassDescription subpass = {
            .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &color_attachment_ref,
        };

        const VkSubpassDependency dependency = {
            .srcSubpass    = VK_SUBPASS_EXTERNAL,
            .dstSubpass    = 0,
            .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};

        const VkRenderPassCreateInfo render_pass_info = {
            .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments    = &color_attachment,
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

    void create_graphics_pipeline()
    {
        // Shader modules

        const auto vert_shader_code = read_bytes("shaders/vert.spv");
        const auto frag_shader_code = read_bytes("shaders/frag.spv");
        const VkShaderModule vert_shader_module =
            create_shader_module(vert_shader_code);
        const VkShaderModule frag_shader_module =
            create_shader_module(frag_shader_code);

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

        const VkPipelineVertexInputStateCreateInfo vertex_input_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount   = 0,
            .pVertexBindingDescriptions      = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions    = nullptr,
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
            .frontFace               = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable         = VK_FALSE,
            .lineWidth               = 1.0f,
        };

        // Multisampling

        const VkPipelineMultisampleStateCreateInfo multisampling = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable  = VK_FALSE,
        };

        // Color blending

        const VkPipelineColorBlendAttachmentState color_blend_attachment = {
            .blendEnable = VK_FALSE,
            .colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};

        const VkPipelineColorBlendStateCreateInfo color_blending = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable   = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments    = &color_blend_attachment,
        };

        // Create pipeline layout

        const VkPipelineLayoutCreateInfo pipeline_layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

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
            .pColorBlendState    = &color_blending,
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

    void create_framebuffers()
    {
        swap_chain_framebuffers_.resize(swap_chain_image_views_.size());
        for (gsl::index i = 0; i < std::ssize(swap_chain_image_views_); ++i)
        {
            const VkImageView attachments[] = {swap_chain_image_views_.at(i)};
            const VkFramebufferCreateInfo framebuffer_info = {
                .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass      = render_pass_,
                .attachmentCount = std::size(attachments),
                .pAttachments    = &attachments[0],
                .width           = swap_chain_extent_.width,
                .height          = swap_chain_extent_.height,
                .layers          = 1,
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
        QueueFamilyIndices queue_family_indices =
            find_queue_families(physical_device_);

        VkCommandPoolCreateInfo pool_info = {
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags            = 0,
            .queueFamilyIndex = queue_family_indices.graphics_family.value()};

        if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create command pool!");
        }

        log_info("Created command pool");
    }

    void create_command_buffers()
    {
        command_buffers_.resize(swap_chain_framebuffers_.size());

        const VkCommandBufferAllocateInfo alloc_info = {
            .sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool_,
            .level       = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = command_buffers_.size(),
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
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

            if (vkBeginCommandBuffer(command_buffers_.at(i), &begin_info) !=
                VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to begin recording command buffer!");
            }

            const VkClearValue clear_color = {0.0f, 0.0f, 0.0f, 1.0f};

            const VkRenderPassBeginInfo render_pass_info = {
                .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass  = render_pass_,
                .framebuffer = swap_chain_framebuffers_.at(i),
                .renderArea  = {.offset = {0, 0}, .extent = swap_chain_extent_},
                .clearValueCount = 1,
                .pClearValues    = &clear_color,
            };

            vkCmdBeginRenderPass(command_buffers_.at(i), &render_pass_info,
                                 VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(command_buffers_.at(i),
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              graphics_pipeline_);
            vkCmdDraw(command_buffers_.at(i), 3, 1, 0, 0);
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
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        const VkFenceCreateInfo fence_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT};

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

        uint32_t image_index = 0;
        vkAcquireNextImageKHR(device_, swap_chain_, UINT64_MAX,
                              image_available_semaphores_.at(current_frame_),
                              VK_NULL_HANDLE, &image_index);

        if (images_in_flight_.at(image_index) != VK_NULL_HANDLE)
        {
            vkWaitForFences(device_, 1, &images_in_flight_.at(image_index),
                            VK_TRUE, UINT64_MAX);
        }
        images_in_flight_.at(image_index) =
            in_flight_fences_.at(current_frame_);

        const VkSemaphore wait_semaphores[] = {
            image_available_semaphores_.at(current_frame_)};
        const VkPipelineStageFlags wait_stages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const VkSemaphore signal_semaphores[] = {
            render_finished_semaphores_.at(current_frame_)};

        const VkSubmitInfo submit_info = {
            .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount   = std::size(wait_semaphores),
            .pWaitSemaphores      = &wait_semaphores[0],
            .pWaitDstStageMask    = &wait_stages[0],
            .commandBufferCount   = std::size(command_buffers_),
            .pCommandBuffers      = &command_buffers_.at(image_index),
            .signalSemaphoreCount = std::size(signal_semaphores),
            .pSignalSemaphores    = &signal_semaphores[0]};

        vkResetFences(device_, 1, &in_flight_fences_.at(current_frame_));
        if (vkQueueSubmit(graphics_queue_, 1, &submit_info,
                          in_flight_fences_.at(current_frame_)) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        const VkSwapchainKHR swap_chains[] = {swap_chain_};

        const VkPresentInfoKHR present_info = {
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = std::size(signal_semaphores),
            .pWaitSemaphores    = &signal_semaphores[0],
            .swapchainCount     = std::size(swap_chains),
            .pSwapchains        = &swap_chains[0],
            .pImageIndices      = &image_index};

        vkQueuePresentKHR(present_queue_, &present_info);

        current_frame_ = (current_frame_ + 1) % max_frames_in_flight_;
    }

    // Helpers

    bool is_device_suitable(VkPhysicalDevice device) const
    {
        const bool extensions_supported =
            check_device_extension_support(device);
        const bool swap_chain_adequate = [this, device,
                                          extensions_supported]() {
            if (!extensions_supported)
            {
                return false;
            }
            const auto swap_details = query_swap_chain_support(device);
            return !swap_details.formats.empty() &&
                   !swap_details.present_modes.empty();
        }();
        const QueueFamilyIndices indices = find_queue_families(device);
        return indices.is_complete() && extensions_supported &&
               swap_chain_adequate;
    }

    bool check_device_extension_support(VkPhysicalDevice device) const
    {
        uint32_t extension_count;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                             nullptr);
        std::vector<VkExtensionProperties> available_extensions(
            extension_count);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                             available_extensions.data());

        // TODO: Better
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

    QueueFamilyIndices find_queue_families(VkPhysicalDevice device) const
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
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_,
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

    SwapChainSupportDetails query_swap_chain_support(
        VkPhysicalDevice device) const
    {
        SwapChainSupportDetails details = {};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_,
                                                  &details.capabilities);
        uint32_t format_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count,
                                             nullptr);
        if (format_count != 0)
        {
            details.formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                device, surface_, &format_count, details.formats.data());
        }

        uint32_t present_mode_count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_,
                                                  &present_mode_count, nullptr);
        if (present_mode_count != 0)
        {
            details.present_modes.resize(present_mode_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                device, surface_, &present_mode_count,
                details.present_modes.data());
        }
        return details;
    }

    VkSurfaceFormatKHR choose_swap_surface_format(
        const std::vector<VkSurfaceFormatKHR> &available_formats) const
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

    VkPresentModeKHR choose_swap_present_mode(
        const std::vector<VkPresentModeKHR> &available_present_modes) const
    {
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

    VkExtent2D choose_swap_extent(
        const VkSurfaceCapabilitiesKHR &capabilities) const noexcept
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        else
        {
            return {std::clamp(gsl::narrow_cast<uint32_t>(width_),
                               capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width),
                    std::clamp(gsl::narrow_cast<uint32_t>(height_),
                               capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height)};
        }
    }

    [[gsl::suppress(26490)]] // reinterpret_cast
    VkShaderModule
    create_shader_module(const std::vector<char> &code) const
    {
        VkShaderModuleCreateInfo create_info = {
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<const uint32_t *>(code.data()),
        };

        VkShaderModule shader_module;
        if (vkCreateShaderModule(device_, &create_info, nullptr,
                                 &shader_module) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create shader module!");
        }
        return shader_module;
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
