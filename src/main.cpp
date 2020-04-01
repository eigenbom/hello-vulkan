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
#include <fstream>
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

static std::vector<char> read_bytes(const std::string &filename)
{
    std::ifstream file{filename, std::ios::ate | std::ios::binary};
    if (!file.is_open())
    {
        throw std::runtime_error(fmt::format("Failed to open {}!", filename).c_str());
    }
    std::size_t size = static_cast<std::size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    file.close();
    return buffer;
}

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
    const int width_    = 800;
    const int height_   = 600;
    GLFWwindow *window_ = nullptr;

    VkInstance instance_                                = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_                   = nullptr;
    VkDevice device_                                    = nullptr;
    VkQueue graphics_queue_                             = nullptr;
    VkQueue present_queue_                              = nullptr;
    VkSurfaceKHR surface_                               = VK_NULL_HANDLE;
    VkSwapchainKHR swap_chain_                          = VK_NULL_HANDLE;
    VkFormat swap_chain_image_format_                   = {};
    VkExtent2D swap_chain_extent_                       = {};
    std::vector<VkImage> swap_chain_images_             = {};
    std::vector<VkImageView> swap_chain_image_views_    = {};
    VkRenderPass render_pass_                           = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_                   = VK_NULL_HANDLE;
    VkPipeline graphics_pipeline_                       = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> swap_chain_framebuffers_ = {};
    VkCommandPool command_pool_                         = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_       = {};

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
            throw std::runtime_error("Couldn't initialise GLFW!");
        }
        log_info("Initialised GLFW");
        glfwSetErrorCallback(error_callback);

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ = glfwCreateWindow(width_, height_, "Hello Vulkan", nullptr, nullptr);
        if (!window_)
        {
            throw std::runtime_error("Window or OpenGL context creation failed!");
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

        const VkApplicationInfo app_info{.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                         .pApplicationName   = "Hello Vulkan",
                                         .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                         .pEngineName        = "No Engine",
                                         .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
                                         .apiVersion         = VK_API_VERSION_1_0};

        uint32_t required_extensions_count = 0;
        const char **required_extensions =
            glfwGetRequiredInstanceExtensions(&required_extensions_count);

        VkInstanceCreateInfo create_info{
            .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledLayerCount =
                enable_validation_ ? static_cast<uint32_t>(validation_layers_.size()) : 0,
            .ppEnabledLayerNames     = enable_validation_ ? validation_layers_.data() : nullptr,
            .enabledExtensionCount   = required_extensions_count,
            .ppEnabledExtensionNames = required_extensions,
        };

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
            throw std::runtime_error("Failed to create instance!");
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
        const auto it = std::find_if(devices.begin(), devices.end(),
                                     [this](auto d) { return is_device_suitable(d); });
        if (it == devices.end())
        {
            throw std::runtime_error("No suitable Vulkan-compatible devices found!");
        };
        physical_device_ = *it;

        log_info("Found physical device");
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
        VkDeviceCreateInfo create_info           = {
            .sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size()),
            .pQueueCreateInfos    = queue_create_infos.data(),
            .enabledLayerCount =
                enable_validation_ ? static_cast<uint32_t>(validation_layers_.size()) : 0,
            .ppEnabledLayerNames     = enable_validation_ ? validation_layers_.data() : nullptr,
            .enabledExtensionCount   = static_cast<uint32_t>(device_extensions_.size()),
            .ppEnabledExtensionNames = device_extensions_.data(),
            .pEnabledFeatures        = &device_features,
        };

        if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create logical device!");
        }
        vkGetDeviceQueue(device_, indices.graphics_family.value(), 0, &graphics_queue_);
        vkGetDeviceQueue(device_, indices.present_family.value(), 0, &present_queue_);

        log_info("Created logical device");
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

        VkSwapchainCreateInfoKHR create_info = {
            .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface          = surface_,
            .minImageCount    = min_images,
            .imageFormat      = surface_format.format,
            .imageColorSpace  = surface_format.colorSpace,
            .imageExtent      = extent,
            .imageArrayLayers = 1,
            .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform     = swap_chain_support.capabilities.currentTransform,
            .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode      = present_mode,
            .clipped          = VK_TRUE,
            .oldSwapchain     = VK_NULL_HANDLE,
        };

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

        if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swap chain!");
        }

        uint32_t swap_chain_image_count;
        vkGetSwapchainImagesKHR(device_, swap_chain_, &swap_chain_image_count, nullptr);
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
        for (std::size_t i = 0; i < swap_chain_images_.size(); ++i)
        {
            // clang-format off
            VkImageViewCreateInfo create_info = {
                .sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image      = swap_chain_images_[i],
                .viewType   = VK_IMAGE_VIEW_TYPE_2D,
                .format     = swap_chain_image_format_,
                .components = {
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
                }
            };
            // clang-format on

            if (vkCreateImageView(device_, &create_info, nullptr, &swap_chain_image_views_[i]) !=
                VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image views!");
            }
        }

        log_info("Created image views");
    }

    void create_render_pass()
    {
        VkAttachmentDescription color_attachment = {
            .format         = swap_chain_image_format_,
            .samples        = VK_SAMPLE_COUNT_1_BIT,
            .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        VkAttachmentReference color_attachment_ref = {
            .attachment = 0,
            .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };


        VkSubpassDescription subpass = {
            .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &color_attachment_ref,
        };

        VkRenderPassCreateInfo render_pass_info = {
            .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments    = &color_attachment,
            .subpassCount    = 1,
            .pSubpasses      = &subpass,
        };

        if (vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_) != VK_SUCCESS)
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
        // log_info("Loaded file shaders/vert.spv ({}b)", vert_shader_code.size());
        // log_info("Loaded file shaders/frag.spv ({}b)", frag_shader_code.size());
        VkShaderModule vert_shader_module = create_shader_module(vert_shader_code);
        VkShaderModule frag_shader_module = create_shader_module(frag_shader_code);

        VkPipelineShaderStageCreateInfo vert_shader_stage_info = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_shader_module,
            .pName  = "main",
        };

        VkPipelineShaderStageCreateInfo frag_shader_stage_info = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_shader_module,
            .pName  = "main",
        };

        VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info,
                                                           frag_shader_stage_info};

        // Vertex input

        VkPipelineVertexInputStateCreateInfo vertex_input_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount   = 0,
            .pVertexBindingDescriptions      = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions    = nullptr,
        };

        // Input assembly

        VkPipelineInputAssemblyStateCreateInfo input_assembly = {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        // Viewport

        VkViewport viewport = {
            .x        = 0.0f,
            .y        = 0.0f,
            .width    = static_cast<float>(swap_chain_extent_.width),
            .height   = static_cast<float>(swap_chain_extent_.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        VkRect2D scissor = {
            .offset = {0, 0},
            .extent = swap_chain_extent_,
        };

        VkPipelineViewportStateCreateInfo viewport_state = {
            .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports    = &viewport,
            .scissorCount  = 1,
            .pScissors     = &scissor,
        };

        // Rasterizer

        VkPipelineRasterizationStateCreateInfo rasterizer = {
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable        = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode             = VK_POLYGON_MODE_FILL,
            .cullMode                = VK_CULL_MODE_BACK_BIT,
            .frontFace               = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable         = VK_FALSE,
            .lineWidth               = 1.0f,
        };

        // Multisampling

        VkPipelineMultisampleStateCreateInfo multisampling{
            .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable  = VK_FALSE,
        };

        // Color blending

        VkPipelineColorBlendAttachmentState color_blend_attachment = {
            .blendEnable    = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};

        VkPipelineColorBlendStateCreateInfo color_blending = {
            .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable   = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments    = &color_blend_attachment,
        };

        // Create pipeline layout

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        // Create graphics pipeline

        const VkGraphicsPipelineCreateInfo pipeline_info{
            .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount          = 2,
            .pStages             = shader_stages,
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

        if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
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
        for (std::size_t i = 0; i < swap_chain_image_views_.size(); ++i)
        {
            VkImageView attachments[] = {swap_chain_image_views_[i]};
            VkFramebufferCreateInfo framebuffer_info = {
                .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass      = render_pass_,
                .attachmentCount = 1,
                .pAttachments    = attachments,
                .width           = swap_chain_extent_.width,
                .height          = swap_chain_extent_.height,
                .layers          = 1,
            };

            if (vkCreateFramebuffer(device_, &framebuffer_info, nullptr,
                                    &swap_chain_framebuffers_[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create framebuffer!");
            }
        }
    }

    void create_command_pool()
    {
        QueueFamilyIndices queue_family_indices = find_queue_families(physical_device_);

        VkCommandPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                          .flags = 0,
                                          .queueFamilyIndex =
                                              queue_family_indices.graphics_family.value()};

        if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create command pool!");
        }
    }

    void create_command_buffers()
    {
        command_buffers_.resize(swap_chain_framebuffers_.size());

        VkCommandBufferAllocateInfo alloc_info = {
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = command_pool_,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(command_buffers_.size()),
        };

        if (vkAllocateCommandBuffers(device_, &alloc_info, command_buffers_.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate command buffers!");
        }

        // Record

        for (std::size_t i = 0; i < command_buffers_.size(); ++i)
        {
            VkCommandBufferBeginInfo begin_info = {.sType =
                                                       VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

            if (vkBeginCommandBuffer(command_buffers_[i], &begin_info) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin recording command buffer!");
            }
        }
    }

    // Helpers

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

    VkShaderModule create_shader_module(const std::vector<char> &code) const
    {
        VkShaderModuleCreateInfo create_info = {
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<const uint32_t *>(code.data()),
        };

        VkShaderModule shader_module;
        if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create shader module!");
        }
        return shader_module;
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
