/* e2_gpu_l0.c - dlopen/dlsym loader for the Level Zero runtime. */
#include "e2_gpu_l0.h"
#include <dlfcn.h>

ze_result_t (*zeInit)(ze_init_flags_t) = 0;
ze_result_t (*zeDriverGet)(uint32_t *, ze_driver_handle_t *) = 0;
ze_result_t (*zeDeviceGet)(ze_driver_handle_t, uint32_t *, ze_device_handle_t *) = 0;
ze_result_t (*zeDeviceGetProperties)(ze_device_handle_t, ze_device_properties_t *) = 0;
ze_result_t (*zeDeviceGetMemoryProperties)(ze_device_handle_t, uint32_t *,
                                           ze_device_memory_properties_t *) = 0;
ze_result_t (*zeContextCreate)(ze_driver_handle_t, const ze_context_desc_t *,
                               ze_context_handle_t *) = 0;
ze_result_t (*zeContextDestroy)(ze_context_handle_t) = 0;
ze_result_t (*zeCommandQueueCreate)(ze_context_handle_t, ze_device_handle_t,
                                    const ze_command_queue_desc_t *,
                                    ze_command_queue_handle_t *) = 0;
ze_result_t (*zeCommandQueueDestroy)(ze_command_queue_handle_t) = 0;
ze_result_t (*zeCommandQueueExecuteCommandLists)(ze_command_queue_handle_t, uint32_t,
                                                 ze_command_list_handle_t *, void *) = 0;
ze_result_t (*zeCommandQueueSynchronize)(ze_command_queue_handle_t, uint64_t) = 0;
ze_result_t (*zeCommandListCreateImmediate)(ze_context_handle_t, ze_device_handle_t,
                                            const ze_command_queue_desc_t *,
                                            ze_command_list_handle_t *) = 0;
ze_result_t (*zeCommandListHostSynchronize)(ze_command_list_handle_t, uint64_t) = 0;
ze_result_t (*zeEventCreate)(ze_event_pool_handle_t, const ze_event_desc_t *,
                             ze_event_handle_t *) = 0;
ze_result_t (*zeEventDestroy)(ze_event_handle_t) = 0;
ze_result_t (*zeEventHostSynchronize)(ze_event_handle_t, uint64_t) = 0;
ze_result_t (*zeEventHostReset)(ze_event_handle_t) = 0;
ze_result_t (*zeEventQueryStatus)(ze_event_handle_t) = 0;
ze_result_t (*zeEventPoolCreate)(ze_context_handle_t, const ze_event_pool_desc_t *,
                                 uint32_t, ze_device_handle_t *,
                                 ze_event_pool_handle_t *) = 0;
ze_result_t (*zeEventPoolDestroy)(ze_event_pool_handle_t) = 0;
ze_result_t (*zeCommandListCreate)(ze_context_handle_t, ze_device_handle_t,
                                   const ze_command_list_desc_t *,
                                   ze_command_list_handle_t *) = 0;
ze_result_t (*zeCommandListDestroy)(ze_command_list_handle_t) = 0;
ze_result_t (*zeCommandListReset)(ze_command_list_handle_t) = 0;
ze_result_t (*zeCommandListClose)(ze_command_list_handle_t) = 0;
ze_result_t (*zeCommandListAppendBarrier)(ze_command_list_handle_t, ze_event_handle_t,
                                          uint32_t, ze_event_handle_t *) = 0;
ze_result_t (*zeCommandListAppendWriteGlobalTimestamp)(
    ze_command_list_handle_t, uint64_t *, ze_event_handle_t, uint32_t,
    ze_event_handle_t *) = 0;
ze_result_t (*zeDeviceGetGlobalTimestamps)(ze_device_handle_t, uint64_t *,
                                           uint64_t *) = 0;
ze_result_t (*zeCommandListAppendLaunchKernel)(ze_command_list_handle_t, ze_kernel_handle_t,
                                               const ze_group_count_t *, ze_event_handle_t,
                                               uint32_t, ze_event_handle_t *) = 0;
ze_result_t (*zeModuleCreate)(ze_context_handle_t, ze_device_handle_t,
                              const ze_module_desc_t *, ze_module_handle_t *,
                              ze_module_build_log_handle_t *) = 0;
ze_result_t (*zeModuleDestroy)(ze_module_handle_t) = 0;
ze_result_t (*zeModuleBuildLogGetString)(ze_module_build_log_handle_t, size_t *, char *) = 0;
ze_result_t (*zeModuleBuildLogDestroy)(ze_module_build_log_handle_t) = 0;
ze_result_t (*zeKernelCreate)(ze_module_handle_t, const ze_kernel_desc_t *,
                              ze_kernel_handle_t *) = 0;
ze_result_t (*zeKernelDestroy)(ze_kernel_handle_t) = 0;
ze_result_t (*zeKernelSetGroupSize)(ze_kernel_handle_t, uint32_t, uint32_t, uint32_t) = 0;
ze_result_t (*zeKernelSetArgumentValue)(ze_kernel_handle_t, uint32_t, size_t,
                                        const void *) = 0;
ze_result_t (*zeMemAllocShared)(ze_context_handle_t, const ze_device_mem_alloc_desc_t *,
                                const ze_host_mem_alloc_desc_t *, size_t, size_t,
                                ze_device_handle_t, void **) = 0;
ze_result_t (*zeMemFree)(ze_context_handle_t, void *) = 0;

static void *g_handle;

static void *l0_sym(void *h, const char *name) {
    return dlsym(h, name);
}

#define LOAD(fn) do { \
    *(void **)(&fn) = l0_sym(g_handle, #fn); \
    if (!(fn)) { dlclose(g_handle); g_handle = 0; return -1; } \
} while (0)

int e2_gpu_l0_open(void) {
    if (g_handle) return 0;
    g_handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!g_handle) return -1;
    LOAD(zeInit);
    LOAD(zeDriverGet);
    LOAD(zeDeviceGet);
    LOAD(zeDeviceGetProperties);
    LOAD(zeDeviceGetMemoryProperties);
    LOAD(zeContextCreate);
    LOAD(zeContextDestroy);
    LOAD(zeCommandQueueCreate);
    LOAD(zeCommandQueueDestroy);
    LOAD(zeCommandQueueExecuteCommandLists);
    LOAD(zeCommandQueueSynchronize);
    LOAD(zeCommandListCreate);
    LOAD(zeCommandListCreateImmediate);
    LOAD(zeCommandListHostSynchronize);
    LOAD(zeEventCreate);
    LOAD(zeEventDestroy);
    LOAD(zeEventHostSynchronize);
    LOAD(zeEventHostReset);
    LOAD(zeEventQueryStatus);
    LOAD(zeEventPoolCreate);
    LOAD(zeEventPoolDestroy);
    LOAD(zeCommandListDestroy);
    LOAD(zeCommandListReset);
    LOAD(zeCommandListClose);
    LOAD(zeCommandListAppendBarrier);
    LOAD(zeCommandListAppendWriteGlobalTimestamp);
    LOAD(zeDeviceGetGlobalTimestamps);
    LOAD(zeCommandListAppendLaunchKernel);
    LOAD(zeModuleCreate);
    LOAD(zeModuleDestroy);
    LOAD(zeModuleBuildLogGetString);
    LOAD(zeModuleBuildLogDestroy);
    LOAD(zeKernelCreate);
    LOAD(zeKernelDestroy);
    LOAD(zeKernelSetGroupSize);
    LOAD(zeKernelSetArgumentValue);
    LOAD(zeMemAllocShared);
    LOAD(zeMemFree);
    return 0;
}

void e2_gpu_l0_close(void) {
    if (g_handle) { dlclose(g_handle); g_handle = 0; }
}
