/* e2_gpu_l0.h - minimal Level Zero (ze_api) subset, loaded at runtime via
 * dlopen("libze_loader.so.1").  No <level_zero/ze_api.h> include, no link
 * against -lze_loader, so the library still builds/runs without Level Zero.
 *
 * Only the types/constants/prototypes the GPU backend actually uses are
 * declared here, in the exact ABI layout of the installed headers.
 */
#ifndef E2_GPU_L0_H
#define E2_GPU_L0_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t  ze_result_t;
typedef uint8_t  ze_bool_t;
typedef int32_t  ze_structure_type_t;
typedef int32_t  ze_device_type_t;
typedef int32_t  ze_command_queue_mode_t;
typedef int32_t  ze_command_queue_priority_t;
typedef int32_t  ze_module_format_t;
typedef uint32_t ze_init_flags_t;
typedef uint32_t ze_context_flags_t;
typedef uint32_t ze_command_queue_flags_t;
typedef uint32_t ze_command_list_flags_t;
typedef uint32_t ze_kernel_flags_t;
typedef uint32_t ze_device_mem_alloc_flags_t;
typedef uint32_t ze_host_mem_alloc_flags_t;
typedef uint32_t ze_event_scope_flags_t;
typedef uint32_t ze_device_property_flags_t;
typedef uint32_t ze_device_memory_property_flags_t;

typedef struct _ze_driver_handle_t            *ze_driver_handle_t;
typedef struct _ze_device_handle_t            *ze_device_handle_t;
typedef struct _ze_context_handle_t           *ze_context_handle_t;
typedef struct _ze_command_queue_handle_t     *ze_command_queue_handle_t;
typedef struct _ze_command_list_handle_t      *ze_command_list_handle_t;
typedef struct _ze_event_handle_t             *ze_event_handle_t;
typedef struct _ze_module_handle_t            *ze_module_handle_t;
typedef struct _ze_kernel_handle_t            *ze_kernel_handle_t;
typedef struct _ze_module_build_log_handle_t  *ze_module_build_log_handle_t;

#define ZE_RESULT_SUCCESS       0
#define ZE_RESULT_NOT_READY     1
#define ZE_RESULT_ERROR_DEVICE_LOST              ((ze_result_t)0x70000001)
#define ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY       ((ze_result_t)0x70000002)
#define ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY     ((ze_result_t)0x70000003)
#define ZE_RESULT_ERROR_MODULE_BUILD_FAILURE     ((ze_result_t)0x70000004)
#define ZE_RESULT_ERROR_UNKNOWN                  ((ze_result_t)0x7ffffffe)
#define ZE_RESULT_ERROR_UNINITIALIZED            ((ze_result_t)0x78000001)
#define ZE_RESULT_ERROR_UNSUPPORTED_VERSION      ((ze_result_t)0x78000002)
#define ZE_RESULT_ERROR_UNSUPPORTED_FEATURE      ((ze_result_t)0x78000003)
#define ZE_RESULT_ERROR_INVALID_ARGUMENT         ((ze_result_t)0x78000004)
#define ZE_RESULT_ERROR_INVALID_NULL_HANDLE      ((ze_result_t)0x78000005)
#define ZE_RESULT_ERROR_INVALID_NULL_POINTER     ((ze_result_t)0x78000007)
#define ZE_RESULT_ERROR_INVALID_SIZE             ((ze_result_t)0x78000008)
#define ZE_RESULT_ERROR_UNSUPPORTED_SIZE         ((ze_result_t)0x78000009)
#define ZE_RESULT_ERROR_INVALID_KERNEL_NAME      ((ze_result_t)0x7800000f)
#define ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE ((ze_result_t)0x7800001a)

#define ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES   0x1
#define ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES   0x3
#define ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES 0x7
#define ZE_STRUCTURE_TYPE_CONTEXT_DESC        0xd
#define ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC  0xe
#define ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC   0xf
#define ZE_STRUCTURE_TYPE_EVENT_DESC          0x11
#define ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC 0x15
#define ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC   0x16
#define ZE_STRUCTURE_TYPE_MODULE_DESC         0x1b
#define ZE_STRUCTURE_TYPE_KERNEL_DESC         0x1d

#define ZE_INIT_FLAG_GPU_ONLY                 (1u << 0)
#define ZE_DEVICE_TYPE_GPU                    1
#define ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS    2
#define ZE_COMMAND_QUEUE_PRIORITY_NORMAL      0
#define ZE_MODULE_FORMAT_IL_SPIRV             0
#define ZE_MAX_DEVICE_NAME                    256
#define ZE_MAX_DEVICE_UUID_SIZE               16

typedef struct {
    uint8_t id[16];
} ze_driver_uuid_t;

typedef struct {
    uint8_t id[ZE_MAX_DEVICE_UUID_SIZE];
} ze_device_uuid_t;

typedef struct {
    ze_structure_type_t stype;
    void *pNext;
    ze_driver_uuid_t uuid;
    uint32_t driverVersion;
} ze_driver_properties_t;

typedef struct {
    ze_structure_type_t stype;
    void *pNext;
    ze_device_type_t type;
    uint32_t vendorId;
    uint32_t deviceId;
    ze_device_property_flags_t flags;
    uint32_t subdeviceId;
    uint32_t coreClockRate;
    uint64_t maxMemAllocSize;
    uint32_t maxHardwareContexts;
    uint32_t maxCommandQueuePriority;
    uint32_t numThreadsPerEU;
    uint32_t physicalEUSimdWidth;
    uint32_t numEUsPerSubslice;
    uint32_t numSubslicesPerSlice;
    uint32_t numSlices;
    uint64_t timerResolution;
    uint32_t timestampValidBits;
    uint32_t kernelTimestampValidBits;
    ze_device_uuid_t uuid;
    char name[ZE_MAX_DEVICE_NAME];
} ze_device_properties_t;

typedef struct {
    ze_structure_type_t stype;
    void *pNext;
    ze_device_memory_property_flags_t flags;
    uint32_t maxClockRate;
    uint32_t maxBusWidth;
    uint64_t totalSize;
    char name[ZE_MAX_DEVICE_NAME];
} ze_device_memory_properties_t;

typedef struct {
    ze_structure_type_t stype;
    const void *pNext;
    ze_context_flags_t flags;
} ze_context_desc_t;

typedef struct {
    ze_structure_type_t stype;
    const void *pNext;
    uint32_t ordinal;
    uint32_t index;
    ze_command_queue_flags_t flags;
    ze_command_queue_mode_t mode;
    ze_command_queue_priority_t priority;
} ze_command_queue_desc_t;

typedef struct {
    ze_structure_type_t stype;
    const void *pNext;
    uint32_t commandQueueGroupOrdinal;
    ze_command_list_flags_t flags;
} ze_command_list_desc_t;

typedef struct {
    ze_structure_type_t stype;
    const void *pNext;
    ze_module_format_t format;
    size_t inputSize;
    const uint8_t *pInputModule;
    const char *pBuildFlags;
    const void *pConstants;
} ze_module_desc_t;

typedef struct {
    ze_structure_type_t stype;
    const void *pNext;
    ze_kernel_flags_t flags;
    const char *pKernelName;
} ze_kernel_desc_t;

typedef struct {
    uint32_t groupCountX;
    uint32_t groupCountY;
    uint32_t groupCountZ;
} ze_group_count_t;

typedef struct {
    ze_structure_type_t stype;
    const void *pNext;
    ze_device_mem_alloc_flags_t flags;
    uint32_t ordinal;
} ze_device_mem_alloc_desc_t;

typedef struct {
    ze_structure_type_t stype;
    const void *pNext;
    ze_host_mem_alloc_flags_t flags;
} ze_host_mem_alloc_desc_t;

typedef struct {
    ze_structure_type_t stype;
    const void *pNext;
    uint32_t index;
    ze_event_scope_flags_t signal;
    ze_event_scope_flags_t wait;
} ze_event_desc_t;

typedef struct _ze_event_pool_handle_t *ze_event_pool_handle_t;

typedef struct {
    ze_structure_type_t stype;
    const void *pNext;
    uint32_t flags;
    uint32_t count;
} ze_event_pool_desc_t;

#define ZE_EVENT_POOL_FLAG_HOST_VISIBLE 1
#define ZE_EVENT_SCOPE_FLAG_HOST        1
#define ZE_STRUCTURE_TYPE_EVENT_POOL_DESC 0x10

/* Function pointers filled by e2_gpu_l0_open(). */
extern ze_result_t (*zeInit)(ze_init_flags_t flags);
extern ze_result_t (*zeDriverGet)(uint32_t *pCount, ze_driver_handle_t *phDrivers);
extern ze_result_t (*zeDeviceGet)(ze_driver_handle_t hDriver, uint32_t *pCount,
                                  ze_device_handle_t *phDevices);
extern ze_result_t (*zeDeviceGetProperties)(ze_device_handle_t hDevice,
                                            ze_device_properties_t *p);
extern ze_result_t (*zeDeviceGetMemoryProperties)(ze_device_handle_t hDevice,
                                                  uint32_t *pCount,
                                                  ze_device_memory_properties_t *p);
extern ze_result_t (*zeContextCreate)(ze_driver_handle_t hDriver,
                                      const ze_context_desc_t *desc,
                                      ze_context_handle_t *phContext);
extern ze_result_t (*zeContextDestroy)(ze_context_handle_t hContext);
extern ze_result_t (*zeCommandQueueCreate)(ze_context_handle_t hContext,
                                           ze_device_handle_t hDevice,
                                           const ze_command_queue_desc_t *desc,
                                           ze_command_queue_handle_t *phQueue);
extern ze_result_t (*zeCommandQueueDestroy)(ze_command_queue_handle_t hQueue);
extern ze_result_t (*zeCommandQueueExecuteCommandLists)(ze_command_queue_handle_t hQueue,
                                                        uint32_t n,
                                                        ze_command_list_handle_t *phLists,
                                                        void *hFence);
extern ze_result_t (*zeCommandQueueSynchronize)(ze_command_queue_handle_t hQueue,
                                                uint64_t timeout);
extern ze_result_t (*zeCommandListCreateImmediate)(ze_context_handle_t hContext,
                                                   ze_device_handle_t hDevice,
                                                   const ze_command_queue_desc_t *desc,
                                                   ze_command_list_handle_t *phCommandList);
extern ze_result_t (*zeCommandListHostSynchronize)(ze_command_list_handle_t hCommandList,
                                                   uint64_t timeout);
extern ze_result_t (*zeEventCreate)(ze_event_pool_handle_t hEventPool,
                                    const ze_event_desc_t *desc,
                                    ze_event_handle_t *phEvent);
extern ze_result_t (*zeEventDestroy)(ze_event_handle_t hEvent);
extern ze_result_t (*zeEventHostSynchronize)(ze_event_handle_t hEvent, uint64_t timeout);
extern ze_result_t (*zeEventHostReset)(ze_event_handle_t hEvent);
extern ze_result_t (*zeEventQueryStatus)(ze_event_handle_t hEvent);
extern ze_result_t (*zeEventPoolCreate)(ze_context_handle_t hContext,
                                        const ze_event_pool_desc_t *desc,
                                        uint32_t numDevices,
                                        ze_device_handle_t *phDevices,
                                        ze_event_pool_handle_t *phEventPool);
extern ze_result_t (*zeEventPoolDestroy)(ze_event_pool_handle_t hEventPool);
extern ze_result_t (*zeCommandListCreate)(ze_context_handle_t hContext,
                                          ze_device_handle_t hDevice,
                                          const ze_command_list_desc_t *desc,
                                          ze_command_list_handle_t *phList);
extern ze_result_t (*zeCommandListDestroy)(ze_command_list_handle_t hList);
extern ze_result_t (*zeCommandListReset)(ze_command_list_handle_t hList);
extern ze_result_t (*zeCommandListClose)(ze_command_list_handle_t hList);
extern ze_result_t (*zeCommandListAppendWriteGlobalTimestamp)(
    ze_command_list_handle_t hList, uint64_t *dstptr, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
extern ze_result_t (*zeDeviceGetGlobalTimestamps)(ze_device_handle_t hDevice,
                                                 uint64_t *hostTimestamp,
                                                 uint64_t *deviceTimestamp);
extern ze_result_t (*zeDeviceGetProperties2)(ze_device_handle_t, void *);
extern ze_result_t (*zeCommandListAppendBarrier)(ze_command_list_handle_t hList,
                                                 ze_event_handle_t hSignalEvent,
                                                 uint32_t numWaitEvents,
                                                 ze_event_handle_t *phWaitEvents);
extern ze_result_t (*zeCommandListAppendLaunchKernel)(ze_command_list_handle_t hList,
                                                      ze_kernel_handle_t hKernel,
                                                      const ze_group_count_t *pLaunch,
                                                      ze_event_handle_t hSignal,
                                                      uint32_t nWait,
                                                      ze_event_handle_t *phWait);
extern ze_result_t (*zeModuleCreate)(ze_context_handle_t hContext,
                                     ze_device_handle_t hDevice,
                                     const ze_module_desc_t *desc,
                                     ze_module_handle_t *phModule,
                                     ze_module_build_log_handle_t *phLog);
extern ze_result_t (*zeModuleDestroy)(ze_module_handle_t hModule);
extern ze_result_t (*zeModuleBuildLogGetString)(ze_module_build_log_handle_t hLog,
                                                size_t *pSize, char *pLog);
extern ze_result_t (*zeModuleBuildLogDestroy)(ze_module_build_log_handle_t hLog);
extern ze_result_t (*zeKernelCreate)(ze_module_handle_t hModule,
                                     const ze_kernel_desc_t *desc,
                                     ze_kernel_handle_t *phKernel);
extern ze_result_t (*zeKernelDestroy)(ze_kernel_handle_t hKernel);
extern ze_result_t (*zeKernelSetGroupSize)(ze_kernel_handle_t hKernel,
                                           uint32_t gx, uint32_t gy, uint32_t gz);
extern ze_result_t (*zeKernelSetArgumentValue)(ze_kernel_handle_t hKernel,
                                               uint32_t index, size_t size,
                                               const void *pValue);
extern ze_result_t (*zeMemAllocShared)(ze_context_handle_t hContext,
                                       const ze_device_mem_alloc_desc_t *device_desc,
                                       const ze_host_mem_alloc_desc_t *host_desc,
                                       size_t size, size_t alignment,
                                       ze_device_handle_t hDevice, void **pptr);
/* Device-local memory: what a read-only buffer the GPU streams should use.
 * Shared USM is host coherent and exists in the same DRAM on an iGPU, but it is
 * mapped as system memory, and that is not the same access path. */
extern ze_result_t (*zeCommandListAppendMemoryCopy)(
    ze_command_list_handle_t hCommandList, void *dstptr, const void *srcptr,
    size_t size, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
    ze_event_handle_t *phWaitEvents);
extern ze_result_t (*zeMemAllocDevice)(ze_context_handle_t hContext,
                                       const ze_device_mem_alloc_desc_t *device_desc,
                                       size_t size, size_t alignment,
                                       ze_device_handle_t hDevice, void **pptr);
extern ze_result_t (*zeMemFree)(ze_context_handle_t hContext, void *ptr);

/* Returns 0 on success.  Safe to call on a machine without Level Zero. */
int  e2_gpu_l0_open(void);
void e2_gpu_l0_close(void);

#ifdef __cplusplus
}
#endif

#endif /* E2_GPU_L0_H */
