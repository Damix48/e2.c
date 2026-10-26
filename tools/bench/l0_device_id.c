/* l0_device_id.c — prints the Level Zero deviceId of the first Intel GPU, in the hex form ocloc
 * accepts as -device. CMake runs this at configure time so the SPIR-V is built for the GPU that is
 * actually present instead of a hardcoded generation; a machine without a GPU falls back to a
 * generic LPG target (the driver JITs SPIR-V anyway, so a mismatch costs code quality, not correct
 * results). */
#include <stdio.h>
#include <level_zero/ze_api.h>

int main(void) {
    if (zeInit(0) != ZE_RESULT_SUCCESS) return 1;
    uint32_t nd = 0;
    if (zeDriverGet(&nd, NULL) != ZE_RESULT_SUCCESS) return 1;
    ze_driver_handle_t drivers[8];
    if (nd > 8) nd = 8;
    if (zeDriverGet(&nd, drivers) != ZE_RESULT_SUCCESS) return 1;
    for (uint32_t d = 0; d < nd; d++) {
        uint32_t n = 0;
        if (zeDeviceGet(drivers[d], &n, NULL) != ZE_RESULT_SUCCESS || !n) continue;
        ze_device_handle_t devs[8];
        if (n > 8) n = 8;
        zeDeviceGet(drivers[d], &n, devs);
        for (uint32_t i = 0; i < n; i++) {
            ze_device_properties_t p = { ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES };
            if (zeDeviceGetProperties(devs[i], &p) != ZE_RESULT_SUCCESS) continue;
            if (p.type != ZE_DEVICE_TYPE_GPU) continue;
            printf("0x%x", p.deviceId);
            return 0;
        }
    }
    return 1;
}
