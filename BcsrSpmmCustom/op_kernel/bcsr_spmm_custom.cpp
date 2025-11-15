#include "kernel_operator.h"

extern "C" __global__ __aicore__ void bcsr_spmm_custom(GM_ADDR a_shape, GM_ADDR rowPtr, GM_ADDR col, GM_ADDR val, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}