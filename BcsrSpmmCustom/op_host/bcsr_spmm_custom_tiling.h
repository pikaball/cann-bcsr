
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BcsrSpmmCustomTilingData)
  TILING_DATA_FIELD_DEF(int32_t, M);
  TILING_DATA_FIELD_DEF(int32_t, N);
  TILING_DATA_FIELD_DEF(int32_t, K);

  // 行窗口总数
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);

  // mmad一次能处理的N维度有限
  // blockLength = N
  TILING_DATA_FIELD_DEF(uint32_t, mmadNum);
  TILING_DATA_FIELD_DEF(uint32_t, mmadN);
  TILING_DATA_FIELD_DEF(uint32_t, lastMmadN);
  TILING_DATA_FIELD_DEF(uint32_t, lastMmadCubeBlockNum);

  // 均分行窗口给每个cube core
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);

  // TODO: 需要考虑K不对齐的情况，A已经处理过了，读取B比较麻烦

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BcsrSpmmCustom, BcsrSpmmCustomTilingData)
}
