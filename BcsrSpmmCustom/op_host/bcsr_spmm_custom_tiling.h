
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BcsrSpmmCustomTilingData)
  TILING_DATA_FIELD_DEF(int32_t, M);
  TILING_DATA_FIELD_DEF(int32_t, N);
  TILING_DATA_FIELD_DEF(int32_t, K);

  TILING_DATA_FIELD_DEF(uint32_t, totalLength);

  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BcsrSpmmCustom, BcsrSpmmCustomTilingData)
}
