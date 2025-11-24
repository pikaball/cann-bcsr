#include "kernel_operator.h"

template<typename aType, typename bType, typename cType>
class BcsrSpmmKernel {
// output C Tile size [16, 16]
constexpr uint32_t CUBE_BLOCK_M = 16;
constexpr uint32_t CUBE_BLOCK_K = 32 / sizeof(aType);
constexpr uint32_t CUBE_BLOCK_SIZE = CUBE_BLOCK_M * CUBE_BLOCK_K;

public:
    __aicore__ inline BcsrSpmmKernel() {}
    __aicore__ inline void Init(
        GM_ADDR row_ptr, GM_ADDR col, GM_ADDR val,
        GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
        int32_t M, int32_t N, int32_t K,
        uint32_t formerNum, uint32_t formerLength,
        uint32_t tailNum, uint32_t tailLength,
        uint32_t MmadNum, uint32_t MmadN,
        uint32_t lastMmadN, uint32_t lastMmadCubeBlockNum
    ) {
        // set cube only
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

        this->mmadNum = MmadNum;
        this->lastMmadN = lastMmadN;
        this->lastMmadCubeBlockNum = lastMmadCubeBlockNum;
        if (AscendC::GetBlockIdx() < formerNum) {
            this->rowWindowNum = formerLength;
            rowPtrGm.SetGlobalBuffer((__gm__ int32_t *)row_ptr + formerLength * AscendC::GetBlockIdx(), formerLength + 1);
            cGm.SetGlobalBuffer((__gm__ cType *)c + formerLength * AscendC::GetBlockIdx() * CUBE_BLOCK_M * N, 
                formerLength * CUBE_BLOCK_M * N);
        } else if (AscendC::GetBlockIdx() < formerNum + tailNum) {
            this->rowWindowNum = tailLength;
            rowPtrGm.SetGlobalBuffer((__gm__ int32_t *)row_ptr + formerLength * formerNum +
                tailLength * (AscendC::GetBlockIdx() - formerNum), tailLength + 1
            );
            cGm.SetGlobalBuffer((__gm__ cType *)c + (formerLength * formerNum +
                tailLength * (AscendC::GetBlockIdx() - formerNum)) * CUBE_BLOCK_M * N,
                tailLength * CUBE_BLOCK_M * N
            );
        }
        colGm.SetGlobalBuffer((__gm__ int32_t *)col + rowPtrGm.GetValue(0), 
            rowPtrGm.GetValue(formerLength) - rowPtrGm.GetValue(0)
        );
        valGm.SetGlobalBuffer((__gm__ aType *)val + CUBE_BLOCK_SIZE * rowPtrGm.GetValue(0),
            CUBE_BLOCK_SIZE * (rowPtrGm.GetValue(formerLength) - rowPtrGm.GetValue(0))
        );
        bGm.SetGlobalBuffer((__gm__ bType *)b, (uint64_t)K * N * sizeof(bType));

        pipe.InitBuffer(inQueueA1, 1, CUBE_BLOCK_SIZE * sizeof(aType)); // 512B
        pipe.InitBuffer(inQueueA2, 1, CUBE_BLOCK_SIZE * sizeof(aType)); // 512B
        pipe.InitBuffer(inQueueB1, 1, CUBE_BLOCK_K * (MmadN * MmadNum) * sizeof(bType));
        pipe.InitBuffer(inQueueB2, 1, CUBE_BLOCK_K * (MmadN * MmadNum) * sizeof(bType));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(cType));
    }

    __aicore__ inline void Process()
    {
        // if (this->rowWindowNum == 0) {
        //     return;
        // }

        for (int32_t row = 0; row < this->rowWindowNum; row++) {
            // 行窗口中的每块
            for (int32_t n = 0; n < rowPtrGm.GetValue(row + 1) - rowPtrGm.GetValue(row); n++) {
                int32_t col = colGm.GetValue(n);
                // TODO
                CopyIn(n);
                SplitA();
                SplitB();
                Compute();
                CopyOut(row);
            }
        }
    }

// TODO
private:
    __aicore__ inline void CopyIn() {} // DataCopy API with ND2NZ
    __aicore__ inline void SplitA() {} // no need
    __aicore__ inline void SplitB() {} // NZ2NZ, LoadData/LoadDataWithTranspose API
    __aicore__ inline void Compute() {}
    __aicore__ inline void CopyOut() {} // Fixpipe API

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<int32_t> rowPtrGm;
    AscendC::GlobalTensor<int32_t> colGm;
    AscendC::GlobalTensor<aType> valGm;

    AscendC::GlobalTensor<bType> bGM;
    AscendC::GlobalTensor<cType> cGM;

    uint32_t rowWindowNum;
    uint32_t mmadNum;
    uint32_t lastMmadN;
    uint32_t lastMmadCubeBlockNum;
};

extern "C" __global__ __aicore__ void bcsr_spmm_custom(
    GM_ADDR a_shape, GM_ADDR row_ptr, GM_ADDR col, GM_ADDR val,
    GM_ADDR b, GM_ADDR c,
    GM_ADDR workspace, GM_ADDR tiling
) {
    GET_TILING_DATA(tiling_data, tiling);

    BcsrSpmmKernel<half, half, float> op;
    op.Init(a_shape, row_ptr, col, val, b, c, workspace,
        tiling_data.M, tiling_data.N, tiling_data.K,
        tiling_data.formerNum, tiling_data.formerLength,
        tiling_data.tailNum, tiling_data.tailLength,
        tiling_data.mmadNum, tiling_data.mmadN,
        tiling_data.lastMmadN, tiling_data.lastMmadCubeBlockNum
    );
    op.Process();
}