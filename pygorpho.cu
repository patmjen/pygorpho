#include "pygorpho.cuh"
#include "general_morph.cuh"
#include "flat_linear_morph.cuh"
#include "strel.cuh"
#include "view.cuh"

template <class Ty>
void doGenDilateErode(void *resPtr, const void *volPtr, const void *strelPtr, int3 volSize, int3 strelSize,
    int op, int3 blockSize)
{
    gpho::HostView<Ty> res(static_cast<Ty *>(resPtr), volSize);
    gpho::HostView<const Ty> vol(static_cast<const Ty *>(volPtr), volSize);
    gpho::HostView<const Ty> strel(static_cast<const Ty *>(strelPtr), strelSize);

    if (op == MOP_DILATE) {
        gpho::genDilate(res, vol, strel, blockSize);
    } else if (op == MOP_ERODE) {
        gpho::genErode(res, vol, strel, blockSize);
    } else {
        throw ERR_BAD_MORPH_OP;
    }
}

template <class Ty>
void doFlatLinearDilateErode(void *resPtr, const void *volPtr, const int *lineStepsPtr, 
    const int *lineLensPtr, int3 volSize, int numLines, int op, int3 blockSize)
{
    std::vector<gpho::LineSeg> lines;
    lines.reserve(numLines);
    for (int i = 0; i < numLines; ++i) {
        int3 step = make_int3(
            lineStepsPtr[i * 3 + 0],
            lineStepsPtr[i * 3 + 1],
            lineStepsPtr[i * 3 + 2]
        );
        lines.push_back(gpho::LineSeg(step, lineLensPtr[i]));
    }
    
    gpho::HostView<Ty> res(static_cast<Ty *>(resPtr), volSize);
    gpho::HostView<const Ty> vol(static_cast<const Ty *>(volPtr), volSize);

    if (op == MOP_DILATE) {
        gpho::flatLinearDilateErode<gpho::MORPH_DILATE>(res, vol, lines, blockSize);
    } else if (op == MOP_ERODE) {
        gpho::flatLinearDilateErode<gpho::MORPH_ERODE>(res, vol, lines, blockSize);
    } else {
        throw ERR_BAD_MORPH_OP;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

PYGORPHO_API int pyDilateOp() { return MOP_DILATE; };
PYGORPHO_API int pyErodeOp() { return MOP_ERODE; };

PYGORPHO_API int pyFlatBallApproxStrel(int *lineSteps, int *lineLens, int radius)
{
    // NOTE: It is assumed that lineSteps and lineLens point to allocated memory block of adequate size
    std::vector<gpho::LineSeg> lines;
    TRY_OR_RETURN_ERROR(
        lines = gpho::flatBallApprox(radius);
    )
    int i = 0;
    for (const auto& line : lines) {
        lineSteps[i * 3 + 0] = line.step.x;
        lineSteps[i * 3 + 1] = line.step.y;
        lineSteps[i * 3 + 2] = line.step.z;
        lineLens[i] = line.length;
        ++i;
    }
    return SUCCESS;
}

PYGORPHO_API int pyGenDilateErode(void *res, const void *vol, const void *strel, 
    int volX, int volY, int volZ, int strelX, int strelY, int strelZ, int type, int op, 
    int blockX, int blockY, int blockZ)
{
    int3 volSize = make_int3(volX, volY, volZ);
    int3 strelSize = make_int3(strelX, strelY, strelZ);
    int3 blockSize = make_int3(blockX, blockY, blockZ);
    TRY_OR_RETURN_ERROR(
        typeDispatch(type, doGenDilateErode, res, vol, strel, volSize, strelSize, op, blockSize);
    )
    return SUCCESS;
}

PYGORPHO_API int pyFlatLinearDilateErode(void *res, const void *vol, const int *lineSteps,
    const int *lineLens, int volX, int volY, int volZ, int numLines, int type, int op, 
    int blockX, int blockY, int blockZ)
{
    int3 volSize = make_int3(volX, volY, volZ);
    int3 blockSize = make_int3(blockX, blockY, blockZ);
    TRY_OR_RETURN_ERROR(
        typeDispatch(type, doFlatLinearDilateErode, res, vol, lineSteps, 
            lineLens, volSize, numLines, op, blockSize);
    )
    return SUCCESS;
}

#ifdef __cplusplus
}
#endif