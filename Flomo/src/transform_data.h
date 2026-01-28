
#pragma once

#include <assert.h>
#include <stdlib.h>
#include <vector>

#include "raylib.h"
#include "raymath.h"
#include "math_utils.h"
#include "bvh_parser.h"

//----------------------------------------------------------------------------------
// Transform Data
//----------------------------------------------------------------------------------

// Structure for containing a sampled pose as joint transforms
typedef struct
{
    int jointCount;
    std::vector<int> parents;
    std::vector<bool> endSite;
    std::vector<Vector3> localPositions;
    std::vector<Quaternion> localRotations;
    std::vector<Vector3> globalPositions;
    std::vector<Quaternion> globalRotations;

} TransformData;

static inline void TransformDataInit(TransformData* data)
{
    data->jointCount = 0;
    data->parents.clear();
    data->endSite.clear();
    data->localPositions.clear();
    data->localRotations.clear();
    data->globalPositions.clear();
    data->globalRotations.clear();
}

// Resize the transform buffer according to the given BVH data and record the joint
// parents and end-sites.
static inline void TransformDataResize(TransformData* data, const BVHData* bvh)
{
    data->jointCount = bvh->jointCount;
    data->parents.resize(data->jointCount);
    data->endSite.resize(data->jointCount);
    data->localPositions.resize(data->jointCount);
    data->localRotations.resize(data->jointCount);
    data->globalPositions.resize(data->jointCount);
    data->globalRotations.resize(data->jointCount);

    for (int i = 0; i < data->jointCount; i++)
    {
        data->endSite[i] = bvh->joints[i].endSite;
        data->parents[i] = bvh->joints[i].parent;
    }
}

static inline void TransformDataFree(TransformData* data)
{
    // Vectors automatically clean up their memory when they go out of scope,
    // but we can clear them explicitly if needed
    data->parents.clear();
    data->endSite.clear();
    data->localPositions.clear();
    data->localRotations.clear();
    data->globalPositions.clear();
    data->globalRotations.clear();

    // If you want to ensure memory is freed, you can use swap trick
    // but usually clear() is sufficient as vectors manage their own memory
}

// Sample joint transforms from a given frame of the BVH file and with a given scale
static void TransformDataSampleFrame(TransformData* data, const BVHData* bvh, const int _frame, float scale)
{
    // Clamp the frame index in range.
    int frame = _frame < 0 ? 0 : _frame >= bvh->frameCount ? bvh->frameCount - 1 : _frame;

    int offset = 0;
    for (int i = 0; i < bvh->jointCount; i++)
    {
        Vector3 position = Vector3Scale(bvh->joints[i].offset, scale);
        Quaternion rotation = QuaternionIdentity();

        for (int c = 0; c < bvh->joints[i].channelCount; c++)
        {
            switch (bvh->joints[i].channels[c])
            {
            case CHANNEL_X_POSITION:
                position.x = scale * bvh->motionData[frame * bvh->channelCount + offset];
                offset++;
                break;

            case CHANNEL_Y_POSITION:
                position.y = scale * bvh->motionData[frame * bvh->channelCount + offset];
                offset++;
                break;

            case CHANNEL_Z_POSITION:
                position.z = scale * bvh->motionData[frame * bvh->channelCount + offset];
                offset++;
                break;

            case CHANNEL_X_ROTATION:
                rotation = QuaternionMultiply(rotation, QuaternionFromAxisAngle(
                    Vector3{ 1, 0, 0 }, DEG2RAD * bvh->motionData[frame * bvh->channelCount + offset]));
                offset++;
                break;

            case CHANNEL_Y_ROTATION:
                rotation = QuaternionMultiply(rotation, QuaternionFromAxisAngle(
                    Vector3{ 0, 1, 0 }, DEG2RAD * bvh->motionData[frame * bvh->channelCount + offset]));
                offset++;
                break;

            case CHANNEL_Z_ROTATION:
                rotation = QuaternionMultiply(rotation, QuaternionFromAxisAngle(
                    Vector3{ 0, 0, 1 }, DEG2RAD * bvh->motionData[frame * bvh->channelCount + offset]));
                offset++;
                break;
            }
        }

        data->localPositions[i] = position;
        data->localRotations[i] = rotation;
    }

    assert(offset == bvh->channelCount);
}

// Sample the nearest frame to the given time
static void TransformDataSampleFrameNearest(TransformData* data, const BVHData* bvh, float time, float scale)
{
    const int frame = ClampInt((int)(time / bvh->frameTime + 0.5f), 0, bvh->frameCount - 1);
    TransformDataSampleFrame(data, bvh, frame, scale);
}

// Perform a basic linear interpolation of the frame data in the BVH file
static void TransformDataSampleFrameLinear(
    TransformData* data,
    TransformData* tmp0,
    TransformData* tmp1,
    const BVHData* bvh,
    float time,
    float scale)
{
    const float alpha = fmodf(time / bvh->frameTime, 1.0f);
    const int frame0 = ClampInt((int)(time / bvh->frameTime) + 0, 0, bvh->frameCount - 1);
    const int frame1 = ClampInt((int)(time / bvh->frameTime) + 1, 0, bvh->frameCount - 1);

    TransformDataSampleFrame(tmp0, bvh, frame0, scale);
    TransformDataSampleFrame(tmp1, bvh, frame1, scale);

    for (int i = 0; i < data->jointCount; i++)
    {
        data->localPositions[i] = Vector3Lerp(tmp0->localPositions[i], tmp1->localPositions[i], alpha);
        data->localRotations[i] = QuaternionSlerp(tmp0->localRotations[i], tmp1->localRotations[i], alpha);
    }
}

// Perform a cubic interpolation of the frame data in the BVH file
static void TransformDataSampleFrameCubic(
    TransformData* data,
    TransformData* tmp0,
    TransformData* tmp1,
    TransformData* tmp2,
    TransformData* tmp3,
    const BVHData* bvh,
    float time,
    float scale)
{
    const float alpha = fmodf(time / bvh->frameTime, 1.0f);
    const int frame0 = ClampInt((int)(time / bvh->frameTime) - 1, 0, bvh->frameCount - 1);
    const int frame1 = ClampInt((int)(time / bvh->frameTime) + 0, 0, bvh->frameCount - 1);
    const int frame2 = ClampInt((int)(time / bvh->frameTime) + 1, 0, bvh->frameCount - 1);
    const int frame3 = ClampInt((int)(time / bvh->frameTime) + 2, 0, bvh->frameCount - 1);

    TransformDataSampleFrame(tmp0, bvh, frame0, scale);
    TransformDataSampleFrame(tmp1, bvh, frame1, scale);
    TransformDataSampleFrame(tmp2, bvh, frame2, scale);
    TransformDataSampleFrame(tmp3, bvh, frame3, scale);

    for (int i = 0; i < data->jointCount; i++)
    {
        data->localPositions[i] = Vector3InterpolateCubic(
            tmp0->localPositions[i], tmp1->localPositions[i],
            tmp2->localPositions[i], tmp3->localPositions[i], alpha);

        data->localRotations[i] = QuaternionInterpolateCubic(
            tmp0->localRotations[i], tmp1->localRotations[i],
            tmp2->localRotations[i], tmp3->localRotations[i], alpha);
    }
}

// Compute forward kinematics on the transform buffer
static void TransformDataForwardKinematics(TransformData* data)
{
    for (int i = 0; i < data->jointCount; i++)
    {
        const int p = data->parents[i];
        assert(p <= i);

        if (p == -1)
        {
            data->globalPositions[i] = data->localPositions[i];
            data->globalRotations[i] = data->localRotations[i];
        }
        else
        {
            data->globalPositions[i] = Vector3Add(
                Vector3RotateByQuaternion(data->localPositions[i], data->globalRotations[p]), 
                data->globalPositions[p]);

            data->globalRotations[i] = QuaternionMultiply(data->globalRotations[p], data->localRotations[i]);
        }
    }
}