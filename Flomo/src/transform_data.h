
#pragma once

#include <assert.h>
#include <stdlib.h>
#include <vector>

#include "definitions.h"
#include "raylib.h"
#include "raymath.h"
#include "math_utils.h"
#include "bvh_parser.h"

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

// Accumulate local rotations from root down to (but not including) joint j
static Quaternion TransformDataParentGlobalRotation(const TransformData* data, int jointIndex)
{
    int chain[64];
    int chainLen = 0;
    int j = data->parents[jointIndex];
    while (j >= 0 && chainLen < 64)
    {
        chain[chainLen++] = j;
        j = data->parents[j];
    }
    Quaternion rot = QuaternionIdentity();
    for (int k = chainLen - 1; k >= 0; k--)
    {
        rot = QuaternionMultiply(rot, data->localRotations[chain[k]]);
    }
    return rot;
}

// sum of channelCounts of joints [0, jointIndex) — gives the offset
// into a frame's channel data where jointIndex's channels begin
static int BVHJointChannelOffset(const BVHData* bvh, int jointIndex)
{
    int offset = 0;
    for (int i = 0; i < jointIndex; i++)
    {
        offset += bvh->joints[i].channelCount;
    }
    return offset;
}

// decompose quaternion into Euler angles (degrees) matching a BVH joint's
// rotation channel order. writes one float per rotation channel found.
// the channels array and channelCount come from BVHJointData.
static void QuaternionToChannelOrder(
    Quaternion q,
    const char* channels,
    int channelCount,
    float* outDegrees)
{
    // build rotation matrix from quaternion
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

    // column-major: m[col][row]
    const float m00 = 1.0f - 2.0f * (yy + zz);
    const float m01 = 2.0f * (xy + wz);
    const float m02 = 2.0f * (xz - wy);
    const float m10 = 2.0f * (xy - wz);
    const float m11 = 1.0f - 2.0f * (xx + zz);
    const float m12 = 2.0f * (yz + wx);
    const float m20 = 2.0f * (xz + wy);
    const float m21 = 2.0f * (yz - wx);
    const float m22 = 1.0f - 2.0f * (xx + yy);

    // figure out rotation axis order from channels
    int axes[3];
    int rotCount = 0;
    for (int c = 0; c < channelCount && rotCount < 3; c++)
    {
        if (channels[c] == CHANNEL_X_ROTATION) axes[rotCount++] = 0;
        else if (channels[c] == CHANNEL_Y_ROTATION) axes[rotCount++] = 1;
        else if (channels[c] == CHANNEL_Z_ROTATION) axes[rotCount++] = 2;
    }
    if (rotCount != 3) return;

    // R = R(axes[0]) * R(axes[1]) * R(axes[2])
    // store full matrix as m[row][col] for indexing
    const float mat[3][3] = {
        { m00, m10, m20 },
        { m01, m11, m21 },
        { m02, m12, m22 }
    };

    const int a0 = axes[0], a1 = axes[1], a2 = axes[2];

    // for intrinsic rotation R = Ra0 * Ra1 * Ra2:
    // middle angle from mat[a0][a2], first and third from atan2
    float angle0, angle1, angle2;

    if (a0 != a2)
    {
        // Tait-Bryan angles (all axes different)
        // sign depends on whether the axes form an even or odd permutation
        const int sign = ((a1 - a0 + 3) % 3 == 1) ? 1 : -1;

        angle1 = asinf(Clamp((float)sign * mat[a0][a2], -1.0f, 1.0f));
        const float cosA1 = cosf(angle1);

        if (fabsf(cosA1) > 1e-6f)
        {
            angle0 = atan2f(-sign * mat[a1][a2], mat[a2][a2]);
            angle2 = atan2f(-sign * mat[a0][a1], mat[a0][a0]);
        }
        else
        {
            // gimbal lock
            angle0 = atan2f(sign * mat[a2][a1], mat[a1][a1]);
            angle2 = 0.0f;
        }
    }
    else
    {
        // Euler angles (first == last axis, e.g. XYX) — rare in BVH
        angle0 = 0.0f;
        angle1 = 0.0f;
        angle2 = 0.0f;
    }

    int rotIdx = 0;
    for (int c = 0; c < channelCount; c++)
    {
        if (channels[c] >= CHANNEL_X_ROTATION)
        {
            if (rotIdx == 0) outDegrees[rotIdx] = angle0 * RAD2DEG;
            else if (rotIdx == 1) outDegrees[rotIdx] = angle1 * RAD2DEG;
            else if (rotIdx == 2) outDegrees[rotIdx] = angle2 * RAD2DEG;
            rotIdx++;
        }
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