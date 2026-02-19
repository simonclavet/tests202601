#pragma once

#include <string>
#include <vector>

#include "raylib.h"
#include "bvh_parser.h"
#include "fbx_loader.h"
#include "transform_data.h"
#include "capsule_data.h"
#include "utils.h"

//----------------------------------------------------------------------------------
// Character Data
//----------------------------------------------------------------------------------

// All the data required for all of the characters we want to have in the scene
struct CharacterData
{
    // Total number of characters
    int count = 0;

    // Character which is "active" or selected
    int active = 0;

    // Character BVH Data
    std::vector<BVHData> bvhData;

    // Scales of each character
    std::vector<float> scales;

    // Names of each character
    std::vector<std::string> names;

    // Automatic scaling for each character
    std::vector<float> autoScales;

    // Color of each character
    std::vector<Color> colors;

    // Opacity of each character
    std::vector<float> opacities;

    // Maximum capsule radius of each character
    std::vector<float> radii;

    // Original file path for each character
    std::vector<std::string> filePaths;

    // Transform buffers for each character
    std::vector<TransformData> xformData;
    std::vector<TransformData> xformTmp0;
    std::vector<TransformData> xformTmp1;
    std::vector<TransformData> xformTmp2;
    std::vector<TransformData> xformTmp3;

    // Joint combo string for each character (for GUI combo box)
    std::vector<std::string> jointNamesCombo;

    // If the color picker is active
    bool colorPickerActive = false;

    // Flag set by UI to request a clear (handled in main update with full state access)
    bool clearRequested = false;

    // Default values for new characters
    float defaultOpacity;
    float defaultRadius;
    std::vector<Color> defaultColors;
};

// Reserve capacity in all parallel vectors to avoid reallocation
static inline void CharacterDataReserve(CharacterData* data, int capacity)
{
    data->bvhData.reserve(capacity);
    data->xformData.reserve(capacity);
    data->xformTmp0.reserve(capacity);
    data->xformTmp1.reserve(capacity);
    data->xformTmp2.reserve(capacity);
    data->xformTmp3.reserve(capacity);
    data->names.reserve(capacity);
    data->filePaths.reserve(capacity);
    data->scales.reserve(capacity);
    data->autoScales.reserve(capacity);
    data->opacities.reserve(capacity);
    data->radii.reserve(capacity);
    data->colors.reserve(capacity);
    data->jointNamesCombo.reserve(capacity);
}

// Initializes all the CharacterData to a safe state
static inline void CharacterDataInit(CharacterData* data, int argc, char** argv)
{
    // Store default values for new characters
    data->defaultOpacity = ArgFloat(argc, argv, "capsuleOpacity", 1.0f);
    data->defaultRadius = ArgFloat(argc, argv, "maxCapsuleRadius", 0.04f);

    // Set up default colors for new characters
    data->defaultColors = {
        ArgColor(argc, argv, "-color0", ORANGE),
        ArgColor(argc, argv, "-color1", Color{ 38, 134, 157, 255 }),
        ArgColor(argc, argv, "-color2", PINK),
        ArgColor(argc, argv, "-color3", LIME),
        ArgColor(argc, argv, "-color4", VIOLET),
        ArgColor(argc, argv, "-color5", MAROON),
    };

    data->colorPickerActive = ArgBool(argc, argv, "colorPickerActive", false);
}

// Helper to get color for a new character
static inline Color CharacterDataGetColorForIndex(const CharacterData* data, int index)
{
    if (index < (int)data->defaultColors.size())
    {
        return data->defaultColors[index];
    }
    // Generate random color for indices beyond defaults
    srand(1234 + index);
    return Color{
        (unsigned char)(rand() % 255),
        (unsigned char)(rand() % 255),
        (unsigned char)(rand() % 255),
        255
    };
}

// Attempt to load a new character from the given file path
// Supports both BVH and FBX files
static bool CharacterDataLoadFromFile(
    CharacterData* data,
    const char* path,
    char* errMsg,
    int errMsgSize)
{
    printf("INFO: Loading '%s'\n", path);

    // Load animation data into a temporary first
    BVHData newBvh;
    BVHDataInit(&newBvh);

    bool loadSuccess = false;

    if (IsFileExtension(path, ".fbx"))
    {
        // Load FBX file
        loadSuccess = FBXDataLoad(&newBvh, path, errMsg, errMsgSize);
    }
    else
    {
        // Default to BVH loader
        loadSuccess = BVHDataLoad(&newBvh, path, errMsg, errMsgSize);
    }

    if (!loadSuccess)
    {
        printf("INFO: Failed to Load '%s'\n", path);
        return false;
    }

    // Success - add to vectors
    const int idx = data->count;

    data->bvhData.push_back(newBvh);
    data->filePaths.push_back(path);

    // Extract filename from path
    const char* filename = path;
    while (strchr(filename, '/')) { filename = strchr(filename, '/') + 1; }
    while (strchr(filename, '\\')) { filename = strchr(filename, '\\') + 1; }
    data->names.push_back(filename);

    data->scales.push_back(1.0f);
    data->autoScales.push_back(1.0f);
    data->opacities.push_back(data->defaultOpacity);
    data->radii.push_back(data->defaultRadius);
    data->colors.push_back(CharacterDataGetColorForIndex(data, idx));

    // Add transform data
    TransformData xform;
    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformData.push_back(xform);

    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformTmp0.push_back(xform);

    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformTmp1.push_back(xform);

    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformTmp2.push_back(xform);

    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformTmp3.push_back(xform);

    // Auto-Scaling and unit detection
    if (data->bvhData[idx].frameCount > 0)
    {
        TransformDataSampleFrame(&data->xformData[idx], &data->bvhData[idx], 0, 1.0f);
        TransformDataForwardKinematics(&data->xformData[idx]);

        float height = 1e-8f;
        for (int j = 0; j < data->xformData[idx].jointCount; j++)
        {
            height = Max(height, data->xformData[idx].globalPositions[j].y);
        }

        data->scales[idx] = height > 10.0f ? 0.01f : 1.0f;
        data->autoScales[idx] = 1.8f / height;
    }

    // Build joint names combo string (semicolon-separated for GUI)
    std::string combo;
    for (int i = 0; i < data->bvhData[idx].jointCount; i++)
    {
        if (i > 0) combo += ";";
        combo += data->bvhData[idx].joints[i].name;
    }
    data->jointNamesCombo.push_back(combo);

    data->count++;
    return true;
}

// find spine3 joint index from BVH joint names (same candidates as AnimDatabaseRebuild)
static int FindSpine3IndexFromBVH(const BVHData& bvh)
{
    const std::vector<std::string> candidates = { "spine3", "spine2", "chest", "upperchest", "upper_chest" };
    return FindJointIndexByNames(&bvh, candidates);
}

// create a deep copy of a BVHData with aim offset baked into spine1/2/3 for every frame.
// the rotation is applied around world Y, distributed equally across spine1, spine2, spine3.
static BVHData CreateAimOffsetBVH(
    const BVHData& src,
    int spine3Index,
    float aimAngleRadians)
{
    BVHData dst = src;  // deep copy (vectors copy their data)

    const int spine2 = src.joints[spine3Index].parent;
    const int spine1 = (spine2 >= 0) ? src.joints[spine2].parent : -1;

    const int spineCount = 1 + (spine2 >= 0 ? 1 : 0) + (spine1 >= 0 ? 1 : 0);
    const float perJointAngle = aimAngleRadians / (float)spineCount;
    const Quaternion worldYawRot = QuaternionFromAxisAngle(
        Vector3{ 0.0f, 1.0f, 0.0f }, perJointAngle);

    const int spineJoints[3] = { spine1, spine2, spine3Index };

    // precompute channel offsets for the spine joints
    int spineChannelOffsets[3] = { -1, -1, -1 };
    for (int s = 0; s < 3; s++)
    {
        if (spineJoints[s] >= 0)
        {
            spineChannelOffsets[s] = BVHJointChannelOffset(&src, spineJoints[s]);
        }
    }

    // allocate a TransformData once and reuse for every frame
    TransformData xd;
    TransformDataInit(&xd);
    TransformDataResize(&xd, &src);

    for (int frame = 0; frame < src.frameCount; frame++)
    {
        // sample this frame into local quaternions
        TransformDataSampleFrame(&xd, &src, frame, 1.0f);

        // apply aim rotation to each spine joint (parent to child order)
        for (int s = 0; s < 3; s++)
        {
            const int sj = spineJoints[s];
            if (sj < 0) continue;

            const Quaternion parentGlobal = TransformDataParentGlobalRotation(&xd, sj);
            const Quaternion parentGlobalInv = QuaternionInvert(parentGlobal);

            // newLocal = inv(parentGlobal) * worldYaw * parentGlobal * oldLocal
            xd.localRotations[sj] = QuaternionMultiply(
                parentGlobalInv,
                QuaternionMultiply(worldYawRot,
                    QuaternionMultiply(parentGlobal, xd.localRotations[sj])));

            // decompose back to Euler degrees and write into dst.motionData
            const BVHJointData& joint = src.joints[sj];
            float degrees[3];
            QuaternionToChannelOrder(xd.localRotations[sj],
                joint.channels, joint.channelCount, degrees);

            const int baseOffset = frame * dst.channelCount + spineChannelOffsets[s];
            int rotIdx = 0;
            for (int c = 0; c < joint.channelCount; c++)
            {
                if (joint.channels[c] >= CHANNEL_X_ROTATION)
                {
                    dst.motionData[baseOffset + c] = degrees[rotIdx++];
                }
            }
        }
    }

    return dst;
}

// create a resampled copy of a BVHData with adjusted playback speed.
// timescale > 1 means faster (fewer frames), < 1 means slower (more frames).
// keeps the same frameTime, resamples through quaternions to avoid Euler interpolation issues.
static BVHData CreateTimescaleBVH(const BVHData& src, float timescale)
{
    BVHData dst = src;

    const int newFrameCount = (int)(src.frameCount / timescale + 0.5f);
    dst.frameCount = newFrameCount;
    dst.motionData.resize((size_t)newFrameCount * (size_t)src.channelCount);

    // precompute channel offsets for each joint
    std::vector<int> channelOffsets(src.jointCount);
    for (int j = 0; j < src.jointCount; j++)
    {
        channelOffsets[j] = BVHJointChannelOffset(&src, j);
    }

    // allocate two TransformData buffers for slerping between frames
    TransformData xd0, xd1;
    TransformDataInit(&xd0);
    TransformDataInit(&xd1);
    TransformDataResize(&xd0, &src);
    TransformDataResize(&xd1, &src);

    for (int i = 0; i < newFrameCount; i++)
    {
        const float srcFrame = i * timescale;
        const int f0 = ClampInt((int)srcFrame, 0, src.frameCount - 1);
        const int f1 = ClampInt(f0 + 1, 0, src.frameCount - 1);
        const float alpha = srcFrame - (float)f0;

        // sample both bracketing frames into quaternions
        TransformDataSampleFrame(&xd0, &src, f0, 1.0f);
        TransformDataSampleFrame(&xd1, &src, f1, 1.0f);

        // slerp rotations, lerp positions, write back as Euler angles
        const int dstBase = i * src.channelCount;

        for (int j = 0; j < src.jointCount; j++)
        {
            const Vector3 pos = Vector3Lerp(xd0.localPositions[j], xd1.localPositions[j], alpha);
            const Quaternion rot = QuaternionSlerp(xd0.localRotations[j], xd1.localRotations[j], alpha);

            const BVHJointData& joint = src.joints[j];
            float degrees[3];
            QuaternionToChannelOrder(rot, joint.channels, joint.channelCount, degrees);

            int rotIdx = 0;
            for (int c = 0; c < joint.channelCount; c++)
            {
                if (joint.channels[c] < CHANNEL_X_ROTATION)
                {
                    // position channel
                    const float val = (joint.channels[c] == CHANNEL_X_POSITION) ? pos.x :
                                      (joint.channels[c] == CHANNEL_Y_POSITION) ? pos.y : pos.z;
                    dst.motionData[dstBase + channelOffsets[j] + c] = val;
                }
                else
                {
                    dst.motionData[dstBase + channelOffsets[j] + c] = degrees[rotIdx++];
                }
            }
        }
    }

    return dst;
}

// add a pre-built BVHData to CharacterData (like loading from file but from memory)
static void CharacterDataAddBVH(
    CharacterData* data,
    BVHData bvh,
    const std::string& name)
{
    const int idx = data->count;

    data->bvhData.push_back(bvh);
    data->filePaths.push_back("(augmented)");
    data->names.push_back(name);

    data->scales.push_back(1.0f);
    data->autoScales.push_back(1.0f);
    data->opacities.push_back(data->defaultOpacity);
    data->radii.push_back(data->defaultRadius);
    data->colors.push_back(CharacterDataGetColorForIndex(data, idx));

    TransformData xform;
    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformData.push_back(xform);

    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformTmp0.push_back(xform);

    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformTmp1.push_back(xform);

    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformTmp2.push_back(xform);

    TransformDataInit(&xform);
    TransformDataResize(&xform, &data->bvhData[idx]);
    data->xformTmp3.push_back(xform);

    // auto-scale detection
    if (data->bvhData[idx].frameCount > 0)
    {
        TransformDataSampleFrame(&data->xformData[idx], &data->bvhData[idx], 0, 1.0f);
        TransformDataForwardKinematics(&data->xformData[idx]);

        float height = 1e-8f;
        for (int j = 0; j < data->xformData[idx].jointCount; j++)
        {
            height = Max(height, data->xformData[idx].globalPositions[j].y);
        }

        data->scales[idx] = height > 10.0f ? 0.01f : 1.0f;
        data->autoScales[idx] = 1.8f / height;
    }

    // build joint names combo string
    std::string combo;
    for (int i = 0; i < data->bvhData[idx].jointCount; i++)
    {
        if (i > 0) combo += ";";
        combo += data->bvhData[idx].joints[i].name;
    }
    data->jointNamesCombo.push_back(combo);

    data->count++;
}

// TODO: get rid of Euler angles in BVHData as early as possible in the pipeline.
// as soon as we finish reading a file, convert to quaternions and never look back.

// Resize so that we have enough capsules in the buffers for the given set of characters
static inline void CapsuleDataUpdateForCharacters(CapsuleData* capsuleData, CharacterData* characterData)
{
    int totalJointCount = 0;
    for (int i = 0; i < characterData->count; i++)
    {
        totalJointCount += characterData->bvhData[i].jointCount;
    }

    CapsuleDataResize(capsuleData, totalJointCount);
}
