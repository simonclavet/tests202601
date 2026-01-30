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
    int count;

    // Character which is "active" or selected
    int active;

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
    bool colorPickerActive;

    // Flag set by UI to request a clear (handled in main update with full state access)
    bool clearRequested;

    // Default values for new characters
    float defaultOpacity;
    float defaultRadius;
    std::vector<Color> defaultColors;
};

// Initializes all the CharacterData to a safe state
static inline void CharacterDataInit(CharacterData* data, int argc, char** argv)
{
    data->count = 0;
    data->active = 0;
    data->clearRequested = false;
    data->colorPickerActive = false;

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

static inline void CharacterDataFree(CharacterData* data)
{
    data->xformData.clear();
    data->xformTmp0.clear();
    data->xformTmp1.clear();
    data->xformTmp2.clear();
    data->xformTmp3.clear();
    data->bvhData.clear();
    data->scales.clear();
    data->names.clear();
    data->autoScales.clear();
    data->colors.clear();
    data->opacities.clear();
    data->radii.clear();
    data->filePaths.clear();
    data->count = 0;
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
