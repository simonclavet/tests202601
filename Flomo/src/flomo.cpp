/*******************************************************************************************
*
*    BVHView - A simple BVH animation viewer written using raylib
*
*  This is a simple viewer for the .bvh animation file format made using raylib. For more
*  info on the motivation behind it and information on features and documentation please
*  see: https://theorangeduck.com/page/bvhview
*
*  The program itself essentially consists of the following components:
*
*     - A parser for the BVH file format
*     - A set of functions for sampling data from particular frames of the BVH file.
*     - A set of functions for creating capsules from the skeleton structure of the BVH data
*       and animation transforms.
*     - A (relatively) efficient and high quality shader for rendering capsules that includes
*       nice lighting, soft shadows, and some CPU based culling to limit the amount of work
*       required by the GPU.
*
*  Coding style is roughly meant to follow the rest of raylib and community contributions
*  are very welcome.
*
*******************************************************************************************/

// Most headers are in pch.h (precompiled header):
// - Standard C headers, Windows headers, raylib core, torch, STL

// These must stay here (implementation headers with #define macros)
//#define RAYGUI_WINDOWBOX_STATUSBAR_HEIGHT 24
//#define GUI_WINDOW_FILE_DIALOG_IMPLEMENTATION
//#include "gui_window_file_dialog.h"
//#define RAYGUI_IMPLEMENTATION
//#include "raygui.h"

#include "math_utils.h"
#include "utils.h"

// Un-comment to enable profiling
//#define ENABLE_PROFILE
#include "profiler.h"


#include "camera.h"
#include "bvh_parser.h"
#include "fbx_loader.h"
#include "transform_data.h"
#include "geometry_utils.h"
#include "capsule_data.h"
#include "mesh_utils.h"
#include "geno_renderer.h"
#include "app_config.h"
#include "balltree.h"
//#include "character_data.h"



// Dear ImGui with raylib backend
#include "imgui.h"
#include "rlImGui.h"

using namespace std;

// Declare the CUDA functions
extern "C" void run_cuda_addition(float* a, float* b, float* c, int n);
extern "C" void cuda_check_error(const char* msg);

static void TestCudaAndLibtorch()
{
    const int N = 1000000;  // 1 million elements
    vector<float> a(N, 1.0f);
    vector<float> b(N, 2.0f);
    vector<float> c(N, 0.0f);

    cout << "Running CUDA addition..." << endl;

    auto start = chrono::high_resolution_clock::now();

    run_cuda_addition(a.data(), b.data(), c.data(), N);
    cuda_check_error("main execution");

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (c[i] != 3.0f) {
            correct = false;
            break;
        }
    }

    cout << "CUDA addition " << (correct ? "PASSED" : "FAILED") << endl;
    cout << "Time: " << duration.count() << " microseconds" << endl;
    cout << "First 5 results: ";
    for (int i = 0; i < 5; i++) {
        cout << c[i] << " ";
    }
    cout << endl;

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        cout << "LibTorch: Using CUDA device" << endl;
    }
    else {
        cout << "LibTorch: Using CPU device" << endl;
    }

    torch::Tensor tensor = torch::rand({ 3, 3 }).to(device);
    auto result = tensor * 2;
    cout << "Random tensor:\n" << tensor << endl;
    cout << "Tensor * 2:\n" << result << endl;
}

//----------------------------------------------------------------------------------
// Character Data
//----------------------------------------------------------------------------------


// All the data required for all of the characters we want to have in the scene
struct CharacterData {

    // Total number of characters
    int count;

    // Character which is "active" or selected
    int active;

    // Character BVH Data
    vector<BVHData> bvhData;

    // Scales of each character
    vector<float> scales;

    // Names of each character
    vector<string> names;

    // Automatic scaling for each character
    vector<float> autoScales;

    // Color of each character
    vector<Color> colors;

    // Opacity of each character
    vector<float> opacities;

    // Maximum capsule radius of each character
    vector<float> radii;

    // Original file path for each character
    vector<string> filePaths;

    // Transform buffers for each character
    vector<TransformData> xformData;
    vector<TransformData> xformTmp0;
    vector<TransformData> xformTmp1;
    vector<TransformData> xformTmp2;
    vector<TransformData> xformTmp3;

    // Joint combo string for each character (for GUI combo box)
    vector<string> jointNamesCombo;

    // If the color picker is active
    bool colorPickerActive;

    // Flag set by UI to request a clear (handled in main update with full state access)
    bool clearRequested;

    // Default values for new characters
    float defaultOpacity;
    float defaultRadius;
    vector<Color> defaultColors;
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
    int idx = data->count;

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
    string combo;
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

//----------------------------------------------------------------------------------
// Animation Database
//----------------------------------------------------------------------------------

// A unified view of all loaded animations, suitable for sampling by ControlledCharacter.
struct AnimDatabase {
    // References to all loaded animations
    int animCount = -1;;

    // Per-animation info
    vector<int> animStartFrame;   // Global frame index where each anim starts
    vector<int> animFrameCount;   // Number of frames in each anim
    vector<float> animFrameTime;  // Frame time for each anim (usually same)

    // Total frames across all animations
    int totalFrames = -1;;

    // Scale to apply when sampling (for unit conversion)
    float scale = 0.0f;

    // Validity: true only if ALL animations are compatible with canonical skeleton
    bool valid = false;

    // ---- Motion-database specific fields ----
    // Canonical joint count (set from first animation's joint count)
    int jointCount = -1;;

    // Number of frames actually stored in the compacted motion DB (may be <= totalFrames
    // if some clips have mismatched skeletons and are skipped)
    int motionFrameCount = -1;;

    // Per-frame joint transforms: [motionFrameCount x jointCount]
    Array2D<Vector3> globalJointPositions;      // global positions
    Array2D<Quaternion> globalJointRotations;   // global rotations
    Array2D<Vector3> globalJointVelocities;     // velocities (defined at midpoint between frames)
    Array2D<Vector3> globalJointAccelerations;  // accelerations (derivative of velocity)

    // local joint transforms (for blending without global->local conversion)
    Array2D<Vector3> localJointPositions;           // local positions [motionFrameCount x jointCount]
    Array2D<Rot6d> localJointRotations6d;           // local rotations as Rot6d [motionFrameCount x jointCount]
    Array2D<Vector3> localJointAngularVelocities;   // local angular velocities [motionFrameCount x jointCount]

    // Segmentation of the compacted motion DB into clips:
    // clipStartFrame[c] .. clipEndFrame[c]-1 are frames for clip c in motion DB frame space.
    vector<int> clipStartFrame;
    vector<int> clipEndFrame;

    // motion matching features [motionFrameCount x featureDim]
    int featureDim = -1;
    Array2D<float> features;
    int hipJointIndex = -1;            // resolved index for "Hips" in canonical skeleton
    int toeIndices[SIDES_COUNT] = { -1, -1 };
    vector<string> featureNames;
};


static inline int FindClipForMotionFrame(const AnimDatabase* db, int frame) {
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        if (frame >= db->clipStartFrame[c] && frame < db->clipEndFrame[c]) return c;
    }
    return -1;
}

static inline int FindJointIndexByNames(const BVHData* bvh, const vector<string>& candidates)
{
    // Exact match pass (case-insensitive)
    for (int j = 0; j < bvh->jointCount; ++j)
    {
        const string& name = bvh->joints[j].name;
        if (name.empty()) continue;
        string lname = ToLowerCopy(name.c_str());
        for (int k = 0; k < (int)candidates.size(); ++k)
        {
            if (lname == candidates[k])
            {
                return j;
            }
        }
    }

    // Substring fallback pass (case-insensitive)
    for (int j = 0; j < bvh->jointCount; ++j)
    {
        const string& name = bvh->joints[j].name;
        if (name.empty()) continue;
        string lname = ToLowerCopy(name.c_str());
        for (int k = 0; k < (int)candidates.size(); ++k)
        {
            if (StrContainsCaseInsensitive(lname, candidates[k]))
            {
                return j;
            }
        }
    }

    return -1;
}

// Free/reset AnimDatabase to empty state
static void AnimDatabaseFree(AnimDatabase* db)
{
    db->animCount = -1;
    db->animStartFrame.clear();
    db->animFrameCount.clear();
    db->animFrameTime.clear();
    db->totalFrames = -1;
    db->scale = 0.0f;
    db->valid = false;
    db->jointCount = -1;
    db->motionFrameCount = -1;
    db->globalJointPositions.clear();
    db->globalJointRotations.clear();
    db->globalJointVelocities.clear();
    db->globalJointAccelerations.clear();
    db->localJointPositions.clear();
    db->localJointRotations6d.clear();
    db->localJointAngularVelocities.clear();
    db->clipStartFrame.clear();
    db->clipEndFrame.clear();
    db->features.clear();
    db->featureDim = 0;
    db->featureNames.clear();
}

// Updated AnimDatabaseRebuild: require all animations to match canonical skeleton.
// Populate localJointPositions/localJointRotations6d as well as global arrays.
// If any clip mismatches jointCount we invalidate the DB (db->valid = false).
static void AnimDatabaseRebuild(AnimDatabase* db, const CharacterData* characterData) {
    db->animCount = characterData->count;
    db->animStartFrame.resize(db->animCount);
    db->animFrameCount.resize(db->animCount);
    db->animFrameTime.resize(db->animCount);
    db->motionFrameCount = 0;
    db->globalJointPositions.clear();
    db->globalJointRotations.clear();
    db->globalJointVelocities.clear();
    db->globalJointAccelerations.clear();
    db->localJointPositions.clear();
    db->localJointRotations6d.clear();
    db->localJointAngularVelocities.clear();
    db->clipStartFrame.clear();
    db->clipEndFrame.clear();
    db->valid = false; // pessimistic until proven valid

    int globalFrame = 0;
    for (int i = 0; i < db->animCount; i++)
    {
        db->animStartFrame[i] = globalFrame;
        db->animFrameCount[i] = characterData->bvhData[i].frameCount;
        db->animFrameTime[i] = characterData->bvhData[i].frameTime;
        globalFrame += db->animFrameCount[i];
    }
    db->totalFrames = globalFrame;

    // Use scale from first animation
    if (db->animCount > 0)
    {
        db->scale = characterData->scales[0];
    }

    printf("AnimDatabase: Rebuilt with %d anims, %d total frames\n", db->animCount, db->totalFrames);

    // -------------------------
    // Build compact Motion Database
    // -------------------------
    // pick canonical skeleton (first animation) if available

    if (db->animCount == 0 || db->totalFrames == 0) {
        TraceLog(LOG_INFO, "AnimDatabase: no animations available for motion DB");
        return;
    }

    const BVHData* canonBvh = &characterData->bvhData[0];
    db->jointCount = canonBvh->jointCount;

    // STRICT: require every clip to have the same jointCount as the canonical skeleton.
    for (int a = 0; a < db->animCount; ++a) {
        const BVHData* bvh = &characterData->bvhData[a];
        if (bvh->jointCount != db->jointCount) {
            TraceLog(LOG_WARNING, "AnimDatabase: incompatible anim %d (%s) - jointCount mismatch (%d != %d). Aborting DB build.",
                a, characterData->filePaths[a].c_str(), bvh->jointCount, db->jointCount);
            db->motionFrameCount = 0;
            db->valid = false;
            return;
        }
    }

    // Compute total frames (all clips included)
    int includedFrames = 0;
    for (int a = 0; a < db->animCount; ++a) {
        includedFrames += characterData->bvhData[a].frameCount;
    }


    if (includedFrames == 0) {
        TraceLog(LOG_WARNING, "AnimDatabase: no compatible animations for motion DB (jointCount=%d)", db->jointCount);
        db->motionFrameCount = 0;
        db->valid = false;
        return;
    }

    // allocate compact storage [motionFrameCount x jointCount]
    db->motionFrameCount = includedFrames;
    db->globalJointPositions.resize(db->motionFrameCount, db->jointCount);
    db->globalJointRotations.resize(db->motionFrameCount, db->jointCount);
    db->globalJointVelocities.resize(db->motionFrameCount, db->jointCount);
    db->globalJointAccelerations.resize(db->motionFrameCount, db->jointCount);
    db->localJointPositions.resize(db->motionFrameCount, db->jointCount);
    db->localJointRotations6d.resize(db->motionFrameCount, db->jointCount);
    db->localJointAngularVelocities.resize(db->motionFrameCount, db->jointCount);

    // sample each compatible clip frame and fill the flat arrays
    TransformData tmpXform;
    TransformDataInit(&tmpXform);
    TransformDataResize(&tmpXform, canonBvh); // sized to canonical skeleton

    int motionFrameIdx = 0;
    for (int a = 0; a < db->animCount; ++a) {
        const BVHData* bvh = &characterData->bvhData[a];
        db->clipStartFrame.push_back(motionFrameIdx);

        for (int f = 0; f < bvh->frameCount; ++f) {
            TransformDataSampleFrame(&tmpXform, bvh, f, characterData->scales[a]);
            TransformDataForwardKinematics(&tmpXform);

            span<Vector3> globalPos = db->globalJointPositions.row_view(motionFrameIdx);
            span<Quaternion> globalRot = db->globalJointRotations.row_view(motionFrameIdx);
            span<Vector3> localPos = db->localJointPositions.row_view(motionFrameIdx);
            span<Rot6d> localRot = db->localJointRotations6d.row_view(motionFrameIdx);

            for (int j = 0; j < db->jointCount; ++j)
            {
                globalPos[j] = tmpXform.globalPositions[j];
                globalRot[j] = tmpXform.globalRotations[j];
                localPos[j] = tmpXform.localPositions[j];
                Rot6dFromQuaternion(tmpXform.localRotations[j], localRot[j]);
            }

            ++motionFrameIdx;
        }

        db->clipEndFrame.push_back(motionFrameIdx);
    }

    TransformDataFree(&tmpXform);

    // Compute velocities for each joint at each frame
    // Velocity at frame i is defined at midpoint between frame i and i+1: v = (pos[i+1] - pos[i]) / frameTime
    // For last frame of each clip, copy from previous frame
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float invFrameTime = 1.0f / frameTime;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);
            const int nextF = isLastFrame ? f : (f + 1);
            const int prevF = isLastFrame ? (f - 1) : f;

            span<Vector3> velRow = db->globalJointVelocities.row_view(f);

            // handle edge case: single-frame clip
            if (clipEnd - clipStart <= 1)
            {
                for (int j = 0; j < db->jointCount; ++j)
                {
                    velRow[j] = Vector3Zero();
                }
                continue;
            }

            span<const Vector3> pos0Row = db->globalJointPositions.row_view(prevF);
            span<const Vector3> pos1Row = db->globalJointPositions.row_view(nextF);

            for (int j = 0; j < db->jointCount; ++j)
            {
                const Vector3 vel = Vector3Scale(Vector3Subtract(pos1Row[j], pos0Row[j]), invFrameTime);
                velRow[j] = vel;
            }
        }
    }

    // Compute accelerations for each joint at each frame
    // Acceleration at frame i: a = (vel[i+1] - vel[i]) / frameTime
    // For last frame of each clip, copy from previous frame
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float invFrameTime = 1.0f / frameTime;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);
            const int nextF = isLastFrame ? f : (f + 1);
            const int prevF = isLastFrame ? (f - 1) : f;

            span<Vector3> accRow = db->globalJointAccelerations.row_view(f);

            // handle edge case: single-frame or two-frame clip
            if (clipEnd - clipStart <= 2)
            {
                for (int j = 0; j < db->jointCount; ++j)
                {
                    accRow[j] = Vector3Zero();
                }
                continue;
            }

            span<const Vector3> vel0Row = db->globalJointVelocities.row_view(prevF);
            span<const Vector3> vel1Row = db->globalJointVelocities.row_view(nextF);

            for (int j = 0; j < db->jointCount; ++j)
            {
                const Vector3 acc = Vector3Scale(Vector3Subtract(vel1Row[j], vel0Row[j]), invFrameTime);
                accRow[j] = acc;
            }
        }
    }

    // Compute local angular velocities for each joint at each frame
    // Angular velocity at frame i is defined at midpoint between frame i and i+1
    // For last frame of each clip, copy from previous frame
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);
            const int nextF = isLastFrame ? f : (f + 1);
            const int prevF = isLastFrame ? (f - 1) : f;

            span<Vector3> angVelRow = db->localJointAngularVelocities.row_view(f);

            // handle edge case: single-frame clip
            if (clipEnd - clipStart <= 1)
            {
                for (int j = 0; j < db->jointCount; ++j)
                {
                    angVelRow[j] = Vector3Zero();
                }
                continue;
            }

            span<const Rot6d> rot0Row = db->localJointRotations6d.row_view(prevF);
            span<const Rot6d> rot1Row = db->localJointRotations6d.row_view(nextF);

            for (int j = 0; j < db->jointCount; ++j)
            {
                Rot6dGetVelocity(rot0Row[j], rot1Row[j], frameTime, angVelRow[j]);
            }
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: built motion DB with %d frames and %d joints",
        db->motionFrameCount, db->jointCount);

    // set db->valid true now that we completed full build
    db->valid = true;

    vector<string> jointNames;
    jointNames.reserve((size_t)canonBvh->jointCount);
    for (int j = 0; j < canonBvh->jointCount; ++j)
    {
        // BVHJointData::name is now string
        jointNames.push_back(canonBvh->joints[j].name);
    }

    // Reset indices
    db->hipJointIndex = -1;
    db->toeIndices[SIDE_LEFT] = -1;
    db->toeIndices[SIDE_RIGHT] = -1;

    // HIP candidates (lowercase)
    vector<string> hipCandidates = { "hips", "hip", "pelvis", "root" };

    // Toe candidates (lowercase)
    const vector<string> leftToeCandidates = { "lefttoebase", "lefttoe", "left_toe", "l_toe" };
    const vector<string> rightToeCandidates = { "righttoebase", "righttoe", "right_toe", "r_toe" };
    const vector<vector<string>> toeCandidates = { leftToeCandidates, rightToeCandidates };

    // Use helper to find hip and toes (exact then substring)
    db->hipJointIndex = FindJointIndexByNames(canonBvh, hipCandidates);

    // iterate integer sides array (no casts)
    for (int side : sides)
    {
        int si = side; // 0 or 1
        db->toeIndices[si] = FindJointIndexByNames(canonBvh, toeCandidates[si]);
    }

    // Log resolved indices
    TraceLog(LOG_INFO, "AnimDatabase: feature joint indices hip=%d leftToe=%d rightToe=%d",
        db->hipJointIndex,
        db->toeIndices[SIDE_LEFT],
        db->toeIndices[SIDE_RIGHT]);

    static constexpr bool FEATURE_TOE_POS = false;       // left+right toe positions (X,Z) => 4 dims
    static constexpr bool FEATURE_TOE_VEL = true;       // left+right toe velocities (X,Z) => 4 dims
    static constexpr bool FEATURE_TOE_DIFF = true;      // left-right difference (X,Z) => 2 dims

    static constexpr int FEATURE_TOE_POS_DIM = FEATURE_TOE_POS ? 4 : 0;
    static constexpr int FEATURE_TOE_VEL_DIM = FEATURE_TOE_VEL ? 4 : 0;
    static constexpr int FEATURE_TOE_DIFF_DIM = FEATURE_TOE_DIFF ? 2 : 0;

    static constexpr int FEATURE_DIM = FEATURE_TOE_POS_DIM + FEATURE_TOE_VEL_DIM + FEATURE_TOE_DIFF_DIM;

    db->featureDim = FEATURE_DIM;
    db->features.clear();
    db->featureNames.clear();

    db->features.resize(db->motionFrameCount, db->featureDim);
    db->features.fill(0.0f);

    // Populate features from jointPositions and jointRotations
    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        const bool isFirstFrame = f == 0;

        span<const Vector3> posRow = db->globalJointPositions.row_view(f);
        span<const Quaternion> rotRow = db->globalJointRotations.row_view(f);
        span<float> featRow = db->features.row_view(f);

        Vector3 hipPos = { 0.0f, 0.0f, 0.0f };
        Vector3 leftPos = { 0.0f, 0.0f, 0.0f };
        Vector3 rightPos = { 0.0f, 0.0f, 0.0f };

        if (db->hipJointIndex >= 0) {
            hipPos = posRow[db->hipJointIndex];
        }

        const int leftIdx = db->toeIndices[SIDE_LEFT];
        const int rightIdx = db->toeIndices[SIDE_RIGHT];

        if (leftIdx >= 0) {
            leftPos = posRow[leftIdx];
        }
        if (rightIdx >= 0) {
            rightPos = posRow[rightIdx];
        }

        // Extract hip yaw (if available) once per frame
        Quaternion hipYaw = QuaternionIdentity();
        if (db->hipJointIndex >= 0) {
            hipYaw = QuaternionYComponent(rotRow[db->hipJointIndex]);
        }
        const Quaternion invHipYaw = QuaternionInvert(hipYaw);

        int currentFeature = 0;

        // Precompute local toe positions (hip horizontal frame) - used by pos and diff
        Vector3 hipToLeft = Vector3Subtract(leftPos, hipPos);
        Vector3 localLeftPos = Vector3RotateByQuaternion(hipToLeft, invHipYaw);

        Vector3 hipToRight = Vector3Subtract(rightPos, hipPos);
        Vector3 localRightPos = Vector3RotateByQuaternion(hipToRight, invHipYaw);

        // POSITION: toePos->Left(X, Z), Right(X, Z)
        if constexpr (FEATURE_TOE_POS) {
            featRow[currentFeature++] = localLeftPos.x;
            featRow[currentFeature++] = localLeftPos.z;
            featRow[currentFeature++] = localRightPos.x;
            featRow[currentFeature++] = localRightPos.z;

            if (isFirstFrame)
            {
                db->featureNames.push_back(string("LeftToePosX"));
                db->featureNames.push_back(string("LeftToePosZ"));
                db->featureNames.push_back(string("RightToePosX"));
                db->featureNames.push_back(string("RightToePosZ"));
            }
        }

        // VELOCITY: toeVel -> compute world finite-difference then rotate into hip frame
        if constexpr (FEATURE_TOE_VEL) {
            Vector3 localLeftVel = Vector3Zero();
            Vector3 localRightVel = Vector3Zero();

            const int clipIdx = FindClipForMotionFrame(db, f);
            if (clipIdx != -1) {
                const int clipStart = db->clipStartFrame[clipIdx];
                const float dt = db->animFrameTime[clipIdx] > 1e-8f ? db->animFrameTime[clipIdx] : 0.0f;

                if (f > clipStart && dt > 0.0f) {
                    span<const Vector3> posPrevRow = db->globalJointPositions.row_view(f - 1);

                    if (leftIdx >= 0) {
                        Vector3 deltaLeft = Vector3Subtract(leftPos, posPrevRow[leftIdx]);
                        const Vector3 velLeftWorld = Vector3Scale(deltaLeft, 1.0f / dt);
                        localLeftVel = Vector3RotateByQuaternion(velLeftWorld, invHipYaw);
                    }

                    if (rightIdx >= 0) {
                        Vector3 deltaRight = Vector3Subtract(rightPos, posPrevRow[rightIdx]);
                        const Vector3 velRightWorld = Vector3Scale(deltaRight, 1.0f / dt);
                        localRightVel = Vector3RotateByQuaternion(velRightWorld, invHipYaw);
                    }
                }
            }

            featRow[currentFeature++] = localLeftVel.x;
            featRow[currentFeature++] = localLeftVel.z;
            featRow[currentFeature++] = localRightVel.x;
            featRow[currentFeature++] = localRightVel.z;

            if (isFirstFrame) {
                db->featureNames.push_back(string("LeftToeVelX"));
                db->featureNames.push_back(string("LeftToeVelZ"));
                db->featureNames.push_back(string("RightToeVelX"));
                db->featureNames.push_back(string("RightToeVelZ"));
            }
        }

        // DIFFERENCE: toeDifference = Left - Right (in hip horizontal frame) => (dx, dz)
        if constexpr (FEATURE_TOE_DIFF) {
            const float diffX = localLeftPos.x - localRightPos.x;
            const float diffZ = localLeftPos.z - localRightPos.z;
            featRow[currentFeature++] = diffX;
            featRow[currentFeature++] = diffZ;

            if (isFirstFrame) {
                db->featureNames.push_back(string("ToeDiffX"));
                db->featureNames.push_back(string("ToeDiffZ"));
            }
        }

        assert(currentFeature == db->featureDim);
    }


    TraceLog(LOG_INFO, "AnimDatabase: built %d 4D features (left-toe pos+vel in hips frame)", db->motionFrameCount);
}

// Convert global frame index to (animIndex, localFrame)
static void AnimDatabaseGlobalToLocal(const AnimDatabase* db, int globalFrame, int* animIndex, int* localFrame)
{
    for (int i = 0; i < db->animCount; i++)
    {
        if (globalFrame < db->animStartFrame[i] + db->animFrameCount[i])
        {
            *animIndex = i;
            *localFrame = globalFrame - db->animStartFrame[i];
            return;
        }
    }
    // Clamp to last frame
    *animIndex = db->animCount - 1;
    *localFrame = db->animFrameCount[*animIndex] - 1;
}

// Get a random (animIndex, time) from the database
static void AnimDatabaseRandomTime(const AnimDatabase* db, int* animIndex, float* time)
{
    if (db->animCount == 0) return;

    *animIndex = rand() % db->animCount;
    const int frameCount = db->animFrameCount[*animIndex];
    const float frameTime = db->animFrameTime[*animIndex];
    const int randomFrame = rand() % frameCount;
    *time = randomFrame * frameTime;
}

// Computes interpolation frame indices and alpha for animation sampling
static inline void GetInterFrameAlpha(
    const AnimDatabase* db,
    int animIndex,
    float animTime,
    int& outF0,
    int& outF1,
    float& outAlpha)
{
    const float frameTime = db->animFrameTime[animIndex];
    const int frameCount = db->animFrameCount[animIndex];

    outF0 = 0;
    outF1 = 0;
    outAlpha = 0.0f;

    if (frameTime > 0.0f && frameCount > 0)
    {
        const float maxFrame = (float)(frameCount - 1);
        float frameF = animTime / frameTime;

        if (frameF < 0.0f) frameF = 0.0f;
        if (frameF > maxFrame) frameF = maxFrame;

        outF0 = (int)floorf(frameF);
        outF1 = outF0 + 1;
        if (outF1 >= frameCount) outF1 = frameCount - 1;
        outAlpha = frameF - (float)outF0;
    }
}


// sample interpolated local pose from AnimDatabase at time animTime, using Rot6d for rotations
// optionally also samples angular velocities if outAngularVelocities is not null
// velocityTimeOffset: offset added to animTime when sampling velocities (use -dt/2 for midpoint sampling)
static inline void SampleCursorPoseLerp6d(
    const AnimDatabase* db,
    int animIndex,
    float animTime,
    float velocityTimeOffset,
    vector<Vector3>& outPositions,
    vector<Rot6d>& outRotations6d,
    vector<Vector3>* outAngularVelocities,
    Vector3* outRootPos,
    Rot6d* outRootRot6d)
{
    const int jointCount = db->jointCount;
    const int clipStart = db->clipStartFrame[animIndex];

    // sample pose at animTime
    int f0, f1;
    float alpha;
    GetInterFrameAlpha(db, animIndex, animTime, f0, f1, alpha);

    span<const Vector3> posRow0 = db->localJointPositions.row_view(clipStart + f0);
    span<const Vector3> posRow1 = db->localJointPositions.row_view(clipStart + f1);
    span<const Rot6d> rotRow0 = db->localJointRotations6d.row_view(clipStart + f0);
    span<const Rot6d> rotRow1 = db->localJointRotations6d.row_view(clipStart + f1);

    for (int j = 0; j < jointCount; ++j)
    {
        outPositions[j] = Vector3Lerp(posRow0[j], posRow1[j], alpha);
        Rot6dLerp(rotRow0[j], rotRow1[j], alpha, outRotations6d[j]);
    }

    // sample angular velocities at animTime + velocityTimeOffset (for midpoint sampling)
    if (outAngularVelocities)
    {
        int vf0, vf1;
        float vAlpha;
        GetInterFrameAlpha(db, animIndex, animTime, vf0, vf1, vAlpha);

        span<const Vector3> velRow0 = db->localJointAngularVelocities.row_view(clipStart + vf0);
        span<const Vector3> velRow1 = db->localJointAngularVelocities.row_view(clipStart + vf1);
        for (int j = 0; j < jointCount; ++j)
        {
            (*outAngularVelocities)[j] = Vector3Lerp(velRow0[j], velRow1[j], vAlpha);
        }
    }

    if (outRootPos) *outRootPos = outPositions[0];
    if (outRootRot6d) *outRootRot6d = outRotations6d[0];
}

// sample global toe velocity from db->globalJointVelocities at animTime
// used for foot IK velocity blending
static inline void SampleGlobalToeVelocity(
    const AnimDatabase* db,
    int animIndex,
    float animTime,
    /*out*/ Vector3 outToeVelocity[SIDES_COUNT])
{
    const int clipStart = db->clipStartFrame[animIndex];

    // sample pose at animTime
    int f0, f1;
    float alpha;
    GetInterFrameAlpha(db, animIndex, animTime, f0, f1, alpha);

    span<const Vector3> velRow0 = db->globalJointVelocities.row_view(clipStart + f0);
    span<const Vector3> velRow1 = db->globalJointVelocities.row_view(clipStart + f1);

    for (int side : sides)
    {
        const int toeIdx = db->toeIndices[side];
        if (toeIdx >= 0)
        {
            outToeVelocity[side] = Vector3Lerp(velRow0[toeIdx], velRow1[toeIdx], alpha);
        }
        else
        {
            outToeVelocity[side] = Vector3Zero();
        }
    }
}

struct BlendCursor {
    int animIndex = -1;                         // which animation/clip
    float animTime = 0.0f;                      // playback time in that clip
    DoubleSpringDamperState weightSpring = {};  // spring state for weight blending (x = current weight)
    float normalizedWeight = 0.0f;              // weight / totalWeight (sums to 1 across active cursors)
    float targetWeight = 0.0f;                  // desired weight
    float blendTime = 0.3f;                     // halflife for double spring damper
    bool active = false;                        // is cursor in use

    // Local-space pose stored per cursor for blending (size = jointCount)
    vector<Vector3> localPositions;
    vector<Rot6d> localRotations6d;
    vector<Vector3> localAngularVelocities;

    // Global-space pose for debug visualization (computed via FK after sampling)
    vector<Vector3> globalPositions;
    vector<Quaternion> globalRotations;

    // Previous local root state used to compute per-cursor root deltas.
    // Stored in the same local-space that we sample into above.
    Vector3 prevLocalRootPos;
    Rot6d prevLocalRootRot6d = Rot6dIdentity();

    // Debug: last computed deltas (for visualization)
    Vector3 lastDeltaWorld;      // XZ world-space position delta this frame
    float lastDeltaYaw;          // Yaw delta this frame (radians)

    // Root motion velocity tracking (for acceleration-based blending)
    Vector3 rootVelocity = Vector3Zero();    // current root velocity (world space)
    float rootYawRate = 0.0f;                // current yaw rate (radians/sec)
    Vector3 prevRootVelocity = Vector3Zero(); // previous frame's velocity
    float prevRootYawRate = 0.0f;            // previous frame's yaw rate

    // Global toe velocities for foot IK (sampled from db->globalJointVelocities)
    Vector3 globalToeVelocity[SIDES_COUNT] = { Vector3Zero(), Vector3Zero() };
};

//----------------------------------------------------------------------------------
// Controlled Character - Root Motion Playback
//----------------------------------------------------------------------------------

// A character that plays animation with root motion extracted and applied to world transform.
// Every N seconds it jumps to a random time in the loaded animations while maintaining
// its world position and facing direction.
struct ControlledCharacter {
    // World placement (accumulated from root motion)
    Vector3 worldPosition;
    Quaternion worldRotation;  // Character's facing direction (Y-axis rotation only)

    // Animation state
    int animIndex;             // Which loaded animation to play from
    float animTime;            // Current playback time in that animation

    // For computing root motion deltas between frames
    Vector3 prevRootPosition;
    Quaternion prevRootRotation;

    // Random switch timer
    float switchTimer;

    // Pose output (local space with root zeroed, then transformed to world)
    TransformData xformData;
    TransformData xformTmp0;
    TransformData xformTmp1;
    TransformData xformTmp2;
    TransformData xformTmp3;

    // Visual properties
    Color color;
    float opacity;
    float radius;
    float scale;

    // Reference to skeleton (first loaded BVH)
    const BVHData* skeleton;
    bool active;


    // -----------------------
    // Blending cursor pool
    // -----------------------
    static constexpr int MAX_BLEND_CURSORS = 10;
    BlendCursor cursors[MAX_BLEND_CURSORS];

    // Debug: last blended root motion delta (for visualization)
    Vector3 lastBlendedDeltaWorld;
    float lastBlendedDeltaYaw;

    // Velocity-based blending state (advances with blended angular velocity, lerps to target)
    vector<Rot6d> velBlendedRotations6d;       // [jointCount] - smoothed rotations
    float blendPosReturnTime = 0.1f;       // time to reach halfway to target (lower = snappier)
    bool velBlendInitialized = false;
    bool useVelBlending = false;               // toggle for enabling velocity-based blending

    // Separate hips rotation smoothing (no vel blending, just lerp with different blend time)
    Rot6d hipsSmoothedRot6d = Rot6dIdentity();
    float hipsRotationBlendTime = 0.3f;         // typically longer than blendPosReturnTime
    bool hipsInitialized = false;

    // Smoothed root motion state
    Vector3 smoothedRootVelocity = Vector3Zero();  // smoothed linear velocity (world space XZ)
    float smoothedRootYawRate = 0.0f;              // smoothed angular velocity (radians/sec)
    bool rootMotionInitialized = false;

    // Toe velocity tracking (for foot IK)
    Vector3 prevToeGlobalPos[SIDES_COUNT];        // previous frame global positions
    Vector3 toeActualVelocity[SIDES_COUNT];       // computed from FK: (current - prev) / dt
    Vector3 toeBlendedVelocity[SIDES_COUNT];      // blended from cursor global toe velocities
    bool toeTrackingInitialized = false;
};

// Initialize the controlled character when first animation is loaded
static void ControlledCharacterInit(
    ControlledCharacter* cc,
    const BVHData* skeleton,
    float scale,
    float switchInterval)
{
    cc->skeleton = skeleton;
    cc->scale = scale;
    cc->active = true;

    // Start offset from origin so we can see both characters
    cc->worldPosition = Vector3{ 2.0f, 0.0f, 0.0f };
    cc->worldRotation = QuaternionIdentity();

    // Animation state
    cc->animIndex = 0;
    cc->animTime = 0.0f;
    cc->switchTimer = switchInterval;

    // Initialize transform buffers
    TransformDataInit(&cc->xformData);
    TransformDataResize(&cc->xformData, skeleton);
    TransformDataInit(&cc->xformTmp0);
    TransformDataResize(&cc->xformTmp0, skeleton);
    TransformDataInit(&cc->xformTmp1);
    TransformDataResize(&cc->xformTmp1, skeleton);
    TransformDataInit(&cc->xformTmp2);
    TransformDataResize(&cc->xformTmp2, skeleton);
    TransformDataInit(&cc->xformTmp3);
    TransformDataResize(&cc->xformTmp3, skeleton);

    // Ensure blend cursors have storage sized to joint count
    for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i) {
        cc->cursors[i].localPositions.resize(cc->xformData.jointCount);
        cc->cursors[i].localRotations6d.resize(cc->xformData.jointCount);
        cc->cursors[i].localAngularVelocities.resize(cc->xformData.jointCount);
        cc->cursors[i].globalPositions.resize(cc->xformData.jointCount);
        cc->cursors[i].globalRotations.resize(cc->xformData.jointCount);
        cc->cursors[i].prevLocalRootPos = cc->xformData.localPositions[0];
        Rot6dFromQuaternion(cc->xformData.localRotations[0], cc->cursors[i].prevLocalRootRot6d);

        cc->cursors[i].active = false;
        cc->cursors[i].weightSpring = {};  // zero all spring state
        cc->cursors[i].targetWeight = 0.0f;
    }

    // Sample initial pose to get starting root state
    TransformDataSampleFrame(&cc->xformData, skeleton, 0, scale);
    cc->prevRootPosition = cc->xformData.localPositions[0];
    cc->prevRootRotation = cc->xformData.localRotations[0];

    // Velocity-based blending state
    cc->velBlendedRotations6d.resize(cc->xformData.jointCount);
    cc->velBlendInitialized = false;
    cc->useVelBlending = false;

    // Hips rotation smoothing state
    cc->hipsSmoothedRot6d = Rot6dIdentity();
    cc->hipsInitialized = false;

    // Smoothed root motion state
    cc->smoothedRootVelocity = Vector3Zero();
    cc->smoothedRootYawRate = 0.0f;
    cc->rootMotionInitialized = false;

    // Visual defaults (cyan-ish to distinguish from orange original)
    cc->color = Color{ 50, 200, 200, 255 };
    cc->opacity = 1.0f;
    cc->radius = 0.04f;
}

static void ControlledCharacterFree(ControlledCharacter* cc)
{
    TransformDataFree(&cc->xformData);
    TransformDataFree(&cc->xformTmp0);
    TransformDataFree(&cc->xformTmp1);
    TransformDataFree(&cc->xformTmp2);
    TransformDataFree(&cc->xformTmp3);

    // Clear cursor storage vectors
    for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i) {
        cc->cursors[i].localPositions.clear();
        cc->cursors[i].localRotations6d.clear();
        cc->cursors[i].localAngularVelocities.clear();
        cc->cursors[i].active = false;
    }

    cc->velBlendedRotations6d.clear();
    cc->active = false;
}


// Helper: find an available cursor (inactive) or the one with smallest weight
static BlendCursor* FindAvailableCursor(ControlledCharacter* cc)
{
    BlendCursor* smallest = nullptr;
    float minWeight = 1e9f;

    for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i)
    {
        if (!cc->cursors[i].active) return &cc->cursors[i];
        if (cc->cursors[i].weightSpring.x < minWeight)
        {
            minWeight = cc->cursors[i].weightSpring.x;
            smallest = &cc->cursors[i];
        }
    }
    return smallest; // may be null only when MAX_BLEND_CURSORS == 0
}

// - requires db != nullptr && db->valid and db->jointCount == cc->xformData.jointCount
// - uses db->localJointPositions / localJointRotations6d for sampling (no per-frame global->local conversion)
// - blends per-cursor root deltas in world-space (XZ + yaw) and applies to cc->world*
static void ControlledCharacterUpdate(
    ControlledCharacter* cc,
    const CharacterData* characterData,
    const AnimDatabase* db,
    float dt,
    int sampleMode,
    float defaultBlendTime,
    float switchInterval)
{
    if (!cc->active || characterData->count == 0) return;

    const int jc = cc->xformData.jointCount;

    // REQUIRE a valid AnimDatabase     
    if (db == nullptr || !db->valid) {
        // Defensive: disable controlled character to avoid inconsistent behaviour.
        TraceLog(LOG_WARNING, "ControlledCharacterUpdate: AnimDatabase not valid - disabling controlled character.");
        cc->active = false;
        return;
    }

    if (db->jointCount != cc->xformData.jointCount) {
        TraceLog(LOG_WARNING, "ControlledCharacterUpdate: jointCount mismatch - disabling controlled character.");
        cc->active = false;
        return;
    }

    const BVHData* bvh = &characterData->bvhData[cc->animIndex];

    bool firstFrame = true;
    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        const BlendCursor& cur = cc->cursors[ci];
        if (cur.active) {
            firstFrame = false;
            break;
        }
    }

    // --- Random switch logic ---
    cc->switchTimer -= dt;
    if (cc->switchTimer <= 0.0f || firstFrame)
    {
        const int newAnim = rand() % characterData->count;
        const BVHData* newBvh = &characterData->bvhData[newAnim];
        const float newMaxTime = (newBvh->frameCount - 1) * newBvh->frameTime;
        const float startTime = ((float)rand() / (float)RAND_MAX) * newMaxTime;

        // Fade out existing cursors
        for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i) {
            if (cc->cursors[i].active) {
                cc->cursors[i].targetWeight = 0.0f;
            }
        }

        // Acquire cursor slot
        BlendCursor* cursor = FindAvailableCursor(cc);
        assert(cursor != nullptr);
        if (cursor)
        {
            cursor->active = true;
            cursor->animIndex = newAnim;
            cursor->animTime = startTime;
            cursor->weightSpring = {};  // reset spring state

            if (firstFrame) {
                // immediate full weight on first frame - set all spring state to 1
                cursor->weightSpring.x = 1.0f;
                cursor->weightSpring.xi = 1.0f;
            }

            cursor->targetWeight = 1.0f;
            cursor->blendTime = defaultBlendTime;

            Vector3 rootPos;
            Rot6d rootRot6d;
            SampleCursorPoseLerp6d(
                db,
                cursor->animIndex,
                cursor->animTime,
                0.0f,  // no velocity offset for initialization
                cursor->localPositions,
                cursor->localRotations6d,
                &cursor->localAngularVelocities,
                &rootPos,
                &rootRot6d);

            cursor->prevLocalRootPos = rootPos;
            cursor->prevLocalRootRot6d = rootRot6d;
        }

        cc->animIndex = newAnim;
        cc->animTime = startTime;
        cc->switchTimer = switchInterval;

        TraceLog(LOG_INFO, "ControlledCharacter: Spawned cursor for anim %d, time %.2f", newAnim, startTime);
    }

    // --- Advance main anim time (kept for legacy semantics if needed) ---
    cc->animTime += dt;
    const float currentMaxTime = (bvh->frameCount - 1) * bvh->frameTime;
    if (cc->animTime >= currentMaxTime)
    {
        cc->animTime = fmodf(cc->animTime, currentMaxTime);
    }

    // --------- Per-cursor update: sample pose, update weights, compute per-cursor root delta ----------
    Vector3 blendedWorldDelta = Vector3Zero();
    float blendedYawDelta = 0.0f;
    float totalRootWeight = 0.0f;

    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        BlendCursor& cur = cc->cursors[ci];
        if (!cur.active) continue;

        // Advance cursor time and clamp
        cur.animTime += dt;
        const BVHData* cbvh = &characterData->bvhData[cur.animIndex];
        const float clipMax = (cbvh->frameCount - 1) * cbvh->frameTime;
        if (cur.animTime > clipMax) cur.animTime = clipMax;

        // Sample interpolated pose from DB (using Rot6d for blending)
        // Velocity sampled at midpoint of frame interval for better accuracy
        Vector3 sampledRootPos;
        Rot6d sampledRootRot6d;
        SampleCursorPoseLerp6d(
            db,
            cur.animIndex,
            cur.animTime,
            -dt * 0.5f,  // sample velocity at midpoint of frame
            cur.localPositions,
            cur.localRotations6d,
            &cur.localAngularVelocities,
            &sampledRootPos,
            &sampledRootRot6d);

        // Sample global toe velocities from database (for foot IK)
        // These are in animation-world space, we'll transform them to character-world after getting root yaw
        Vector3 animSpaceToeVel[SIDES_COUNT];
        SampleGlobalToeVelocity(db, cur.animIndex, cur.animTime - dt * 0.5f, animSpaceToeVel);

        // Update weight via spring integrator
        DoubleSpringDamper(cur.weightSpring, cur.targetWeight, cur.blendTime, dt);

        // clamp the output weight
        if (cur.weightSpring.x < 0.0f) cur.weightSpring.x = 0.0f;
        if (cur.weightSpring.x > 1.0f) cur.weightSpring.x = 1.0f;

        // Compute per-cursor root delta in animation-local space and convert to world-space
        const Vector3 currLocalRootPos = sampledRootPos;
        const Rot6d currLocalRootRot6d = sampledRootRot6d;
        const Vector3 prevLocalRootPos = cur.prevLocalRootPos;
        const Rot6d prevLocalRootRot6d = cur.prevLocalRootRot6d;

        Vector3 deltaPosAnim = Vector3Subtract(currLocalRootPos, prevLocalRootPos);
        deltaPosAnim.y = 0.0f;

        // extract yaw from Rot6d and rotate delta position to local space
        const float prevYaw = Rot6dGetYaw(prevLocalRootRot6d);
        const float currYaw = Rot6dGetYaw(currLocalRootRot6d);
        const Rot6d invPrevYawRot = Rot6dFromYaw(-prevYaw);
        const Rot6d invCurrYawRot = Rot6dFromYaw(-currYaw);

        // Transform toe velocities from animation-world to character-world
        // 1. Remove animation's root yaw (to get heading-relative velocity)
        // 2. Rotate by character's worldRotation
        for (int side : sides)
        {
            Vector3 toeVelLocal;
            Rot6dTransformVector(invCurrYawRot, animSpaceToeVel[side], toeVelLocal);
            cur.globalToeVelocity[side] = Vector3RotateByQuaternion(toeVelLocal, cc->worldRotation);
        }

        Vector3 deltaLocal;
        Rot6dTransformVector(invPrevYawRot, deltaPosAnim, deltaLocal);
        const Vector3 deltaWorld = Vector3RotateByQuaternion(deltaLocal, cc->worldRotation);

        // compute yaw delta (simple subtraction since we have angles directly)
        float deltaYaw = currYaw - prevYaw;
        if (deltaYaw > PI) deltaYaw -= 2.0f * PI;
        else if (deltaYaw < -PI) deltaYaw += 2.0f * PI;

        // Store deltas for debug visualization
        cur.lastDeltaWorld = deltaWorld;
        cur.lastDeltaYaw = deltaYaw;

        // Compute velocity from delta
        if (dt > 1e-6f)
        {
            cur.prevRootVelocity = cur.rootVelocity;
            cur.prevRootYawRate = cur.rootYawRate;
            cur.rootVelocity = Vector3Scale(deltaWorld, 1.0f / dt);
            cur.rootYawRate = deltaYaw / dt;
        }

        const float wgt = cur.weightSpring.x;
        if (wgt > 1e-6f)
        {
            blendedWorldDelta = Vector3Add(blendedWorldDelta, Vector3Scale(deltaWorld, wgt));
            blendedYawDelta += deltaYaw * wgt;
            totalRootWeight += wgt;
        }

        // store current root as prev for next frame
        cur.prevLocalRootPos = currLocalRootPos;
        cur.prevLocalRootRot6d = currLocalRootRot6d;

        // Strip yaw from root rotation BEFORE blending to avoid Rot6d singularity
        // when blending anims facing opposite directions.
        // Problem: Rot6d averaging of 0° and 180° yaw gives a zero-length vector.
        // Solution: blend hip "tilt/roll relative to heading" instead of absolute rotation.
        Rot6dRemoveYComponent(cur.localRotations6d[0], cur.localRotations6d[0]);

        // deactivate tiny-weight cursors
        if (cur.weightSpring.x <= 1e-4f && cur.targetWeight <= 1e-4f)
        {
            cur.active = false;
            cur.animIndex = -1;
        }
    }

    // Compute normalized weights for all cursors
    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        BlendCursor& cur = cc->cursors[ci];
        if (!cur.active)
        {
            cur.normalizedWeight = 0.0f;
            continue;
        }
        cur.normalizedWeight = (totalRootWeight > 1e-6f) ? (cur.weightSpring.x / totalRootWeight) : 0.0f;
    }

    // Blend toe velocities from cursors using normalized weights
    for (int side : sides)
    {
        cc->toeBlendedVelocity[side] = Vector3Zero();
    }
    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        const BlendCursor& cur = cc->cursors[ci];
        if (!cur.active) continue;
        const float w = cur.normalizedWeight;
        if (w <= 1e-6f) continue;

        for (int side : sides)
        {
            cc->toeBlendedVelocity[side] = Vector3Add(
                cc->toeBlendedVelocity[side],
                Vector3Scale(cur.globalToeVelocity[side], w));
        }
    }

    // Compute FK for each active cursor (for debug visualization)
    // We do this with root zeroed (same treatment as final blended pose)
    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        BlendCursor& cur = cc->cursors[ci];
        if (!cur.active) continue;

        // Copy local pose to global (we'll compute FK in-place)
        // Convert from Rot6d to Quaternion for FK computation
        for (int j = 0; j < jc; ++j)
        {
            cur.globalPositions[j] = cur.localPositions[j];
            Rot6dToQuaternion(cur.localRotations6d[j], cur.globalRotations[j]);
        }

        // Zero out root XZ translation
        // Note: Y rotation was already stripped from localRotations6d[0] earlier
        cur.globalPositions[0].x = 0.0f;
        cur.globalPositions[0].z = 0.0f;

        // Forward kinematics
        for (int j = 1; j < jc; ++j)
        {
            const int p = cc->xformData.parents[j];
            cur.globalPositions[j] = Vector3Add(
                Vector3RotateByQuaternion(cur.globalPositions[j], cur.globalRotations[p]),
                cur.globalPositions[p]);
            cur.globalRotations[j] = QuaternionMultiply(cur.globalRotations[p], cur.globalRotations[j]);
        }

        // Transform to world space
        for (int j = 0; j < jc; ++j)
        {
            cur.globalPositions[j] = Vector3Add(
                Vector3RotateByQuaternion(cur.globalPositions[j], cc->worldRotation),
                cc->worldPosition);
            cur.globalRotations[j] = QuaternionMultiply(cc->worldRotation, cur.globalRotations[j]);
        }
    }

    assert(totalRootWeight > 1e-6f);

    const Vector3 finalWorldDelta = Vector3Scale(blendedWorldDelta, 1.0f / totalRootWeight);
    const float finalYawDelta = blendedYawDelta / totalRootWeight;

    // Store for debug visualization
    cc->lastBlendedDeltaWorld = finalWorldDelta;
    cc->lastBlendedDeltaYaw = finalYawDelta;

    // Apply root motion (with optional velocity-based smoothing)
    if (cc->useVelBlending && dt > 1e-6f)
    {
        // Blend velocities and accelerations from cursors using normalized weights
        Vector3 blendedVelocity = Vector3Zero();
        float blendedYawRate = 0.0f;
        Vector3 blendedAcceleration = Vector3Zero();
        float blendedYawAccel = 0.0f;

        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc->cursors[ci];
            if (!cur.active) continue;
            const float w = cur.normalizedWeight;
            if (w <= 1e-6f) continue;

            // accumulate weighted velocity
            blendedVelocity = Vector3Add(blendedVelocity, Vector3Scale(cur.rootVelocity, w));
            blendedYawRate += cur.rootYawRate * w;

            // compute and accumulate weighted acceleration
            const Vector3 acc = Vector3Scale(Vector3Subtract(cur.rootVelocity, cur.prevRootVelocity), 1.0f / dt);
            const float yawAcc = (cur.rootYawRate - cur.prevRootYawRate) / dt;
            blendedAcceleration = Vector3Add(blendedAcceleration, Vector3Scale(acc, w));
            blendedYawAccel += yawAcc * w;
        }

        // Initialize on first frame
        if (!cc->rootMotionInitialized)
        {
            cc->smoothedRootVelocity = blendedVelocity;
            cc->smoothedRootYawRate = blendedYawRate;
            cc->rootMotionInitialized = true;
        }

        // Step 1: advance smoothed velocity using blended acceleration
        cc->smoothedRootVelocity = Vector3Add(cc->smoothedRootVelocity, Vector3Scale(blendedAcceleration, dt));
        cc->smoothedRootYawRate += blendedYawAccel * dt;

        // Step 2: lerp towards blended target velocity
        const float blendTime = cc->blendPosReturnTime;
        if (blendTime > 1e-6f)
        {
            const float alpha = 1.0f - powf(0.5f, dt / blendTime);
            cc->smoothedRootVelocity = Vector3Lerp(cc->smoothedRootVelocity, blendedVelocity, alpha);
            cc->smoothedRootYawRate = Lerp(cc->smoothedRootYawRate, blendedYawRate, alpha);
        }
        else
        {
            cc->smoothedRootVelocity = blendedVelocity;
            cc->smoothedRootYawRate = blendedYawRate;
        }

        // Apply smoothed velocity
        const Vector3 smoothedDelta = Vector3Scale(cc->smoothedRootVelocity, dt);
        const float smoothedYawDelta = cc->smoothedRootYawRate * dt;

        cc->worldPosition = Vector3Add(cc->worldPosition, smoothedDelta);
        const Quaternion yawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, smoothedYawDelta);
        cc->worldRotation = QuaternionNormalize(QuaternionMultiply(yawQ, cc->worldRotation));
    }
    else
    {
        // Direct application without smoothing
        cc->worldPosition = Vector3Add(cc->worldPosition, finalWorldDelta);
        const Quaternion yawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, finalYawDelta);
        cc->worldRotation = QuaternionNormalize(QuaternionMultiply(yawQ, cc->worldRotation));

        // Still update smoothedRootVelocity for visualization
        if (dt > 1e-6f)
        {
            cc->smoothedRootVelocity = Vector3Scale(finalWorldDelta, 1.0f / dt);
            cc->smoothedRootYawRate = finalYawDelta / dt;
        }
    }

    // --- Rot6d blending using normalized weights (no double-cover issues, simple weighted average then normalize) ---
    {
        vector<Vector3> posAccum(jc, Vector3Zero());
        vector<Rot6d> rot6dAccum(jc, Rot6d{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
        vector<Vector3> angVelAccum(jc, Vector3Zero());

        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc->cursors[ci];
            if (!cur.active) continue;

            // use precomputed normalized weight (sums to 1 across active cursors)
            const float w = cur.normalizedWeight;
            if (w <= 1e-6f) continue;

            for (int j = 0; j < jc; ++j)
            {
                posAccum[j] = Vector3Add(posAccum[j], Vector3Scale(cur.localPositions[j], w));
                angVelAccum[j] = Vector3Add(angVelAccum[j], Vector3Scale(cur.localAngularVelocities[j], w));

                // weighted accumulation of Rot6d (just add scaled components)
                const Rot6d& r = cur.localRotations6d[j];
                rot6dAccum[j].ax += r.ax * w;
                rot6dAccum[j].ay += r.ay * w;
                rot6dAccum[j].az += r.az * w;
                rot6dAccum[j].bx += r.bx * w;
                rot6dAccum[j].by += r.by * w;
                rot6dAccum[j].bz += r.bz * w;
            }
        }

        // positions use normalized weights directly (no division needed)
        for (int j = 0; j < jc; ++j)
        {
            cc->xformData.localPositions[j] = posAccum[j];
        }

        // normalize blended Rot6d to get target rotations
        vector<Rot6d> blendedRot6d(jc);
        for (int j = 0; j < jc; ++j)
        {
            Rot6d blended = rot6dAccum[j];
            const float lenA = sqrtf(blended.ax * blended.ax + blended.ay * blended.ay + blended.az * blended.az);
            if (lenA > 1e-6f)
            {
                Rot6dNormalize(blended);
                blendedRot6d[j] = blended;
            }
            else
            {
                blendedRot6d[j] = Rot6dIdentity();
            }
        }

        // Hips (joint 0) uses separate smoothing with its own blend time (no velocity integration)
        {
            if (!cc->hipsInitialized)
            {
                cc->hipsSmoothedRot6d = blendedRot6d[0];
                cc->hipsInitialized = true;
            }

            const float hipsBlendTime = cc->hipsRotationBlendTime;
            if (hipsBlendTime > 1e-6f)
            {
                const float alpha = 1.0f - powf(0.5f, dt / hipsBlendTime);
                Rot6dLerp(cc->hipsSmoothedRot6d, blendedRot6d[0], alpha, cc->hipsSmoothedRot6d);
            }
            else
            {
                cc->hipsSmoothedRot6d = blendedRot6d[0];
            }

            Rot6dToQuaternion(cc->hipsSmoothedRot6d, cc->xformData.localRotations[0]);
        }

        // Other joints (1+): velocity-based blending if enabled
        if (cc->useVelBlending)
        {
            // initialize on first frame
            if (!cc->velBlendInitialized)
            {
                for (int j = 1; j < jc; ++j)
                {
                    cc->velBlendedRotations6d[j] = blendedRot6d[j];
                }
                cc->velBlendInitialized = true;
            }

            // step 1: advance rotations using blended angular velocity (skip hips)
            for (int j = 1; j < jc; ++j)
            {
                Rot6dRotate(cc->velBlendedRotations6d[j], angVelAccum[j], dt);
            }

            // step 2: lerp towards blended target
            const float blendTime = cc->blendPosReturnTime;
            if (blendTime > 1e-6f)
            {
                const float alpha = 1.0f - powf(0.5f, dt / blendTime);
                for (int j = 1; j < jc; ++j)
                {
                    Rot6dLerp(cc->velBlendedRotations6d[j], blendedRot6d[j], alpha, cc->velBlendedRotations6d[j]);
                }
            }
            else
            {
                for (int j = 1; j < jc; ++j)
                {
                    cc->velBlendedRotations6d[j] = blendedRot6d[j];
                }
            }

            // convert to quaternion for FK (skip hips, already done above)
            for (int j = 1; j < jc; ++j)
            {
                Rot6dToQuaternion(cc->velBlendedRotations6d[j], cc->xformData.localRotations[j]);
            }
        }
        else
        {
            // standard blending: directly use blended rotations (skip hips, already done above)
            for (int j = 1; j < jc; ++j)
            {
                Rot6dToQuaternion(blendedRot6d[j], cc->xformData.localRotations[j]);
            }
        }
    }

    // --- After blending local pose, update prev-root bookkeeping ---
    const Vector3 currentRootPos = cc->xformData.localPositions[0];
    const Quaternion currentRootRot = cc->xformData.localRotations[0];

    cc->prevRootPosition = currentRootPos;
    cc->prevRootRotation = currentRootRot;

    // Zero out root translation XZ for rendering
    cc->xformData.localPositions[0].x = 0.0f;
    cc->xformData.localPositions[0].z = 0.0f;
    // Note: Y rotation was already stripped from localRotations6d[0] before blending

    // Forward kinematics (local space)
    TransformDataForwardKinematics(&cc->xformData);

    // Transform to world space using updated cc->world*
    for (int i = 0; i < cc->xformData.jointCount; ++i)
    {
        cc->xformData.globalPositions[i] = Vector3Add(Vector3RotateByQuaternion(cc->xformData.globalPositions[i], cc->worldRotation), cc->worldPosition);
        cc->xformData.globalRotations[i] = QuaternionMultiply(cc->worldRotation, cc->xformData.globalRotations[i]);
    }

    // Compute actual toe velocity from FK result
    for (int side : sides)
    {
        const int toeIdx = db->toeIndices[side];
        if (toeIdx < 0 || toeIdx >= cc->xformData.jointCount) continue;

        const Vector3 currentPos = cc->xformData.globalPositions[toeIdx];
        if (cc->toeTrackingInitialized && dt > 1e-6f)
        {
            cc->toeActualVelocity[side] = Vector3Scale(
                Vector3Subtract(currentPos, cc->prevToeGlobalPos[side]), 1.0f / dt);
        }
        else
        {
            cc->toeActualVelocity[side] = Vector3Zero();
        }
        cc->prevToeGlobalPos[side] = currentPos;
    }
    cc->toeTrackingInitialized = true;
}






//----------------------------------------------------------------------------------
// Shaders
//----------------------------------------------------------------------------------


#define AO_CAPSULES_MAX 32
#define SHADOW_CAPSULES_MAX 64


// Shader uniform location indices (cached for performance)
struct ShaderUniforms {
    // Capsule geometry
    int isCapsule;
    int capsulePosition;
    int capsuleRotation;
    int capsuleHalfLength;
    int capsuleRadius;
    int capsuleStart;
    int capsuleVector;

    // Shadow casting capsules
    int shadowCapsuleCount;
    int shadowCapsuleStarts;
    int shadowCapsuleVectors;
    int shadowCapsuleRadii;
    int shadowLookupTable;
    int shadowLookupResolution;

    // Ambient occlusion capsules
    int aoCapsuleCount;
    int aoCapsuleStarts;
    int aoCapsuleVectors;
    int aoCapsuleRadii;
    int aoLookupTable;
    int aoLookupResolution;

    // Camera
    int cameraPosition;

    // Material properties
    int objectColor;
    int objectSpecularity;
    int objectGlossiness;
    int objectOpacity;

    // Lighting
    int sunStrength;
    int sunDir;
    int sunColor;
    int skyStrength;
    int skyColor;
    int ambientStrength;
    int groundStrength;

    // Tonemapping
    int exposure;
};

// Lookup all shader uniform indices
static void ShaderUniformsInit(ShaderUniforms* uniforms, Shader shader)
{
    uniforms->isCapsule = GetShaderLocation(shader, "isCapsule");
    uniforms->capsulePosition = GetShaderLocation(shader, "capsulePosition");
    uniforms->capsuleRotation = GetShaderLocation(shader, "capsuleRotation");
    uniforms->capsuleHalfLength = GetShaderLocation(shader, "capsuleHalfLength");
    uniforms->capsuleRadius = GetShaderLocation(shader, "capsuleRadius");
    uniforms->capsuleStart = GetShaderLocation(shader, "capsuleStart");
    uniforms->capsuleVector = GetShaderLocation(shader, "capsuleVector");

    uniforms->shadowCapsuleCount = GetShaderLocation(shader, "shadowCapsuleCount");
    uniforms->shadowCapsuleStarts = GetShaderLocation(shader, "shadowCapsuleStarts");
    uniforms->shadowCapsuleVectors = GetShaderLocation(shader, "shadowCapsuleVectors");
    uniforms->shadowCapsuleRadii = GetShaderLocation(shader, "shadowCapsuleRadii");
    uniforms->shadowLookupTable = GetShaderLocation(shader, "shadowLookupTable");
    uniforms->shadowLookupResolution = GetShaderLocation(shader, "shadowLookupResolution");

    uniforms->aoCapsuleCount = GetShaderLocation(shader, "aoCapsuleCount");
    uniforms->aoCapsuleStarts = GetShaderLocation(shader, "aoCapsuleStarts");
    uniforms->aoCapsuleVectors = GetShaderLocation(shader, "aoCapsuleVectors");
    uniforms->aoCapsuleRadii = GetShaderLocation(shader, "aoCapsuleRadii");
    uniforms->aoLookupTable = GetShaderLocation(shader, "aoLookupTable");
    uniforms->aoLookupResolution = GetShaderLocation(shader, "aoLookupResolution");

    uniforms->cameraPosition = GetShaderLocation(shader, "cameraPosition");

    uniforms->objectColor = GetShaderLocation(shader, "objectColor");
    uniforms->objectSpecularity = GetShaderLocation(shader, "objectSpecularity");
    uniforms->objectGlossiness = GetShaderLocation(shader, "objectGlossiness");
    uniforms->objectOpacity = GetShaderLocation(shader, "objectOpacity");

    uniforms->sunStrength = GetShaderLocation(shader, "sunStrength");
    uniforms->sunDir = GetShaderLocation(shader, "sunDir");
    uniforms->sunColor = GetShaderLocation(shader, "sunColor");
    uniforms->skyStrength = GetShaderLocation(shader, "skyStrength");
    uniforms->skyColor = GetShaderLocation(shader, "skyColor");
    uniforms->ambientStrength = GetShaderLocation(shader, "ambientStrength");
    uniforms->groundStrength = GetShaderLocation(shader, "groundStrength");

    uniforms->exposure = GetShaderLocation(shader, "exposure");

    TraceLog(LOG_INFO, "Shader uniform locations:");
    TraceLog(LOG_INFO, "isCapsule: %d", uniforms->isCapsule);
    TraceLog(LOG_INFO, "capsulePosition: %d", uniforms->capsulePosition);
}

//--------------------------------------
// Scrubber
//--------------------------------------

// Animation playback state and settings
struct ScrubberSettings {
    // Playback controls
    bool playing;
    bool looping;
    bool inplace;       // Lock root position during playback
    float playTime;
    bool frameSnap;     // Snap to frame boundaries
    int sampleMode;     // 0=nearest, 1=linear, 2=cubic

    // Frame range limits
    float timeLimit;
    int frameLimit;
    int frameMin;
    int frameMax;
    int frameMinSelect;
    int frameMaxSelect;
    bool frameMinEdit;
    bool frameMaxEdit;
    float timeMin;
    float timeMax;
};

static inline void ScrubberSettingsInit(ScrubberSettings* settings, int argc, char** argv)
{
    settings->playing = ArgBool(argc, argv, "playing", true);
    settings->looping = ArgBool(argc, argv, "looping", false);
    settings->inplace = ArgBool(argc, argv, "inplace", false);
    settings->playTime = ArgFloat(argc, argv, "playTime", 0.0f);
    settings->frameSnap = ArgBool(argc, argv, "frameSnap", true);
    static const char* sampleModeOptions[] = { "nearest", "linear", "cubic" };
    settings->sampleMode = ArgEnum(argc, argv, "sampleMode", 3, sampleModeOptions, 1);

    settings->timeLimit = 0.0f;
    settings->frameLimit = 0;
    settings->frameMin = 0;
    settings->frameMax = 0;
    settings->frameMinSelect = 0;
    settings->frameMaxSelect = 0;
    settings->frameMinEdit = false;
    settings->frameMaxEdit = false;
    settings->timeMin = 0.0f;
    settings->timeMax = 0.0f;
}

static inline void ScrubberSettingsRecomputeLimits(ScrubberSettings* settings, CharacterData* characterData)
{
    settings->frameLimit = 0;
    settings->timeLimit = 0.0f;
    for (int i = 0; i < characterData->count; i++)
    {
        settings->frameLimit = MaxInt(settings->frameLimit, characterData->bvhData[i].frameCount - 1);
        settings->timeLimit = Max(settings->timeLimit, (characterData->bvhData[i].frameCount - 1) * characterData->bvhData[i].frameTime);
    }
}

static inline void ScrubberSettingsInitMaxs(ScrubberSettings* settings, CharacterData* characterData)
{
    if (characterData->count == 0) { return; }

    settings->frameMax = characterData->bvhData[characterData->active].frameCount - 1;
    settings->frameMaxSelect = settings->frameMax;
    settings->timeMax = settings->frameMax * characterData->bvhData[characterData->active].frameTime;

    settings->frameMin = 0;
    settings->frameMinSelect = settings->frameMin;
    settings->timeMin = 0.0f;
}

static inline void ScrubberSettingsClamp(ScrubberSettings* settings, CharacterData* characterData)
{
    if (characterData->count == 0) { return; }

    settings->frameMax = ClampInt(settings->frameMax, 0, settings->frameLimit);
    settings->frameMaxSelect = settings->frameMax;
    settings->timeMax = settings->frameMax * characterData->bvhData[characterData->active].frameTime;

    settings->frameMin = ClampInt(settings->frameMin, 0, settings->frameMax);
    settings->frameMinSelect = settings->frameMin;
    settings->timeMin = settings->frameMin * characterData->bvhData[characterData->active].frameTime;

    settings->playTime = Clamp(settings->playTime, settings->timeMin, settings->timeMax);
}

//----------------------------------------------------------------------------------
// Drawing
//----------------------------------------------------------------------------------

static inline void DrawTransform(const Vector3 position, const Quaternion rotation, const float size)
{
    DrawLine3D(position, Vector3Add(position, Vector3RotateByQuaternion(Vector3{ size, 0.0, 0.0 }, rotation)), RED);
    DrawLine3D(position, Vector3Add(position, Vector3RotateByQuaternion(Vector3{ 0.0, size, 0.0 }, rotation)), GREEN);
    DrawLine3D(position, Vector3Add(position, Vector3RotateByQuaternion(Vector3{ 0.0, 0.0, size }, rotation)), BLUE);
}

static inline void DrawSkeleton(TransformData* xformData, bool drawEndSites, Color color, Color endSiteColor)
{
    for (int i = 0; i < xformData->jointCount; i++)
    {
        if (!xformData->endSite[i])
        {
            DrawSphereWires(
                xformData->globalPositions[i],
                0.01f,
                4,
                6,
                color);
        }
        else if (drawEndSites)
        {
            DrawCubeWiresV(
                xformData->globalPositions[i],
                Vector3{ 0.02f, 0.02f, 0.02f },
                endSiteColor);
        }

        if (xformData->parents[i] != -1)
        {
            if (!xformData->endSite[i])
            {
                DrawLine3D(
                    xformData->globalPositions[i],
                    xformData->globalPositions[xformData->parents[i]],
                    color);
            }
            else if (drawEndSites)
            {
                DrawLine3D(
                    xformData->globalPositions[i],
                    xformData->globalPositions[xformData->parents[i]],
                    endSiteColor);
            }
        }
    }
}

static inline void DrawTransforms(TransformData* xformData)
{
    for (int i = 0; i < xformData->jointCount; i++)
    {
        if (!xformData->endSite[i])
        {
            DrawTransform(
                xformData->globalPositions[i],
                xformData->globalRotations[i],
                0.1f);
        }
    }
}

static inline void DrawWireFrames(CapsuleData* capsuleData, Color color)
{
    for (int i = 0; i < capsuleData->capsuleCount; i++)
    {
        const Vector3 capsuleStart = CapsuleStart(capsuleData->capsulePositions[i], capsuleData->capsuleRotations[i], capsuleData->capsuleHalfLengths[i]);
        const Vector3 capsuleEnd = CapsuleEnd(capsuleData->capsulePositions[i], capsuleData->capsuleRotations[i], capsuleData->capsuleHalfLengths[i]);
        const float capsuleRadius = capsuleData->capsuleRadii[i];

        DrawSphereWires(capsuleStart, capsuleRadius, 4, 6, color);
        DrawSphereWires(capsuleEnd, capsuleRadius, 4, 6, color);
        DrawCylinderWiresEx(capsuleStart, capsuleEnd, capsuleRadius, capsuleRadius, 6, color);
    }
}

//----------------------------------------------------------------------------------
// GUI
//----------------------------------------------------------------------------------
static inline void ImGuiCamera(CameraSystem* camera, CharacterData* characterData,
    const ControlledCharacter* controlledCharacter, int argc, char** argv)
{
    ImGui::SetNextWindowPos(ImVec2(20, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(220, 320), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Camera")) {
        // Camera mode selector
        const char* modeNames[] = { "Orbit", "Unreal" };
        int currentMode = static_cast<int>(camera->mode);
        if (ImGui::Combo("Mode", &currentMode, modeNames, 2)) {
            camera->mode = static_cast<FlomoCameraMode>(currentMode);
        }

        ImGui::Separator();

        if (camera->mode == FlomoCameraMode::Orbit)
        {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "RMB - Rotate");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "MMB - Pan");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Scroll - Zoom");
            ImGui::Separator();

            ImGui::Text("Target: [%.2f, %.2f, %.2f]",
                camera->cam3d.target.x, camera->cam3d.target.y, camera->cam3d.target.z);
            ImGui::Text("Azimuth: %.2f", camera->orbit.azimuth);
            ImGui::Text("Altitude: %.2f", camera->orbit.altitude);
            ImGui::Text("Distance: %.2f", camera->orbit.distance);

            if (ImGui::Button("Reset Orbit")) {
                camera->orbit.azimuth = ArgFloat(argc, argv, "cameraAzimuth", 0.0f);
                camera->orbit.altitude = ArgFloat(argc, argv, "cameraAltitude", 0.4f);
                camera->orbit.distance = ArgFloat(argc, argv, "cameraDistance", 4.0f);
                camera->orbit.offset = ArgVector3(argc, argv, "cameraOffset", Vector3Zero());
            }

            ImGui::Separator();

            if (characterData->count > 0) {
                ImGui::Checkbox("Track Bone", &camera->orbit.track);

                // Option to track controlled character
                if (controlledCharacter->active) {
                    ImGui::SameLine();
                    ImGui::Checkbox("Controlled", &camera->orbit.trackControlledCharacter);
                }

                // Build joint name list for combo
                vector<string> joints;
                string comboStr = characterData->jointNamesCombo[characterData->active];
                stringstream ss(comboStr);
                string token;
                while (getline(ss, token, ';')) {
                    joints.push_back(token);
                }
                vector<const char*> items;
                for (const string& s : joints) items.push_back(s.c_str());

                ImGui::Combo("##trackbone", &camera->orbit.trackBone, items.data(), (int)items.size());
            }
        }
        else if (camera->mode == FlomoCameraMode::Unreal)
        {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "RMB + WASD - Move");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "RMB + Q/E - Down/Up");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "MMB - Pan");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Scroll - Dolly");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "RMB + Scroll - Speed");
            ImGui::Separator();

            ImGui::Text("Position: [%.2f, %.2f, %.2f]",
                camera->unreal.position.x, camera->unreal.position.y, camera->unreal.position.z);
            ImGui::Text("Yaw: %.2f  Pitch: %.2f", camera->unreal.yaw, camera->unreal.pitch);
            ImGui::SliderFloat("Move Speed", &camera->unreal.moveSpeed,
                camera->unreal.minSpeed, camera->unreal.maxSpeed, "%.1f");

            if (ImGui::Button("Reset Unreal")) {
                camera->unreal.position = Vector3{ 2.0f, 1.5f, 5.0f };
                camera->unreal.yaw = PI;
                camera->unreal.pitch = 0.0f;
                camera->unreal.moveSpeed = 5.0f;
            }
        }
    }
    ImGui::End();
}

static inline void ImGuiRenderSettings(AppConfig* config,
    CapsuleData* capsuleData, int screenWidth, int screenHeight,
    bool* genoRenderMode, bool genoModelLoaded)
{
    const float sw = (float)screenWidth;
    ImGui::SetNextWindowPos(ImVec2(sw - 260, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(240, 430), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Rendering")) {
        ImGui::SliderFloat("Exposure", &config->exposure, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("Sun Light", &config->sunLightStrength, 0.0f, 1.0f, "%.2f");
        if (ImGui::SliderFloat("Sun Softness", &config->sunLightConeAngle, 0.02f, PI / 4.0f, "%.2f")) {
            CapsuleDataUpdateShadowLookupTable(capsuleData, config->sunLightConeAngle);
        }
        ImGui::SliderFloat("Sky Light", &config->skyLightStrength, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Ambient Light", &config->ambientLightStrength, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Ground Light", &config->groundLightStrength, 0.0f, 0.5f, "%.2f");
        ImGui::SliderFloat("Sun Azimuth", &config->sunAzimuth, -PI, PI, "%.2f");
        ImGui::SliderFloat("Sun Altitude", &config->sunAltitude, 0.0f, 0.49f * PI, "%.2f");

        ImGui::Columns(2);
        ImGui::Checkbox("Draw Origin", &config->drawOrigin);
        ImGui::NextColumn();
        ImGui::Checkbox("Draw Grid", &config->drawGrid);
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Checkbox("Draw Checker", &config->drawChecker);
        ImGui::NextColumn();
        ImGui::Checkbox("Draw Capsules", &config->drawCapsules);
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Checkbox("Draw Wireframes", &config->drawWireframes);
        ImGui::NextColumn();
        ImGui::Checkbox("Draw Skeleton", &config->drawSkeleton);
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Checkbox("Draw Transforms", &config->drawTransforms);
        ImGui::NextColumn();
        ImGui::Checkbox("Draw AO", &config->drawAO);
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Checkbox("Draw Shadows", &config->drawShadows);
        ImGui::NextColumn();
        ImGui::Checkbox("Draw End Sites", &config->drawEndSites);
        ImGui::Columns(1);

        ImGui::Columns(2);
        ImGui::Checkbox("Draw FPS", &config->drawFPS);
        ImGui::NextColumn();
        if (genoModelLoaded) {
            ImGui::Checkbox("Mesh Character", genoRenderMode);
        }
        ImGui::Columns(1);

        ImGui::Separator();
        ImGui::Checkbox("Draw Features", &config->drawFeatures);
        ImGui::Checkbox("Draw Blend Cursors", &config->drawBlendCursors);
        ImGui::Checkbox("Draw Velocities", &config->drawVelocities);
        ImGui::Checkbox("Draw Accelerations", &config->drawAccelerations);
        ImGui::Checkbox("Draw Root Velocities", &config->drawRootVelocities);
        ImGui::Checkbox("Draw Toe Velocities", &config->drawToeVelocities);

    }
    ImGui::End();
}

static inline void ImGuiCharacterData(
    CharacterData* characterData,
    //GuiWindowFileDialogState* fileDialogState,
    ScrubberSettings* scrubberSettings,
    char* errMsg,
    int argc,
    char** argv)
{
    float offsetHeight = 280.0f;
    ImGui::SetNextWindowPos(ImVec2(20, offsetHeight), ImGuiCond_FirstUseEver);

    // Maximum characters to show in the GUI layout (cosmetic limit only)
    constexpr int CHARACTERS_GUI_SLOTS = 15;

    ImGui::SetNextWindowSize(ImVec2(190, (CHARACTERS_GUI_SLOTS - 1) * 30 + 150), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Characters")) {
        //#if !defined(PLATFORM_WEB)
        //        if (ImGui::Button("Open")) {
        //            fileDialogState->windowActive = true;
        //        }
        //        ImGui::SameLine();
        //#endif
        if (ImGui::Button("Clear")) {
            characterData->clearRequested = true;  // Handled in main update with full state access
            errMsg[0] = '\0';
        }

        for (int i = 0; i < characterData->count; i++) {
            string bvhNameShort;
            if (characterData->names[i].length() < 100) {
                bvhNameShort = characterData->names[i];
            }
            else {
                bvhNameShort = characterData->names[i].substr(0, 96) + "...";
            }
            //bool bvhSelected = i == characterData->active;
            ImGui::RadioButton(bvhNameShort.c_str(), &characterData->active, i); // Using RadioButton for selection

            ImGui::SameLine();
            Color color = characterData->colors[i];
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(color.r / 255.f, color.g / 255.f, color.b / 255.f, color.a / 255.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(color.r / 255.f, color.g / 255.f, color.b / 255.f, color.a / 255.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(color.r / 255.f, color.g / 255.f, color.b / 255.f, color.a / 255.f));
            if (ImGui::Button((string("##color") + to_string(i)).c_str(), ImVec2(20, 20))) {
                characterData->colorPickerActive = !characterData->colorPickerActive;
            }
            ImGui::PopStyleColor(3);
            // Add border
            ImVec2 min = ImGui::GetItemRectMin();
            ImVec2 max = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddRect(min, max, IM_COL32(128, 128, 128, 255));
        }

        if (characterData->count > 0) {
            bool scaleM = characterData->scales[characterData->active] == 1.0f;
            ImGui::Checkbox("m", &scaleM);
            if (scaleM) characterData->scales[characterData->active] = 1.0f;
            ImGui::SameLine();

            bool scaleCM = characterData->scales[characterData->active] == 0.01f;
            ImGui::Checkbox("cm", &scaleCM);
            if (scaleCM) characterData->scales[characterData->active] = 0.01f;
            ImGui::SameLine();

            bool scaleInches = characterData->scales[characterData->active] == 0.0254f;
            ImGui::Checkbox("inch", &scaleInches);
            if (scaleInches) characterData->scales[characterData->active] = 0.0254f;
            ImGui::SameLine();

            bool scaleFeet = characterData->scales[characterData->active] == 0.3048f;
            ImGui::Checkbox("feet", &scaleFeet);
            if (scaleFeet) characterData->scales[characterData->active] = 0.3048f;
            ImGui::SameLine();

            bool scaleAuto = characterData->scales[characterData->active] == characterData->autoScales[characterData->active];
            ImGui::Checkbox("auto", &scaleAuto);
            if (scaleAuto) characterData->scales[characterData->active] = characterData->autoScales[characterData->active];

            // Enforce mutual exclusivity manually
            if (scaleM) { scaleCM = scaleInches = scaleFeet = scaleAuto = false; }
            if (scaleCM) { scaleM = scaleInches = scaleFeet = scaleAuto = false; }
            if (scaleInches) { scaleM = scaleCM = scaleFeet = scaleAuto = false; }
            if (scaleFeet) { scaleM = scaleCM = scaleInches = scaleAuto = false; }
            if (scaleAuto) { scaleM = scaleCM = scaleInches = scaleFeet = false; }

            ImGui::SliderFloat("Radius", &characterData->radii[characterData->active], 0.01f, 0.1f, "%.2f");
            ImGui::SliderFloat("Opacity", &characterData->opacities[characterData->active], 0.0f, 1.0f, "%.2f");
        }
    }
    ImGui::End();
}

static inline void ImGuiScrubberSettings(
    ScrubberSettings* settings,
    CharacterData* characterData,
    int screenWidth,
    int screenHeight)
{
    if (characterData->count == 0) { return; }
    const float sw = (float)screenWidth;
    const float sh = (float)screenHeight;
    const float frameTime = characterData->bvhData[characterData->active].frameTime;

    const float padding = 20.0f;
    ImGui::SetNextWindowPos(ImVec2(padding, sh - 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(sw - padding * 2, 90), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Scrubber")) {
        ImGui::Text("Frame Time: %f", frameTime);
        ImGui::SameLine();
        ImGui::Checkbox("Snap to Frame", &settings->frameSnap);
        ImGui::SameLine();

        const char* sampleModes[] = { "Nearest", "Linear", "Cubic" };
        ImGui::Combo("##samplemode", &settings->sampleMode, sampleModes, IM_ARRAYSIZE(sampleModes));
        ImGui::SameLine();

        ImGui::Checkbox("Inplace", &settings->inplace);
        ImGui::SameLine();
        ImGui::Checkbox("Loop", &settings->looping);
        ImGui::SameLine();
        ImGui::Checkbox("Play", &settings->playing);
        if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Numpad+: unpause\nNumpad*: pause (hold to advance slow)\nNumpad-/+: adjust speed"); }

        int frame = ClampInt((int)(settings->playTime / frameTime + 0.5f), settings->frameMin, settings->frameMax);

        ImGui::InputInt("Min", &settings->frameMinSelect);
        settings->frameMinSelect = ClampInt(settings->frameMinSelect, 0, settings->frameLimit);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            settings->frameMin = settings->frameMinSelect;
            ScrubberSettingsClamp(settings, characterData);
        }
        ImGui::SameLine();

        ImGui::InputInt("Max", &settings->frameMaxSelect);
        settings->frameMaxSelect = ClampInt(settings->frameMaxSelect, 0, settings->frameLimit);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            settings->frameMax = settings->frameMaxSelect;
            ScrubberSettingsClamp(settings, characterData);
        }
        ImGui::SameLine();
        ImGui::Text("of %i", settings->frameLimit);

        float frameFloatPrev = settings->frameSnap ? (float)frame : settings->playTime / frameTime;
        float frameFloat = frameFloatPrev;
        ImGui::Text("%5.2f", settings->playTime);
        ImGui::SameLine();
        ImGui::SliderFloat("##framefloat", &frameFloat, (float)settings->frameMin, (float)settings->frameMax, "");
        ImGui::SameLine();
        ImGui::Text("%i", frame);

        if (frameFloat != frameFloatPrev) {
            if (settings->frameSnap) {
                frame = ClampInt((int)(frameFloat + 0.5f), settings->frameMin, settings->frameMax);
                settings->playTime = Clamp(frame * frameTime, settings->timeMin, settings->timeMax);
            }
            else {
                settings->playTime = Clamp(frameFloat * frameTime, settings->timeMin, settings->timeMax);
            }
        }
    }
    ImGui::End();
}

static inline void ImGuiAnimSettings(AppConfig* config)
{
    ImGui::SetNextWindowPos(ImVec2(250, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(200, 180), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Anim Settings")) {
        ImGui::SliderFloat("Blend Time", &config->defaultBlendTime, 0.0f, 2.0f, "%.2f s");
        ImGui::SliderFloat("Switch Interval", &config->switchInterval, 0.1f, 5.0f, "%.2f s");
        ImGui::Separator();
        ImGui::Checkbox("Vel Blending", &config->useVelBlending);
        if (config->useVelBlending) {
            ImGui::SliderFloat("Return Blend Time", &config->blendPosReturnTime, 0.01f, 1.0f, "%.2f s");
        }
        ImGui::SliderFloat("Hips Blend Time", &config->hipsRotationBlendTime, 0.01f, 1.0f, "%.2f s");
    }
    ImGui::End();
}

//----------------------------------------------------------------------------------
// Application
//----------------------------------------------------------------------------------

// Main application state - passed to update/render functions
struct ApplicationState {
    int argc;
    char** argv;

    // Window
    int screenWidth;
    int screenHeight;

    // Camera
    CameraSystem camera;

    // Rendering resources
    Shader shader;
    ShaderUniforms uniforms;
    Mesh groundPlaneMesh;
    Model groundPlaneModel;
    Model capsuleModel;

    // Animation data
    CharacterData characterData;
    CapsuleData capsuleData;
    AnimDatabase animDatabase;
    ControlledCharacter controlledCharacter;

    // UI state
    ScrubberSettings scrubberSettings;
    AppConfig config;
    //GuiWindowFileDialogState fileDialogState;

    char errMsg[512];

    // Geno character rendering (experimental skinned mesh)
    bool genoRenderMode;
    Model genoModel;
    ModelAnimation genoAnimation;
    Shader genoBasicShader;
    bool genoModelLoaded;
    vector<BVHGenoMapping> genoMappings;

    // Debug timescale system
    // numpad-: halve debugTimescale
    // numpad+: double debugTimescale (max 1.0), also unpause
    // numpad*: toggle pause, hold while paused to advance at half speed
    float debugTimescale = 1.0f;
    bool debugPaused = false;
};

// Update function - what is called to "tick" the application.
static void ApplicationUpdate(void* voidApplicationState)
{
    ApplicationState* app = (ApplicationState*)voidApplicationState;


    // Update window dimensions if resized
    if (IsWindowResized()) {
        app->screenWidth = GetScreenWidth();
        app->screenHeight = GetScreenHeight();
    }

    // Process Dragged and Dropped Files

    if (IsFileDropped())
    {
        FilePathList droppedFiles = LoadDroppedFiles();

        int prevBvhCount = app->characterData.count;

        for (int i = 0; i < droppedFiles.count; i++)
        {
            if (CharacterDataLoadFromFile(&app->characterData, droppedFiles.paths[i], app->errMsg, 512))
            {
                app->characterData.active = app->characterData.count - 1;
            }
        }

        UnloadDroppedFiles(droppedFiles);

        if (app->characterData.count > prevBvhCount)
        {
            // Ensure active character is valid
            if (app->characterData.active < 0 || app->characterData.active >= app->characterData.count)
            {
                app->characterData.active = app->characterData.count - 1;
            }

            // Reset scrubber to known state before updating
            ScrubberSettingsInit(&app->scrubberSettings, app->argc, app->argv);

            ScrubberSettingsRecomputeLimits(&app->scrubberSettings, &app->characterData);
            ScrubberSettingsInitMaxs(&app->scrubberSettings, &app->characterData);

            // Rebuild animation database
            AnimDatabaseRebuild(&app->animDatabase, &app->characterData);
            if (!app->animDatabase.valid) {
                TraceLog(LOG_WARNING, "AnimDatabase invalid after rebuild - disabling controlled character.");
                app->controlledCharacter.active = false;
            }
            else {
                // initialize controlled character
                if (!app->controlledCharacter.active) {
                    ControlledCharacterInit(
                        &app->controlledCharacter,
                        &app->characterData.bvhData[0],
                        app->characterData.scales[0],
                        app->config.switchInterval);
                }
            }

            // Resize capsule buffer for all characters + controlled character
            CapsuleDataUpdateForCharacters(&app->capsuleData, &app->characterData);
            if (app->controlledCharacter.active)
            {
                const int totalJoints = (int)app->capsuleData.capsulePositions.size() +
                    app->controlledCharacter.xformData.jointCount;
                CapsuleDataResize(&app->capsuleData, totalJoints);
            }

            string windowTitle = app->characterData.filePaths[app->characterData.active] + " - BVHView";
            SetWindowTitle(windowTitle.c_str());
        }
    }

    // Handle clear request (with full state access)
    if (app->characterData.clearRequested)
    {
        app->characterData.clearRequested = false;

        // Free and reset character data
        CharacterDataFree(&app->characterData);
        CharacterDataInit(&app->characterData, app->argc, app->argv);

        // Reset AnimDatabase
        AnimDatabaseFree(&app->animDatabase);

        // Disable and free controlled character
        if (app->controlledCharacter.active)
        {
            ControlledCharacterFree(&app->controlledCharacter);
            app->controlledCharacter.active = false;
        }

        // Reset scrubber
        ScrubberSettingsInit(&app->scrubberSettings, app->argc, app->argv);

        // Reset capsule data
        CapsuleDataReset(&app->capsuleData);

        SetWindowTitle("Flomo");
        TraceLog(LOG_INFO, "Cleared all animations");
    }

    // Process Key Presses

    if (IsKeyPressed(KEY_H))// && !app->fileDialogState.windowActive)
    {
        app->config.drawUI = !app->config.drawUI;
    }

    PROFILE_BEGIN(Update);

    // Compute effective dt based on debug timescale
    const float rawDt = GetFrameTime();
    float effectiveDt = 0.0f;
    {
        // Check numpad input for debug timescale (need to check here before ImGui captures input)
        const bool numpadMinusPressed = IsKeyPressed(KEY_KP_SUBTRACT);
        const bool numpadPlusPressed = IsKeyPressed(KEY_KP_ADD);
        const bool numpadMultiplyPressed = IsKeyPressed(KEY_KP_MULTIPLY);
        const bool numpadMultiplyHeld = IsKeyDown(KEY_KP_MULTIPLY);
        const bool shiftHeld = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);

        if (numpadMinusPressed)
        {
            app->debugTimescale *= 0.5f;
            TraceLog(LOG_INFO, "Debug timescale: %.4f", app->debugTimescale);
        }
        if (numpadPlusPressed)
        {
            if (app->debugPaused)
            {
                // Just unpause, don't change timescale
                app->debugPaused = false;
                TraceLog(LOG_INFO, "Unpaused at timescale: %.4f", app->debugTimescale);
            }
            else if (shiftHeld)
            {
                // Shift+numpad+: double timescale without clamping (allows >1x speed)
                app->debugTimescale *= 2.0f;
                TraceLog(LOG_INFO, "Debug timescale: %.4f (fast forward)", app->debugTimescale);
            }
            else
            {
                // Double timescale up to max 1.0
                app->debugTimescale = Clamp(app->debugTimescale * 2.0f, 0.0f, 1.0f);
                TraceLog(LOG_INFO, "Debug timescale: %.4f", app->debugTimescale);
            }
        }
        if (numpadMultiplyPressed)
        {
            app->debugPaused = true;
            TraceLog(LOG_INFO, "Paused (hold * to advance at half speed)");
        }

        // Compute effective dt
        if (app->debugPaused)
        {
            if (numpadMultiplyHeld)
            {
                // Holding * while paused: advance at half the debug timescale
                effectiveDt = rawDt * app->debugTimescale * 0.5f;
            }
            else
            {
                effectiveDt = 0.0f;
            }
        }
        else
        {
            effectiveDt = rawDt * app->debugTimescale;
        }
    }

    // Tick time forward

    if (app->scrubberSettings.playing)
    {
        app->scrubberSettings.playTime += effectiveDt;

        if (app->scrubberSettings.playTime >= app->scrubberSettings.timeMax)
        {
            app->scrubberSettings.playTime = (app->scrubberSettings.looping && app->scrubberSettings.timeMax >= 1e-8f) ?
                fmodf(app->scrubberSettings.playTime, app->scrubberSettings.timeMax) + app->scrubberSettings.timeMin :
                app->scrubberSettings.timeMax;
        }
    }

    // Sample Animation Data

    for (int i = 0; i < app->characterData.count; i++)
    {
        if (app->scrubberSettings.sampleMode == 0)
        {
            TransformDataSampleFrameNearest(
                &app->characterData.xformData[i],
                &app->characterData.bvhData[i],
                app->scrubberSettings.playTime,
                app->characterData.scales[i]);
        }
        else if (app->scrubberSettings.sampleMode == 1)
        {
            TransformDataSampleFrameLinear(
                &app->characterData.xformData[i],
                &app->characterData.xformTmp0[i],
                &app->characterData.xformTmp1[i],
                &app->characterData.bvhData[i],
                app->scrubberSettings.playTime,
                app->characterData.scales[i]);
        }
        else
        {
            TransformDataSampleFrameCubic(
                &app->characterData.xformData[i],
                &app->characterData.xformTmp0[i],
                &app->characterData.xformTmp1[i],
                &app->characterData.xformTmp2[i],
                &app->characterData.xformTmp3[i],
                &app->characterData.bvhData[i],
                app->scrubberSettings.playTime,
                app->characterData.scales[i]);
        }

        if (app->scrubberSettings.inplace)
        {
            // Remove Translation on ground Plane

            app->characterData.xformData[i].localPositions[0].x = 0.0f;
            app->characterData.xformData[i].localPositions[0].z = 0.0f;

            // Attempt to extract rotation around vertical axis (this does not work 
            // for all animations but is pretty effective for almost all of them)

            Quaternion verticalRotation = QuaternionInvert(QuaternionNormalize(Quaternion{
                0.0f,
                app->characterData.xformData[i].localRotations[0].y,
                0.0f,
                app->characterData.xformData[i].localRotations[0].w,
                }));

            // Remove rotation around vertical axis

            app->characterData.xformData[i].localRotations[0] = QuaternionMultiply(
                verticalRotation,
                app->characterData.xformData[i].localRotations[0]);
        }

        TransformDataForwardKinematics(&app->characterData.xformData[i]);
    }

    // Update Controlled Character (root motion playback)

    const float dt = GetFrameTime();

    if (app->controlledCharacter.active && effectiveDt > 0.0f)
    {
        // sync velocity blending settings from config
        app->controlledCharacter.useVelBlending = app->config.useVelBlending;
        app->controlledCharacter.blendPosReturnTime = app->config.blendPosReturnTime;
        app->controlledCharacter.hipsRotationBlendTime = app->config.hipsRotationBlendTime;

        ControlledCharacterUpdate(
            &app->controlledCharacter,
            &app->characterData,
            &app->animDatabase,
            effectiveDt,
            app->scrubberSettings.sampleMode,
            app->config.defaultBlendTime,
            app->config.switchInterval);
    }

    // Update Camera
    const Vector2 mouseDelta = GetMouseDelta();
    const float mouseWheel = GetMouseWheelMove();
    const bool imguiWantsMouse = ImGui::GetIO().WantCaptureMouse;
    const bool imguiWantsKeyboard = ImGui::GetIO().WantCaptureKeyboard;

    // Get bone target for orbit camera (used for mode switching too)
    Vector3 boneTarget = Vector3{ 0.0f, 1.0f, 0.0f };
    if (app->camera.orbit.track)
    {
        if (app->camera.orbit.trackControlledCharacter && app->controlledCharacter.active)
        {
            // Track controlled character's root (joint 0) or hips
            const int trackBone = MinInt(app->camera.orbit.trackBone,
                app->controlledCharacter.xformData.jointCount - 1);
            boneTarget = app->controlledCharacter.xformData.globalPositions[trackBone];
        }
        else if (app->characterData.count > 0 &&
            app->camera.orbit.trackBone < app->characterData.xformData[app->characterData.active].jointCount)
        {
            boneTarget = app->characterData.xformData[app->characterData.active].globalPositions[app->camera.orbit.trackBone];
        }
    }

    // 'F' key toggles camera mode
    if (!imguiWantsKeyboard && IsKeyPressed(KEY_F))
    {
        if (app->camera.mode == FlomoCameraMode::Orbit)
        {
            CameraSwitchToUnreal(&app->camera);
        }
        else
        {
            CameraSwitchToOrbit(&app->camera, boneTarget);
        }
    }

    // Debug timescale controls (numpad)
    // numpad-: halve debugTimescale
    // numpad+: double debugTimescale (max 1.0), also unpause if paused
    // numpad*: pause, hold while paused to advance at half speed
    if (!imguiWantsKeyboard)
    {
        if (IsKeyPressed(KEY_KP_SUBTRACT))
        {
            app->debugTimescale *= 0.5f;
            TraceLog(LOG_INFO, "Debug timescale: %.4f", app->debugTimescale);
        }
        if (IsKeyPressed(KEY_KP_ADD))
        {
            if (app->debugPaused)
            {
                // Just unpause, don't change timescale
                app->debugPaused = false;
                TraceLog(LOG_INFO, "Unpaused at timescale: %.4f", app->debugTimescale);
            }
            else
            {
                // Double timescale up to max 1.0
                app->debugTimescale = Clamp(app->debugTimescale * 2.0f, 0.0f, 1.0f);
                TraceLog(LOG_INFO, "Debug timescale: %.4f", app->debugTimescale);
            }
        }
        if (IsKeyPressed(KEY_KP_MULTIPLY))
        {
            // Toggle pause
            app->debugPaused = true;
            TraceLog(LOG_INFO, "Paused (hold * to advance at half speed)");
        }
    }

    if (app->camera.mode == FlomoCameraMode::Orbit)
    {
        // Orbit camera: RMB rotates, MMB pans, scroll zooms
        // Always update to follow target, but zero input when ImGui has mouse
        const bool acceptInput = !imguiWantsMouse;
        const float scrollInput = (acceptInput && !IsMouseButtonDown(2)) ? mouseWheel : 0.0f;

        OrbitCameraUpdate(
            &app->camera,
            boneTarget,
            (acceptInput && IsMouseButtonDown(1)) ? mouseDelta.x : 0.0f,  // RMB rotates
            (acceptInput && IsMouseButtonDown(1)) ? mouseDelta.y : 0.0f,
            (acceptInput && IsMouseButtonDown(2)) ? mouseDelta.x : 0.0f,  // MMB pans
            (acceptInput && IsMouseButtonDown(2)) ? mouseDelta.y : 0.0f,
            scrollInput,  // Scroll zooms
            dt);
    }
    else if (app->camera.mode == FlomoCameraMode::Unreal)
    {
        // Unreal camera: RMB + WASD/QE moves, scroll adjusts speed
        const bool isActive = IsMouseButtonDown(1) && !imguiWantsMouse;
        // Ignore scroll wheel when MMB is held
        const float scrollInput = (imguiWantsMouse || IsMouseButtonDown(2)) ? 0.0f : mouseWheel;

        const bool isPanning = IsMouseButtonDown(2) && !imguiWantsMouse;

        UnrealCameraUpdate(
            &app->camera,
            mouseDelta.x,
            mouseDelta.y,
            scrollInput,
            !imguiWantsKeyboard && IsKeyDown(KEY_W),
            !imguiWantsKeyboard && IsKeyDown(KEY_S),
            !imguiWantsKeyboard && IsKeyDown(KEY_A),
            !imguiWantsKeyboard && IsKeyDown(KEY_D),
            !imguiWantsKeyboard && IsKeyDown(KEY_E),
            !imguiWantsKeyboard && IsKeyDown(KEY_Q),
            isActive,
            isPanning,
            dt);
    }

    // Create Capsules

    CapsuleDataReset(&app->capsuleData);
    for (int i = 0; i < app->characterData.count; i++)
    {
        CapsuleDataAppendFromTransformData(
            &app->capsuleData,
            &app->characterData.xformData[i],
            app->characterData.radii[i],
            app->characterData.colors[i],
            app->characterData.opacities[i],
            !app->config.drawEndSites);
    }

    // Add controlled character's capsules
    if (app->controlledCharacter.active)
    {
        CapsuleDataAppendFromTransformData(
            &app->capsuleData,
            &app->controlledCharacter.xformData,
            app->controlledCharacter.radius,
            app->controlledCharacter.color,
            app->controlledCharacter.opacity,
            !app->config.drawEndSites);
    }

    PROFILE_END(Update);

    // Rendering

    Frustum frustum;
    FrustumFromCameraMatrices(
        //GetCameraProjectionMatrix(&app->camera.cam3d, (float)app->screenHeight / (float)app->screenWidth),
        GetCameraProjectionMatrix(&app->camera.cam3d, (float)app->screenWidth / (float)app->screenHeight),
        GetCameraViewMatrix(&app->camera.cam3d),
        frustum);

    BeginDrawing();

    PROFILE_BEGIN(Rendering);

    ClearBackground(app->config.backgroundColor);

    BeginMode3D(app->camera.cam3d);

    //DrawSphere(Vector3{ 0, 2, 0 }, 0.5f, RED);  // Simple red sphere
    //DrawCube(Vector3{ 2, 1, 0 }, 1, 1, 1, BLUE); // Simple blue cube


    // Set shader uniforms that don't change based on the object being drawn

    const Vector3 sunColorValue = { app->config.sunColor.r / 255.0f, app->config.sunColor.g / 255.0f, app->config.sunColor.b / 255.0f };
    const Vector3 skyColorValue = { app->config.skyColor.r / 255.0f, app->config.skyColor.g / 255.0f, app->config.skyColor.b / 255.0f };
    const float objectSpecularity = 0.5f;
    const float objectGlossiness = 10.0f;
    const float objectOpacity = 1.0f;

    const Vector3 sunLightPosition = Vector3RotateByQuaternion(Vector3{ 0.0f, 0.0f, 1.0f }, QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, app->config.sunAzimuth));
    const Vector3 sunLightAxis = Vector3Normalize(Vector3CrossProduct(sunLightPosition, Vector3{ 0.0f, 1.0f, 0.0f }));
    const Vector3 sunLightDir = Vector3Negate(Vector3RotateByQuaternion(sunLightPosition, QuaternionFromAxisAngle(sunLightAxis, app->config.sunAltitude)));

    SetShaderValue(app->shader, app->uniforms.cameraPosition, &app->camera.cam3d.position, SHADER_UNIFORM_VEC3);
    SetShaderValue(app->shader, app->uniforms.exposure, &app->config.exposure, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.sunDir, &sunLightDir, SHADER_UNIFORM_VEC3);
    SetShaderValue(app->shader, app->uniforms.sunStrength, &app->config.sunLightStrength, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.sunColor, &sunColorValue, SHADER_UNIFORM_VEC3);
    SetShaderValue(app->shader, app->uniforms.skyStrength, &app->config.skyLightStrength, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.skyColor, &skyColorValue, SHADER_UNIFORM_VEC3);
    SetShaderValue(app->shader, app->uniforms.ambientStrength, &app->config.ambientLightStrength, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.groundStrength, &app->config.groundLightStrength, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.objectSpecularity, &objectSpecularity, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.objectGlossiness, &objectGlossiness, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.objectOpacity, &objectOpacity, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.aoLookupResolution, &app->capsuleData.aoLookupResolution, SHADER_UNIFORM_VEC2);
    SetShaderValue(app->shader, app->uniforms.shadowLookupResolution, &app->capsuleData.shadowLookupResolution, SHADER_UNIFORM_VEC2);
    SetShaderValueTexture(app->shader, app->uniforms.aoLookupTable, app->capsuleData.aoLookupTable);
    SetShaderValueTexture(app->shader, app->uniforms.shadowLookupTable, app->capsuleData.shadowLookupTable);

    // Draw Ground

    PROFILE_BEGIN(RenderingGround);

    if (app->config.drawChecker)
    {
        const int groundIsCapsule = 0;
        const Vector3 groundColor = { 0.75f, 0.75f, 0.75f };

        SetShaderValue(app->shader, app->uniforms.isCapsule, &groundIsCapsule, SHADER_UNIFORM_INT);
        SetShaderValue(app->shader, app->uniforms.objectColor, &groundColor, SHADER_UNIFORM_VEC3);

        // Ground tile parameters
        const float tileSize = 2.0f;       // Size of each tile (matches groundPlaneMesh)
        const int tilesPerSide = 25;       // How many tiles to draw in each direction from center
        const float segmentRadius = tileSize * 0.707f;  // For frustum culling

        // Center the grid around the camera target (snapped to tile boundaries)
        const Vector3 camTarget = app->camera.cam3d.target;
        const float centerX = floorf(camTarget.x / tileSize) * tileSize;
        const float centerZ = floorf(camTarget.z / tileSize) * tileSize;

        for (int i = -tilesPerSide; i <= tilesPerSide; i++)
        {
            for (int j = -tilesPerSide; j <= tilesPerSide; j++)
            {
                // Tile position at fixed world coordinates
                const Vector3 groundSegmentPosition =
                {
                    centerX + i * tileSize,
                    0.0f,
                    centerZ + j * tileSize,
                };

                // Frustum culling
                if (!FrustumContainsSphere(frustum, groundSegmentPosition, segmentRadius * 1.1f))
                {
                    continue;
                }

                PROFILE_BEGIN(RenderingGroundSegment);

                // Gather all capsules casting AO on this ground segment

                PROFILE_BEGIN(RenderingGroundSegmentAO);

                app->capsuleData.aoCapsuleCount = 0;
                if (app->config.drawCapsules && app->config.drawAO)
                {
                    CapsuleDataUpdateAOCapsulesForGroundSegment(&app->capsuleData, groundSegmentPosition);
                }
                int aoCapsuleCount = MinInt(app->capsuleData.aoCapsuleCount, AO_CAPSULES_MAX);

                PROFILE_END(RenderingGroundSegmentAO);

                SetShaderValue(app->shader, app->uniforms.aoCapsuleCount, &aoCapsuleCount, SHADER_UNIFORM_INT);
                SetShaderValueV(app->shader, app->uniforms.aoCapsuleStarts, app->capsuleData.aoCapsuleStarts.data(), SHADER_UNIFORM_VEC3, aoCapsuleCount);
                SetShaderValueV(app->shader, app->uniforms.aoCapsuleVectors, app->capsuleData.aoCapsuleVectors.data(), SHADER_UNIFORM_VEC3, aoCapsuleCount);
                SetShaderValueV(app->shader, app->uniforms.aoCapsuleRadii, app->capsuleData.aoCapsuleRadii.data(), SHADER_UNIFORM_FLOAT, aoCapsuleCount);

                // Gather all capsules casting shadows on this ground segment

                PROFILE_BEGIN(RenderingGroundSegmentShadow);

                app->capsuleData.shadowCapsuleCount = 0;
                if (app->config.drawCapsules && app->config.drawShadows)
                {
                    CapsuleDataUpdateShadowCapsulesForGroundSegment(&app->capsuleData, groundSegmentPosition, sunLightDir, app->config.sunLightConeAngle);
                }
                const int shadowCapsuleCount = MinInt(app->capsuleData.shadowCapsuleCount, SHADOW_CAPSULES_MAX);

                PROFILE_END(RenderingGroundSegmentShadow);

                SetShaderValue(app->shader, app->uniforms.shadowCapsuleCount, &shadowCapsuleCount, SHADER_UNIFORM_INT);
                SetShaderValueV(app->shader, app->uniforms.shadowCapsuleStarts, app->capsuleData.shadowCapsuleStarts.data(), SHADER_UNIFORM_VEC3, shadowCapsuleCount);
                SetShaderValueV(app->shader, app->uniforms.shadowCapsuleVectors, app->capsuleData.shadowCapsuleVectors.data(), SHADER_UNIFORM_VEC3, shadowCapsuleCount);
                SetShaderValueV(app->shader, app->uniforms.shadowCapsuleRadii, app->capsuleData.shadowCapsuleRadii.data(), SHADER_UNIFORM_FLOAT, shadowCapsuleCount);

                // Draw

                DrawModel(app->groundPlaneModel, groundSegmentPosition, 1.0f, WHITE);

                PROFILE_END(RenderingGroundSegment);
            }
        }
    }

    PROFILE_END(RenderingGround);

    // Draw Capsules

    PROFILE_BEGIN(RenderingCapsules);

    if (app->config.drawCapsules && !app->genoRenderMode)
    {
        // Depth sort back to front for transparency

        for (int i = 0; i < app->capsuleData.capsuleCount; i++)
        {
            app->capsuleData.capsuleSort[i].index = i;
            app->capsuleData.capsuleSort[i].value = Vector3Distance(app->camera.cam3d.position, app->capsuleData.capsulePositions[i]);
        }

        //qsort(app->capsuleData.capsuleSort, app->capsuleData.capsuleCount, sizeof(CapsuleSort), CapsuleSortCompareLess);
        sort(app->capsuleData.capsuleSort.begin(), app->capsuleData.capsuleSort.begin() + app->capsuleData.capsuleCount,
            [](const CapsuleSort& a, const CapsuleSort& b) { return a.value < b.value; });

        // Render

        const int capsuleIsCapsule = 1;
        SetShaderValue(app->shader, app->uniforms.isCapsule, &capsuleIsCapsule, SHADER_UNIFORM_INT);

        for (int i = 0; i < app->capsuleData.capsuleCount; i++)
        {
            const int j = app->capsuleData.capsuleSort[i].index;

            // Check if we can cull capsule

            const Vector3 capsulePosition = app->capsuleData.capsulePositions[j];
            const float capsuleHalfLength = app->capsuleData.capsuleHalfLengths[j];
            const float capsuleRadius = app->capsuleData.capsuleRadii[j];

            if (!FrustumContainsSphere(frustum, capsulePosition, capsuleHalfLength + capsuleRadius))
            {
                continue;
            }

            PROFILE_BEGIN(RenderingCapsulesCapsule);

            // If capsule is semi-transparent disable depth mask

            if (app->capsuleData.capsuleOpacities[j] < 1.0f)
            {
                rlDrawRenderBatchActive();
                rlDisableDepthMask();
            }

            // Set shader properties

            const Quaternion capsuleRotation = app->capsuleData.capsuleRotations[j];
            const Vector3 capsuleStart = CapsuleStart(capsulePosition, capsuleRotation, capsuleHalfLength);
            const Vector3 capsuleVector = CapsuleVector(capsulePosition, capsuleRotation, capsuleHalfLength);

            SetShaderValue(app->shader, app->uniforms.objectColor, &app->capsuleData.capsuleColors[j], SHADER_UNIFORM_VEC3);
            SetShaderValue(app->shader, app->uniforms.objectOpacity, &app->capsuleData.capsuleOpacities[j], SHADER_UNIFORM_FLOAT);
            SetShaderValue(app->shader, app->uniforms.capsulePosition, &app->capsuleData.capsulePositions[j], SHADER_UNIFORM_VEC3);
            SetShaderValue(app->shader, app->uniforms.capsuleRotation, &app->capsuleData.capsuleRotations[j], SHADER_UNIFORM_VEC4);
            SetShaderValue(app->shader, app->uniforms.capsuleHalfLength, &app->capsuleData.capsuleHalfLengths[j], SHADER_UNIFORM_FLOAT);
            SetShaderValue(app->shader, app->uniforms.capsuleRadius, &app->capsuleData.capsuleRadii[j], SHADER_UNIFORM_FLOAT);
            SetShaderValue(app->shader, app->uniforms.capsuleStart, &capsuleStart, SHADER_UNIFORM_VEC3);
            SetShaderValue(app->shader, app->uniforms.capsuleVector, &capsuleVector, SHADER_UNIFORM_VEC3);

            // Find all capsules casting AO on this capsule

            PROFILE_BEGIN(RenderingCapsulesCapsuleAO);

            app->capsuleData.aoCapsuleCount = 0;
            if (app->config.drawAO)
            {
                CapsuleDataUpdateAOCapsulesForCapsule(&app->capsuleData, j);
            }
            const int aoCapsuleCount = MinInt(app->capsuleData.aoCapsuleCount, AO_CAPSULES_MAX);

            PROFILE_END(RenderingCapsulesCapsuleAO);

            SetShaderValue(app->shader, app->uniforms.aoCapsuleCount, &aoCapsuleCount, SHADER_UNIFORM_INT);
            SetShaderValueV(app->shader, app->uniforms.aoCapsuleStarts, app->capsuleData.aoCapsuleStarts.data(), SHADER_UNIFORM_VEC3, aoCapsuleCount);
            SetShaderValueV(app->shader, app->uniforms.aoCapsuleVectors, app->capsuleData.aoCapsuleVectors.data(), SHADER_UNIFORM_VEC3, aoCapsuleCount);
            SetShaderValueV(app->shader, app->uniforms.aoCapsuleRadii, app->capsuleData.aoCapsuleRadii.data(), SHADER_UNIFORM_FLOAT, aoCapsuleCount);

            // Find all capsules casting shadows on this capsule

            PROFILE_BEGIN(RenderingCapsulesCapsuleShadow);

            app->capsuleData.shadowCapsuleCount = 0;
            if (app->config.drawShadows)
            {
                CapsuleDataUpdateShadowCapsulesForCapsule(&app->capsuleData, j, sunLightDir, app->config.sunLightConeAngle);
            }
            const int shadowCapsuleCount = MinInt(app->capsuleData.shadowCapsuleCount, SHADOW_CAPSULES_MAX);

            PROFILE_END(RenderingCapsulesCapsuleShadow);

            SetShaderValue(app->shader, app->uniforms.shadowCapsuleCount, &shadowCapsuleCount, SHADER_UNIFORM_INT);
            SetShaderValueV(app->shader, app->uniforms.shadowCapsuleStarts, app->capsuleData.shadowCapsuleStarts.data(), SHADER_UNIFORM_VEC3, shadowCapsuleCount);
            SetShaderValueV(app->shader, app->uniforms.shadowCapsuleVectors, app->capsuleData.shadowCapsuleVectors.data(), SHADER_UNIFORM_VEC3, shadowCapsuleCount);
            SetShaderValueV(app->shader, app->uniforms.shadowCapsuleRadii, app->capsuleData.shadowCapsuleRadii.data(), SHADER_UNIFORM_FLOAT, shadowCapsuleCount);

            // Draw

            DrawModel(app->capsuleModel, Vector3Zero(), 1.0f, WHITE);

            // Reset depth mask if rendered semi-transparent

            if (app->capsuleData.capsuleOpacities[j] < 1.0f)
            {
                rlDrawRenderBatchActive();
                rlEnableDepthMask();
            }

            PROFILE_END(RenderingCapsulesCapsule);
        }
    }

    PROFILE_END(RenderingCapsules);

    // Geno Character Rendering

    if (app->genoRenderMode && app->genoModelLoaded && app->characterData.count > 0)
    {
        // Ensure we have mappings for all characters
        while ((int)app->genoMappings.size() < app->characterData.count)
        {
            const int idx = (int)app->genoMappings.size();
            BVHGenoMapping mapping = CreateBVHGenoMapping(
                &app->characterData.bvhData[idx],
                &app->genoModel);
            app->genoMappings.push_back(mapping);
        }

        // Draw all characters
        for (int c = 0; c < app->characterData.count; c++)
        {
            // Update Geno animation from current BVH pose
            UpdateGenoAnimationFromBVH(
                &app->genoAnimation,
                &app->characterData.xformData[c],
                &app->genoMappings[c],
                1.0f);  // Scale already applied by TransformDataSampleFrame

            // Update animation bones (raylib handles GPU skinning internally)
            UpdateModelAnimationBones(app->genoModel, app->genoAnimation, 0);

            // Draw the Geno model with character's color
            const Color charColor = app->characterData.colors[c];
            DrawModel(app->genoModel, Vector3Zero(), 1.0f, charColor);
        }
    }

    // Grid

    if (app->config.drawGrid)
    {
        DrawGrid(20, 1.0f);
    }

    // Origin

    if (app->config.drawOrigin)
    {
        DrawTransform(
            Vector3{ 0.0f, 0.01f, 0.0f },
            QuaternionIdentity(),
            1.0f);
    }

    // Disable Depth Test

    rlDrawRenderBatchActive();
    rlDisableDepthTest();

    // Draw Capsule Wireframes

    if (app->config.drawWireframes)
    {
        DrawWireFrames(&app->capsuleData, DARKGRAY);
    }

    // Draw Bones

    if (app->config.drawSkeleton)
    {
        for (int i = 0; i < app->characterData.count; i++)
        {
            DrawSkeleton(
                &app->characterData.xformData[i],
                app->config.drawEndSites,
                DARKGRAY,
                GRAY);
        }
    }

    // Draw joint velocities
    if (app->config.drawVelocities && app->animDatabase.valid)
    {
        const float velScale = 0.1f;  // scale factor for velocity visualization

        for (int c = 0; c < app->characterData.count; ++c)
        {
            if (c >= (int)app->animDatabase.clipStartFrame.size()) continue;

            // compute current motion frame index for this character
            const float frameTime = app->animDatabase.animFrameTime[c];
            const int clipStart = app->animDatabase.clipStartFrame[c];
            const int clipEnd = app->animDatabase.clipEndFrame[c];
            const int clipFrameCount = clipEnd - clipStart;
            if (clipFrameCount <= 0 || frameTime <= 0.0f) continue;

            int localFrame = (int)(app->scrubberSettings.playTime / frameTime);
            localFrame = ClampInt(localFrame, 0, clipFrameCount - 1);
            const int motionFrame = clipStart + localFrame;

            const int jointCount = app->animDatabase.jointCount;
            const TransformData& xform = app->characterData.xformData[c];
            span<const Vector3> velRow = app->animDatabase.globalJointVelocities.row_view(motionFrame);

            for (int j = 0; j < jointCount && j < xform.jointCount; ++j)
            {
                if (xform.endSite[j]) continue;  // skip end sites

                const Vector3 pos = xform.globalPositions[j];
                const Vector3 endPos = Vector3Add(pos, Vector3Scale(velRow[j], velScale));

                DrawLine3D(pos, endPos, BLUE);
            }
        }
    }

    // Draw joint accelerations
    if (app->config.drawAccelerations && app->animDatabase.valid)
    {
        const float accScale = 0.01f;  // scale factor for acceleration visualization (smaller since acc is larger)

        for (int c = 0; c < app->characterData.count; ++c)
        {
            if (c >= (int)app->animDatabase.clipStartFrame.size()) continue;

            const float frameTime = app->animDatabase.animFrameTime[c];
            const int clipStart = app->animDatabase.clipStartFrame[c];
            const int clipEnd = app->animDatabase.clipEndFrame[c];
            const int clipFrameCount = clipEnd - clipStart;
            if (clipFrameCount <= 0 || frameTime <= 0.0f) continue;

            int localFrame = (int)(app->scrubberSettings.playTime / frameTime);
            localFrame = ClampInt(localFrame, 0, clipFrameCount - 1);
            const int motionFrame = clipStart + localFrame;

            const int jointCount = app->animDatabase.jointCount;
            const TransformData& xform = app->characterData.xformData[c];
            span<const Vector3> accRow = app->animDatabase.globalJointAccelerations.row_view(motionFrame);

            for (int j = 0; j < jointCount && j < xform.jointCount; ++j)
            {
                if (xform.endSite[j]) continue;  // skip end sites

                const Vector3 pos = xform.globalPositions[j];
                const Vector3 endPos = Vector3Add(pos, Vector3Scale(accRow[j], accScale));

                DrawLine3D(pos, endPos, RED);
            }
        }
    }

    // Shared colors for cursor visualization (used by root velocities and skeleton drawing)
    const Color cursorColors[ControlledCharacter::MAX_BLEND_CURSORS] = {
        RED, GREEN, BLUE, YELLOW, MAGENTA,
        ORANGE, PINK, SKYBLUE, LIME, VIOLET
    };

    // Draw root motion velocities from each cursor
    if (app->controlledCharacter.active && app->config.drawRootVelocities)
    {
        const ControlledCharacter& cc = app->controlledCharacter;
        const float velScale = 0.5f;  // scale for velocity visualization
        const float yOffset = 0.05f;  // slight vertical offset to avoid z-fighting

        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc.cursors[ci];
            if (!cur.active) continue;
            if (cur.normalizedWeight < 0.01f) continue;  // skip very low weight cursors

            // draw from character's world position (all cursors share same world pos)
            const Vector3 startPos = Vector3Add(cc.worldPosition, Vector3{ 0.0f, yOffset * (ci + 1), 0.0f });
            const Vector3 endPos = Vector3Add(startPos, Vector3Scale(cur.rootVelocity, velScale));

            const Color col = cursorColors[ci];
            DrawLine3D(startPos, endPos, col);

            // draw small sphere at end to make it more visible
            DrawSphere(endPos, 0.02f, col);

            // also draw yaw rate as a small arc/line perpendicular to velocity
            // (positive yaw = counter-clockwise when viewed from above)
            //const float yawVis = cur.rootYawRate * velScale * 0.5f;
            //const Vector3 yawDir = Vector3{ -sinf(yawVis), 0.0f, cosf(yawVis) };
            //const Vector3 yawEnd = Vector3Add(startPos, Vector3Scale(yawDir, 0.3f));
            //DrawLine3D(startPos, yawEnd, col);
        }

        // also draw the smoothed velocity in white
        {
            const Vector3 startPos = Vector3Add(cc.worldPosition, Vector3{ 0.0f, yOffset * 0.5f, 0.0f });
            const Vector3 endPos = Vector3Add(startPos, Vector3Scale(cc.smoothedRootVelocity, velScale));
            DrawLine3D(startPos, endPos, WHITE);
            DrawSphere(endPos, 0.025f, WHITE);
        }
    }

    // Draw toe velocities (actual vs blended)
    if (app->controlledCharacter.active && app->config.drawToeVelocities && app->animDatabase.valid)
    {
        const ControlledCharacter& cc = app->controlledCharacter;
        const AnimDatabase& db = app->animDatabase;
        const float velScale = 0.3f;  // scale for velocity visualization

        for (int side : sides)
        {
            const int toeIdx = db.toeIndices[side];
            if (toeIdx < 0 || toeIdx >= cc.xformData.jointCount) continue;

            const Vector3 toePos = cc.xformData.globalPositions[toeIdx];

            // Actual velocity (yellow) - computed from FK result
            const Vector3 actualEnd = Vector3Add(toePos, Vector3Scale(cc.toeActualVelocity[side], velScale));
            DrawLine3D(toePos, actualEnd, YELLOW);
            DrawSphere(actualEnd, 0.015f, YELLOW);

            // Blended velocity (cyan) - weighted average from cursors
            const Vector3 blendedEnd = Vector3Add(toePos, Vector3Scale(cc.toeBlendedVelocity[side], velScale));
            DrawLine3D(toePos, blendedEnd, SKYBLUE);
            DrawSphere(blendedEnd, 0.015f, SKYBLUE);
        }
    }

    // Draw Blend Cursor Skeletons (for debug visualization)
    if (app->controlledCharacter.active && app->config.drawBlendCursors)
    {
        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = app->controlledCharacter.cursors[ci];
            if (!cur.active) continue;

            const Color baseColor = cursorColors[ci];
            // Fade color based on normalized weight
            const unsigned char alpha = (unsigned char)(cur.normalizedWeight * 255.0f);
            const Color drawColor = Color{ baseColor.r, baseColor.g, baseColor.b, alpha };

            const int jc = app->controlledCharacter.xformData.jointCount;
            for (int j = 0; j < jc; ++j)
            {
                // Draw joint sphere
                if (!app->controlledCharacter.xformData.endSite[j])
                {
                    DrawSphereWires(cur.globalPositions[j], 0.015f, 4, 6, drawColor);
                }

                // Draw bone to parent
                const int p = app->controlledCharacter.xformData.parents[j];
                if (p != -1 && !app->controlledCharacter.xformData.endSite[j])
                {
                    DrawLine3D(cur.globalPositions[j], cur.globalPositions[p], drawColor);
                }
            }
        }
    }

    // Draw Joint Transforms

    if (app->config.drawTransforms)
    {
        for (int i = 0; i < app->characterData.count; i++)
        {
            DrawTransforms(&app->characterData.xformData[i]);
        }
    }

    // Re-Enable Depth Test

    rlDrawRenderBatchActive();
    rlEnableDepthTest();



    //if (app->animDatabase.motionFrameCount > 0 && app->animDatabase.jointCount > 0)
    //{
    //    // How far into the future to visualize (seconds)
    //    const float futureOffsetSeconds = 1.0f;

    //    // For every loaded character (these are the entries in CharacterData)
    //    for (int c = 0; c < app->characterData.count; ++c)
    //    {
    //        // Make sure this character has an associated clip in the motion DB
    //        if (c < (int)app->animDatabase.clipStartFrame.size() &&
    //            app->animDatabase.clipStartFrame[c] < app->animDatabase.clipEndFrame[c])
    //        {
    //            //const BVHData& bvh = app->characterData.bvhData[c];

    //            // Compute future local frame index for this clip (nearest sampling)
    //            const float clipFrameTime = app->animDatabase.animFrameTime[c];
    //            const int clipFrameCount = app->animDatabase.animFrameCount[c];
    //            const float futureTime = app->scrubberSettings.playTime + futureOffsetSeconds;

    //            int localFrame = 0;
    //            if (clipFrameTime > 0.0f)
    //            {
    //                localFrame = ClampInt((int)(futureTime / clipFrameTime + 0.5f), 0, clipFrameCount - 1);
    //            }

    //            // Map to motion-DB frame index (compacted DB)
    //            const int motionFrameIndex = app->animDatabase.clipStartFrame[c] + localFrame;
    //            if (motionFrameIndex < 0 || motionFrameIndex >= app->animDatabase.motionFrameCount)
    //            {
    //                continue; // out-of-range (shouldn't usually happen)
    //            }

    //            // Draw skeleton using positions from jointPositions
    //            const int jointCount = app->animDatabase.jointCount;
    //            const Color drawColor = app->characterData.colors[c];

    //            for (int j = 0; j < jointCount; ++j)
    //            {
    //                const size_t idx = (size_t)motionFrameIndex * jointCount + j;
    //                const Vector3 jp = app->animDatabase.globalJointPositions[idx];

    //                // If this joint is an end-site for this character, draw slightly different primitive
    //                bool isEnd = false;
    //                if (j < app->characterData.xformData[c].jointCount)
    //                {
    //                    isEnd = app->characterData.xformData[c].endSite[j];
    //                }

    //                if (!isEnd)
    //                {
    //                    DrawSphereWires(jp, 0.01f, 4, 6, drawColor);
    //                }
    //                else
    //                {
    //                    DrawCubeWiresV(jp, Vector3{ 0.02f, 0.02f, 0.02f }, Color{ 150, 150, 150, 255 });
    //                }

    //                // Draw line to parent when valid
    //                int parent = -1;
    //                if (j < app->characterData.xformData[c].jointCount)
    //                {
    //                    parent = app->characterData.xformData[c].parents[j];
    //                }

    //                if (parent != -1 && parent < jointCount)
    //                {
    //                    const size_t pidx = (size_t)motionFrameIndex * jointCount + parent;
    //                    const Vector3 pjp = app->animDatabase.globalJointPositions[pidx];
    //                    DrawLine3D(jp, pjp, drawColor);
    //                }
    //            }
    //        }
    //    }
    //}



    // Draw small red spheres at the location of the bone named "LeftToeBase"
    // for every loaded character and the controlled character (if active).
    // This draws filled red spheres at the joint's global position.
    //{
    //    // Characters loaded from BVH/FBX files
    //    for (int c = 0; c < app->characterData.count; ++c)
    //    {
    //        const BVHData& bvh = app->characterData.bvhData[c];
    //        int leftToeIdx = -1;
    //        for (int j = 0; j < bvh.jointCount; ++j)
    //        {
    //            if (bvh.joints[j].name != nullptr && strcmp(bvh.joints[j].name, "LeftToeBase") == 0)
    //            {
    //                leftToeIdx = j;
    //                break;
    //            }
    //        }

    //        if (leftToeIdx != -1 && leftToeIdx < app->characterData.xformData[c].jointCount)
    //        {
    //            const Vector3 pos = app->characterData.xformData[c].globalPositions[leftToeIdx];
    //            DrawSphere(pos, 0.02f, RED);
    //        }
    //    }

    //    // Controlled character (if active)
    //    if (app->controlledCharacter.active && app->controlledCharacter.skeleton)
    //    {
    //        const BVHData* cbvh = app->controlledCharacter.skeleton;
    //        int leftToeIdx = -1;
    //        for (int j = 0; j < cbvh->jointCount; ++j)
    //        {
    //            if (cbvh->joints[j].name != nullptr && strcmp(cbvh->joints[j].name, "LeftToeBase") == 0)
    //            {
    //                leftToeIdx = j;
    //                break;
    //            }
    //        }

    //        if (leftToeIdx != -1 && leftToeIdx < app->controlledCharacter.xformData.jointCount)
    //        {
    //            const Vector3 pos = app->controlledCharacter.xformData.globalPositions[leftToeIdx];
    //            DrawSphere(pos, 0.02f, RED);
    //        }
    //    }
    //}

    // Rendering Done

    EndMode3D();

    PROFILE_END(Rendering);

    // Begin ImGui frame
    rlImGuiBegin();

    // Draw UI

    PROFILE_BEGIN(Gui);

    if (app->config.drawUI) {
        // Error Message (only show if there's an error)
        if (app->errMsg[0] != '\0') {
            ImGui::SetNextWindowPos(ImVec2(250, 20), ImGuiCond_FirstUseEver);
            ImGui::Begin("Error", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", app->errMsg);
            ImGui::End();
        }

        if (app->characterData.count == 0) {
            ImGui::SetNextWindowPos(ImVec2((float)app->screenWidth / 2 - 330, (float)app->screenHeight / 2 - 15), ImGuiCond_FirstUseEver);
            ImGui::Begin("Info", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("Drag and Drop .bvh or .fbx files to open them.");
            ImGui::End();
        }

        // Render Settings
        ImGuiRenderSettings(&app->config, &app->capsuleData, app->screenWidth, app->screenHeight,
            &app->genoRenderMode, app->genoModelLoaded);

        // FPS
        if (app->config.drawFPS) {
            ImGui::SetNextWindowPos(ImVec2(230, 10), ImGuiCond_FirstUseEver);
            ImGui::Begin("FPS", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("FPS: %d", GetFPS());
            ImGui::End();
        }

        // Camera Settings
        ImGuiCamera(&app->camera, &app->characterData, &app->controlledCharacter, app->argc, app->argv);

        // Characters
        ImGuiCharacterData(&app->characterData,
            //&app->fileDialogState, 
            &app->scrubberSettings,
            app->errMsg, app->argc, app->argv);

        // Color Picker
        if (app->characterData.colorPickerActive) {
            ImGui::SetNextWindowPos(ImVec2((float)app->screenWidth - 180, 450), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(160, 140), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Color Picker", &app->characterData.colorPickerActive)) {
                float col[3] = { app->characterData.colors[app->characterData.active].r / 255.f,
                                 app->characterData.colors[app->characterData.active].g / 255.f,
                                 app->characterData.colors[app->characterData.active].b / 255.f };
                ImGui::ColorPicker3("##picker", col);
                app->characterData.colors[app->characterData.active] =
                    Color{ (unsigned char)(col[0] * 255), (unsigned char)(col[1] * 255), (unsigned char)(col[2] * 255), 255 };
            }
            ImGui::End();
        }

        // Scrubber
        ImGuiScrubberSettings(&app->scrubberSettings, &app->characterData, app->screenWidth, app->screenHeight);

        // Animation settings
        ImGuiAnimSettings(&app->config);

        // File Dialog
        //ImGuiWindowFileDialog(&app->fileDialogState);
    }

    if (app->config.drawFeatures && app->animDatabase.motionFrameCount > 0)
    {
        ImGui::SetNextWindowPos(ImVec2(60, 60), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360, 260), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Features"))
        {
            // Plotting
            const int count = app->animDatabase.motionFrameCount;
            int plotCount = count;
            if (plotCount > 1024) plotCount = 1024;

            vector<float> featX(plotCount);
            vector<float> featZ(plotCount);

            const int startIdx = count - plotCount;
            int pi = 0;
            for (int f = startIdx; f < count; ++f)
            {
                span<const float> featRow = app->animDatabase.features.row_view(f);
                featX[pi] = featRow[0];
                featZ[pi] = featRow[1];
                ++pi;
            }

            ImGui::Text("%s / %s",
                (app->animDatabase.featureNames.size() > 0) ? app->animDatabase.featureNames[0].c_str() : "Feat0",
                (app->animDatabase.featureNames.size() > 1) ? app->animDatabase.featureNames[1].c_str() : "Feat1");
            ImGui::PlotLines("##featX", featX.data(), plotCount, 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));
            ImGui::PlotLines("##featZ", featZ.data(), plotCount, 0, NULL, FLT_MAX, FLT_MAX, ImVec2(0, 80));

            // NEW: show current feature values for the currently selected/active character
            ImGui::Separator();

            const int activeChar = app->characterData.active;
            if (activeChar >= 0 && activeChar < app->animDatabase.animCount &&
                activeChar < (int)app->animDatabase.clipStartFrame.size() &&
                app->animDatabase.clipStartFrame[activeChar] < app->animDatabase.clipEndFrame[activeChar])
            {
                const float playTime = app->scrubberSettings.playTime;
                const float frameTime = app->animDatabase.animFrameTime[activeChar];
                const int clipFrameCount = app->animDatabase.animFrameCount[activeChar];

                int localFrame = 0;
                if (frameTime > 0.0f)
                {
                    localFrame = ClampInt((int)(playTime / frameTime + 0.5f), 0, clipFrameCount - 1);
                }

                const int motionIndex = app->animDatabase.clipStartFrame[activeChar] + localFrame;
                if (motionIndex >= 0 && motionIndex < app->animDatabase.motionFrameCount)
                {
                    ImGui::Text("Active: %s", app->characterData.names[activeChar].c_str());
                    ImGui::SameLine();
                    ImGui::Text("LocalFrame: %d  MotionIndex: %d", localFrame, motionIndex);

                    // Display each named feature value
                    const int fd = app->animDatabase.featureDim;
                    span<const float> featRow = app->animDatabase.features.row_view(motionIndex);
                    for (int fi = 0; fi < fd; ++fi)
                    {
                        const char* fname = (fi < (int)app->animDatabase.featureNames.size()) ?
                            app->animDatabase.featureNames[fi].c_str() : "Feature";
                        ImGui::Text("%s: % .6f", fname, featRow[fi]);
                    }
                }
                else
                {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Active clip has no motion-DB frames (not compatible with canonical skeleton).");
                }
            }
            else
            {
                ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Active character not present in motion DB.");
            }
        }
        ImGui::End();
    }

    // Blend Stack Debug Window
    if (app->controlledCharacter.active && app->config.drawBlendCursors)
    {
        // Cursor colors matching the 3D visualization
        const ImVec4 cursorImColors[ControlledCharacter::MAX_BLEND_CURSORS] = {
            ImVec4(1.0f, 0.0f, 0.0f, 1.0f),    // RED
            ImVec4(0.0f, 1.0f, 0.0f, 1.0f),    // GREEN
            ImVec4(0.0f, 0.0f, 1.0f, 1.0f),    // BLUE
            ImVec4(1.0f, 1.0f, 0.0f, 1.0f),    // YELLOW
            ImVec4(1.0f, 0.0f, 1.0f, 1.0f),    // MAGENTA
            ImVec4(1.0f, 0.5f, 0.0f, 1.0f),    // ORANGE
            ImVec4(1.0f, 0.75f, 0.8f, 1.0f),   // PINK
            ImVec4(0.5f, 0.8f, 1.0f, 1.0f),    // SKYBLUE
            ImVec4(0.0f, 1.0f, 0.5f, 1.0f),    // LIME
            ImVec4(0.5f, 0.0f, 1.0f, 1.0f),    // VIOLET
        };

        ImGui::SetNextWindowPos(ImVec2(20, 400), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(320, 200), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Blend Stack"))
        {
            // Debug timescale display
            if (app->debugPaused)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "PAUSED");
                ImGui::SameLine();
                ImGui::Text("(hold * to advance)");
            }
            else
            {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "PLAYING");
            }
            ImGui::Text("Timescale: %.4f", app->debugTimescale);
            ImGui::Text("Switch Timer: %.2f", app->controlledCharacter.switchTimer);
            ImGui::Separator();

            int activeCursorCount = 0;
            for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
            {
                const BlendCursor& cur = app->controlledCharacter.cursors[ci];
                if (cur.active) activeCursorCount++;
            }
            ImGui::Text("Active Cursors: %d", activeCursorCount);

            for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
            {
                const BlendCursor& cur = app->controlledCharacter.cursors[ci];
                if (!cur.active) continue;

                ImGui::PushID(ci);

                // Color indicator
                ImGui::TextColored(cursorImColors[ci], "[%d]", ci);
                ImGui::SameLine();

                // Anim name (if available)
                const char* animName = (cur.animIndex >= 0 && cur.animIndex < (int)app->characterData.names.size())
                    ? app->characterData.names[cur.animIndex].c_str()
                    : "???";
                ImGui::Text("%s", animName);

                // Weight bar
                ImGui::Text("  W: %.3f -> %.3f (norm: %.3f)",
                    cur.weightSpring.x, cur.targetWeight, cur.normalizedWeight);

                // Progress bar showing normalized weight
                ImGui::ProgressBar(cur.normalizedWeight, ImVec2(-1, 0), "");

                // Time info
                const float maxTime = (cur.animIndex >= 0 && cur.animIndex < app->characterData.count)
                    ? (app->characterData.bvhData[cur.animIndex].frameCount - 1) * app->characterData.bvhData[cur.animIndex].frameTime
                    : 0.0f;
                ImGui::Text("  Time: %.2f / %.2f", cur.animTime, maxTime);

                // Root motion delta info
                const float deltaLen = Vector3Length(cur.lastDeltaWorld);
                ImGui::Text("  dPos: (%.4f, %.4f) len=%.4f",
                    cur.lastDeltaWorld.x, cur.lastDeltaWorld.z, deltaLen);
                ImGui::Text("  dYaw: %.4f rad (%.2f deg)",
                    cur.lastDeltaYaw, cur.lastDeltaYaw * RAD2DEG);

                ImGui::PopID();
            }

            // Blended result
            ImGui::Separator();
            ImGui::Text("Blended Result:");
            const float blendedLen = Vector3Length(app->controlledCharacter.lastBlendedDeltaWorld);
            ImGui::Text("  dPos: (%.4f, %.4f) len=%.4f",
                app->controlledCharacter.lastBlendedDeltaWorld.x,
                app->controlledCharacter.lastBlendedDeltaWorld.z,
                blendedLen);
            ImGui::Text("  dYaw: %.4f rad (%.2f deg)",
                app->controlledCharacter.lastBlendedDeltaYaw,
                app->controlledCharacter.lastBlendedDeltaYaw * RAD2DEG);
        }
        ImGui::End();
    }

#if defined(ENABLE_PROFILE) && defined(_WIN32)
    // Display Profile Records
    PROFILE_TICKERS_UPDATE();

    ImGui::SetNextWindowPos(ImVec2(260, 10), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Profile", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        for (int i = 0; i < globalProfileRecords.num; i++) {
            ImGui::Text("%s", globalProfileRecords.records[i]->name);
            ImGui::SameLine();
            ImGui::Text("%6.1f us", globalProfileTickers.times[i]);
            ImGui::SameLine();
            ImGui::Text("%i calls", globalProfileTickers.samples[i]);
        }
    }
    ImGui::End();
#endif




    PROFILE_END(Gui);

    // End ImGui frame and render
    rlImGuiEnd();

    // Done

    EndDrawing();
}

//----------------------------------------------------------------------------------
// Main
//----------------------------------------------------------------------------------

// Command-line FBX to BVH conversion (runs without GUI)
static int ConvertFBXtoBVH(const char* inputPath)
{
    // Generate output path: input.fbx -> input.fbx.bvh
    char outputPath[512];
    snprintf(outputPath, sizeof(outputPath), "%s.bvh", inputPath);

    printf("Converting: %s -> %s\n", inputPath, outputPath);

    // Load FBX
    BVHData bvh;
    BVHDataInit(&bvh);
    char errMsg[512];

    if (!FBXDataLoad(&bvh, inputPath, errMsg, sizeof(errMsg)))
    {
        fprintf(stderr, "Error loading FBX: %s\n", errMsg);
        return 1;
    }

    printf("Loaded: %d joints, %d frames\n", bvh.jointCount, bvh.frameCount);

    // Save as BVH
    if (!BVHDataSave(&bvh, outputPath, errMsg, sizeof(errMsg)))
    {
        fprintf(stderr, "Error saving BVH: %s\n", errMsg);
        BVHDataFree(&bvh);
        return 1;
    }

    BVHDataFree(&bvh);
    printf("Success: %s\n", outputPath);
    return 0;
}

int main(int argc, char** argv)
{
    //TestBallTree();
    //if (true) return 0;

    // Handle command-line utilities (no GUI)
    if (argc >= 3 && strcmp(argv[1], "-fbx2bvh") == 0)
    {
        return ConvertFBXtoBVH(argv[2]);
    }

    // Set current working directory to source root for file access
    // This helps find shader files and other resources
#if defined(SOURCE_ROOT_PATH)
    // On Windows, use SetCurrentDirectory
#if defined(_WIN32)
    SetCurrentDirectory(SOURCE_ROOT_PATH);
#else
    chdir(SOURCE_ROOT_PATH);
#endif
    printf("Working directory set to: %s\n", SOURCE_ROOT_PATH);
#endif

    PROFILE_INIT();
    PROFILE_TICKERS_INIT();

    //TestCudaAndLibtorch();

    // Init Application State

    ApplicationState app;
    app.argc = argc;
    app.argv = argv;

    // Load saved window config
    app.config = LoadAppConfig(argc, argv);
    app.screenWidth = app.config.windowWidth;
    app.screenHeight = app.config.windowHeight;

    // Init Window
    SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(app.screenWidth, app.screenHeight, "Flomo");
    SetTargetFPS(60);

    // Restore window position if we have a valid config
    if (app.config.valid) {
        SetWindowPosition(app.config.windowX, app.config.windowY);
    }

    // Init Dear ImGui with 2x scale
    rlImGuiBeginInitImGui();
    ImGui::GetIO().FontGlobalScale = 2.0f;
    ImGui::StyleColorsDark();
    rlImGuiEndInitImGui();

    SetTraceLogLevel(LOG_DEBUG);

    // Camera
    CameraSystemInit(&app.camera, argc, argv);

    // Restore camera state from config
    if (app.config.valid) {
        app.camera.unreal.position = Vector3{ app.config.cameraPosX, app.config.cameraPosY, app.config.cameraPosZ };
        app.camera.unreal.yaw = app.config.cameraYaw;
        app.camera.unreal.pitch = app.config.cameraPitch;
        app.camera.unreal.moveSpeed = app.config.cameraMoveSpeed;
        app.camera.mode = (app.config.cameraMode == 0) ? FlomoCameraMode::Orbit : FlomoCameraMode::Unreal;
    }

    // Shader

    app.shader = LoadShader("shaders/shader.vert", "shaders/shader.frag");
    if (app.shader.id == 0 || app.shader.locs == NULL) {
        TraceLog(LOG_ERROR, "Failed to load shader!");
    }

    ShaderUniformsInit(&app.uniforms, app.shader);

    // Models

    app.groundPlaneMesh = GenMeshPlane(2.0f, 2.0f, 1, 1);
    app.groundPlaneModel = LoadModelFromMesh(app.groundPlaneMesh);
    app.groundPlaneModel.materials[0].shader = app.shader;

    app.capsuleModel = LoadCapsuleModel();
    app.capsuleModel.materials[0].shader = app.shader;

    // Character Data

    CharacterDataInit(&app.characterData, argc, argv);

    // Capsule Data

    CapsuleDataInit(&app.capsuleData);

    // Controlled Character (starts inactive until first animation is loaded)

    app.controlledCharacter.active = false;

    // Scrubber Settings

    ScrubberSettingsInit(&app.scrubberSettings, argc, argv);

    // Render Settings

    CapsuleDataUpdateShadowLookupTable(&app.capsuleData, app.config.sunLightConeAngle);

    // Geno Character Rendering

    app.genoRenderMode = false;
    app.genoModelLoaded = false;

    // Try to load Geno model
    app.genoModel = LoadGenoModel("data/Geno.bin");
    if (app.genoModel.meshCount > 0 && app.genoModel.boneCount > 0)
    {
        app.genoModelLoaded = true;
        app.genoAnimation = LoadEmptyModelAnimation(app.genoModel);

        // Load skinned shader for GPU skeletal animation
        app.genoBasicShader = LoadShader("shaders/skinnedBasic.vs", "shaders/skinnedForward.fs");
        if (app.genoBasicShader.id > 0)
        {
            app.genoModel.materials[0].shader = app.genoBasicShader;
            TraceLog(LOG_INFO, "GENO: Loaded skinned shader successfully");
        }
        else
        {
            TraceLog(LOG_WARNING, "GENO: Failed to load skinned shader, animation may not work");
        }

        TraceLog(LOG_INFO, "GENO: Model loaded successfully, %d bones", app.genoModel.boneCount);
    }
    else
    {
        TraceLog(LOG_WARNING, "GENO: Failed to load Geno model from data/Geno.bin");
    }

    // File Dialog

    //app.fileDialogState = InitGuiWindowFileDialog(GetWorkingDirectory());

    // Reset Error Message

    app.errMsg[0] = '\0';

    // Load any files given as command line arguments

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-') { continue; }

        CharacterDataLoadFromFile(&app.characterData, argv[i], app.errMsg, 512);
    }

    const bool loadDefaultFiles = true;
    if (loadDefaultFiles && app.characterData.count == 0)
    {
        app.errMsg[0] = '\0';

        // Auto-load a default scene on startup
        {
            vector<const char*> autoFiles = 
            { 
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-013.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-014.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-015.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-017.fbx"
            };
            for (const char* file : autoFiles)
            {
                if (CharacterDataLoadFromFile(&app.characterData, file, app.errMsg, sizeof(app.errMsg)))
                {
                    // Make the auto-loaded file the active character (if loaded)
                    app.characterData.active = app.characterData.count - 1;
                    TraceLog(LOG_INFO, "Auto-loaded default file at startup: %s", file);
                }
            }
        }
    }
    else
    {
        TraceLog(LOG_INFO, "Loaded %d character(s) at startup.", app.characterData.count);
    }

    // If any characters loaded, update capsules and scrubber

    if (app.characterData.count > 0)
    {
        app.characterData.active = app.characterData.count - 1;

        CapsuleDataUpdateForCharacters(&app.capsuleData, &app.characterData);
        ScrubberSettingsRecomputeLimits(&app.scrubberSettings, &app.characterData);
        ScrubberSettingsInitMaxs(&app.scrubberSettings, &app.characterData);

        // Build animation database and initialize controlled character
        AnimDatabaseRebuild(&app.animDatabase, &app.characterData);
        if (!app.animDatabase.valid) {
            TraceLog(LOG_WARNING, "AnimDatabase invalid at startup - controlled character disabled.");
            app.controlledCharacter.active = false;
        }
        else {
            ControlledCharacterInit(
                &app.controlledCharacter,
                &app.characterData.bvhData[0],
                app.characterData.scales[0],
                app.config.switchInterval);
        }

        // Resize capsule buffer to include controlled character
        {
            const int totalJoints = (int)app.capsuleData.capsulePositions.size() +
                app.controlledCharacter.xformData.jointCount;
            CapsuleDataResize(&app.capsuleData, totalJoints);
        }

        string windowTitle = app.characterData.filePaths[app.characterData.active] + " - BVHView";
        SetWindowTitle(windowTitle.c_str());
    }

    // Game Loop

#if defined(PLATFORM_WEB)
    emscripten_set_main_loop_arg(ApplicationUpdate, &app, 0, 1);
#else
    while (!WindowShouldClose())
    {
        ApplicationUpdate(&app);
    }
#endif

    // Unload and finish

    CapsuleDataFree(&app.capsuleData);
    CharacterDataFree(&app.characterData);
    if (app.controlledCharacter.active)
    {
        ControlledCharacterFree(&app.controlledCharacter);
    }

    // Unload Geno resources
    if (app.genoModelLoaded)
    {
        app.genoMappings.clear();
        UnloadModelAnimation(app.genoAnimation);
        UnloadModel(app.genoModel);
        if (app.genoBasicShader.id > 0)
        {
            UnloadShader(app.genoBasicShader);
        }
    }

    UnloadModel(app.capsuleModel);
    UnloadModel(app.groundPlaneModel);
    UnloadShader(app.shader);

    // Shutdown Dear ImGui
    rlImGuiShutdown();

    // Save config before closing (window + camera)
    const Vector2 windowPos = GetWindowPosition();
    app.config.windowX = (int)windowPos.x;
    app.config.windowY = (int)windowPos.y;
    app.config.windowWidth = app.screenWidth;
    app.config.windowHeight = app.screenHeight;

    app.config.cameraPosX = app.camera.unreal.position.x;
    app.config.cameraPosY = app.camera.unreal.position.y;
    app.config.cameraPosZ = app.camera.unreal.position.z;
    app.config.cameraYaw = app.camera.unreal.yaw;
    app.config.cameraPitch = app.camera.unreal.pitch;
    app.config.cameraMoveSpeed = app.camera.unreal.moveSpeed;
    app.config.cameraMode = (app.camera.mode == FlomoCameraMode::Orbit) ? 0 : 1;


    SaveAppConfig(app.config);

    CloseWindow();

    return 0;
}
