#pragma once

//#include "raylib.h"


// Animation playback mode for controlled character
enum class AnimationMode : int
{
    RandomSwitch = 0,   // randomly switch between animations
    MotionMatching,     // use motion matching to find best animation
    COUNT
};

static inline const char* AnimationModeName(AnimationMode mode)
{
    switch (mode)
    {
    case AnimationMode::RandomSwitch: return "Random Switch";
    case AnimationMode::MotionMatching: return "Motion Matching";
    default: return "Unknown";
    }
}

// Cursor blend mode - how cursor rotations are blended
enum class CursorBlendMode : int
{
    Basic = 0,           // direct weighted average of cursor rotations
    LookaheadDragging,   // lerp towards extrapolated future pose
    COUNT
};

static inline const char* CursorBlendModeName(CursorBlendMode mode)
{
    switch (mode)
    {
    case CursorBlendMode::Basic: return "Basic";

    case CursorBlendMode::LookaheadDragging: return "Lookahead Dragging";
    default: return "Unknown";
    }
}

// Motion Matching Feature Types
enum class FeatureType : int
{
    ToePos = 0,          // left+right toe positions (X,Z) => 4 dims
    ToeVel,              // left+right toe velocities (X,Z) => 4 dims
    ToeDiff,             // left-right difference (X,Z) => 2 dims
    FutureVel,           // future root velocity (XZ) at sample points => 2 * points
    FutureVelClamped,    // future root velocity clamped to max magnitude (XZ) => 2 * points
    FutureSpeed,         // future root speed (scalar) at sample points => 1 * points
    PastPosition,        // past hip position (XZ) in current hip horizontal frame => 2 dims
    AimDirection,        // aim direction (head→rightHand) at trajectory times => 2 * points
    HeadToSlowestToe,    // head to slowest foot vector (XZ) in root space => 2 dims
    FutureAccelClamped,  // future root acceleration clamped: dead zone below 1m/s2, capped at 3m/s2 => 2 * points

    COUNT                // Must be last - used for array sizing
};

// Returns human-readable name for feature type
static inline const char* FeatureTypeName(FeatureType type)
{
    switch (type)
    {
    case FeatureType::ToePos: return "Toe Position";
    case FeatureType::ToeVel: return "Toe Velocity";
    case FeatureType::ToeDiff: return "Toe Difference";
    case FeatureType::FutureVel: return "Future Velocity";
    case FeatureType::FutureVelClamped: return "Future Vel Clamped";
    case FeatureType::FutureSpeed: return "Future Speed";
    case FeatureType::PastPosition: return "Past Position";
    case FeatureType::AimDirection: return "Aim Direction";
    case FeatureType::HeadToSlowestToe: return "Head To Slowest Toe";
    case FeatureType::FutureAccelClamped: return "Future Accel Clamped";
    default: return "Unknown";
    }
}

// Blend root mode for position (which point to use as character root for blending)
enum class BlendRootModePosition : int
{
    Hips = 0,
    CenterOfMass,
    COUNT
};

static inline const char* BlendRootModePositionName(BlendRootModePosition mode)
{
    switch (mode)
    {
    case BlendRootModePosition::Hips: return "Hips";
    case BlendRootModePosition::CenterOfMass: return "Center of Mass";
    default: return "Unknown";
    }
}

// Blend root mode for rotation (how to compute character facing direction)
enum class BlendRootModeRotation : int
{
    Hips = 0,
    HeadToRightHand,
    COUNT
};

static inline const char* BlendRootModeRotationName(BlendRootModeRotation mode)
{
    switch (mode)
    {
    case BlendRootModeRotation::Hips: return "Hips";
    case BlendRootModeRotation::HeadToRightHand: return "Head to Right Hand";
    default: return "Unknown";
    }
}

struct MotionMatchingFeaturesConfig
{
    float featureTypeWeights[static_cast<int>(FeatureType::COUNT)];
    std::vector<float> futureTrajPointTimes = { 0.2f, 0.4f, 0.8f };
    float pastTimeOffset = 0.1f;

    // Pose drag lookahead time (seconds) - used for precomputing lookahead poses
    float poseDragLookaheadTime = 0.1f;

    // Blend root mode settings
    BlendRootModePosition blendRootModePosition = BlendRootModePosition::Hips;
    BlendRootModeRotation blendRootModeRotation = BlendRootModeRotation::Hips;

    MotionMatchingFeaturesConfig()
    {
        for (int i = 0; i < static_cast<int>(FeatureType::COUNT); ++i) featureTypeWeights[i] = 1.0f;
    }

    bool IsFeatureEnabled(FeatureType type) const
    {
        return featureTypeWeights[static_cast<int>(type)] > 0.0f;
    }
};



// Simple app config that persists between runs
struct AppConfig {
    // Window
    int windowX = 100;
    int windowY = 100;
    int windowWidth = 2200;
    int windowHeight = 1500;

    // Camera (Unreal mode state - most general representation)
    float cameraPosX = 2.0f;
    float cameraPosY = 1.5f;
    float cameraPosZ = 5.0f;
    float cameraYaw = 3.14159f;  // PI - facing towards -Z
    float cameraPitch = 0.0f;
    float cameraMoveSpeed = 5.0f;
    int cameraMode = 1;  // 0 = Orbit, 1 = Unreal, 2 = TurretFollower
    float cameraTargetBlendtime = 0.2f;  // Smooth target following blendtime for Orbit and LazyTurretFollower
    bool trackHipsProjectedOnGround = false;  // Track bone 0 at Y=1m (more stable camera)

    // Render settings (persisted)
    // Colors stored as separate ints so parsing/writing JSON remains simple here.
    Color backgroundColor = { 255, 255, 255, 255 };
    float sunLightConeAngle = 0.2f;
    float sunLightStrength = 0.25f;
    float sunAzimuth = 3.14159f / 4.0f;
    float sunAltitude = 0.8f;
    Color sunColor = { 253, 255, 232, 255 };

    float skyLightStrength = 0.15f;
    Color skyColor = { 174, 183, 190, 255 };

    float groundLightStrength = 0.1f;
    float ambientLightStrength = 1.0f;

    float exposure = 0.9f;

    // Toggles
    bool drawOrigin = true;
    bool drawGrid = false;
    bool drawChecker = true;
    bool drawCapsules = true;
    bool drawWireframes = false;
    bool drawSkeleton = true;
    bool drawTransforms = false;
    bool drawAO = true;
    bool drawShadows = true;
    bool drawEndSites = true;
    bool drawFPS = false;
    bool drawUI = true;

    bool drawFeatures = false;
    bool drawBlendCursors = true;  // Debug: show individual blend cursor skeletons
    bool drawVelocities = false;   // Draw joint velocity vectors
    bool drawAccelerations = false; // Draw joint acceleration vectors
    bool drawRootVelocities = false; // Draw root motion velocity from each cursor
    bool drawToeVelocities = false;  // Draw toe velocity vectors (actual vs blended)
    bool drawFootIK = false;         // Draw foot IK debug (virtual toe positions, etc.)
    bool drawBasicBlend = false;     // Draw basic blend result (before lookahead dragging)
    bool drawMagicAnchor = false;    // Draw Magic anchor (spine3 projected + head→hand yaw)
    bool drawPastHistory = false;    // Draw past position history for motion matching

    // Animation settings
    AnimationMode animationMode = AnimationMode::RandomSwitch;  // animation playback mode
    float defaultBlendTime = 0.1f;  // time for blend cursor spring to reach 95% of target
    float switchInterval = 3.0f;    // time between random animation switches
    float mmSearchPeriod = 0.1f;    // time between motion matching searches
    float virtualControlMaxAcceleration = 4.0f;   // maximum acceleration for virtual control velocity (m/s^2)

    // Cursor blend mode settings
    CursorBlendMode cursorBlendMode = CursorBlendMode::Basic;
    float blendPosReturnTime = 0.1f; // time for velblending to lerp towards target

    // Foot IK
    bool enableFootIK = true;  // enable/disable foot IK towards virtual toe positions
    bool enableTimedUnlocking = true;  // enable/disable timed unlock mechanism for virtual toes
    float unlockDistance = 0.2f;  // distance threshold to unlock virtual toe (meters)
    float unlockDuration = 0.3f;  // time to gradually re-lock virtual toe (seconds)
    
    
    bool drawPlayerInput = false;

    // Neural Network settings
    bool useMMFeatureDenoiser = false;

    // Motion Matching Configuration, version that is editable: copied to AnimDatabase on build
    MotionMatchingFeaturesConfig mmConfigEditor;


    // Validity
    bool valid = false;
};


struct PlayerControlInput
{
    Vector3 desiredVelocity = Vector3Zero();  // Desired velocity in world space (XZ plane)
    float maxSpeed = 2.0f;                    // Maximum movement speed (m/s)
    Vector3 desiredAimDirection = { 0.0f, 0.0f, 1.0f };  // Desired aim direction (world space XZ, unit length)
};



//----------------------------------------------------------------------------------
// Pose Features - Neural Network Target
//----------------------------------------------------------------------------------

// Structured representation of pose features for neural network training/inference
// This is what the network outputs given motion matching features as input
// These are the targets that ControlledCharacterUpdateNetwork will use to advance the pose
//
// Usage example for network inference in ControlledCharacterUpdateNetwork:
//   PoseFeatures pose;
//   torch::Tensor output = network.forward(motionMatchingFeatures);
//   pose.DeserializeFrom(outputSpan, jointCount);
//   // Apply pose.lookaheadLocalRotations/Positions to character
//   // Update root motion using pose.lookaheadRootVelocity and pose.rootYawRate
//   // Update foot IK using pose.lookaheadToePositions and pose.toeVelocities
struct PoseFeatures
{
    // Lookahead pose (what the pose will be after lookahead time)
    std::vector<Rot6d> lookaheadLocalRotations;   // [jointCount]
    std::vector<Vector3> lookaheadLocalPositions;  // [jointCount]

    // Root motion (lookahead for velocity, current for yaw rate)
    Vector3 lookaheadRootVelocity;                 // root velocity in root space (lookahead)
    float rootYawRate;                             // current yaw rate (rad/s)

    // Foot IK data (lookahead positions, current velocities for speed clamping)
    Vector3 lookaheadToePositionsRootSpace[SIDES_COUNT];  // [left, right]
    Vector3 toeVelocitiesRootSpace[SIDES_COUNT];          // [left, right] current velocities

    // Get dimension needed for flat array representation
    static int GetDim(int jointCount)
    {
        int dim = 0;
        dim += jointCount * 6;  // lookaheadLocalRotations (Rot6d per joint)
        dim += jointCount * 3;  // lookaheadLocalPositions (Vector3 per joint)
        dim += 3;               // lookaheadRootVelocity (Vector3)
        dim += 1;               // rootYawRate (float)
        dim += 3 * 2;           // lookaheadToePositionsRootSpace (Vector3 x 2 sides)
        dim += 3 * 2;           // toeVelocitiesRootSpace (Vector3 x 2 sides)
        return dim;
    }

    // Resize internal storage for given joint count
    void Resize(int jointCount)
    {
        lookaheadLocalRotations.resize(jointCount);
        lookaheadLocalPositions.resize(jointCount);
        lookaheadRootVelocity = Vector3Zero();
        rootYawRate = 0.0f;
        for (int side : sides)
        {
            lookaheadToePositionsRootSpace[side] = Vector3Zero();
            toeVelocitiesRootSpace[side] = Vector3Zero();
        }
    }

    // Serialize to flat array
    // dest must have size >= GetDim(jointCount)
    void SerializeTo(std::span<float> dest) const
    {
        int idx = 0;

        // Pack lookahead local rotations (Rot6d: 6 floats per joint)
        for (const Rot6d& rot : lookaheadLocalRotations)
        {
            dest[idx++] = rot.ax;
            dest[idx++] = rot.ay;
            dest[idx++] = rot.az;
            dest[idx++] = rot.bx;
            dest[idx++] = rot.by;
            dest[idx++] = rot.bz;
        }

        // Pack lookahead local positions (Vector3: 3 floats per joint)
        for (const Vector3& pos : lookaheadLocalPositions)
        {
            dest[idx++] = pos.x;
            dest[idx++] = pos.y;
            dest[idx++] = pos.z;
        }

        // Pack lookahead root velocity (Vector3: 3 floats)
        dest[idx++] = lookaheadRootVelocity.x;
        dest[idx++] = lookaheadRootVelocity.y;
        dest[idx++] = lookaheadRootVelocity.z;

        // Pack root yaw rate (float: 1 float)
        dest[idx++] = rootYawRate;

        // Pack lookahead toe positions (Vector3 x 2: 6 floats)
        for (int side : sides)
        {
            dest[idx++] = lookaheadToePositionsRootSpace[side].x;
            dest[idx++] = lookaheadToePositionsRootSpace[side].y;
            dest[idx++] = lookaheadToePositionsRootSpace[side].z;
        }

        // Pack toe velocities (Vector3 x 2: 6 floats)
        for (int side : sides)
        {
            dest[idx++] = toeVelocitiesRootSpace[side].x;
            dest[idx++] = toeVelocitiesRootSpace[side].y;
            dest[idx++] = toeVelocitiesRootSpace[side].z;
        }
    }

    // Deserialize from flat array
    // src must have size >= GetDim(jointCount)
    void DeserializeFrom(std::span<const float> src, int jointCount)
    {
        Resize(jointCount);
        int idx = 0;

        // Unpack lookahead local rotations
        for (int j = 0; j < jointCount; ++j)
        {
            Rot6d& rot = lookaheadLocalRotations[j];
            rot.ax = src[idx++];
            rot.ay = src[idx++];
            rot.az = src[idx++];
            rot.bx = src[idx++];
            rot.by = src[idx++];
            rot.bz = src[idx++];
        }

        // Unpack lookahead local positions
        for (int j = 0; j < jointCount; ++j)
        {
            Vector3& pos = lookaheadLocalPositions[j];
            pos.x = src[idx++];
            pos.y = src[idx++];
            pos.z = src[idx++];
        }

        // Unpack lookahead root velocity
        lookaheadRootVelocity.x = src[idx++];
        lookaheadRootVelocity.y = src[idx++];
        lookaheadRootVelocity.z = src[idx++];

        // Unpack root yaw rate
        rootYawRate = src[idx++];

        // Unpack lookahead toe positions
        for (int side : sides)
        {
            lookaheadToePositionsRootSpace[side].x = src[idx++];
            lookaheadToePositionsRootSpace[side].y = src[idx++];
            lookaheadToePositionsRootSpace[side].z = src[idx++];
        }

        // Unpack toe velocities
        for (int side : sides)
        {
            toeVelocitiesRootSpace[side].x = src[idx++];
            toeVelocitiesRootSpace[side].y = src[idx++];
            toeVelocitiesRootSpace[side].z = src[idx++];
        }
    }
};


//----------------------------------------------------------------------------------
// Animation Database
//----------------------------------------------------------------------------------

// A unified view of all loaded animations, suitable for sampling by ControlledCharacter.
struct AnimDatabase
{
    // Motion matching feature configuration (includes lookahead time and blend root modes)
    MotionMatchingFeaturesConfig featuresConfig;

    // References to all loaded animations
    int animCount = -1;

    // Per-animation info
    std::vector<int> animStartFrame;   // Global frame index where each anim starts
    std::vector<int> animFrameCount;   // Number of frames in each anim
    std::vector<float> animFrameTime;  // Frame time for each anim (usually same)

    // Scale to apply when sampling (for unit conversion)
    float scale = 0.0f;

    // Validity: true only if ALL animations are compatible with canonical skeleton
    bool valid = false;

    // ---- Motion-database specific fields ----
    // Canonical joint count (set from first animation's joint count)
    int jointCount = -1;

    // Number of frames actually stored in the compacted motion DB (may be <= totalFrames
    // if some clips have mismatched skeletons and are skipped)
    int motionFrameCount = -1;

    // Per-frame joint transforms in animation space: [motionFrameCount x jointCount]
    Array2D<Vector3> jointPositionsAnimSpace;       // positions in animation world space
    Array2D<Rot6d> jointRotationsAnimSpace;         // rotations in animation world space

    // Per-frame joint velocities/accelerations in root space (magic anchor space)
    Array2D<Vector3> jointVelocitiesRootSpace;      // velocities relative to character heading
    Array2D<Vector3> jointAccelerationsRootSpace;   // accelerations relative to character heading

    // Joint-local transforms (relative to parent joint, for blending)
    Array2D<Vector3> localJointPositions;           // local positions [motionFrameCount x jointCount]
    Array2D<Rot6d> localJointRotations;             // local rotations [motionFrameCount x jointCount]

    // lookahead pose for inertial dragging
    Array2D<Rot6d> lookaheadLocalRotations;         // [motionFrameCount x jointCount]
    Array2D<Vector3> lookaheadLocalPositions;       // [motionFrameCount x jointCount]

    // Root motion velocities in root space (heading-relative, XZ only)
    // Velocity at frame f is transformed by inverse of root yaw at frame f
    std::vector<Vector3> rootMotionVelocitiesRootSpace;   // [motionFrameCount] - XZ velocity in root space
    std::vector<float> rootMotionYawRates;                // [motionFrameCount] - yaw angular velocity (rad/s)

    // Lookahead root motion velocities (extrapolated, also in root space)
    std::vector<Vector3> lookaheadRootMotionVelocitiesRootSpace;  // [motionFrameCount] - extrapolated XZ velocity
    std::vector<float> lookaheadRootMotionYawRates;               // [motionFrameCount] - extrapolated yaw rate (rad/s)



    // Toe positions in root space (relative to hip on ground, heading-aligned)
    std::vector<Vector3> toePositionsRootSpace[SIDES_COUNT];           // [motionFrameCount]
    std::vector<Vector3> lookaheadToePositionsRootSpace[SIDES_COUNT];  // [motionFrameCount] - extrapolated

    // Segmentation of the compacted motion DB into clips:
    // clipStartFrame[c] .. clipEndFrame[c]-1 are frames for clip c in motion DB frame space.
    std::vector<int> clipStartFrame;
    std::vector<int> clipEndFrame;

    // motion matching features [motionFrameCount x featureDim]
    int featureDim = -1;
    Array2D<float> features;
    int hipJointIndex = -1;            // resolved index for "Hips" in canonical skeleton
    int toeIndices[SIDES_COUNT] = { -1, -1 };
    int footIndices[SIDES_COUNT] = { -1, -1 };    // ankle
    int lowlegIndices[SIDES_COUNT] = { -1, -1 };  // shin/calf
    int uplegIndices[SIDES_COUNT] = { -1, -1 };   // thigh
    int handIndices[SIDES_COUNT] = { -1, -1 };    // hands

    // Magic anchor system - alternative reference frame for blending
    int spine3Index = -1;              // upper spine for Magic position
    int spine1Index = -1;              // lower spine for Magic position
    int headIndex = -1;                // head for Magic orientation

    // Magic anchor transforms per frame (position = spine3 on ground, yaw = head→rightHand direction)
    std::vector<Vector3> magicPosition;           // [motionFrameCount] - (spine3.x, 0, spine3.z)
    std::vector<float> magicYaw;                  // [motionFrameCount] - yaw from head→rightHand
    std::vector<Vector3> magicSmoothedVelocityAnimSpace;  // [motionFrameCount] - gaussian-smoothed velocity in animation world space
    std::vector<Vector3> magicSmoothedAccelerationAnimSpace; // [motionFrameCount] - gaussian-smoothed acceleration in animation world space
    std::vector<Vector3> magicVelocityRootSpace;           // [motionFrameCount] - XZ velocity in magic space
    std::vector<float> magicYawRate;              // [motionFrameCount] - yaw rate (rad/s)
    std::vector<Vector3> lookaheadMagicVelocity;  // [motionFrameCount] - extrapolated
    std::vector<float> lookaheadMagicYawRate;     // [motionFrameCount] - extrapolated

    // Hip transform relative to Magic anchor (for placing skeleton when using Magic root motion)
    std::vector<Vector3> hipPositionInMagicSpace;        // [motionFrameCount] - hip offset from magic, in magic-heading space
    std::vector<Rot6d> hipRotationInMagicSpace;          // [motionFrameCount] - full hip rotation relative to magic yaw
    std::vector<std::string> featureNames;
    std::vector<FeatureType> featureTypes;      // which FeatureType each feature dimension belongs to

    std::vector<float> featuresMean;            // mean of each feature dimension [featureDim]
    float featureTypesStd[static_cast<int>(FeatureType::COUNT)] = {};  // std shared by all features of same type
    Array2D<float> normalizedFeatures;          // normalized features [motionFrameCount x featureDim]

    // Neural network training targets: pose generation features [motionFrameCount x poseGenFeaturesComputeDim]
    // Contains lookahead local rotations, positions, root motion, and foot IK data
    // This is what the network should output given the motion matching features as input
    int poseGenFeaturesComputeDim = -1;        // dimension: jointCount*9 + 16
    Array2D<float> poseGenFeatures;            // [motionFrameCount x poseGenFeaturesComputeDim]
};


// Free/reset AnimDatabase to empty state
static void AnimDatabaseFree(AnimDatabase* db)
{
    db->animCount = -1;
    db->animStartFrame.clear();
    db->animFrameCount.clear();
    db->animFrameTime.clear();
    db->scale = 0.0f;
    db->valid = false;
    db->jointCount = -1;
    db->motionFrameCount = -1;
    db->jointPositionsAnimSpace.clear();
    db->jointRotationsAnimSpace.clear();
    db->jointVelocitiesRootSpace.clear();
    db->jointAccelerationsRootSpace.clear();
    db->localJointPositions.clear();
    db->localJointRotations.clear();
    db->lookaheadLocalRotations.clear();
    db->lookaheadLocalPositions.clear();
    db->rootMotionVelocitiesRootSpace.clear();
    db->rootMotionYawRates.clear();
    db->lookaheadRootMotionVelocitiesRootSpace.clear();
    db->lookaheadRootMotionYawRates.clear();
    db->magicPosition.clear();
    db->magicYaw.clear();
    db->magicSmoothedVelocityAnimSpace.clear();
    db->magicSmoothedAccelerationAnimSpace.clear();
    db->magicVelocityRootSpace.clear();
    db->magicYawRate.clear();
    db->lookaheadMagicVelocity.clear();
    db->lookaheadMagicYawRate.clear();
    for (int side : sides)
    {
        db->toePositionsRootSpace[side].clear();
        db->lookaheadToePositionsRootSpace[side].clear();
    }
    db->clipStartFrame.clear();
    db->clipEndFrame.clear();
    db->features.clear();
    db->featureDim = 0;
    db->featureNames.clear();
    db->featureTypes.clear();
    db->featuresMean.clear();
    for (int i = 0; i < static_cast<int>(FeatureType::COUNT); ++i) db->featureTypesStd[i] = 0.0f;
    db->normalizedFeatures.clear();
    db->poseGenFeaturesComputeDim = -1;
    db->poseGenFeatures.clear();
}


// Types of "channels" that are possible in the BVH format
enum
{
    CHANNEL_X_POSITION = 0,
    CHANNEL_Y_POSITION = 1,
    CHANNEL_Z_POSITION = 2,
    CHANNEL_X_ROTATION = 3,
    CHANNEL_Y_ROTATION = 4,
    CHANNEL_Z_ROTATION = 5,
    CHANNELS_MAX = 6,
};

// Data associated with a single "joint" in the BVH format
struct BVHJointData
{
    int parent;
    std::string name;              // changed to std::string
    Vector3 offset;
    int channelCount;
    char channels[CHANNELS_MAX];
    bool endSite;
};


//----------------------------------------------------------------------------------
// Transform Data
//----------------------------------------------------------------------------------

// Structure for containing a sampled pose as joint transforms
struct TransformData
{
    int jointCount;
    std::vector<int> parents;
    std::vector<bool> endSite;
    std::vector<Vector3> localPositions;
    std::vector<Quaternion> localRotations;
    std::vector<Vector3> globalPositions;
    std::vector<Quaternion> globalRotations;

};


// Data structure matching what is present in the BVH file format
struct BVHData
{
    // Hierarchy Data
    int jointCount;
    std::vector<BVHJointData> joints;

    // Motion Data
    int frameCount;
    int channelCount;
    float frameTime;
    std::vector<float> motionData;
};

//----------------------------------------------------------------------------------
// Blend Cursor - individual animation playback cursor with weight
//----------------------------------------------------------------------------------

struct BlendCursor {
    int animIndex = -1;                         // which animation/clip
    float animTime = 0.0f;                      // playback time in that clip
    DoubleSpringDamperState weightSpring = {};  // spring state for weight blending (x = current weight)
    float normalizedWeight = 0.0f;              // weight / totalWeight (sums to 1 across active cursors)
    DoubleSpringDamperState fastWeightSpring = {}; // faster spring for yaw rate
    float fastNormalizedWeight = 0.0f;          // fastWeight / totalFastWeight
    float targetWeight = 0.0f;                  // desired weight
    float blendTime = 0.3f;                     // halflife for double spring damper
    bool active = false;                        // is cursor in use

    // Local-space pose stored per cursor for blending (size = jointCount)
    std::vector<Vector3> localPositions;
    std::vector<Rot6d> localRotations6d;
    std::vector<Rot6d> lookaheadRotations6d;  // extrapolated pose for lookahead dragging
    std::vector<Vector3> lookaheadLocalPositions;  // extrapolated positions for lookahead dragging

    // Sampled root motion velocities from database (root space = heading-relative)
    Vector3 sampledRootVelocityRootSpace = Vector3Zero();  // XZ velocity in root space
    float sampledRootYawRate = 0.0f;                       // yaw rate (rad/s)

    // Sampled lookahead root motion velocities (extrapolated, also root space)
    Vector3 sampledLookaheadRootVelocityRootSpace = Vector3Zero();  // lookahead XZ velocity
    float sampledLookaheadRootYawRate = 0.0f;                       // lookahead yaw rate (rad/s)

    // Sampled lookahead toe positions (root space, for predictive foot IK)
    Vector3 sampledLookaheadToePosRootSpace[SIDES_COUNT] = { Vector3Zero(), Vector3Zero() };

    // Global-space pose for debug visualization (computed via FK after sampling)
    std::vector<Vector3> globalPositions;
    std::vector<Quaternion> globalRotations;

    // Previous local root state used to compute per-cursor root deltas.
    // Stored in the same local-space that we sample into above.
    Vector3 prevLocalRootPos;
    Rot6d prevLocalRootRot6d = Rot6dIdentity();

    // Root motion velocity tracking (for acceleration-based blending)
    Vector3 rootVelocityWorldForDisplayOnly = Vector3Zero();    // current root velocity (world space)
    float rootYawRate = 0.0f;                // current yaw rate (radians/sec)
};



// History point for past position tracking
struct HistoryPoint
{
    Vector3 position;
    double timestamp;
};

//----------------------------------------------------------------------------------
// Controlled Character - Root Motion Playback
//----------------------------------------------------------------------------------

// A character that plays animation with root motion extracted and applied to world transform.
// Every N seconds it jumps to a random time in the loaded animations while maintaining
// its world position and facing direction.
struct ControlledCharacter {
    // World placement of bone 0 (root)
    Vector3 worldPosition;
    Quaternion worldRotation;  // Character's facing direction (Y-axis rotation only)


    // Animation mode
    AnimationMode animMode = AnimationMode::RandomSwitch;

    // Animation state
    int animIndex;             // Which loaded animation to play from
    float animTime;            // Current playback time in that animation



    // Random switch timer (used by RandomSwitch mode)
    float switchTimer;

    // Motion matching state
    int mmBestFrame = -1;           // best matching frame from last search
    float mmBestCost = 0.0f;        // cost of best match
    float mmSearchTimer = 0.0f;     // time since last search

    // Pose output (local space with root zeroed, then transformed to world)
    TransformData xformData;


    // Pre-IK FK state for debugging (saved before IK is applied)
    TransformData xformBeforeIK;
    bool debugSaveBeforeIK;  // toggle to enable saving pre-IK state

    // Basic blend result (before lookahead dragging) for debugging
    TransformData xformBasicBlend;

    // Visual properties
    Color color;
    float opacity;
    float radius;
    float scale;

    // Reference to skeleton (first loaded BVH)
    const BVHData* skeleton;
    bool active;

    // Joint names combo string for UI (semicolon-separated)
    std::string jointNamesCombo;

    // -----------------------
    // Blending cursor pool
    // -----------------------
    static constexpr int MAX_BLEND_CURSORS = 10;
    BlendCursor cursors[MAX_BLEND_CURSORS];

    // Cursor blend mode and state
    CursorBlendMode cursorBlendMode = CursorBlendMode::Basic;
    float blendPosReturnTime = 0.1f;       // time for velblending to lerp towards target



    // LookaheadDragging state
    std::vector<Rot6d> lookaheadDragLocalRotations6d;         // running pose state for lookahead dragging
    std::vector<Vector3> lookaheadDragLocalPositions; // running position state for lookahead dragging
    bool lookaheadDragInitialized = false;

    // root motion state
    Vector3 rootVelocityWorld = Vector3Zero();  // smoothed linear velocity (world space XZ)
    float rootYawRate = 0.0f;              // smoothed angular velocity (radians/sec)

    // Lookahead dragging for magic anchor
    Vector3 lookaheadDragRootVelocityRootSpace = Vector3Zero();
    float lookaheadDragYawRate = 0.0f;

    // Motion matching feature velocity (updated independently from actual velocity)
    Vector3 virtualControlSmoothedVelocity = Vector3Zero();     // velocity used for motion matching features (world space XZ)
    bool virtualControlSmoothedVelocityInitialized = false;

    // Toe velocity tracking
    Vector3 prevToeGlobalPosPreIK[SIDES_COUNT];   // previous frame positions (before IK)
    Vector3 toeVelocityPreIK[SIDES_COUNT];        // velocity from FK result (before IK)
    Vector3 toeBlendedVelocityWorld[SIDES_COUNT];      // blended from cursor toe velocities
    Vector3 toeBlendedPositionWorld[SIDES_COUNT];      // blended from cursor toe positions
    bool toeTrackingPreIKInitialized = false;

    Vector3 prevToeGlobalPos[SIDES_COUNT];        // previous frame positions (after IK)
    Vector3 toeVelocity[SIDES_COUNT];             // velocity from final pose (after IK)
    bool toeTrackingInitialized = false;

    // Virtual toe positions - move with blended velocity, used for IK targets
    Vector3 virtualToePos[SIDES_COUNT];                 // constrained (speed-clamped) for IK
    Vector3 lookaheadDragToePosRootSpace[SIDES_COUNT];  // unconstrained, drags toward lookahead target (in root space)
    Vector3 lookaheadDragToePosWorld[SIDES_COUNT];      // same as above, but in world space (cached for debug/unlock)
    bool lookaheadDragToePosInitialized = false;

    // Virtual toe locking system
    float virtualToeUnlockTimer[SIDES_COUNT] = { -1.0f, -1.0f };  // -1 = locked, >=0 = unlocked
    float virtualToeUnlockClampRadius[SIDES_COUNT] = { 0.0f, 0.0f };  // current clamp radius for debug viz
    float virtualToeUnlockStartDistance[SIDES_COUNT] = { 0.0f, 0.0f };  // distance at unlock moment (for smooth shrink)   

    PlayerControlInput playerInput;

    // Motion matching query (runtime features computed from current state)
    std::vector<float> mmQuery;

    // Past position history for motion matching
    std::vector<HistoryPoint> positionHistory;
    double lastHistorySampleTime = 0.0f;

};

//----------------------------------------------------------------------------------
// Neural Network state
//----------------------------------------------------------------------------------

struct NetworkState {
    bool isTraining = false;
    float currentLoss = 0.0f;
    int iterations = 0;

    torch::Device device = torch::kCPU;

    torch::nn::Sequential model = nullptr;
    torch::nn::Sequential featuresAutoEncoder = nullptr;
    std::shared_ptr<torch::optim::Adam> optimizer = nullptr;
};