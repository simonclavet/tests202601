#pragma once

#include <thread>
#include <atomic>
#include <mutex>

//#include "raylib.h"

constexpr int PCA_SEGMENT_K = 150;
constexpr int PCA_FEATURE_K = 20;

// how many future poses the glimpse flow predicts, and at what times
constexpr int GLIMPSE_POSE_COUNT = 2;
constexpr float GLIMPSE_POSE_TIMES[GLIMPSE_POSE_COUNT] = { 0.1f, 0.3f };

constexpr int GLIMPSE_TOE_PCA_K = 10;

// Structured representation of the glimpse flow's toe output.
// For each future time step and each side (left/right), stores position and velocity in root space XZ.
struct GlimpseFeatures
{
    struct ToeSample
    {
        float posX = 0.0f;
        float posZ = 0.0f;
        float velX = 0.0f;
        float velZ = 0.0f;
    };

    // toes[timeStep][side]: predicted toe position/velocity in current root space
    ToeSample toes[GLIMPSE_POSE_COUNT][SIDES_COUNT] = {};

    static constexpr int GetDim() { return GLIMPSE_POSE_COUNT * SIDES_COUNT * 4; }  // 16

    void SerializeTo(float* dest) const
    {
        int idx = 0;
        for (int t = 0; t < GLIMPSE_POSE_COUNT; ++t)
        {
            for (int side = 0; side < SIDES_COUNT; ++side)
            {
                const ToeSample& s = toes[t][side];
                dest[idx++] = s.posX;
                dest[idx++] = s.posZ;
                dest[idx++] = s.velX;
                dest[idx++] = s.velZ;
            }
        }
        assert(idx == GetDim());
    }

    void DeserializeFrom(const float* src)
    {
        int idx = 0;
        for (int t = 0; t < GLIMPSE_POSE_COUNT; ++t)
        {
            for (int side = 0; side < SIDES_COUNT; ++side)
            {
                ToeSample& s = toes[t][side];
                s.posX = src[idx++];
                s.posZ = src[idx++];
                s.velX = src[idx++];
                s.velZ = src[idx++];
            }
        }
        assert(idx == GetDim());
    }
};

constexpr int GLIMPSE_TOE_RAW_DIM = GlimpseFeatures::GetDim();  // 16

// Animation playback mode for controlled character
enum class AnimationMode : int
{
    RandomSwitch = 0,              // randomly switch between animations
    MotionMatching,                // use motion matching to find best animation
    GlimpseMode,                   // flow-sampled future pose -> deterministic segment decompression
    COUNT
};

static inline const char* AnimationModeName(AnimationMode mode)
{
    switch (mode)
    {
    case AnimationMode::RandomSwitch: return "Random Switch";
    case AnimationMode::MotionMatching: return "Motion Matching";
    case AnimationMode::GlimpseMode: return "Glimpse Mode";
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
    ToePosDiff,          // left-right difference (X,Z) => 2 dims
    FutureVel,           // future root velocity (XZ) at sample points => 2 * points
    FutureVelClamped,    // future root velocity clamped to max magnitude (XZ) => 2 * points
    FutureSpeed,         // future root speed (scalar) at sample points => 1 * points
    PastPosition,        // past hip position (XZ) in current hip horizontal frame => 2 dims
    FutureAimDirection,  // aim direction at trajectory times => 2 * points
    FutureAimVelocity,   // aim angular velocity (rad/s around Y) at trajectory times => 1 * points
    HeadToSlowestToe,    // head to slowest foot vector (XZ) in root space => 2 dims
    HeadToToeAverage,    // head to average of both toe positions (XZ) in root space => 2 dims
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
    case FeatureType::ToePosDiff: return "Toe Pos Difference";
    case FeatureType::FutureVel: return "Future Velocity";
    case FeatureType::FutureVelClamped: return "Future Vel Clamped";
    case FeatureType::FutureSpeed: return "Future Speed";
    case FeatureType::PastPosition: return "Past Position";
    case FeatureType::FutureAimDirection: return "Future Aim Direction";
    case FeatureType::FutureAimVelocity: return "Future Aim Velocity";
    case FeatureType::HeadToSlowestToe: return "Head To Slowest Toe";
    case FeatureType::HeadToToeAverage: return "Head To Toe Average";
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

enum class AimDirectionMode : int
{
    HeadToRightHand = 0,
    HeadDirection,
    HipsDirection,
    COUNT
};

static inline const char* AimDirectionModeName(
    AimDirectionMode mode)
{
    switch (mode)
    {
    case AimDirectionMode::HeadToRightHand:
        return "Head to Right Hand";
    case AimDirectionMode::HeadDirection:
        return "Head Direction";
    case AimDirectionMode::HipsDirection:
        return "Hips Direction";
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

    // What direction the AimDirection feature tracks
    AimDirectionMode aimDirectionMode =
        AimDirectionMode::HeadToRightHand;

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

    // Character visibility
    bool showControlledCharacter = true;
    bool showSequenceCharacters = true;

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
    bool drawLookaheadPose = false;  // Draw blended lookahead pose skeleton
    bool drawBasicBlend = false;     // Draw basic blend result (before lookahead dragging)
    bool drawMagicAnchor = false;    // Draw Magic anchor (spine3 projected + head→hand yaw)
    bool drawPastHistory = false;    // Draw past position history for motion matching

    // Animation settings
    AnimationMode animationMode = AnimationMode::RandomSwitch;  // animation playback mode
    float defaultBlendTime = 0.1f;  // time for blend cursor spring to reach 95% of target
    float switchInterval = 3.0f;    // time between random animation switches
    float mmSearchPeriod = 0.1f;    // time between motion matching searches
    float inputDecidedSearchPeriod = 0.2f;  // cooldown for input-decided early searches
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
    bool testPcaReconstruction = false;

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
struct PoseFeatures
{
    // Lookahead pose (what the pose will be after lookahead time)
    std::vector<Rot6d> lookaheadLocalRotations;   // [jointCount]
    Vector3 rootLocalPosition;                     // bone 0 only (relative to magic anchor)
    // bones 1+ local positions are constant skeleton offsets, not stored here

    // Root motion (lookahead for velocity, current for yaw rate)
    Vector3 lookaheadRootVelocity;                 // root velocity in root space (lookahead)
    float rootYawRate;                             // current yaw rate (rad/s)

    // Foot IK data (lookahead positions, current velocities for speed clamping)
    Vector3 lookaheadToePositionsRootSpace[SIDES_COUNT];  // [left, right]
    Vector3 toeVelocitiesRootSpace[SIDES_COUNT];          // [left, right] current velocities

    // 2D (XZ) difference between left and right toe positions in root space (current frame)
    // crisp direct signal from joint world positions — doesn't go through FK chain
    Vector3 toePosDiffRootSpace;                              // only x,z used; serialized as 2 floats

    // magnitude of XZ speed difference between left and right toes in root space
    // always >= 0, so the network can't regress it to zero without penalty
    // at runtime we pick the fast foot from blended velocities, then enforce this contrast
    float toeSpeedDiff = 0.0f;

    // flat layout: [jc*6 rotations] [3 rootPos] [2 rootVelXZ] [1 yawRate] [6 toePos] [6 toeVel] [2 toePosDiff] [1 toeSpeedDiff]
    static int GetDim(int jointCount)
    {
        int dim = 0;
        dim += jointCount * 6;  // lookaheadLocalRotations (Rot6d per joint)
        dim += 3;               // rootLocalPosition (bone 0 only)
        dim += 2;               // lookaheadRootVelocity (horizontal XZ only)
        dim += 1;               // rootYawRate (float)
        dim += 3 * 2;           // lookaheadToePositionsRootSpace (Vector3 x 2 sides)
        dim += 3 * 2;           // toeVelocitiesRootSpace (Vector3 x 2 sides)
        dim += 2;               // toePosDiffRootSpace (XZ only)
        dim += 1;               // toeSpeedDiff (scalar)
        return dim;             // = jc*6 + 21
    }

    void Resize(int jointCount)
    {
        lookaheadLocalRotations.resize(jointCount);
        rootLocalPosition = Vector3Zero();
        lookaheadRootVelocity = Vector3Zero();
        rootYawRate = 0.0f;
        for (int side : sides)
        {
            lookaheadToePositionsRootSpace[side] = Vector3Zero();
            toeVelocitiesRootSpace[side] = Vector3Zero();
        }
        toePosDiffRootSpace = Vector3Zero();
        toeSpeedDiff = 0.0f;
    }

    void SerializeTo(std::span<float> dest) const
    {
        int idx = 0;

        for (const Rot6d& rot : lookaheadLocalRotations)
        {
            dest[idx++] = rot.ax;
            dest[idx++] = rot.ay;
            dest[idx++] = rot.az;
            dest[idx++] = rot.bx;
            dest[idx++] = rot.by;
            dest[idx++] = rot.bz;
        }

        dest[idx++] = rootLocalPosition.x;
        dest[idx++] = rootLocalPosition.y;
        dest[idx++] = rootLocalPosition.z;

        dest[idx++] = lookaheadRootVelocity.x;
        dest[idx++] = lookaheadRootVelocity.z;

        dest[idx++] = rootYawRate;

        for (int side : sides)
        {
            dest[idx++] = lookaheadToePositionsRootSpace[side].x;
            dest[idx++] = lookaheadToePositionsRootSpace[side].y;
            dest[idx++] = lookaheadToePositionsRootSpace[side].z;
        }

        for (int side : sides)
        {
            dest[idx++] = toeVelocitiesRootSpace[side].x;
            dest[idx++] = toeVelocitiesRootSpace[side].y;
            dest[idx++] = toeVelocitiesRootSpace[side].z;
        }

        dest[idx++] = toePosDiffRootSpace.x;
        dest[idx++] = toePosDiffRootSpace.z;

        dest[idx++] = toeSpeedDiff;

        assert(idx == GetDim((int)lookaheadLocalRotations.size()));
    }

    void DeserializeFrom(std::span<const float> src, int jointCount)
    {
        Resize(jointCount);
        int idx = 0;

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

        rootLocalPosition.x = src[idx++];
        rootLocalPosition.y = src[idx++];
        rootLocalPosition.z = src[idx++];

        lookaheadRootVelocity.x = src[idx++];
        lookaheadRootVelocity.y = 0.0f;
        lookaheadRootVelocity.z = src[idx++];

        rootYawRate = src[idx++];

        for (int side : sides)
        {
            lookaheadToePositionsRootSpace[side].x = src[idx++];
            lookaheadToePositionsRootSpace[side].y = src[idx++];
            lookaheadToePositionsRootSpace[side].z = src[idx++];
        }

        for (int side : sides)
        {
            toeVelocitiesRootSpace[side].x = src[idx++];
            toeVelocitiesRootSpace[side].y = src[idx++];
            toeVelocitiesRootSpace[side].z = src[idx++];
        }

        toePosDiffRootSpace.x = src[idx++];
        toePosDiffRootSpace.y = 0.0f;
        toePosDiffRootSpace.z = src[idx++];

        toeSpeedDiff = src[idx++];

        assert(idx == GetDim((int)lookaheadLocalRotations.size()));

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
    std::vector<int> legalStartFrames;

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

    // precomputed aim direction per frame (unit vector in anim-world XZ plane)
    std::vector<Vector3> aimDirectionAnimSpace;   // [motionFrameCount]
    std::vector<float> aimYawRate;                // [motionFrameCount] rad/s

    // Hip transform relative to Magic anchor (for placing skeleton when using Magic root motion)
    std::vector<Vector3> hipPositionInMagicSpace;        // [motionFrameCount] - hip offset from magic, in magic-heading space
    std::vector<Rot6d> hipRotationInMagicSpace;          // [motionFrameCount] - full hip rotation relative to magic yaw
    std::vector<std::string> featureNames;
    std::vector<FeatureType> featureTypes;      // which FeatureType each feature dimension belongs to

    std::vector<float> featuresMean;            // mean of each feature dimension [featureDim]
    std::vector<float> featuresMin;             // per-dim min from mocap data [featureDim]
    std::vector<float> featuresMax;             // per-dim max from mocap data [featureDim]
    float featureTypesStd[static_cast<int>(FeatureType::COUNT)] = {};  // std shared by all features of same type
    Array2D<float> normalizedFeatures;          // normalized features [motionFrameCount x featureDim]

    // Neural network training targets: pose generation features [motionFrameCount x poseGenFeaturesComputeDim]
    // Contains lookahead local rotations, positions, root motion, and foot IK data
    // This is what the network should output given the motion matching features as input
    int poseGenFeaturesComputeDim = -1;        
    Array2D<float> poseGenFeatures;            // [motionFrameCount x poseGenFeaturesComputeDim]
    float poseGenFeaturesSegmentLength = 0.3f; // how many seconds of poseGenFeatures a cursor copies

    // normalization for poseGenFeatures (for segment autoencoder training)
    std::vector<float> poseGenFeaturesMean;    // per-dim mean [poseGenFeaturesComputeDim]
    std::vector<float> poseGenFeaturesStd;     // per-dim std [poseGenFeaturesComputeDim]
    std::vector<float> poseGenFeaturesMin;     // per-dim min from mocap data [poseGenFeaturesComputeDim]
    std::vector<float> poseGenFeaturesMax;     // per-dim max from mocap data [poseGenFeaturesComputeDim]
    std::vector<float> poseGenFeaturesWeight;  // per-dim bone weight [poseGenFeaturesComputeDim]
    Array2D<float> normalizedPoseGenFeatures;  // (raw - mean) / std * weight [motionFrameCount x poseGenFeaturesComputeDim]

    // segment autoencoder sizing (computed at build time from frame rate and segmentLength)
    int poseGenSegmentFrameCount = -1;         // how many frames in a segment
    int poseGenSegmentFlatDim = -1;            // poseGenSegmentFrameCount * poseGenFeaturesComputeDim

    // PCA on flat normalized segments (computed at build time)
    int pcaSegmentK = -1;                       // number of PCA components kept
    std::vector<float> pcaSegmentMean;          // [flatDim] mean of flat normalized segments
    std::vector<float> pcaSegmentBasis;         // [K * flatDim] row-major, each row is a principal component

    // PCA on glimpse toe data (2 times x 2 toes x (pos_xz + vel_xz) = 16 raw dims -> 8)
    int pcaGlimpseToeK = -1;                           // number of PCA components kept
    std::vector<float> pcaGlimpseToeMean;              // [GLIMPSE_TOE_RAW_DIM]
    std::vector<float> pcaGlimpseToeBasis;             // [K * GLIMPSE_TOE_RAW_DIM] row-major

    // PCA on normalized MM features (for compact glimpse conditioning)
    int pcaFeatureK = -1;                              // number of PCA components kept
    std::vector<float> pcaFeatureMean;                 // [featureDim]
    std::vector<float> pcaFeatureBasis;                // [K * featureDim] row-major

    // precomputed per-frame training data (computed once, used by training loops)
    Array2D<float> precompSegmentPCA;    // [motionFrameCount x pcaSegmentK] segment PCA coefficients
    Array2D<float> precompFeaturePCA;    // [motionFrameCount x pcaFeatureK] feature PCA coefficients
    Array2D<float> precompRawToe;        // [motionFrameCount x GLIMPSE_TOE_RAW_DIM] raw toe data

    // k-means clusters on normalizedFeatures for stratified training sampling
    // clusterFrames[c] holds the global frame indices belonging to cluster c
    int clusterCount = 0;
    std::vector<std::vector<int>> clusterFrames;
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
    db->legalStartFrames.clear();
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
    db->aimDirectionAnimSpace.clear();
    db->aimYawRate.clear();
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
    db->featuresMin.clear();
    db->featuresMax.clear();
    for (int i = 0; i < static_cast<int>(FeatureType::COUNT); ++i) db->featureTypesStd[i] = 0.0f;
    db->normalizedFeatures.clear();
    db->poseGenFeaturesComputeDim = -1;
    db->poseGenFeatures.clear();
    db->poseGenFeaturesMean.clear();
    db->poseGenFeaturesStd.clear();
    db->poseGenFeaturesMin.clear();
    db->poseGenFeaturesMax.clear();
    db->poseGenFeaturesWeight.clear();
    db->normalizedPoseGenFeatures.clear();
    db->poseGenSegmentFrameCount = -1;
    db->poseGenSegmentFlatDim = -1;
    db->pcaGlimpseToeK = -1;
    db->pcaGlimpseToeMean.clear();
    db->pcaGlimpseToeBasis.clear();
    db->pcaFeatureK = -1;
    db->pcaFeatureMean.clear();
    db->pcaFeatureBasis.clear();
    db->precompSegmentPCA.clear();
    db->precompFeaturePCA.clear();
    db->precompRawToe.clear();
    db->clusterCount = 0;
    db->clusterFrames.clear();
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
    float segmentAnimTime = 0.0f;               // playback time within the segment (0 to segmentMaxTime)
    DoubleSpringDamperState weightSpring = {};  // spring state for weight blending (x = current weight)
    float normalizedWeight = 0.0f;              // weight / totalWeight (sums to 1 across active cursors)
    DoubleSpringDamperState fastWeightSpring = {}; // faster spring for yaw rate
    float fastNormalizedWeight = 0.0f;          // fastWeight / totalFastWeight
    float targetWeight = 0.0f;                  // desired weight
    float blendTime = 0.3f;                     // halflife for double spring damper
    bool active = false;                        // is cursor in use


    std::vector<Rot6d> localRotations6d;
    std::vector<Rot6d> lookaheadRotations6d;  // extrapolated pose for lookahead dragging
    Vector3 rootLocalPosition;


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

    // poseGenFeatures segment: a local copy of N frames of poseGenFeatures data
    // segmentAnimTime is relative to segment start (0 to segmentMaxTime)
    // use segment.rows() for frame count, segment.cols() for dim
    Array2D<float> segment;              // [segmentFrameCount x poseGenFeaturesComputeDim]
    float segmentFrameTime = 0.0f;       // seconds per frame
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
    Vector3 prevDesiredVelocity = Vector3Zero();
    Vector3 prevDesiredAimDirection = { 0.0f, 0.0f, 1.0f };
    float prevInputUpdateTimer = 0.0f;
    float inputDecidedSearchCooldown = 0.0f;

    // Pose output (local space with root zeroed, then transformed to world)
    TransformData xformData;


    // Pre-IK FK state for debugging (saved before IK is applied)
    TransformData xformBeforeIK;
    bool debugSaveBeforeIK;  // toggle to enable saving pre-IK state

    // Basic blend result (before lookahead dragging) for debugging
    TransformData xformBasicBlend;

    // Blended lookahead pose for debugging
    TransformData xformLookahead;

    // Visual properties
    Color color;
    float opacity;
    float radius;
    float scale;

    // Copy of skeleton (so we don't hold a pointer into a resizable vector)
    BVHData skeleton;
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
    Vector3 toeBlendedVelocityWorld[SIDES_COUNT];      // blended from cursor toe velocities (current frame)
    Vector3 toeBlendedPositionWorld[SIDES_COUNT];      // blended from cursor toe positions
    float blendedToeSpeedDiff = 0.0f;                 // from network-predicted pose features
    Vector3 toeBlendedPosDiffRootSpace;                // blended (left-right) toe diff from segments
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

    // Glimpse flow predicted toe data (updated each prediction)
    GlimpseFeatures predictedGlimpse = {};
    bool predictedGlimpseValid = false;

    // Motion matching query (runtime features computed from current state)
    std::vector<float> mmQuery;

    // Past position history for motion matching
    std::vector<HistoryPoint> positionHistory;
    double lastHistorySampleTime = 0.0f;

};

// Glimpse flow: noise -> pose PCA coefficients, conditioned on MM features + time
// deeper than FullFlowModel: 256 -> 512 -> 256 with condition re-injected at each layer
struct GlimpseFlowModelImpl : torch::nn::Module
{
    torch::nn::Linear layer1{nullptr};
    torch::nn::Linear layer2{nullptr};
    torch::nn::Linear layer3{nullptr};
    torch::nn::Linear outputLayer{nullptr};
    int condTimeDim = 0;

    GlimpseFlowModelImpl(int featureDim, int posePcaDim);
    torch::Tensor forward(const torch::Tensor& xt,
                          const torch::Tensor& condTime);
};
TORCH_MODULE(GlimpseFlowModel);

// Glimpse decompressor: (futurePose, features) -> segment, features re-injected at each layer
struct GlimpseDecompressorModelImpl : torch::nn::Module
{
    torch::nn::Linear layer1{nullptr};
    torch::nn::Linear layer2{nullptr};
    torch::nn::Linear layer3{nullptr};
    torch::nn::Linear outputLayer{nullptr};
    int condDim = 0;
    int poseDim = 0;

    GlimpseDecompressorModelImpl(int featureDim, int pgDim, int segmentFlatDim);
    torch::Tensor forward(const torch::Tensor& futurePose,
                          const torch::Tensor& cond);
};
TORCH_MODULE(GlimpseDecompressorModel);

//----------------------------------------------------------------------------------
// Neural Network state
//----------------------------------------------------------------------------------

struct NetworkState {
    bool isTraining = false;
    torch::Device device = torch::kCPU;

    // glimpse flow: noise -> pose PCA coefficients at 0.3s, conditioned on MM features
    GlimpseFlowModel glimpseFlow = nullptr;
    std::shared_ptr<torch::optim::Adam> glimpseFlowOptimizer = nullptr;
    float glimpseFlowLoss = 0.0f;
    float glimpseFlowLossSmoothed = 0.0f;
    int glimpseFlowIterations = 0;
    std::vector<float> glimpseFlowLossHistory;

    // per-dim std of normalizedPoseGenFeatures (for noise scaling in glimpse flow)
    std::vector<float> glimpseTargetStd;

    // glimpse decompressor: (MM features, future pose) -> full segment
    GlimpseDecompressorModel glimpseDecompressor = nullptr;
    std::shared_ptr<torch::optim::Adam> glimpseDecompressorOptimizer = nullptr;
    float glimpseDecompressorLoss = 0.0f;
    float glimpseDecompressorLossSmoothed = 0.0f;
    int glimpseDecompressorIterations = 0;
    std::vector<float> glimpseDecompressorLossHistory;

    // loss history — one sample every LOSS_LOG_INTERVAL_SECONDS,
    // all networks logged at the same time so curves are directly comparable
    std::vector<float> lossHistoryTime;         // training elapsed seconds
    double timeSinceLastLossLog = 0.0;

    // unified training timing
    double trainingElapsedSeconds = 0.0;
    double timeSinceLastAutoSave = 0.0;

    // model architecture dimensions (stored at init, needed for cloning)
    int glimpseFlowCondDim = 0;
    int glimpseFlowGlimpseDim = 0;
    int glimpseDecompCondDim = 0;
    int glimpseDecompGlimpseDim = 0;
    int glimpseDecompSegK = 0;
};

//----------------------------------------------------------------------------------
// Training thread control
//----------------------------------------------------------------------------------

struct TrainingThreadControl
{
    std::thread thread;

    // main → training thread signals
    std::atomic<bool> stopRequested = false;

    // training → main thread status (atomics for lock-free reads)
    std::atomic<bool> isRunning = false;
    std::atomic<float> glimpseFlowLossSmoothed = 0.0f;
    std::atomic<int> glimpseFlowIterations = 0;
    std::atomic<float> glimpseDecompressorLossSmoothed = 0.0f;
    std::atomic<int> glimpseDecompressorIterations = 0;
    std::atomic<double> trainingElapsedSeconds = 0.0;

    // staged weights: training thread writes under mutex, main thread consumes
    std::mutex stagingMutex;
    bool stagingReady = false;
    GlimpseFlowModel stagedGlimpseFlow = nullptr;
    GlimpseDecompressorModel stagedGlimpseDecompressor = nullptr;
    std::vector<float> stagedGlimpseTargetStd;

    // staged loss history (copied under same mutex)
    std::vector<float> stagedLossHistoryTime;
    std::vector<float> stagedGlimpseFlowLossHistory;
    std::vector<float> stagedGlimpseDecompressorLossHistory;
    bool lossHistoryDirty = false;
};
