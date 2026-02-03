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
    VelBlending,         // velocity-driven blending with lerp to target
    LookaheadDragging,   // lerp towards extrapolated future pose
    COUNT
};

static inline const char* CursorBlendModeName(CursorBlendMode mode)
{
    switch (mode)
    {
    case CursorBlendMode::Basic: return "Basic";
    case CursorBlendMode::VelBlending: return "Vel Blending";
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
    FutureVel,        // future root velocity (XZ) at sample points => 2 * points
    PastPosition,        // past hip position (XZ) in current hip horizontal frame => 2 dims

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
    case FeatureType::PastPosition: return "Past Position";
    default: return "Unknown";
    }
}

// Descriptor for a single feature type
struct FeatureTypeDescriptor
{
    float weight = 1.0f;  // Weight/importance of this feature in matching (0 = disabled)
};

struct MotionMatchingFeaturesConfig
{
    // Per-feature-type descriptors (indexed by FeatureType enum)
    FeatureTypeDescriptor features[static_cast<int>(FeatureType::COUNT)];

    // Future trajectory configuration (only used if FutureVelDir weight > 0)
    std::vector<float> futureTrajPointTimes = { 0.2f, 0.4f, 0.8f };  // time offsets (seconds)

    // Past pose configuration
    float pastTimeOffset = 0.1f;  // time offset for past position feature (seconds)

    // Helper: check if a feature is enabled (weight > 0)
    bool IsFeatureEnabled(FeatureType type) const
    {
        const int idx = static_cast<int>(type);
        return features[idx].weight > 0.0f;
    }

    // Helper: get weight for a feature type
    float GetFeatureWeight(FeatureType type) const
    {
        const int idx = static_cast<int>(type);
        return features[idx].weight;
    }

    // Helper: set weight for a feature type
    void SetFeatureWeight(FeatureType type, float weight)
    {
        const int idx = static_cast<int>(type);
        features[idx].weight = weight;
    }
};

// Initialize motion matching config with default values
static inline void MotionMatchingConfigInit(MotionMatchingFeaturesConfig& config)
{
    // Initialize all feature weights to 1.0
    for (int i = 0; i < static_cast<int>(FeatureType::COUNT); ++i)
    {
        config.features[i].weight = 1.0f;
    }

    // Default future trajectory sample times
    config.futureTrajPointTimes = { 0.2f, 0.4f, 0.8f };
}



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
    int cameraMode = 1;  // 0 = Orbit, 1 = Unreal
    float cameraTargetBlendtime = 0.2f;  // Smooth target following blendtime for Orbit and LazyTurretFollower

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
    bool drawPastHistory = false;         // Draw foot IK debug (virtual toe positions, etc.)

    // Animation settings
    AnimationMode animationMode = AnimationMode::RandomSwitch;  // animation playback mode
    float defaultBlendTime = 0.1f;  // time for blend cursor spring to reach 95% of target
    float switchInterval = 3.0f;    // time between random animation switches
    float mmSearchPeriod = 0.1f;    // time between motion matching searches
    float poseDragLookaheadTime = 0.1f;  // lookahead time for pose dragging (seconds)

    // Cursor blend mode settings
    CursorBlendMode cursorBlendMode = CursorBlendMode::Basic;
    float blendPosReturnTime = 0.1f; // time for velblending to lerp towards target

    // Foot IK
    bool enableFootIK = true;  // enable/disable foot IK towards virtual toe positions
    bool enableTimedUnlocking = true;  // enable/disable timed unlock mechanism for virtual toes
    float unlockDistance = 0.2f;  // distance threshold to unlock virtual toe (meters)
    float unlockDuration = 0.3f;  // time to gradually re-lock virtual toe (seconds)
    
    
    bool drawPlayerInput = false;

    // Motion Matching Configuration
    MotionMatchingFeaturesConfig mmConfigEditor; // version for edition: use featureConfig in animdatabase for the one actually used


    // Validity
    bool valid = false;
};


struct PlayerControlInput 
{
    Vector3 desiredVelocity = Vector3Zero();  // Desired velocity in world space (XZ plane)
    float maxSpeed = 2.0f;                     // Maximum movement speed (m/s)
};



//----------------------------------------------------------------------------------
// Animation Database
//----------------------------------------------------------------------------------

// A unified view of all loaded animations, suitable for sampling by ControlledCharacter.
struct AnimDatabase
{
    // Motion matching feature configuration
    MotionMatchingFeaturesConfig featuresConfig;

    // Pose drag lookahead time (seconds) - used for precomputing lookahead poses
    float poseDragLookaheadTime = 0.1f;

    // References to all loaded animations
    int animCount = -1;

    // Per-animation info
    std::vector<int> animStartFrame;   // Global frame index where each anim starts
    std::vector<int> animFrameCount;   // Number of frames in each anim
    std::vector<float> animFrameTime;  // Frame time for each anim (usually same)

    // Total frames across all animations
    int totalFrames = -1;

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

    // Per-frame joint transforms: [motionFrameCount x jointCount]
    Array2D<Vector3> globalJointPositions;      // global positions
    Array2D<Quaternion> globalJointRotations;   // global rotations
    Array2D<Vector3> globalJointVelocities;     // velocities (defined at midpoint between frames)
    Array2D<Vector3> globalJointAccelerations;  // accelerations (derivative of velocity)

    // local joint transforms (for blending without global->local conversion)
    Array2D<Vector3> localJointPositions;           // local positions [motionFrameCount x jointCount]
    Array2D<Rot6d> localJointRotations6d;           // local rotations as Rot6d [motionFrameCount x jointCount]
    Array2D<Vector3> localJointAngularVelocities;   // local angular velocities [motionFrameCount x jointCount]

    // lookahead pose for inertial dragging: pose[f] + 3*(pose[f] - pose[f-1])
    Array2D<Rot6d> lookaheadLocalRotations6d;       // [motionFrameCount x jointCount]

    // root motion velocities (stored separately for clarity)
    // These are linear XZ velocities defined at frame midpoints
    std::vector<Vector3> rootMotionVelocities;     // [motionFrameCount] - XZ velocity only
    std::vector<float> rootMotionYawRates;         // [motionFrameCount] - yaw angular velocity (rad/s)

    // lookahead root motion velocities (extrapolated for smooth anticipation)
    std::vector<Vector3> lookaheadRootMotionVelocities;  // [motionFrameCount] - extrapolated XZ velocity
    std::vector<float> lookaheadRootMotionYawRates;      // [motionFrameCount] - extrapolated yaw rate (rad/s)

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
    std::vector<std::string> featureNames;
    std::vector<FeatureType> featureTypes;      // which FeatureType each feature dimension belongs to

    std::vector<float> featuresMean;            // mean of each feature dimension [featureDim]
    std::vector<float> featuresStd;             // standard deviation of each feature dimension [featureDim]
    Array2D<float> normalizedFeatures;          // normalized features [motionFrameCount x featureDim]
};


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
    db->lookaheadLocalRotations6d.clear();
    db->rootMotionVelocities.clear();
    db->rootMotionYawRates.clear();
    db->lookaheadRootMotionVelocities.clear();
    db->lookaheadRootMotionYawRates.clear();
    db->clipStartFrame.clear();
    db->clipEndFrame.clear();
    db->features.clear();
    db->featureDim = 0;
    db->featureNames.clear();
    db->featureTypes.clear();
    db->featuresMean.clear();
    db->featuresStd.clear();
    db->normalizedFeatures.clear();
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
    float targetWeight = 0.0f;                  // desired weight
    float blendTime = 0.3f;                     // halflife for double spring damper
    bool active = false;                        // is cursor in use

    // Local-space pose stored per cursor for blending (size = jointCount)
    std::vector<Vector3> localPositions;
    std::vector<Rot6d> localRotations6d;
    std::vector<Vector3> localAngularVelocities;
    std::vector<Rot6d> lookaheadRotations6d;  // extrapolated pose for lookahead dragging

    // Sampled root motion velocities from database (animation space, before world transform)
    Vector3 sampledRootVelocityAnim = Vector3Zero();  // XZ velocity in animation space
    float sampledRootYawRate = 0.0f;                   // yaw rate (rad/s)

    // Sampled lookahead root motion velocities (extrapolated for anticipation)
    Vector3 sampledLookaheadRootVelocityAnim = Vector3Zero();  // lookahead XZ velocity
    float sampledLookaheadRootYawRate = 0.0f;                   // lookahead yaw rate (rad/s)

    // Global-space pose for debug visualization (computed via FK after sampling)
    std::vector<Vector3> globalPositions;
    std::vector<Quaternion> globalRotations;

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
    // World placement (accumulated from root motion)
    Vector3 worldPosition;
    Quaternion worldRotation;  // Character's facing direction (Y-axis rotation only)

    // Animation mode
    AnimationMode animMode = AnimationMode::RandomSwitch;

    // Animation state
    int animIndex;             // Which loaded animation to play from
    float animTime;            // Current playback time in that animation

    // For computing root motion deltas between frames
    Vector3 prevRootPosition;
    Quaternion prevRootRotation;

    // Random switch timer (used by RandomSwitch mode)
    float switchTimer;

    // Motion matching state
    int mmBestFrame = -1;           // best matching frame from last search
    float mmBestCost = 0.0f;        // cost of best match
    float mmSearchTimer = 0.0f;     // time since last search

    // Pose output (local space with root zeroed, then transformed to world)
    TransformData xformData;
    TransformData xformTmp0;
    TransformData xformTmp1;
    TransformData xformTmp2;
    TransformData xformTmp3;

    // Pre-IK FK state for debugging (saved before IK is applied)
    TransformData xformBeforeIK;
    bool debugSaveBeforeIK;  // toggle to enable saving pre-IK state

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

    // Cursor blend mode and state
    CursorBlendMode cursorBlendMode = CursorBlendMode::Basic;
    float blendPosReturnTime = 0.1f;       // time for velblending to lerp towards target

    // VelBlending state
    std::vector<Rot6d> velBlendedRotations6d;       // [jointCount] - smoothed rotations
    bool velBlendInitialized = false;

    // LookaheadDragging state
    std::vector<Rot6d> lookaheadDragPose6d;         // running pose state for lookahead dragging
    bool lookaheadDragInitialized = false;

    // Lookahead root motion velocity state
    Vector3 lookaheadDragVelocity = Vector3Zero();   // running velocity state for lookahead dragging (world space)
    float lookaheadDragYawRate = 0.0f;               // running yaw rate for lookahead dragging (rad/s)
    bool lookaheadVelocityInitialized = false;

    // Smoothed root motion state
    Vector3 smoothedRootVelocity = Vector3Zero();  // smoothed linear velocity (world space XZ)
    float smoothedRootYawRate = 0.0f;              // smoothed angular velocity (radians/sec)
    bool rootMotionInitialized = false;

    // Toe velocity tracking
    Vector3 prevToeGlobalPosPreIK[SIDES_COUNT];   // previous frame positions (before IK)
    Vector3 toeVelocityPreIK[SIDES_COUNT];        // velocity from FK result (before IK)
    Vector3 toeBlendedVelocity[SIDES_COUNT];      // blended from cursor global toe velocities
    bool toeTrackingPreIKInitialized = false;

    Vector3 prevToeGlobalPos[SIDES_COUNT];        // previous frame positions (after IK)
    Vector3 toeVelocity[SIDES_COUNT];             // velocity from final pose (after IK)
    bool toeTrackingInitialized = false;

    // Virtual toe positions - move with blended velocity, used for IK targets
    Vector3 virtualToePos[SIDES_COUNT];
    Vector3 intermediateVirtualToePos[SIDES_COUNT];  // velocity + viscous return (no speed clamp)

    bool virtualToeInitialized = false;

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