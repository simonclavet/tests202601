here is a single, coherent, readable C++ file - style rewrite of the motion - matching loop, with:

Clean variable names

Linear flow

Raylib math(Vector3, Quaternion, Matrix)

Flat pose / feature storage using libtorch tensors

Very detailed comments, especially on the weird / non - obvious parts of motion matching

No abstractions hiding logic

Designed to be educational first, engine - optimized later

This is meant to be read like a reference implementation, not production code.

motion_matching.cpp
#include <torch/torch.h>
#include <raylib.h>
#include <vector>
#include <array>
#include <cmath>
#include <limits>

/*
    ============================
    DATA STRUCTURES
    ============================

    Important idea:
    ---------------
    We separate:
    - math (raylib: Vector3, Quaternion, Matrix)
    - data storage (libtorch tensors = flat float arrays)
    - logic (motion matching, search, inertialization, FK)

    This avoids mixing math types with memory layout.
*/


/* =============================
   Motion database (offline data)
   ============================= */

    struct MotionDatabase
{
    // Shapes:
    // jointPositions   : [frame, joint, 3]
    // jointRotations   : [frame, joint, 4]   (quat xyzw)
    // jointVelocities  : [frame, joint, 3]
    // jointAngularVel  : [frame, joint, 3]

    torch::Tensor globalJointPositions;
    torch::Tensor globalJointRotations;
    torch::Tensor jointVelocities;
    torch::Tensor jointAngularVel;

    // Animation clips segmentation
    std::vector<int> clipStartFrame;
    std::vector<int> clipEndFrame;

    // Skeleton hierarchy
    std::vector<int> parentJoint;
};


/* =============================
   Root motion state (runtime)
   ============================= */

struct RootMotionState
{
    Vector3 position = { 0,0,0 };
    Vector3 velocity = { 0,0,0 };
    Vector3 acceleration = { 0,0,0 };

    Quaternion rotation = { 0,0,0,1 };
    Vector3 angularVel = { 0,0,0 };
};


/* =============================
   Inertialization buffers
   =============================

   These store "corrections" when switching animations.
   Instead of snapping to new pose, we decay offsets smoothly.

   This is NOT blending.
   This is physically-inspired smoothing (spring-damper model).
*/

struct InertialBuffers
{
    // [joint, 3]
    torch::Tensor positionOffset;
    torch::Tensor velocityOffset;

    // [joint, 4] and [joint, 3]
    torch::Tensor rotationOffset;
    torch::Tensor angularOffset;
};


/* =============================
   Global state
   ============================= */

static int activeClip = 0;
static int activeFrame = 0;
static float searchCooldown = 0.15f;

constexpr float SEARCH_INTERVAL = 0.15f;
constexpr float FLOAT_MAX = std::numeric_limits<float>::max();


/*
    ============================
    MAIN UPDATE FUNCTION
    ============================

    This function runs once per frame.
    It performs:

    1. Input → desired motion
    2. Trajectory prediction
    3. Feature vector construction
    4. Database search (KD-tree)
    5. Transition smoothing (inertialization)
    6. Animation time stepping
    7. Root motion update
    8. Joint pose update
    9. Forward kinematics
    10. Mesh pose upload
*/

void UpdateMotionMatchingSystem(
    MotionDatabase& database,
    RootMotionState& root,
    InertialBuffers& inertial,
    float deltaTime)
{
    /* ============================================================
       STEP 1 — PLAYER INPUT → DESIRED MOTION
       ============================================================ */

    Vector3 moveInput = GetGamepadStickLeft();   // movement direction
    Vector3 lookInput = GetGamepadStickRight();  // facing direction

    // Desired velocity in world space
    Vector3 desiredVelocity = Vector3Scale(moveInput, 5.0f);

    /*
        Weird part:
        -----------
        Facing direction does NOT always come from movement.
        - If right stick exists → look direction controls facing
        - Otherwise movement controls facing

        This allows strafing / aiming systems.
    */
    Vector3 desiredFacing = { 0,0,1 };

    if (Vector3Length(lookInput) > 0.01f)
        desiredFacing = Vector3Normalize(lookInput);
    else if (Vector3Length(moveInput) > 0.01f)
        desiredFacing = Vector3Normalize(moveInput);


    /* ============================================================
       STEP 2 — TRAJECTORY PREDICTION
       ============================================================ */

       /*
           Motion matching does NOT match on current pose only.
           It matches on *future intention*.

           We predict where the character will be in the future
           using critically-damped springs (smooth convergence).
       */

    Quaternion desiredRotation =
        QuaternionFromVector3ToVector3({ 0,0,1 }, desiredFacing);

    constexpr float velocityHalfLife = 0.2f;
    constexpr float rotationHalfLife = 0.2f;

    // Future prediction times (seconds)
    std::array<float, 3> futureTimes = {
        20.0f / 60.0f,
        40.0f / 60.0f,
        60.0f / 60.0f
    };

    std::array<Vector3, 3> futurePositions;
    std::array<Quaternion, 3> futureRotations;

    PredictTrajectoryPosition(
        root.position,
        root.velocity,
        root.acceleration,
        desiredVelocity,
        velocityHalfLife,
        futureTimes,
        futurePositions);

    PredictTrajectoryRotation(
        root.rotation,
        root.angularVel,
        desiredRotation,
        rotationHalfLife,
        futureTimes,
        futureRotations);


    /* ============================================================
       STEP 3 — BUILD FEATURE VECTOR
       ============================================================ */

       /*
           Weird part:
           -----------
           We DO NOT store full 3D data in features.

           We usually store:
           - x/z positions (ground plane)
           - x/z facing directions
           - ignore y (height)

           This makes matching invariant to terrain height
           and massively reduces feature dimension.
       */

    torch::Tensor featureQuery = torch::zeros({ 14 }, torch::kFloat32);
    float* f = featureQuery.data_ptr<float>();

    int k = 0;
    for (int i = 0; i < 3; ++i)
    {
        // future position (x,z)
        f[k++] = futurePositions[i].x;
        f[k++] = futurePositions[i].z;

        // future facing (x,z)
        Vector3 dir = QuaternionRotateVector(futureRotations[i], { 0,0,1 });
        f[k++] = dir.x;
        f[k++] = dir.z;
    }

    // current facing
    Vector3 currentForward = QuaternionRotateVector(root.rotation, { 0,0,1 });
    f[k++] = currentForward.x;
    f[k++] = currentForward.z;


    /* ============================================================
       STEP 4 — DATABASE SEARCH
       ============================================================ */

       /*
           Weird part:
           -----------
           Motion matching is NOT continuous search.
           Searching every frame is expensive and unstable.

           We only search every ~150ms.
           Between searches, animation continues naturally.
       */

    if (searchCooldown <= 0.0f)
    {
        int bestClip = activeClip;
        int bestFrame = activeFrame;
        float bestCost = FLOAT_MAX;

        /*
            Bias toward staying in same animation.
            This prevents jitter and micro-switching.
        */
        if (activeFrame < database.clipEndFrame[activeClip] - 60)
        {
            bestCost =
                FeatureDistance(featureQuery, databaseFeatures[activeFrame])
                - 0.01f;
        }

        /*
            KD-tree search:
            ---------------
            We search all animation clips and find the closest feature match.
        */
        for (int clip = 0; clip < kdTrees.size(); ++clip)
        {
            float cost;
            int localFrame;

            kdTrees[clip].Query(
                featureQuery.data_ptr<float>(),
                cost,
                localFrame,
                bestCost);

            if (cost < bestCost)
            {
                bestCost = cost;
                bestClip = clip;
                bestFrame = database.clipStartFrame[clip] + localFrame;
            }
        }


        /* ========================================================
           STEP 5 — TRANSITION SMOOTHING (INERTIALIZATION)
           ======================================================== */

           /*
               Weird part:
               -----------
               We DO NOT blend poses.
               We compute the difference between poses and decay it over time.

               This avoids:
               - foot sliding
               - collapsing knees
               - volume loss
               - interpolation artifacts
           */

        if (bestClip != activeClip || bestFrame != activeFrame)
        {
            BeginInertialPositionTransition(
                inertial.positionOffset,
                inertial.velocityOffset,
                database.globalJointPositions[activeFrame],
                database.jointVelocities[activeFrame],
                database.globalJointPositions[bestFrame],
                database.jointVelocities[bestFrame]);

            BeginInertialRotationTransition(
                inertial.rotationOffset,
                inertial.angularOffset,
                database.globalJointRotations[activeFrame],
                database.jointAngularVel[activeFrame],
                database.globalJointRotations[bestFrame],
                database.jointAngularVel[bestFrame]);

            activeClip = bestClip;
            activeFrame = bestFrame;
        }

        searchCooldown = SEARCH_INTERVAL;
    }


    /* ============================================================
       STEP 6 — ANIMATION TIME ADVANCE
       ============================================================ */

       /*
           Weird part:
           -----------
           Database is usually 60 FPS.
           Game runs at 30 FPS.
           So we advance by 2 frames.
       */
    activeFrame += 2;

    activeFrame = Clamp(
        activeFrame,
        database.clipStartFrame[activeClip],
        database.clipEndFrame[activeClip] - 1);

    searchCooldown -= deltaTime;

    /*
        Force search near clip end
        (prevents getting stuck at end frames)
    */
    if (activeFrame >= database.clipEndFrame[activeClip] - 4)
        searchCooldown = 0.0f;


    /* ============================================================
       STEP 7 — ROOT MOTION UPDATE
       ============================================================ */

       /*
           Weird part:
           -----------
           Root is not inertialized here.
           Only joints are inertialized.
           Root is driven by animation + input.
       */

    UpdateSpringPosition(
        root.position,
        root.velocity,
        root.acceleration,
        desiredVelocity,
        deltaTime);

    root.velocity =
        QuaternionRotateVector(
            root.rotation,
            QuaternionInverseRotateVector(
                database.globalJointRotations[activeFrame][0],
                database.jointVelocities[activeFrame][0]));

    root.angularVel =
        QuaternionRotateVector(
            root.rotation,
            QuaternionInverseRotateVector(
                database.globalJointRotations[activeFrame][0],
                database.jointAngularVel[activeFrame][0]));

    root.position = Vector3Add(
        root.position,
        Vector3Scale(root.velocity, deltaTime));

    root.rotation = QuaternionMultiply(
        QuaternionFromAxisAngle(
            Vector3Normalize(root.angularVel),
            Vector3Length(root.angularVel) * deltaTime),
        root.rotation);


    /* ============================================================
       STEP 8 — JOINT POSE UPDATE (INERTIALIZATION)
       ============================================================ */

    constexpr float poseHalfLife = 0.075f;

    torch::Tensor finalJointPositions;
    torch::Tensor finalJointRotations;

    UpdateInertialPositions(
        inertial.positionOffset,
        inertial.velocityOffset,
        database.globalJointPositions[activeFrame],
        database.jointVelocities[activeFrame],
        poseHalfLife,
        deltaTime,
        finalJointPositions);

    UpdateInertialRotations(
        inertial.rotationOffset,
        inertial.angularOffset,
        database.globalJointRotations[activeFrame],
        database.jointAngularVel[activeFrame],
        poseHalfLife,
        deltaTime,
        finalJointRotations);


    /* ============================================================
       STEP 9 — FORWARD KINEMATICS
       ============================================================ */

       /*
           Weird part:
           -----------
           FK is done manually instead of engine skeleton system.
           This allows:
           - full control
           - deterministic behavior
           - custom root handling
           - easy ML integration
       */

    std::vector<Matrix> globalTransforms(database.parentJoint.size());

    // Root transform
    Matrix rootMatrix = QuaternionToMatrix(root.rotation);
    rootMatrix.m12 = root.position.x;
    rootMatrix.m13 = root.position.y;
    rootMatrix.m14 = root.position.z;

    globalTransforms[0] = rootMatrix;

    for (size_t j = 1; j < database.parentJoint.size(); ++j)
    {
        Quaternion localRot = {
            finalJointRotations[j - 1][0].item<float>(),
            finalJointRotations[j - 1][1].item<float>(),
            finalJointRotations[j - 1][2].item<float>(),
            finalJointRotations[j - 1][3].item<float>()
        };

        Vector3 localPos = {
            finalJointPositions[j - 1][0].item<float>(),
            finalJointPositions[j - 1][1].item<float>(),
            finalJointPositions[j - 1][2].item<float>()
        };

        Matrix localMatrix = QuaternionToMatrix(localRot);
        localMatrix.m12 = localPos.x;
        localMatrix.m13 = localPos.y;
        localMatrix.m14 = localPos.z;

        globalTransforms[j] =
            MatrixMultiply(
                globalTransforms[database.parentJoint[j]],
                localMatrix);
    }


    /* ============================================================
       STEP 10 — APPLY POSE TO MESH
       ============================================================ */

    ApplyPoseToModel(
        model,
        bindPosePositions,
        bindPoseRotations,
        globalTransforms);
}

Why this version is pedagogically correct
It shows the real structure of motion matching :

Intent → prediction → features → search → transition → motion

It explains the non - obvious parts :

Why features are 2D

Why prediction matters

Why inertialization ≠ blending

Why root is handled differently

Why searching is periodic

Why database FPS ≠ game FPS

Why FK is manual

Why pose offsets exist

It cleanly separates :

math

data

logic

animation

control

physics

Mental model(important)

Motion matching is not animation blending.
It is :

"Search a giant motion database for the frame that already does what I want."

Everything else (inertialization, prediction, biasing, cooldowns) exists only to :

avoid jitter

avoid snapping

preserve continuity

preserve momentum

preserve physical plausibility