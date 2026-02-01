#pragma once

#include <vector>
#include <cassert>

#include "raylib.h"
#include "math_utils.h"
#include "utils.h"
#include "transform_data.h"
#include "bvh_parser.h"
#include "anim_database.h"
#include "character_data.h"
#include "leg_ik.h"
#include "app_config.h"

using std::vector;


//----------------------------------------------------------------------------------
// Player Control Input
//----------------------------------------------------------------------------------

struct PlayerControlInput {
    Vector3 desiredVelocity = Vector3Zero();  // Desired velocity in world space (XZ plane)
    float maxSpeed = 2.0f;                     // Maximum movement speed (m/s)
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

    // Velocity-based blending state (advances with blended angular velocity, lerps to target)
    vector<Rot6d> velBlendedRotations6d;       // [jointCount] - smoothed rotations
    float blendPosReturnTime = 0.1f;       // time to reach halfway to target (lower = snappier)
    bool velBlendInitialized = false;
    bool useVelBlending = false;               // toggle for enabling velocity-based blending

    // Smoothed root motion state
    Vector3 smoothedRootVelocity = Vector3Zero();  // smoothed linear velocity (world space XZ)
    float smoothedRootYawRate = 0.0f;              // smoothed angular velocity (radians/sec)
    bool rootMotionInitialized = false;

    // Toe velocity tracking (for foot IK)
    Vector3 prevToeGlobalPos[SIDES_COUNT];        // previous frame global positions
    Vector3 toeActualVelocity[SIDES_COUNT];       // computed from FK: (current - prev) / dt
    Vector3 toeBlendedVelocity[SIDES_COUNT];      // blended from cursor global toe velocities
    bool toeTrackingInitialized = false;

    // Virtual toe positions - move with blended velocity, used for IK targets
    Vector3 virtualToePos[SIDES_COUNT];
    bool virtualToeInitialized = false;
    // Virtual toe locking system
    float virtualToeUnlockTimer[SIDES_COUNT] = { -1.0f, -1.0f };  // -1 = locked, >=0 = unlocked

    PlayerControlInput playerInput;


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

    // Initialize pre-IK debug transform buffer
    TransformDataInit(&cc->xformBeforeIK);
    TransformDataResize(&cc->xformBeforeIK, skeleton);
    cc->debugSaveBeforeIK = true;  // enable by default for debugging


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
    TransformDataFree(&cc->xformBeforeIK);  // Add this line

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
    const AppConfig& config)
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
            cursor->blendTime = config.defaultBlendTime;

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
        cc->switchTimer = config.switchInterval;

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
        // Problem: Rot6d averaging of 0 and 180 yaw gives a zero-length vector.
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

        // Hips (joint 0) - no additional smoothing, no velblending
        // Just use the blended rotation directly, cursor weight blending is enough
        Rot6dToQuaternion(blendedRot6d[0], cc->xformData.localRotations[0]);

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

    // Update virtual toe positions
    // Move towards blended cursor XZ target at speed limited by blended cursor toe speed
    for (int side : sides)
    {
        const int toeIdx = db->toeIndices[side];
        if (toeIdx < 0 || toeIdx >= cc->xformData.jointCount)
        {
            if (!cc->virtualToeInitialized)
            {
                TraceLog(LOG_WARNING, "Virtual toe: can't find %s toe joint (idx=%d, jointCount=%d)",
                    StringFromSide(side), toeIdx, cc->xformData.jointCount);
            }
            continue;
        }

        // Blend toe position and speed from cursors using normalized weights
        float blendedToeX = 0.0f;
        float blendedToeY = 0.0f;
        float blendedToeZ = 0.0f;
        float blendedSpeed = 0.0f;
        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc->cursors[ci];
            if (!cur.active) continue;
            const float w = cur.normalizedWeight;
            if (w <= 1e-6f) continue;

            if (toeIdx < (int)cur.globalPositions.size())
            {
                blendedToeX += cur.globalPositions[toeIdx].x * w;
                blendedToeY += cur.globalPositions[toeIdx].y * w;
                blendedToeZ += cur.globalPositions[toeIdx].z * w;
            }

            // blend speed magnitude (XZ only)
            const float velX = cur.globalToeVelocity[side].x;
            const float velZ = cur.globalToeVelocity[side].z;
            const float speedXZ = sqrtf(velX * velX + velZ * velZ);
            blendedSpeed += speedXZ * w;
        }

        static constexpr float UNLOCK_DISTANCE = 0.2f;        // distance threshold to unlock
        static constexpr float UNLOCK_DURATION = 0.3f;        // time to gradually re-lock (seconds)


        if (!cc->virtualToeInitialized)
        {
            cc->virtualToePos[side] = Vector3{ blendedToeX, blendedToeY, blendedToeZ };
            TraceLog(LOG_INFO, "Virtual toe: initialized %s toe at (%.2f, %.2f, %.2f)",
                StringFromSide(side), cc->virtualToePos[side].x, cc->virtualToePos[side].y, cc->virtualToePos[side].z);
        }
        else
        {
            // Y always follows cursor-blended height directly
            cc->virtualToePos[side].y = blendedToeY;

            // XZ: move towards blended target, speed limit scales with blended cursor speed
            // At low speeds (<=0.5 m/s): 1.2x multiplier
            // At high speeds (>=2.0 m/s): faster (cubic interpolation in between)
            const float dx = blendedToeX - cc->virtualToePos[side].x;
            const float dz = blendedToeZ - cc->virtualToePos[side].z;
            const float distXZ = sqrtf(dx * dx + dz * dz);


            // Check if we should unlock (distance exceeds threshold and not already unlocked)
            if (distXZ > UNLOCK_DISTANCE && cc->virtualToeUnlockTimer[side] < 0.0f)
            {
                cc->virtualToeUnlockTimer[side] = UNLOCK_DURATION;
                TraceLog(LOG_INFO, "Virtual toe %s unlocked: distance %.3f > %.3f",
                    StringFromSide(side), distXZ, UNLOCK_DISTANCE);
            }

            // Update unlock timer (countdown to re-lock)
            if (cc->virtualToeUnlockTimer[side] >= 0.0f)
            {
                cc->virtualToeUnlockTimer[side] -= dt;
                if (cc->virtualToeUnlockTimer[side] < 0.0f)
                {
                    cc->virtualToeUnlockTimer[side] = -1.0f;  // back to locked
                }
            }

            if (distXZ > 1e-6f && dt > 1e-6f)
            {
                const float lowSpeed = 0.5f;
                const float highSpeed = 2.0f;
                const float lowMult = 1.2f;
                const float highMult = 1.5f;
                const float t = Clamp((blendedSpeed - lowSpeed) / (highSpeed - lowSpeed), 0.0f, 1.0f);
                const float speedMultiplier = lowMult + (highMult - lowMult) * Smoothstep(t);
                const float maxSpeed = blendedSpeed * speedMultiplier;
                const float maxDist = maxSpeed * dt;

                if (distXZ <= maxDist)
                {
                    // can reach target this frame
                    cc->virtualToePos[side].x = blendedToeX;
                    cc->virtualToePos[side].z = blendedToeZ;
                }
                else
                {
                    // move towards target at max speed
                    const float scale = maxDist / distXZ;
                    cc->virtualToePos[side].x += dx * scale;
                    cc->virtualToePos[side].z += dz * scale;
                }
            }
        }


        // If unlocked, apply final clamp towards FK blended position
        // Clamp distance decreases linearly from UNLOCK_DISTANCE to 0 over UNLOCK_DURATION
        if (cc->virtualToeUnlockTimer[side] >= 0.0f)
        {
            float unlockProgress = cc->virtualToeUnlockTimer[side] / UNLOCK_DURATION;
            unlockProgress = Smoothstep(unlockProgress);  
            const float maxClampDist = (UNLOCK_DISTANCE * 1.05f) * unlockProgress;

            const float dxFinal = blendedToeX - cc->virtualToePos[side].x;
            const float dzFinal = blendedToeZ - cc->virtualToePos[side].z;
            const float distFinal = sqrtf(dxFinal * dxFinal + dzFinal * dzFinal);

            if (distFinal > maxClampDist)
            {
                const float clampScale = maxClampDist / distFinal;
                cc->virtualToePos[side].x = blendedToeX - dxFinal * clampScale;
                cc->virtualToePos[side].z = blendedToeZ - dzFinal * clampScale;
            }
        }
    }
    cc->virtualToeInitialized = true;

    // Save pre-IK state for debugging visualization
    if (cc->debugSaveBeforeIK)
    {
        // Copy current FK result before IK modifies it
        for (int i = 0; i < cc->xformData.jointCount; ++i)
        {
            cc->xformBeforeIK.globalPositions[i] = cc->xformData.globalPositions[i];
            cc->xformBeforeIK.globalRotations[i] = cc->xformData.globalRotations[i];
            cc->xformBeforeIK.localPositions[i] = cc->xformData.localPositions[i];
            cc->xformBeforeIK.localRotations[i] = cc->xformData.localRotations[i];
        }
    }

    if (config.enableFootIK)
    {
        // Apply leg IK to pull toes toward virtual toe positions
        // At this point globalPositions/Rotations are in world space, so IK target is world space
        // and RecomputeLegFK will produce world-space results (using world-space parent transforms)
        for (int side : sides)
        {
            LegChainIndices legIdx;
            legIdx.hip = db->hipJointIndex;
            legIdx.upleg = db->uplegIndices[side];
            legIdx.lowleg = db->lowlegIndices[side];
            legIdx.foot = db->footIndices[side];
            legIdx.toe = db->toeIndices[side];

            if (!legIdx.IsValid()) continue;

            SolveLegIKInPlace(&cc->xformData, legIdx, cc->virtualToePos[side]);
        }
    }
}
