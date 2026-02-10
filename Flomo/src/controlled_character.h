#pragma once

#include <vector>
#include <span>
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
#include "motion_matching.h"





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

    // Build joint names combo string for UI
    cc->jointNamesCombo.clear();
    for (int j = 0; j < skeleton->jointCount; ++j)
    {
        if (j > 0) cc->jointNamesCombo += ";";
        cc->jointNamesCombo += skeleton->joints[j].name;
    }

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


    // Initialize pre-IK debug transform buffer
    TransformDataInit(&cc->xformBeforeIK);
    TransformDataResize(&cc->xformBeforeIK, skeleton);
    cc->debugSaveBeforeIK = true;  // enable by default for debugging

    // Initialize basic blend debug transform buffer
    TransformDataInit(&cc->xformBasicBlend);
    TransformDataResize(&cc->xformBasicBlend, skeleton);

    // Ensure blend cursors have storage sized to joint count
    for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i) {
        cc->cursors[i].localPositions.resize(cc->xformData.jointCount);
        cc->cursors[i].localRotations6d.resize(cc->xformData.jointCount);
        cc->cursors[i].lookaheadRotations6d.resize(cc->xformData.jointCount);
        cc->cursors[i].lookaheadLocalPositions.resize(cc->xformData.jointCount);
        cc->cursors[i].globalPositions.resize(cc->xformData.jointCount);
        cc->cursors[i].globalRotations.resize(cc->xformData.jointCount);
        cc->cursors[i].prevLocalRootPos = cc->xformData.localPositions[0];
        Rot6dFromQuaternion(cc->xformData.localRotations[0], cc->cursors[i].prevLocalRootRot6d);

        cc->cursors[i].active = false;
        cc->cursors[i].weightSpring = {};  // zero all spring state
        cc->cursors[i].fastWeightSpring = {};
        cc->cursors[i].targetWeight = 0.0f;
        cc->cursors[i].fastNormalizedWeight = 0.0f;
    }

    // Magic anchor root motion state
    cc->lookaheadDragRootVelocityRootSpace = Vector3Zero();
    cc->lookaheadDragYawRate = 0.0f;
    //cc->lookaheadMagicVelocityInitialized = false;

    // Sample initial pose to get starting root state
    TransformDataSampleFrame(&cc->xformData, skeleton, 0, scale);


    // Cursor blend mode state
    cc->cursorBlendMode = CursorBlendMode::Basic;

    cc->lookaheadDragLocalRotations6d.resize(cc->xformData.jointCount);
    cc->lookaheadDragLocalPositions.resize(cc->xformData.jointCount);
    cc->lookaheadDragInitialized = false;

    // Smoothed root motion state
    cc->rootVelocityWorld = Vector3Zero();
    cc->rootYawRate = 0.0f;


    // Visual defaults (cyan-ish to distinguish from orange original)
    cc->color = Color{ 50, 200, 200, 255 };
    cc->opacity = 1.0f;
    cc->radius = 0.04f;
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

// Helper: spawn a new blend cursor at a given animation and time
// Fades out existing cursors and spawns the new one
static void SpawnBlendCursor(
    ControlledCharacter* cc,
    int animIndex,
    float animTime,
    float blendTime,
    bool immediate)
{
    // fade out existing cursors
    for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i)
    {
        if (cc->cursors[i].active)
        {
            cc->cursors[i].targetWeight = 0.0f;
        }
    }

    BlendCursor* cursor = FindAvailableCursor(cc);
    assert(cursor != nullptr);
    if (!cursor) return;

    cursor->active = true;
    cursor->animIndex = animIndex;
    cursor->animTime = animTime;
    cursor->weightSpring = {};
    cursor->fastWeightSpring = {};

    if (immediate)
    {
        cursor->weightSpring.x = 1.0f;
        cursor->weightSpring.xi = 1.0f;
        cursor->fastWeightSpring.x = 1.0f;
        cursor->fastWeightSpring.xi = 1.0f;
    }

    cursor->targetWeight = 1.0f;
    cursor->blendTime = blendTime;

    // No need to pre-sample - the normal cursor loop will sample on first frame
    cc->animIndex = animIndex;
    cc->animTime = animTime;
}



// - requires db != nullptr && db->valid and db->jointCount == cc->xformData.jointCount
// - uses db->localJointPositions / localJointRotations6d for sampling (no per-frame global->local conversion)
// - blends per-cursor root deltas in world-space (XZ + yaw) and applies to cc->world*
static void ControlledCharacterUpdate(
    ControlledCharacter* cc,
    const CharacterData* characterData,
    const AnimDatabase* db,
    float dt,
    double worldTime,
    const AppConfig& config,
    NetworkState* networkState = nullptr)
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

    // Early exit if timestep is too small (prevents division by zero and unnecessary updates)
    if (dt <= 1e-6f) return;

    //const BVHData* bvh = &characterData->bvhData[cc->animIndex];

    bool firstFrame = true;
    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        const BlendCursor& cur = cc->cursors[ci];
        if (cur.active)
        {
            firstFrame = false;
            break;
        }
    }

    // --- Animation mode logic ---
    if (firstFrame)
    {
        // spawn initial cursor at random position
        const int newAnim = rand() % characterData->count;
        const BVHData* newBvh = &characterData->bvhData[newAnim];
        const float newMaxTime = (newBvh->frameCount - 1) * newBvh->frameTime;
        const float startTime = ((float)rand() / (float)RAND_MAX) * newMaxTime;
        cc->lookaheadDragInitialized = false;
        SpawnBlendCursor(cc, newAnim, startTime, config.defaultBlendTime, true);
        cc->switchTimer = config.switchInterval;
    }
    else if (cc->animMode == AnimationMode::RandomSwitch)
    {
        cc->switchTimer -= dt;
        if (cc->switchTimer <= 0.0f)
        {
            const int newAnim = rand() % characterData->count;
            const BVHData* newBvh = &characterData->bvhData[newAnim];
            const float newMaxTime = (newBvh->frameCount - 1) * newBvh->frameTime;
            const float startTime = ((float)rand() / (float)RAND_MAX) * newMaxTime;

            SpawnBlendCursor(cc, newAnim, startTime, config.defaultBlendTime, false);
            cc->switchTimer = config.switchInterval;
        }
    }
    else if (cc->animMode == AnimationMode::MotionMatching)
    {
        // motion matching: search and spawn new cursor every mmSearchPeriod
        cc->mmSearchTimer -= dt;
        if (cc->mmSearchTimer <= 0.0f)
        {
            // Compute query from current pose before searching
            // This uses the pose from the end of the previous frame
            ComputeMotionFeatures(
                db,
                cc,
                cc->mmQuery);

            cc->mmSearchTimer = config.mmSearchPeriod;

            const int skipBoundary = 60;
            float bestCost = 0.0f;
            const int bestFrame = MotionMatchingSearch(db, cc->mmQuery, skipBoundary, &bestCost, config, networkState);

            cc->mmBestFrame = bestFrame;
            cc->mmBestCost = bestCost;

            if (bestFrame >= 0)
            {
                const int clipIdx = FindClipForMotionFrame(db, bestFrame);
                if (clipIdx >= 0)
                {
                    const int clipStart = db->clipStartFrame[clipIdx];
                    const int localFrame = bestFrame - clipStart;
                    const float frameTime = db->animFrameTime[clipIdx];
                    const float targetTime = localFrame * frameTime;

                    // Don't spawn new cursor if an existing cursor is already playing the same anim at nearby time
                    const float minTimeDiff = 0.2f;  // seconds
                    bool tooCloseToExisting = false;
                    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
                    {
                        const BlendCursor& cur = cc->cursors[ci];
                        if (cur.active && cur.animIndex == clipIdx)
                        {
                            const float timeDiff = fabsf(cur.animTime - targetTime);
                            if (timeDiff < minTimeDiff)
                            {
                                tooCloseToExisting = true;
                                break;
                            }
                        }
                    }

                    if (!tooCloseToExisting)
                    {
                        SpawnBlendCursor(cc, clipIdx, targetTime, config.defaultBlendTime, false);
                    }
                }
            }
        }
    }

    // --- Update motion matching feature velocity (independent from actual velocity) ---
    // This velocity is used for FutureVel features and updated using maxAcceleration
    {
        if (!cc->virtualControlSmoothedVelocityInitialized)
        {
            cc->virtualControlSmoothedVelocity = Vector3Zero();
            cc->virtualControlSmoothedVelocityInitialized = true;
        }

        // Update feature velocity towards desired velocity using maximum acceleration
        const Vector3 desiredVel = cc->playerInput.desiredVelocity;
        const float maxAccel = config.virtualControlMaxAcceleration;
        const float maxDeltaVelMag = maxAccel * dt;

        const Vector3 velDelta = Vector3Subtract(cc->playerInput.desiredVelocity, cc->virtualControlSmoothedVelocity);
        const float velDeltaMag = Vector3Length(velDelta);

        if (velDeltaMag <= maxDeltaVelMag)
        {
            // Can reach desired velocity within this timestep
            cc->virtualControlSmoothedVelocity = cc->playerInput.desiredVelocity;
        }
        else if (velDeltaMag > 1e-6f)
        {
            // Clamp to max achievable change
            const Vector3 velDeltaDir = Vector3Scale(velDelta, 1.0f / velDeltaMag);
            cc->virtualControlSmoothedVelocity =
                Vector3Add(cc->virtualControlSmoothedVelocity, Vector3Scale(velDeltaDir, maxDeltaVelMag));
        }
    }

    // advance cursor times, update weights, compute total weight and normalize weights
    {
        float totalCursorWeight = 0.0f;
        float totalFastCursorWeight = 0.0f;

        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            BlendCursor& cur = cc->cursors[ci];
            if (!cur.active) continue;

            // Advance cursor time and clamp
            cur.animTime += dt;
            const BVHData* cbvh = &characterData->bvhData[cur.animIndex];
            const float clipMax = (cbvh->frameCount - 1) * cbvh->frameTime;
            if (cur.animTime > clipMax)
            {
                cur.animTime = clipMax;
                // Warn once when cursor reaches end (only log on first frame it hits the end)
                static int lastWarnedCursor = -1;
                static float lastWarnedTime = -1.0f;
                if (lastWarnedCursor != ci || fabsf(lastWarnedTime - clipMax) > 0.01f)
                {
                    TraceLog(LOG_WARNING, "Cursor %d reached end of animation %d at time %.2f - stopped advancing",
                        ci, cur.animIndex, clipMax);
                    lastWarnedCursor = ci;
                    lastWarnedTime = clipMax;
                }
            }

            // Update weight via spring integrator
            DoubleSpringDamper(cur.weightSpring, cur.targetWeight, cur.blendTime, dt);
            // Update fast weight via spring integrator (hardcoded 0.05s blend time)
            DoubleSpringDamper(cur.fastWeightSpring, cur.targetWeight, 0.05f, dt);

            // Clamp the output weights to [0, 1]
            cur.weightSpring.x = ClampZeroOne(cur.weightSpring.x);
            cur.fastWeightSpring.x = ClampZeroOne(cur.fastWeightSpring.x);

            const float wgt = cur.weightSpring.x;
            if (wgt > 1e-6f)
            {
                totalCursorWeight += wgt;
            }

            const float fastWgt = cur.fastWeightSpring.x;
            if (fastWgt > 1e-6f)
            {
                totalFastCursorWeight += fastWgt;
            }

            // deactivate tiny-weight cursors
            if (cur.weightSpring.x <= 1e-4f && cur.fastWeightSpring.x <= 1e-4f && cur.targetWeight <= 1e-4f)
            {
                cur.active = false;
                cur.animIndex = -1;
            }
        }

        // Warn if no active cursors
        if (totalCursorWeight <= 1e-6f)
        {
            TraceLog(LOG_WARNING, "No active cursors with non-zero weight - character has no animation");
        }

        // Compute normalized weights for all cursors
        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            BlendCursor& cur = cc->cursors[ci];
            if (!cur.active)
            {
                cur.normalizedWeight = 0.0f;
                cur.fastNormalizedWeight = 0.0f;
                continue;
            }
            cur.normalizedWeight = cur.weightSpring.x / totalCursorWeight;
            cur.fastNormalizedWeight = (totalFastCursorWeight > 1e-6f) ? (cur.fastWeightSpring.x / totalFastCursorWeight) : 0.0f;

            // deactivate tiny normalized weight cursors
            if (cur.normalizedWeight <= 1e-4f && cur.fastNormalizedWeight <= 1e-4f && cur.targetWeight <= 1e-4f)
            {
                cur.active = false;
                cur.animIndex = -1;
            }
        }
    }

    // --------- Per-cursor update: sample pose, update weights, blend velocities ----------
    Vector3 toeBlendedPositionRootSpace[SIDES_COUNT] = { Vector3Zero(), Vector3Zero() };
    Vector3 toeBlendedLookaheadPositionRootSpace[SIDES_COUNT] = { Vector3Zero(), Vector3Zero() };
    Vector3 toeBlendedVelocityRootSpace[SIDES_COUNT] = { Vector3Zero(), Vector3Zero() };
    
    Vector3 blendedRootVelocityRootSpace = Vector3Zero();
    Vector3 blendedLookaheadRootVelocityRootSpace = Vector3Zero();
    float blendedYawRate = 0.0f;
    float blendedLookaheadYawRate = 0.0f;

    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        BlendCursor& cur = cc->cursors[ci];
        if (!cur.active) continue;
        const float w = cur.normalizedWeight;

        const int clipStart = db->clipStartFrame[cur.animIndex];

        // Sample pose at animTime
        int f0, f1;
        float interFrameAlpha;
        GetInterFrameAlpha(db, cur.animIndex, cur.animTime, f0, f1, interFrameAlpha);
        const int baseFrame = clipStart + f0;

        // Sample velocities at midpoint for better accuracy
        int vf0, vf1;
        float vInterFrameAlpha;
        GetInterFrameAlpha(db, cur.animIndex, cur.animTime - dt * 0.5f, vf0, vf1, vInterFrameAlpha);
        const int vBaseFrame = clipStart + vf0;

        // Sample local positions and rotations
        std::span<const Vector3> posRow0 = db->localJointPositions.row_view(baseFrame);
        std::span<const Vector3> posRow1 = db->localJointPositions.row_view(baseFrame + 1);
        std::span<const Rot6d> rotRow0 = db->localJointRotations.row_view(baseFrame);
        std::span<const Rot6d> rotRow1 = db->localJointRotations.row_view(baseFrame + 1);
        for (int j = 0; j < jc; ++j)
        {
            cur.localPositions[j] = Vector3Lerp(posRow0[j], posRow1[j], interFrameAlpha);
        }
        for (int j = 0; j < jc; ++j)
        {
            cur.localRotations6d[j] = Rot6dLerp(rotRow0[j], rotRow1[j], interFrameAlpha);
        }

        // Sample lookahead rotations and positions (for lookahead dragging)
        std::span<const Rot6d> lookaheadRotRow0 = db->lookaheadLocalRotations.row_view(baseFrame);
        std::span<const Rot6d> lookaheadRotRow1 = db->lookaheadLocalRotations.row_view(baseFrame + 1);
        std::span<const Vector3> lookaheadPosRow0 = db->lookaheadLocalPositions.row_view(baseFrame);
        std::span<const Vector3> lookaheadPosRow1 = db->lookaheadLocalPositions.row_view(baseFrame + 1);
        for (int j = 0; j < jc; ++j)
        {
            cur.lookaheadRotations6d[j] = Rot6dLerp(lookaheadRotRow0[j], lookaheadRotRow1[j], interFrameAlpha);
        }
        for (int j = 0; j < jc; ++j)
        {
            cur.lookaheadLocalPositions[j] = Vector3Lerp(lookaheadPosRow0[j], lookaheadPosRow1[j], interFrameAlpha);
        }

        // Sample root motion velocities and yaw rates
        cur.sampledRootVelocityRootSpace = LerpFrames(&db->rootMotionVelocitiesRootSpace[baseFrame], interFrameAlpha);
        blendedRootVelocityRootSpace = Vector3Add(blendedRootVelocityRootSpace, Vector3Scale(cur.sampledRootVelocityRootSpace, w));

        cur.sampledLookaheadRootVelocityRootSpace = LerpFrames(&db->lookaheadRootMotionVelocitiesRootSpace[baseFrame], interFrameAlpha);
        blendedLookaheadRootVelocityRootSpace = Vector3Add(blendedLookaheadRootVelocityRootSpace, Vector3Scale(cur.sampledLookaheadRootVelocityRootSpace, w));

        cur.sampledRootYawRate = LerpFrames(&db->rootMotionYawRates[baseFrame], interFrameAlpha);
        blendedYawRate += cur.sampledRootYawRate * cur.fastNormalizedWeight;

        cur.sampledLookaheadRootYawRate = LerpFrames(&db->lookaheadRootMotionYawRates[baseFrame], interFrameAlpha);
        blendedLookaheadYawRate += cur.sampledLookaheadRootYawRate * cur.fastNormalizedWeight;

        // For display only: world-space velocity (not used for blending)
        cur.rootVelocityWorldForDisplayOnly = Vector3RotateByQuaternion(cur.sampledRootVelocityRootSpace, cc->worldRotation);

        // Sample toe positions and velocities (for foot IK)
        std::span<const Vector3> jointVelRow0 = db->jointVelocitiesRootSpace.row_view(vBaseFrame);
        std::span<const Vector3> jointVelRow1 = db->jointVelocitiesRootSpace.row_view(vBaseFrame + 1);
        for (int side : sides)
        {
            assertEvenInRelease(db->toeIndices[side] >= 0);
            const int toeIdx = db->toeIndices[side];

            // Current toe position (root space)
            const Vector3 cursorToePosRootSpace = LerpFrames(&db->toePositionsRootSpace[side][baseFrame], interFrameAlpha);
            toeBlendedPositionRootSpace[side] = Vector3Add(toeBlendedPositionRootSpace[side], Vector3Scale(cursorToePosRootSpace, w));

            // Lookahead toe position (root space)
            const Vector3 cursorToeLookaheadPosRootSpace = LerpFrames(&db->lookaheadToePositionsRootSpace[side][baseFrame], interFrameAlpha);
            toeBlendedLookaheadPositionRootSpace[side] = Vector3Add(toeBlendedLookaheadPositionRootSpace[side], 
                Vector3Scale(cursorToeLookaheadPosRootSpace, w));

            // Toe velocity sampled at midpoint (root space)
            const Vector3 cursorToeVelRootSpace = Vector3Lerp(jointVelRow0[toeIdx], jointVelRow1[toeIdx], vInterFrameAlpha);
            toeBlendedVelocityRootSpace[side] = Vector3Add(toeBlendedVelocityRootSpace[side], Vector3Scale(cursorToeVelRootSpace, w));
        }
    }


    for (int side : sides)
    {
        // toes velocity in world space from blended toe velocities in root space
        cc->toeBlendedVelocityWorld[side] = Vector3RotateByQuaternion(toeBlendedVelocityRootSpace[side], cc->worldRotation);
        // toes position in world space from blended toe positions in root space
        cc->toeBlendedPositionWorld[side] = Vector3Add(
            Vector3RotateByQuaternion(toeBlendedPositionRootSpace[side], cc->worldRotation),
            cc->worldPosition);
    }
    

 

    // --- Rot6d blending using normalized weights
    std::vector<Vector3> blendedLocalPositions(jc, Vector3Zero());
    std::vector<Vector3> blendedLookaheadLocalPositions(jc, Vector3Zero());

    std::vector<Rot6d> blendedLocalRotations(jc, Rot6dZero());
    std::vector<Rot6d> blendedLookaheadLocalRotations(jc, Rot6dZero());
       
    //std::vector<Vector3> angVelAccum(jc, Vector3Zero());

    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        const BlendCursor& cur = cc->cursors[ci];
        if (!cur.active) continue;
        const float w = cur.normalizedWeight;

        for (int j = 0; j < jc; ++j)
        {
            blendedLocalPositions[j] = Vector3Add(
                blendedLocalPositions[j], Vector3Scale(cur.localPositions[j], w));
        }
        for (int j = 0; j < jc; ++j)
        {
            //angVelAccum[j] = Vector3Add(angVelAccum[j], Vector3Scale(cur.localAngularVelocities[j], w));
            // weighted accumulation of lookahead positions
            blendedLookaheadLocalPositions[j] = Vector3Add(
                blendedLookaheadLocalPositions[j], Vector3Scale(cur.lookaheadLocalPositions[j], w));
        }
        for (int j = 0; j < jc; ++j)
        {
           // weighted accumulation of Rot6d using helper
            Rot6dScaledAdd(w, cur.localRotations6d[j], blendedLocalRotations[j]);
        }
        for (int j = 0; j < jc; ++j)
        {
            Rot6dScaledAdd(w, cur.lookaheadRotations6d[j], blendedLookaheadLocalRotations[j]);
        }
    }

    // normalize blended Rot6d to get target rotations
    for (int j = 0; j < jc; ++j)
    {
        // normalize current rotations
        Rot6d blended = blendedLocalRotations[j];
        const float lenA = sqrtf(blended.ax * blended.ax + blended.ay * blended.ay + blended.az * blended.az);
        assertEvenInRelease(lenA > 1e-6f);
        Rot6dNormalize(blendedLocalRotations[j]);
    }

    // Save basic blend pose (before lookahead modifications) for debug visualization
    if (config.drawBasicBlend)
    {
        // Copy positions (already accumulated in posAccum)
        for (int j = 0; j < jc; ++j)
        {
            cc->xformBasicBlend.localPositions[j] = blendedLocalPositions[j];
        }
        // Convert blendedRot6d to quaternions
        for (int j = 0; j < jc; ++j)
        {
            Rot6dToQuaternion(blendedLocalRotations[j], cc->xformBasicBlend.localRotations[j]);
        }
        // NOTE: No longer zeroing root XZ - hip position is relative to Magic anchor
        // FK
        TransformDataForwardKinematics(&cc->xformBasicBlend);
        // Transform to world space
        for (int i = 0; i < jc; ++i)
        {
            cc->xformBasicBlend.globalPositions[i] = Vector3Add(
                Vector3RotateByQuaternion(cc->xformBasicBlend.globalPositions[i], cc->worldRotation),
                cc->worldPosition);
            cc->xformBasicBlend.globalRotations[i] = QuaternionMultiply(
                cc->worldRotation, cc->xformBasicBlend.globalRotations[i]);
        }
    }

    // Transform blended velocity from root space to world space
    const Vector3 blendedRootVelocityWorld =
        Vector3RotateByQuaternion(blendedRootVelocityRootSpace, cc->worldRotation);


    switch (cc->cursorBlendMode)
    {
    case CursorBlendMode::LookaheadDragging:
    {
        // Initialize drag states
        if (!cc->lookaheadDragInitialized)
        {
            // Initialize rotations and positions for all joints
            for (int j = 0; j < jc; ++j)
            {
                cc->lookaheadDragLocalRotations6d[j] = blendedLocalRotations[j];
            }
            for (int j = 0; j < jc; ++j)
            {
                cc->lookaheadDragLocalPositions[j] = blendedLocalPositions[j];
            }

            cc->lookaheadDragRootVelocityRootSpace = blendedRootVelocityRootSpace;
            cc->lookaheadDragYawRate = blendedYawRate;

            cc->lookaheadDragInitialized = true;
        }

        // lookaheadTime >= dt ensures alpha <= 1 (no overshoot)
        const float lookaheadTime = Max(dt, db->featuresConfig.poseDragLookaheadTime);
        // ratio of how much we go toward lookahead target 
        const float lookaheadProjectionAlpha = dt / lookaheadTime;
        assert(lookaheadProjectionAlpha <= 1.0f);

        // lerp toward extrapolated targets for all joints (rotations and positions)
        for (int j = 0; j < jc; ++j)
        {
            //                const Rot6d effectiveTarget = Rot6dLerp(blendedRot6d[j], blendedLookaheadRot6d[j], alpha);
            cc->lookaheadDragLocalRotations6d[j] = Rot6dLerp(
                cc->lookaheadDragLocalRotations6d[j],
                blendedLookaheadLocalRotations[j], lookaheadProjectionAlpha);
        }

        for (int j = 0; j < jc; ++j)
        {
            cc->lookaheadDragLocalPositions[j] =
                Vector3Lerp(cc->lookaheadDragLocalPositions[j], blendedLookaheadLocalPositions[j], lookaheadProjectionAlpha);
        }

        // Apply dragged rotations (convert from Rot6d to quaternions)
        for (int j = 0; j < jc; ++j)
        {
            Rot6dToQuaternion(cc->lookaheadDragLocalRotations6d[j], cc->xformData.localRotations[j]);
        }

        // Apply dragged positions
        for (int j = 0; j < jc; ++j)
        {
            cc->xformData.localPositions[j] = cc->lookaheadDragLocalPositions[j];
        }

        cc->lookaheadDragRootVelocityRootSpace = Vector3Lerp(
            cc->lookaheadDragRootVelocityRootSpace,
            blendedLookaheadRootVelocityRootSpace, lookaheadProjectionAlpha);

        // lookahead is too slow for yaw rate, so we just directly set it to the blended yaw rate. 
        // This fixes turnonspot and turnstart precision
        constexpr bool doLookaheadForYawRate = false;
        if (doLookaheadForYawRate) 
        {
            cc->lookaheadDragYawRate = Lerp(
                cc->lookaheadDragYawRate,
                blendedLookaheadYawRate, lookaheadProjectionAlpha);
        }
        else
        {
            cc->lookaheadDragYawRate = blendedYawRate;
        }

        // Transform lerped velocity from root space to world space
        cc->rootVelocityWorld = Vector3RotateByQuaternion(cc->lookaheadDragRootVelocityRootSpace, cc->worldRotation);
        cc->rootYawRate = cc->lookaheadDragYawRate;
        

        break;
    }

    case CursorBlendMode::Basic:
    default:
    {
        // standard blending: directly use blended rotations and positions
        for (int j = 0; j < jc; ++j)
        {
            Rot6dToQuaternion(blendedLocalRotations[j], cc->xformData.localRotations[j]);
            cc->xformData.localPositions[j] = blendedLocalPositions[j];
        }

        // Store for debug visualization
        cc->rootVelocityWorld = blendedRootVelocityWorld;
        cc->rootYawRate = blendedYawRate;



        break;
    }
    } // end switch on blend mode
        
    // Compute position and rotation deltas for this timestep
    const Vector3 positionDelta = Vector3Scale(cc->rootVelocityWorld, dt);
    const float yawDelta = cc->rootYawRate * dt;

    // Update world position and rotation
    cc->worldPosition = Vector3Add(cc->worldPosition, positionDelta);
    const Quaternion yawRotation = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, yawDelta);
    cc->worldRotation = QuaternionMultiply(yawRotation, cc->worldRotation);

    // Keep only Y component to prevent pitch/roll accumulation
    cc->worldRotation = QuaternionYComponent(cc->worldRotation);

    // Forward kinematics (local space)
    TransformDataForwardKinematics(&cc->xformData);

    // Transform to world space using updated cc->world*
    for (int i = 0; i < cc->xformData.jointCount; ++i)
    {
        cc->xformData.globalPositions[i] = 
            Vector3Add(Vector3RotateByQuaternion(cc->xformData.globalPositions[i], cc->worldRotation), 
                cc->worldPosition);
        cc->xformData.globalRotations[i] = QuaternionMultiply(cc->worldRotation, cc->xformData.globalRotations[i]);
    }

    //// Compute FK for each active cursor (for debug visualization)
    //// We do this with root zeroed (same treatment as final blended pose)
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

        // NOTE: No longer zeroing root XZ - hip position is relative to Magic anchor

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


    // Compute toe velocity from FK result (before IK)
    for (int side : sides)
    {
        const int toeIdx = db->toeIndices[side];
        if (toeIdx < 0 || toeIdx >= cc->xformData.jointCount) continue;

        const Vector3 currentPos = cc->xformData.globalPositions[toeIdx];
        if (cc->toeTrackingPreIKInitialized)
        {
            cc->toeVelocityPreIK[side] = Vector3Scale(
                Vector3Subtract(currentPos, cc->prevToeGlobalPosPreIK[side]), 1.0f / dt);
        }
        else
        {
            cc->toeVelocityPreIK[side] = Vector3Zero();
        }
        cc->prevToeGlobalPosPreIK[side] = currentPos;
    }
    cc->toeTrackingPreIKInitialized = true;

    // Update virtual toe positions using lookahead dragging
    // lookaheadDragToePos: drags toward blended lookahead target (unconstrained, for unlock detection)
    // virtualToePos: speed-clamped for IK (constrained)
    const float lookaheadTime = Max(dt, db->featuresConfig.poseDragLookaheadTime);
    const float toeAlpha = dt / lookaheadTime;
    assert(toeAlpha <= 1.0f);

    for (int side : sides)
    {
        if (!cc->lookaheadDragToePosInitialized)
        {
            cc->lookaheadDragToePosRootSpace[side] = toeBlendedPositionRootSpace[side];
            cc->virtualToePos[side] = cc->toeBlendedPositionWorld[side];
            TraceLog(LOG_INFO, "Virtual toe: initialized %s toe at (%.2f, %.2f, %.2f)",
                StringFromSide(side), cc->virtualToePos[side].x, cc->virtualToePos[side].y, cc->virtualToePos[side].z);
        }
        else
        {
            // Advance lookahead drag toe position in root space
            if (cc->cursorBlendMode == CursorBlendMode::Basic)
            {
                cc->lookaheadDragToePosRootSpace[side] = toeBlendedPositionRootSpace[side];
            }
            else if (cc->cursorBlendMode == CursorBlendMode::LookaheadDragging)
            {
                cc->lookaheadDragToePosRootSpace[side] = Vector3Lerp(
                    cc->lookaheadDragToePosRootSpace[side],
                    toeBlendedLookaheadPositionRootSpace[side], toeAlpha);
            }

            // Convert lookahead drag position from root space to world space for virtual toe target
            cc->lookaheadDragToePosWorld[side] = Vector3Add(
                Vector3RotateByQuaternion(cc->lookaheadDragToePosRootSpace[side], cc->worldRotation),
                cc->worldPosition);

            // Update virtualToePos: try to go DIRECTLY to target, but speed-clamped
            const Vector3 prevVirtualToePos = cc->virtualToePos[side];
            Vector3 newVirtualToePos = cc->lookaheadDragToePosWorld[side];  // target directly

            const bool doSpeedClamp = true;
            if (doSpeedClamp)
            {
                // Speed clamp (XZ only) based on blended toe velocity
                const Vector3 blendedToeVel = cc->toeBlendedVelocityWorld[side];
                const Vector3 displacement = Vector3Subtract(newVirtualToePos, prevVirtualToePos);
                const float distXZ = Vector3Length2D(displacement);

                if (distXZ > 1e-6f)
                {
                    const float blendedSpeedXZ = Vector3Length2D(blendedToeVel);

                    const float lowSpeed = 0.5f;
                    const float highSpeed = 2.0f;
                    const float howFast = ClampedInvLerp(lowSpeed, highSpeed, blendedSpeedXZ);
                    const float lowMult = 1.2f;
                    const float highMult = 1.4f;
                    const float speedMultiplier = SmoothLerp(lowMult, highMult, howFast);
                    const float maxSpeed = blendedSpeedXZ * speedMultiplier;
                    const float maxDistXZ = maxSpeed * dt;

                    if (distXZ > maxDistXZ)
                    {
                        const float scale = maxDistXZ / distXZ;
                        newVirtualToePos.x = prevVirtualToePos.x + displacement.x * scale;
                        newVirtualToePos.z = prevVirtualToePos.z + displacement.z * scale;
                        // Y is NOT clamped - height should track target directly
                    }
                }
            }

            cc->virtualToePos[side] = newVirtualToePos;

            // Check unlock condition: distance between lookaheadDrag (unconstrained) and virtual (constrained)
            const Vector3 unlockDelta = Vector3Subtract(cc->lookaheadDragToePosWorld[side], cc->virtualToePos[side]);
            const float distXZUnlock = Vector3Length2D(unlockDelta);

            // Trigger unlock when constrained can't keep up with unconstrained
            const int otherSide = OtherSideInt(side);
            const bool otherFootUnlocked = (cc->virtualToeUnlockTimer[otherSide] >= 0.0f);

            if (config.enableTimedUnlocking &&
                distXZUnlock > config.unlockDistance &&
                cc->virtualToeUnlockTimer[side] < 0.0f &&
                !otherFootUnlocked)
            {
                cc->virtualToeUnlockTimer[side] = config.unlockDuration;
                cc->virtualToeUnlockStartDistance[side] = distXZUnlock;
            }

            // Update unlock timer
            if (cc->virtualToeUnlockTimer[side] >= 0.0f)
            {
                cc->virtualToeUnlockTimer[side] -= dt;
                if (cc->virtualToeUnlockTimer[side] < 0.0f)
                {
                    cc->virtualToeUnlockTimer[side] = -1.0f;
                }
            }

            // If unlocked, pull constrained toward unconstrained (shrinking sphere)
            if (config.enableTimedUnlocking && cc->virtualToeUnlockTimer[side] >= 0.0f)
            {
                const float unlockProgress = cc->virtualToeUnlockTimer[side] / config.unlockDuration;
                const float smoothUnlockProgress = SmoothStep(unlockProgress);
                const float maxClampDist = cc->virtualToeUnlockStartDistance[side] * smoothUnlockProgress;

                cc->virtualToeUnlockClampRadius[side] = maxClampDist;

                const Vector3 clampDelta = Vector3Subtract(cc->lookaheadDragToePosWorld[side], cc->virtualToePos[side]);
                const float distXZClamp = Vector3Length2D(clampDelta);

                if (distXZClamp > maxClampDist)
                {
                    const float clampScale = maxClampDist / distXZClamp;
                    cc->virtualToePos[side].x = cc->lookaheadDragToePosWorld[side].x - clampDelta.x * clampScale;
                    cc->virtualToePos[side].z = cc->lookaheadDragToePosWorld[side].z - clampDelta.z * clampScale;
                }
            }
            else
            {
                cc->virtualToeUnlockClampRadius[side] = 0.0f;
            }
        }
    }

    cc->lookaheadDragToePosInitialized = true;   
    


    // Update position history every HISTORY_SAMPLE_INTERVAL seconds
    // This maintains a ring buffer of recent positions for motion matching past position feature
    {
        static constexpr float HISTORY_SAMPLE_INTERVAL = 0.01f;  // 10ms between samples
        static constexpr float HISTORY_MAX_DURATION = 0.5f;      // keep 500ms of history

        // Check if we should sample (interval elapsed)
        if ((worldTime - cc->lastHistorySampleTime) >= HISTORY_SAMPLE_INTERVAL)
        {
            // Add new history point at the back
            HistoryPoint newPoint;  
            newPoint.position = cc->worldPosition;
            newPoint.timestamp = worldTime;
            cc->positionHistory.push_back(newPoint);

            cc->lastHistorySampleTime = worldTime;

            // Remove old history points from the front (keep only HISTORY_MAX_DURATION seconds)
            const double cutoffTime = worldTime - HISTORY_MAX_DURATION;
            while (!cc->positionHistory.empty() && cc->positionHistory.front().timestamp < cutoffTime)
            {
                cc->positionHistory.erase(cc->positionHistory.begin());
            }
        }
    }

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

    // Compute toe velocity from final pose (after IK)
    for (int side : sides)
    {
        const int toeIdx = db->toeIndices[side];
        if (toeIdx < 0 || toeIdx >= cc->xformData.jointCount) continue;

        const Vector3 currentPos = cc->xformData.globalPositions[toeIdx];
        if (cc->toeTrackingInitialized)
        {
            cc->toeVelocity[side] = Vector3Scale(
                Vector3Subtract(currentPos, cc->prevToeGlobalPos[side]), 1.0f / dt);
        }
        else
        {
            cc->toeVelocity[side] = Vector3Zero();
        }
        cc->prevToeGlobalPos[side] = currentPos;
    }
    cc->toeTrackingInitialized = true;
}
