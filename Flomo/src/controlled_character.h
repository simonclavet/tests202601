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

    // Initialize basic blend debug transform buffer
    TransformDataInit(&cc->xformBasicBlend);
    TransformDataResize(&cc->xformBasicBlend, skeleton);

    // Ensure blend cursors have storage sized to joint count
    for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i) {
        cc->cursors[i].localPositions.resize(cc->xformData.jointCount);
        cc->cursors[i].localRotations6d.resize(cc->xformData.jointCount);
        cc->cursors[i].localAngularVelocities.resize(cc->xformData.jointCount);
        cc->cursors[i].lookaheadRotations6d.resize(cc->xformData.jointCount);
        cc->cursors[i].globalPositions.resize(cc->xformData.jointCount);
        cc->cursors[i].globalRotations.resize(cc->xformData.jointCount);
        cc->cursors[i].prevLocalRootPos = cc->xformData.localPositions[0];
        Rot6dFromQuaternion(cc->xformData.localRotations[0], cc->cursors[i].prevLocalRootRot6d);

        cc->cursors[i].active = false;
        cc->cursors[i].weightSpring = {};  // zero all spring state
        cc->cursors[i].targetWeight = 0.0f;
    }

    // Magic anchor root motion state
    cc->magicWorldPosition = Vector3{ 2.0f, 0.0f, 0.0f };  // Same starting position as worldPosition
    cc->magicWorldRotation = QuaternionIdentity();
    cc->magicAnchorInitialized = false;
    cc->smoothedMagicVelocity = Vector3Zero();
    cc->smoothedMagicYawRate = 0.0f;
    cc->lookaheadDragMagicVelocity = Vector3Zero();
    cc->lookaheadDragMagicYawRate = 0.0f;
    cc->lookaheadMagicVelocityInitialized = false;

    // Sample initial pose to get starting root state
    TransformDataSampleFrame(&cc->xformData, skeleton, 0, scale);
    cc->prevRootPosition = cc->xformData.localPositions[0];
    cc->prevRootRotation = cc->xformData.localRotations[0];

    // Cursor blend mode state
    cc->cursorBlendMode = CursorBlendMode::Basic;
    cc->velBlendedRotations6d.resize(cc->xformData.jointCount);
    cc->velBlendInitialized = false;
    cc->lookaheadDragPose6d.resize(cc->xformData.jointCount);
    cc->lookaheadDragInitialized = false;

    // Smoothed root motion state
    cc->smoothedRootVelocity = Vector3Zero();
    cc->smoothedRootYawRate = 0.0f;
    cc->rootMotionInitialized = false;

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
    const AnimDatabase* db,
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

    if (immediate)
    {
        cursor->weightSpring.x = 1.0f;
        cursor->weightSpring.xi = 1.0f;
    }

    cursor->targetWeight = 1.0f;
    cursor->blendTime = blendTime;

    Vector3 rootPos;
    Rot6d rootRot6d;
    SamplePoseAndMotion(
        db,
        cursor->animIndex,
        cursor->animTime,
        0.0f,
        cursor->localPositions,
        cursor->localRotations6d,
        &cursor->localAngularVelocities,
        &cursor->lookaheadRotations6d,
        &rootPos,
        &rootRot6d,
        nullptr,  // outRootVelocityRootSpace
        nullptr,  // outRootYawRate
        nullptr,  // outLookaheadRootVelocityRootSpace
        nullptr,  // outLookaheadRootYawRate
        &cursor->sampledLookaheadHipsHeight,
        &cursor->sampledHipRotationYawFree,
        &cursor->sampledLookaheadHipRotationYawFree,
        nullptr,  // outMagicVelocity
        nullptr,  // outMagicYawRate
        nullptr,  // outLookaheadMagicVelocity
        nullptr,  // outLookaheadMagicYawRate
        &cursor->sampledHipPositionInMagicSpace,
        &cursor->sampledHipRotationInMagicSpace,
        &cursor->sampledLookaheadHipRotationInMagicSpace);

    cursor->prevLocalRootPos = rootPos;
    cursor->prevLocalRootRot6d = rootRot6d;

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

    // Early exit if timestep is too small (prevents division by zero and unnecessary updates)
    if (dt <= 1e-6f) return;

    const BVHData* bvh = &characterData->bvhData[cc->animIndex];

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

        SpawnBlendCursor(cc, db, newAnim, startTime, config.defaultBlendTime, true);
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

            SpawnBlendCursor(cc, db, newAnim, startTime, config.defaultBlendTime, false);
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

            const int skipBoundary = 5;
            float bestCost = 0.0f;
            const int bestFrame = MotionMatchingSearch(db, cc->mmQuery, skipBoundary, &bestCost);

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

                    SpawnBlendCursor(cc, db, clipIdx, targetTime, config.defaultBlendTime, false);
                }
            }
        }
    }

    // --- Advance main anim time (kept for legacy semantics if needed) ---
    cc->animTime += dt;
    const float currentMaxTime = (bvh->frameCount - 1) * bvh->frameTime;
    if (cc->animTime >= currentMaxTime)
    {
        cc->animTime = fmodf(cc->animTime, currentMaxTime);
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
            cc->virtualControlSmoothedVelocity = Vector3Add(cc->virtualControlSmoothedVelocity, Vector3Scale(velDeltaDir, maxDeltaVelMag));
        }
    }


    // --------- Per-cursor update: sample pose, update weights, blend velocities ----------
    Vector3 blendedVelocity = Vector3Zero();
    float blendedYawRate = 0.0f;
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
        Vector3 sampledRootVelRootSpace;
        float sampledRootYawRate;
        Vector3 sampledLookaheadRootVelRootSpace;
        float sampledLookaheadRootYawRate;
        float sampledLookaheadHipsHeight;
        Rot6d sampledHipRotationYawFree;
        Rot6d sampledLookaheadHipRotationYawFree;
        Vector3 sampledMagicVelocity;
        float sampledMagicYawRate;
        Vector3 sampledLookaheadMagicVelocity;
        float sampledLookaheadMagicYawRate;
        Vector3 sampledHipPositionInMagicSpace;
        Rot6d sampledHipRotationInMagicSpace;
        Rot6d sampledLookaheadHipRotationInMagicSpace;
        SamplePoseAndMotion(
            db,
            cur.animIndex,
            cur.animTime,
            -dt * 0.5f,  // sample velocity at midpoint of frame
            cur.localPositions,
            cur.localRotations6d,
            &cur.localAngularVelocities,
            &cur.lookaheadRotations6d,
            &sampledRootPos,
            &sampledRootRot6d,
            &sampledRootVelRootSpace,
            &sampledRootYawRate,
            &sampledLookaheadRootVelRootSpace,
            &sampledLookaheadRootYawRate,
            &sampledLookaheadHipsHeight,
            &sampledHipRotationYawFree,
            &sampledLookaheadHipRotationYawFree,
            &sampledMagicVelocity,
            &sampledMagicYawRate,
            &sampledLookaheadMagicVelocity,
            &sampledLookaheadMagicYawRate,
            &sampledHipPositionInMagicSpace,
            &sampledHipRotationInMagicSpace,
            &sampledLookaheadHipRotationInMagicSpace);

        // Store sampled values in cursor
        cur.sampledRootVelocityRootSpace = sampledRootVelRootSpace;
        cur.sampledRootYawRate = sampledRootYawRate;
        cur.sampledLookaheadRootVelocityRootSpace = sampledLookaheadRootVelRootSpace;
        cur.sampledLookaheadRootYawRate = sampledLookaheadRootYawRate;
        cur.sampledLookaheadHipsHeight = sampledLookaheadHipsHeight;
        cur.sampledHipRotationYawFree = sampledHipRotationYawFree;
        cur.sampledLookaheadHipRotationYawFree = sampledLookaheadHipRotationYawFree;
        cur.sampledMagicVelocity = sampledMagicVelocity;
        cur.sampledMagicYawRate = sampledMagicYawRate;
        cur.sampledLookaheadMagicVelocity = sampledLookaheadMagicVelocity;
        cur.sampledLookaheadMagicYawRate = sampledLookaheadMagicYawRate;
        cur.sampledHipPositionInMagicSpace = sampledHipPositionInMagicSpace;
        cur.sampledHipRotationInMagicSpace = sampledHipRotationInMagicSpace;
        cur.sampledLookaheadHipRotationInMagicSpace = sampledLookaheadHipRotationInMagicSpace;

        // Sample toe velocities from database (already in root space)
        Vector3 toeVelRootSpace[SIDES_COUNT];
        SampleToeVelocityRootSpace(db, cur.animIndex, cur.animTime - dt * 0.5f, toeVelRootSpace);

        // Sample lookahead toe positions (for predictive foot IK)
        SampleLookaheadToePosRootSpace(db, cur.animIndex, cur.animTime, cur.sampledLookaheadToePosRootSpace);

        // Update weight via spring integrator
        DoubleSpringDamper(cur.weightSpring, cur.targetWeight, cur.blendTime, dt);

        // Clamp the output weight to [0, 1]
        cur.weightSpring.x = ClampZeroOne(cur.weightSpring.x);

        // Transform root velocity from root space to world space
        // Root space is heading-relative, just rotate by character's worldRotation
        const Vector3 rootVelWorld = Vector3RotateByQuaternion(sampledRootVelRootSpace, cc->worldRotation);

        // Store velocities in cursor (for acceleration-based blending)
        cur.prevRootVelocity = cur.rootVelocity;
        cur.prevRootYawRate = cur.rootYawRate;
        cur.rootVelocity = rootVelWorld;
        cur.rootYawRate = sampledRootYawRate;

        // Transform toe velocities from root space to world space
        for (int side : sides)
        {
            cur.toeVelocityWorld[side] = Vector3RotateByQuaternion(toeVelRootSpace[side], cc->worldRotation);
        }

        const float wgt = cur.weightSpring.x;
        if (wgt > 1e-6f)
        {
            blendedVelocity = Vector3Add(blendedVelocity, Vector3Scale(cur.rootVelocity, wgt));
            blendedYawRate += cur.rootYawRate * wgt;
            totalRootWeight += wgt;
        }

        // Store current root state for next frame (still needed for some features)
        cur.prevLocalRootPos = sampledRootPos;
        cur.prevLocalRootRot6d = sampledRootRot6d;

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
                Vector3Scale(cur.toeVelocityWorld[side], w));
        }
    }

    assert(totalRootWeight > 1e-6f);

    const Vector3 finalVelocity = Vector3Scale(blendedVelocity, 1.0f / totalRootWeight);
    const float finalYawRate = blendedYawRate / totalRootWeight;

    // For lookahead dragging mode, also blend lookahead velocities
    Vector3 blendedLookaheadVelocity = Vector3Zero();
    float blendedLookaheadYawRate = 0.0f;
    if (cc->cursorBlendMode == CursorBlendMode::LookaheadDragging)
    {
        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc->cursors[ci];
            if (!cur.active) continue;
            const float w = cur.normalizedWeight;
            if (w <= 1e-6f) continue;

            // Transform lookahead velocities to world space
            // They're already in local space, just apply worldRotation
            const Vector3 lookaheadVelWorld = Vector3RotateByQuaternion(cur.sampledLookaheadRootVelocityRootSpace, cc->worldRotation);

            blendedLookaheadVelocity = Vector3Add(blendedLookaheadVelocity, Vector3Scale(lookaheadVelWorld, w));
            blendedLookaheadYawRate += cur.sampledLookaheadRootYawRate * w;
        }
        blendedLookaheadVelocity = Vector3Scale(blendedLookaheadVelocity, 1.0f / totalRootWeight);
        blendedLookaheadYawRate /= totalRootWeight;
    }

    // Store deltas for debug visualization (computed from velocity)
    const Vector3 finalWorldDelta = Vector3Scale(finalVelocity, dt);
    const float finalYawDelta = finalYawRate * dt;
    cc->lastBlendedDeltaWorld = finalWorldDelta;
    cc->lastBlendedDeltaYaw = finalYawDelta;




    // --------- Apply root motion: Magic Anchor OR Hip-based ---------
    if (true)
    {
        // ===== MAGIC ANCHOR ROOT MOTION =====
        // Blend magic velocities from cursors (instead of hip velocities)
        Vector3 blendedMagicVelocity = Vector3Zero();
        float blendedMagicYawRate = 0.0f;

        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc->cursors[ci];
            if (!cur.active) continue;
            const float w = cur.normalizedWeight;
            if (w <= 1e-6f) continue;

            // Transform magic velocity from magic space to world space
            const Vector3 magicVelWorld = Vector3RotateByQuaternion(
                cur.sampledMagicVelocity, cc->magicWorldRotation);

            blendedMagicVelocity = Vector3Add(blendedMagicVelocity, Vector3Scale(magicVelWorld, w));
            blendedMagicYawRate += cur.sampledMagicYawRate * w;
        }

        // Normalize
        const Vector3 finalMagicVelocity = Vector3Scale(blendedMagicVelocity, 1.0f / totalRootWeight);
        const float finalMagicYawRate = blendedMagicYawRate / totalRootWeight;

        // For lookahead dragging mode, blend lookahead magic velocities
        Vector3 blendedLookaheadMagicVelocity = Vector3Zero();
        float blendedLookaheadMagicYawRate = 0.0f;
        if (cc->cursorBlendMode == CursorBlendMode::LookaheadDragging)
        {
            for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
            {
                const BlendCursor& cur = cc->cursors[ci];
                if (!cur.active) continue;
                const float w = cur.normalizedWeight;
                if (w <= 1e-6f) continue;

                const Vector3 lookaheadMagicVelWorld = Vector3RotateByQuaternion(
                    cur.sampledLookaheadMagicVelocity, cc->magicWorldRotation);

                blendedLookaheadMagicVelocity = Vector3Add(blendedLookaheadMagicVelocity,
                    Vector3Scale(lookaheadMagicVelWorld, w));
                blendedLookaheadMagicYawRate += cur.sampledLookaheadMagicYawRate * w;
            }
            blendedLookaheadMagicVelocity = Vector3Scale(blendedLookaheadMagicVelocity, 1.0f / totalRootWeight);
            blendedLookaheadMagicYawRate /= totalRootWeight;
        }

        // Initialize magic anchor on first frame
        if (!cc->magicAnchorInitialized)
        {
            cc->smoothedMagicVelocity = finalMagicVelocity;
            cc->smoothedMagicYawRate = finalMagicYawRate;
            cc->magicAnchorInitialized = true;
        }

        // Apply magic root motion based on blend mode
        //if (cc->cursorBlendMode == CursorBlendMode::VelBlending)
        //{
        //    // VelBlending mode (simplified - no per-cursor acceleration tracking yet)
        //    // Just lerp smoothed velocity towards target
        //    const float blendTime = cc->blendPosReturnTime;
        //    if (blendTime > 1e-6f)
        //    {
        //        const float alpha = 1.0f - powf(0.5f, dt / blendTime);
        //        cc->smoothedMagicVelocity = Vector3Lerp(cc->smoothedMagicVelocity, finalMagicVelocity, alpha);
        //        cc->smoothedMagicYawRate = Lerp(cc->smoothedMagicYawRate, finalMagicYawRate, alpha);
        //    }
        //    else
        //    {
        //        cc->smoothedMagicVelocity = finalMagicVelocity;
        //        cc->smoothedMagicYawRate = finalMagicYawRate;
        //    }

        //    const Vector3 smoothedMagicDelta = Vector3Scale(cc->smoothedMagicVelocity, dt);
        //    const float smoothedMagicYawDelta = cc->smoothedMagicYawRate * dt;

        //    cc->magicWorldPosition = Vector3Add(cc->magicWorldPosition, smoothedMagicDelta);
        //    const Quaternion magicYawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, smoothedMagicYawDelta);
        //    cc->magicWorldRotation = QuaternionNormalize(QuaternionMultiply(magicYawQ, cc->magicWorldRotation));
        //}
        //else 
        // Apply magic root motion based on blend mode
        if (cc->cursorBlendMode == CursorBlendMode::LookaheadDragging)
        {
            if (!cc->lookaheadMagicVelocityInitialized)
            {
                cc->lookaheadDragMagicVelocity = finalMagicVelocity;
                cc->lookaheadDragMagicYawRate = finalMagicYawRate;
                cc->lookaheadMagicVelocityInitialized = true;
            }

            const float lookaheadTime = Max(dt, db->poseDragLookaheadTime);
            const float alpha = dt / lookaheadTime;
            cc->lookaheadDragMagicVelocity = Vector3Lerp(cc->lookaheadDragMagicVelocity, blendedLookaheadMagicVelocity, alpha);
            cc->lookaheadDragMagicYawRate = Lerp(cc->lookaheadDragMagicYawRate, blendedLookaheadMagicYawRate, alpha);

            const Vector3 magicDragDelta = Vector3Scale(cc->lookaheadDragMagicVelocity, dt);
            const float magicDragYawDelta = cc->lookaheadDragMagicYawRate * dt;

            cc->magicWorldPosition = Vector3Add(cc->magicWorldPosition, magicDragDelta);
            const Quaternion magicYawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, magicDragYawDelta);
            cc->magicWorldRotation = QuaternionMultiply(magicYawQ, cc->magicWorldRotation);

            // CRITICAL FIX: Keep only Y component to prevent pitch/roll accumulation
            cc->magicWorldRotation = QuaternionYComponent(cc->magicWorldRotation);

            cc->smoothedMagicVelocity = cc->lookaheadDragMagicVelocity;
            cc->smoothedMagicYawRate = cc->lookaheadDragMagicYawRate;
        }
        else
        {
            // Basic mode
            const Vector3 finalMagicDelta = Vector3Scale(finalMagicVelocity, dt);
            const float finalMagicYawDelta = finalMagicYawRate * dt;

            cc->magicWorldPosition = Vector3Add(cc->magicWorldPosition, finalMagicDelta);
            const Quaternion magicYawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, finalMagicYawDelta);
            cc->magicWorldRotation = QuaternionMultiply(magicYawQ, cc->magicWorldRotation);

            // CRITICAL FIX: Keep only Y component to prevent pitch/roll accumulation
            cc->magicWorldRotation = QuaternionYComponent(cc->magicWorldRotation);

            cc->smoothedMagicVelocity = finalMagicVelocity;
            cc->smoothedMagicYawRate = finalMagicYawRate;
        }

        // Place skeleton (hip) relative to magic anchor
        Vector3 blendedHipPosInMagicSpace = Vector3Zero();
        Rot6d blendedHipRotInMagicSpace = Rot6dZero();

        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc->cursors[ci];
            if (!cur.active) continue;
            const float w = cur.normalizedWeight;
            if (w <= 1e-6f) continue;

            blendedHipPosInMagicSpace = Vector3Add(blendedHipPosInMagicSpace,
                Vector3Scale(cur.sampledHipPositionInMagicSpace, w));
            Rot6dScaledAdd(w, cur.sampledHipRotationInMagicSpace, blendedHipRotInMagicSpace);
        }
        Rot6dNormalize(blendedHipRotInMagicSpace);

        // Transform hip position from magic space to world space
        // IMPORTANT: magicWorldRotation must be PURE YAW (no pitch/roll) for Y to pass through unchanged
        const Vector3 hipOffsetWorld = Vector3RotateByQuaternion(
            blendedHipPosInMagicSpace, cc->magicWorldRotation);
        cc->worldPosition = Vector3Add(cc->magicWorldPosition, hipOffsetWorld);

        // Transform hip rotation from magic space to world space
        Quaternion hipRotRelativeToMagic;
        Rot6dToQuaternion(blendedHipRotInMagicSpace, hipRotRelativeToMagic);

        // Set worldRotation to full hip rotation (magic yaw + hip tilt/roll)
        cc->worldRotation = QuaternionMultiply(cc->magicWorldRotation, hipRotRelativeToMagic);

        // Update smoothedRootVelocity for compatibility
        cc->smoothedRootVelocity = cc->smoothedMagicVelocity;
        cc->smoothedRootYawRate = cc->smoothedMagicYawRate;
    }
    //else



    //{







    //    // Apply root motion (with optional velocity-based smoothing)
    //    if (cc->cursorBlendMode == CursorBlendMode::VelBlending)
    //    {
    //        // Compute blended acceleration from cursor velocities (velocities already blended in main loop)
    //        Vector3 blendedAcceleration = Vector3Zero();
    //        float blendedYawAccel = 0.0f;

    //        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    //        {
    //            const BlendCursor& cur = cc->cursors[ci];
    //            if (!cur.active) continue;
    //            const float w = cur.normalizedWeight;
    //            if (w <= 1e-6f) continue;

    //            // compute and accumulate weighted acceleration
    //            const Vector3 acc = Vector3Scale(Vector3Subtract(cur.rootVelocity, cur.prevRootVelocity), 1.0f / dt);
    //            const float yawAcc = (cur.rootYawRate - cur.prevRootYawRate) / dt;
    //            blendedAcceleration = Vector3Add(blendedAcceleration, Vector3Scale(acc, w));
    //            blendedYawAccel += yawAcc * w;
    //        }

    //        // Normalize the already-blended velocity from main loop
    //        const Vector3 blendedVelNormalized = Vector3Scale(blendedVelocity, 1.0f / totalRootWeight);
    //        const float blendedYawRateNormalized = blendedYawRate / totalRootWeight;
    //        blendedVelocity = blendedVelNormalized;
    //        blendedYawRate = blendedYawRateNormalized;

    //        // Initialize on first frame
    //        if (!cc->rootMotionInitialized)
    //        {
    //            cc->smoothedRootVelocity = blendedVelocity;
    //            cc->smoothedRootYawRate = blendedYawRate;
    //            cc->rootMotionInitialized = true;
    //        }

    //        // Step 1: advance smoothed velocity using blended acceleration
    //        cc->smoothedRootVelocity = Vector3Add(cc->smoothedRootVelocity, Vector3Scale(blendedAcceleration, dt));
    //        cc->smoothedRootYawRate += blendedYawAccel * dt;

    //        // Step 2: lerp towards blended target velocity
    //        const float blendTime = cc->blendPosReturnTime;
    //        if (blendTime > 1e-6f)
    //        {
    //            const float alpha = 1.0f - powf(0.5f, dt / blendTime);
    //            cc->smoothedRootVelocity = Vector3Lerp(cc->smoothedRootVelocity, blendedVelocity, alpha);
    //            cc->smoothedRootYawRate = Lerp(cc->smoothedRootYawRate, blendedYawRate, alpha);
    //        }
    //        else
    //        {
    //            cc->smoothedRootVelocity = blendedVelocity;
    //            cc->smoothedRootYawRate = blendedYawRate;
    //        }

    //        // Apply smoothed velocity
    //        const Vector3 smoothedDelta = Vector3Scale(cc->smoothedRootVelocity, dt);
    //        const float smoothedYawDelta = cc->smoothedRootYawRate * dt;

    //        cc->worldPosition = Vector3Add(cc->worldPosition, smoothedDelta);
    //        const Quaternion yawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, smoothedYawDelta);
    //        cc->worldRotation = QuaternionNormalize(QuaternionMultiply(yawQ, cc->worldRotation));
    //    }
    //    else if (cc->cursorBlendMode == CursorBlendMode::LookaheadDragging)
    //    {
    //        // Lookahead dragging: lerp running velocity towards extrapolated future velocity
    //        // Initialize on first frame
    //        if (!cc->lookaheadVelocityInitialized)
    //        {
    //            cc->lookaheadDragVelocity = finalVelocity;
    //            cc->lookaheadDragYawRate = finalYawRate;
    //            cc->lookaheadVelocityInitialized = true;
    //        }

    //        // Lerp towards lookahead target with alpha = dt / lookaheadTime
    //        const float lookaheadTime = Max(dt, db->poseDragLookaheadTime);
    //        const float alpha = dt / lookaheadTime;
    //        cc->lookaheadDragVelocity = Vector3Lerp(cc->lookaheadDragVelocity, blendedLookaheadVelocity, alpha);
    //        cc->lookaheadDragYawRate = Lerp(cc->lookaheadDragYawRate, blendedLookaheadYawRate, alpha);

    //        // Apply the running velocity
    //        const Vector3 dragDelta = Vector3Scale(cc->lookaheadDragVelocity, dt);
    //        const float dragYawDelta = cc->lookaheadDragYawRate * dt;

    //        cc->worldPosition = Vector3Add(cc->worldPosition, dragDelta);
    //        const Quaternion yawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, dragYawDelta);
    //        cc->worldRotation = QuaternionNormalize(QuaternionMultiply(yawQ, cc->worldRotation));

    //        // Update smoothed velocity for visualization
    //        cc->smoothedRootVelocity = cc->lookaheadDragVelocity;
    //        cc->smoothedRootYawRate = cc->lookaheadDragYawRate;
    //    }
    //    else
    //    {
    //        // Basic mode: direct application of blended velocity
    //        cc->worldPosition = Vector3Add(cc->worldPosition, finalWorldDelta);
    //        const Quaternion yawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, finalYawDelta);
    //        cc->worldRotation = QuaternionNormalize(QuaternionMultiply(yawQ, cc->worldRotation));

    //        // Update smoothedRootVelocity for visualization
    //        cc->smoothedRootVelocity = finalVelocity;
    //        cc->smoothedRootYawRate = finalYawRate;
    //    }
    //}

    // --- Rot6d blending using normalized weights 
    {
        std::vector<Vector3> posAccum(jc, Vector3Zero());
        std::vector<Rot6d> rot6dAccum(jc, Rot6d{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
        std::vector<Rot6d> lookaheadRot6dAccum(jc, Rot6d{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
        std::vector<Vector3> angVelAccum(jc, Vector3Zero());

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

                // weighted accumulation of Rot6d using helper
                Rot6dScaledAdd(w, cur.localRotations6d[j], rot6dAccum[j]);
                Rot6dScaledAdd(w, cur.lookaheadRotations6d[j], lookaheadRot6dAccum[j]);
            }
        }

        // positions use normalized weights
        for (int j = 0; j < jc; ++j)
        {
            cc->xformData.localPositions[j] = posAccum[j];
        }

        // normalize blended Rot6d to get target rotations
        std::vector<Rot6d> blendedRot6d(jc);
        std::vector<Rot6d> blendedLookaheadRot6d(jc);
        for (int j = 0; j < jc; ++j)
        {
            // normalize current rotations
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

            // normalize lookahead rotations
            Rot6d lookahead = lookaheadRot6dAccum[j];
            const float lenLA = sqrtf(lookahead.ax * lookahead.ax + lookahead.ay * lookahead.ay + lookahead.az * lookahead.az);
            if (lenLA > 1e-6f)
            {
                Rot6dNormalize(lookahead);
                blendedLookaheadRot6d[j] = lookahead;
            }
            else
            {
                blendedLookaheadRot6d[j] = Rot6dIdentity();
            }
        }

        // Save basic blend pose (before lookahead modifications) for debug visualization
        if (config.drawBasicBlend)
        {
            // Copy positions (already accumulated in posAccum)
            for (int j = 0; j < jc; ++j)
            {
                cc->xformBasicBlend.localPositions[j] = posAccum[j];
            }
            // Convert blendedRot6d to quaternions
            for (int j = 0; j < jc; ++j)
            {
                Rot6dToQuaternion(blendedRot6d[j], cc->xformBasicBlend.localRotations[j]);
            }
            // Zero out root XZ translation
            cc->xformBasicBlend.localPositions[0].x = 0.0f;
            cc->xformBasicBlend.localPositions[0].z = 0.0f;
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

        // Other joints (1+): choose blend mode
        switch (cc->cursorBlendMode)
        {
        case CursorBlendMode::LookaheadDragging:
        {
            // Lookahead dragging: lerp running pose towards extrapolated future
            // with alpha = dt / lookaheadTime so animation tracks perfectly when not transitioning

            // Hip rotation: use the dedicated yaw-free tracks from the database
            // This avoids all the runtime yaw-stripping issues
            // Blend both current and lookahead yaw-free hip rotations from cursors
            Rot6d blendedHipYawFree = Rot6dZero();

            Rot6d blendedLookaheadHipYawFree = Rot6dZero();

            for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
            {
                const BlendCursor& cur = cc->cursors[ci];
                if (!cur.active) continue;
                const float w = cur.normalizedWeight;
                if (w <= 1e-6f) continue;

                // Use the precomputed yaw-free rotations from database
                Rot6dScaledAdd(w, cur.sampledHipRotationYawFree, blendedHipYawFree);
                Rot6dScaledAdd(w, cur.sampledLookaheadHipRotationYawFree, blendedLookaheadHipYawFree);
            }

            // Normalize the blended current hip
            {
                const float len = sqrtf(blendedHipYawFree.ax * blendedHipYawFree.ax +
                    blendedHipYawFree.ay * blendedHipYawFree.ay +
                    blendedHipYawFree.az * blendedHipYawFree.az);
                if (len > 1e-6f)
                {
                    Rot6dNormalize(blendedHipYawFree);
                }
                else
                {
                    blendedHipYawFree = Rot6dIdentity();
                }
            }

            // Normalize the blended lookahead hip
            {
                const float len = sqrtf(blendedLookaheadHipYawFree.ax * blendedLookaheadHipYawFree.ax +
                    blendedLookaheadHipYawFree.ay * blendedLookaheadHipYawFree.ay +
                    blendedLookaheadHipYawFree.az * blendedLookaheadHipYawFree.az);
                if (len > 1e-6f)
                {
                    Rot6dNormalize(blendedLookaheadHipYawFree);
                }
                else
                {
                    blendedLookaheadHipYawFree = Rot6dIdentity();
                }
            }

            // Initialize drag states
            if (!cc->lookaheadDragInitialized)
            {
                // Initialize other joints (not hip - hip uses separate track)
                for (int j = 1; j < jc; ++j)
                {
                    cc->lookaheadDragPose6d[j] = blendedRot6d[j];
                }
                cc->lookaheadDragInitialized = true;
            }

            if (!cc->lookaheadDragHipRotationInitialized)
            {
                // Initialize hip drag state from current blended yaw-free rotation (from database)
                cc->lookaheadDragHipRotationYawFree = blendedHipYawFree;
                cc->lookaheadDragHipRotationInitialized = true;
            }

            // lookaheadTime >= dt ensures alpha <= 1 (no overshoot)
            const float lookaheadTime = Max(dt, db->poseDragLookaheadTime);
            const float alpha = dt / lookaheadTime;
            assert(alpha <= 1.0f);

            // Apply extrapolation multiplier: effectiveTarget = blended + (lookahead - blended) * mult
            const float extrapMult = config.lookaheadExtrapolationMult;

            // Hip: drag using the dedicated yaw-free tracks (both from database)
            {
                Rot6d effectiveHipTarget;
                Rot6dLerp(blendedHipYawFree, blendedLookaheadHipYawFree, extrapMult, effectiveHipTarget);
                Rot6dLerp(cc->lookaheadDragHipRotationYawFree, effectiveHipTarget, alpha, cc->lookaheadDragHipRotationYawFree);
            }

            // Other joints: extrapolate then lerp
            for (int j = 1; j < jc; ++j)
            {
                Rot6d effectiveTarget;
                Rot6dLerp(blendedRot6d[j], blendedLookaheadRot6d[j], extrapMult, effectiveTarget);
                Rot6dLerp(cc->lookaheadDragPose6d[j], effectiveTarget, alpha, cc->lookaheadDragPose6d[j]);
            }

            // Convert to quaternions
            // TEST: use blended yaw-free directly (no dragging) to verify database values
            //Rot6dToQuaternion(blendedHipYawFree, cc->xformData.localRotations[0]);
            Rot6dToQuaternion(cc->lookaheadDragHipRotationYawFree, cc->xformData.localRotations[0]);
            for (int j = 1; j < jc; ++j)
            {
                Rot6dToQuaternion(cc->lookaheadDragPose6d[j], cc->xformData.localRotations[j]);
            }

            // Lookahead hips height dragging
            // Blend lookahead heights from cursors
            float blendedLookaheadHipsHeight = 0.0f;
            for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
            {
                const BlendCursor& cur = cc->cursors[ci];
                if (!cur.active) continue;
                const float w = cur.normalizedWeight;
                if (w <= 1e-6f) continue;

                blendedLookaheadHipsHeight += cur.sampledLookaheadHipsHeight * w;
            }

            // Initialize hips height dragging state
            if (!cc->lookaheadDragHipsHeightInitialized)
            {
                // Start from current blended height (not lookahead)
                cc->lookaheadDragHipsHeight = posAccum[0].y;
                cc->lookaheadDragHipsHeightInitialized = true;
            }

            // Apply extrapolation multiplier then drag towards target (reuses alpha from above)
            const float blendedHeight = posAccum[0].y;
            const float effectiveHeightTarget = blendedHeight + (blendedLookaheadHipsHeight - blendedHeight) * extrapMult;
            cc->lookaheadDragHipsHeight = Lerp(cc->lookaheadDragHipsHeight, effectiveHeightTarget, alpha);

            // Apply dragged height to hip Y position
            cc->xformData.localPositions[0].y = cc->lookaheadDragHipsHeight;

            break;
        }
        case CursorBlendMode::VelBlending:
        {
            // Hips (joint 0) - no additional smoothing, no velblending or lookahead dragging
            // Just use the blended rotation directly, cursor weight blending is enough
            Rot6dToQuaternion(blendedRot6d[0], cc->xformData.localRotations[0]);

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
            break;
        }
        case CursorBlendMode::Basic:
        default:
        {
            // standard blending: directly use blended rotations (skip hips, already done above)
            for (int j = 0; j < jc; ++j)
            {
                Rot6dToQuaternion(blendedRot6d[j], cc->xformData.localRotations[j]);
            }
            break;
        }
        } // end switch
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
    const float lookaheadTime = Max(dt, db->poseDragLookaheadTime);
    const float toeAlpha = dt / lookaheadTime;
    assert(toeAlpha <= 1.0f);

    for (int side : sides)
    {
        const int toeIdx = db->toeIndices[side];
        if (toeIdx < 0 || toeIdx >= cc->xformData.jointCount)
        {
            if (!cc->lookaheadDragToePosInitialized)
            {
                TraceLog(LOG_WARNING, "Virtual toe: can't find %s toe joint (idx=%d, jointCount=%d)",
                    StringFromSide(side), toeIdx, cc->xformData.jointCount);
            }
            continue;
        }

        // Blend lookahead toe positions and hip position from cursors
        Vector3 blendedLookaheadToePosRootSpace = Vector3Zero();
        Vector3 blendedHipPosWorld = Vector3Zero();
        Vector3 blendedToePos = Vector3Zero();
        const int hipIdx = db->hipJointIndex;

        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc->cursors[ci];
            if (!cur.active) continue;
            const float w = cur.normalizedWeight;
            if (w <= 1e-6f) continue;

            blendedLookaheadToePosRootSpace = Vector3Add(
                blendedLookaheadToePosRootSpace,
                Vector3Scale(cur.sampledLookaheadToePosRootSpace[side], w));

            // Get hip world position for root space -> world space transform
            if (hipIdx >= 0 && hipIdx < (int)cur.globalPositions.size())
            {
                blendedHipPosWorld = Vector3Add(blendedHipPosWorld, Vector3Scale(cur.globalPositions[hipIdx], w));
            }

            // Get toe position for speed clamp reference
            if (toeIdx < (int)cur.globalPositions.size())
            {
                blendedToePos = Vector3Add(blendedToePos, Vector3Scale(cur.globalPositions[toeIdx], w));
            }
        }

        // Transform lookahead toe from root space to world space
        // Root space is relative to ground-projected hip (hipPos.x, 0, hipPos.z), heading-aligned
        // So: rotate by worldRotation, then add blended hip world position (ground-projected)
        const Vector3 blendedLookaheadToePosWorld = Vector3Add(
            Vector3RotateByQuaternion(blendedLookaheadToePosRootSpace, cc->worldRotation),
            Vector3{ blendedHipPosWorld.x, 0.0f, blendedHipPosWorld.z });

        if (!cc->lookaheadDragToePosInitialized)
        {
            cc->lookaheadDragToePos[side] = blendedToePos;
            cc->virtualToePos[side] = blendedToePos;
            TraceLog(LOG_INFO, "Virtual toe: initialized %s toe at (%.2f, %.2f, %.2f)",
                StringFromSide(side), cc->virtualToePos[side].x, cc->virtualToePos[side].y, cc->virtualToePos[side].z);
        }
        else
        {
            // Apply extrapolation multiplier then drag toward target (unconstrained, for unlock detection)
            const Vector3 effectiveToeTarget = Vector3Add(blendedToePos,
                Vector3Scale(Vector3Subtract(blendedLookaheadToePosWorld, blendedToePos), config.lookaheadExtrapolationMult));
            cc->lookaheadDragToePos[side] = Vector3Lerp(cc->lookaheadDragToePos[side], effectiveToeTarget, toeAlpha);

            // Update virtualToePos: try to go DIRECTLY to target, but speed-clamped
            // No dragging - infinite stiffness, just clamped max speed
            const Vector3 prevVirtualToePos = cc->virtualToePos[side];
            Vector3 newVirtualToePos = cc->lookaheadDragToePos[side];  // target directly

            const bool doSpeedClamp = true;
            if (doSpeedClamp)
            {
                // Speed clamp (XZ only) based on blended toe velocity
                const Vector3 blendedToeVel = cc->toeBlendedVelocity[side];
                const Vector3 displacement = Vector3Subtract(newVirtualToePos, prevVirtualToePos);
                const float distXZ = Vector3Length2D(displacement);

                if (distXZ > 1e-6f)
                {
                    const float blendedSpeedXZ = Vector3Length2D(blendedToeVel);

                    const float lowSpeed = 0.5f;
                    const float highSpeed = 2.0f;
                    const float howFast = ClampedInvLerp(lowSpeed, highSpeed, blendedSpeedXZ);
                    const float lowMult = 1.1f;
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
            const Vector3 unlockDelta = Vector3Subtract(cc->lookaheadDragToePos[side], cc->virtualToePos[side]);
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

                const Vector3 clampDelta = Vector3Subtract(cc->lookaheadDragToePos[side], cc->virtualToePos[side]);
                const float distXZClamp = Vector3Length2D(clampDelta);

                if (distXZClamp > maxClampDist)
                {
                    const float clampScale = maxClampDist / distXZClamp;
                    cc->virtualToePos[side].x = cc->lookaheadDragToePos[side].x - clampDelta.x * clampScale;
                    cc->virtualToePos[side].z = cc->lookaheadDragToePos[side].z - clampDelta.z * clampScale;
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
