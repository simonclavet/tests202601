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
        nullptr,  // outRootVelocityLocal
        nullptr,  // outRootYawRate
        nullptr,  // outLookaheadRootVelocityLocal
        nullptr); // outLookaheadRootYawRate

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
        Vector3 sampledRootVelLocal;
        float sampledRootYawRate;
        Vector3 sampledLookaheadRootVelLocal;
        float sampledLookaheadRootYawRate;
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
            &sampledRootVelLocal,
            &sampledRootYawRate,
            &sampledLookaheadRootVelLocal,
            &sampledLookaheadRootYawRate);

        // Store sampled velocities in cursor (now in local/heading-relative space)
        cur.sampledRootVelocityLocal = sampledRootVelLocal;
        cur.sampledRootYawRate = sampledRootYawRate;
        cur.sampledLookaheadRootVelocityLocal = sampledLookaheadRootVelLocal;
        cur.sampledLookaheadRootYawRate = sampledLookaheadRootYawRate;

        // Sample global toe velocities from database (for foot IK)
        // These are still in animation-world space, need to transform
        Vector3 animSpaceToeVel[SIDES_COUNT];
        SampleGlobalToeVelocity(db, cur.animIndex, cur.animTime - dt * 0.5f, animSpaceToeVel);

        // Update weight via spring integrator
        DoubleSpringDamper(cur.weightSpring, cur.targetWeight, cur.blendTime, dt);

        // Clamp the output weight to [0, 1]
        cur.weightSpring.x = ClampZeroOne(cur.weightSpring.x);

        // Transform root velocity from local space to world space
        // Local space is heading-relative, so just rotate by character's worldRotation
        const Vector3 rootVelWorld = Vector3RotateByQuaternion(sampledRootVelLocal, cc->worldRotation);

        // Store velocities in cursor (for acceleration-based blending)
        cur.prevRootVelocity = cur.rootVelocity;
        cur.prevRootYawRate = cur.rootYawRate;
        cur.rootVelocity = rootVelWorld;
        cur.rootYawRate = sampledRootYawRate;

        // Transform toe velocities from animation-world to character-world
        // Need to extract yaw to convert toe velocities (they're still in anim space)
        const float currYaw = Rot6dGetYaw(sampledRootRot6d);
        const Rot6d invCurrYawRot = Rot6dFromYaw(-currYaw);
        for (int side : sides)
        {
            Vector3 toeVelLocal;
            Rot6dTransformVector(invCurrYawRot, animSpaceToeVel[side], toeVelLocal);
            cur.globalToeVelocity[side] = Vector3RotateByQuaternion(toeVelLocal, cc->worldRotation);
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
                Vector3Scale(cur.globalToeVelocity[side], w));
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
            const Vector3 lookaheadVelWorld = Vector3RotateByQuaternion(cur.sampledLookaheadRootVelocityLocal, cc->worldRotation);

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

    // Apply root motion (with optional velocity-based smoothing)
    if (cc->cursorBlendMode == CursorBlendMode::VelBlending)
    {
        // Compute blended acceleration from cursor velocities (velocities already blended in main loop)
        Vector3 blendedAcceleration = Vector3Zero();
        float blendedYawAccel = 0.0f;

        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc->cursors[ci];
            if (!cur.active) continue;
            const float w = cur.normalizedWeight;
            if (w <= 1e-6f) continue;

            // compute and accumulate weighted acceleration
            const Vector3 acc = Vector3Scale(Vector3Subtract(cur.rootVelocity, cur.prevRootVelocity), 1.0f / dt);
            const float yawAcc = (cur.rootYawRate - cur.prevRootYawRate) / dt;
            blendedAcceleration = Vector3Add(blendedAcceleration, Vector3Scale(acc, w));
            blendedYawAccel += yawAcc * w;
        }

        // Normalize the already-blended velocity from main loop
        const Vector3 blendedVelNormalized = Vector3Scale(blendedVelocity, 1.0f / totalRootWeight);
        const float blendedYawRateNormalized = blendedYawRate / totalRootWeight;
        blendedVelocity = blendedVelNormalized;
        blendedYawRate = blendedYawRateNormalized;

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
    else if (cc->cursorBlendMode == CursorBlendMode::LookaheadDragging)
    {
        // Lookahead dragging: lerp running velocity towards extrapolated future velocity
        // Initialize on first frame
        if (!cc->lookaheadVelocityInitialized)
        {
            cc->lookaheadDragVelocity = finalVelocity;
            cc->lookaheadDragYawRate = finalYawRate;
            cc->lookaheadVelocityInitialized = true;
        }

        // Lerp towards lookahead target with alpha = dt / lookaheadTime
        const float lookaheadTime = Max(dt, config.poseDragLookaheadTime);
        const float alpha = dt / lookaheadTime;
        cc->lookaheadDragVelocity = Vector3Lerp(cc->lookaheadDragVelocity, blendedLookaheadVelocity, alpha);
        cc->lookaheadDragYawRate = Lerp(cc->lookaheadDragYawRate, blendedLookaheadYawRate, alpha);

        // Apply the running velocity
        const Vector3 dragDelta = Vector3Scale(cc->lookaheadDragVelocity, dt);
        const float dragYawDelta = cc->lookaheadDragYawRate * dt;

        cc->worldPosition = Vector3Add(cc->worldPosition, dragDelta);
        const Quaternion yawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, dragYawDelta);
        cc->worldRotation = QuaternionNormalize(QuaternionMultiply(yawQ, cc->worldRotation));

        // Update smoothed velocity for visualization
        cc->smoothedRootVelocity = cc->lookaheadDragVelocity;
        cc->smoothedRootYawRate = cc->lookaheadDragYawRate;
    }
    else
    {
        // Basic mode: direct application of blended velocity
        cc->worldPosition = Vector3Add(cc->worldPosition, finalWorldDelta);
        const Quaternion yawQ = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, finalYawDelta);
        cc->worldRotation = QuaternionNormalize(QuaternionMultiply(yawQ, cc->worldRotation));

        // Update smoothedRootVelocity for visualization
        cc->smoothedRootVelocity = finalVelocity;
        cc->smoothedRootYawRate = finalYawRate;
    }

    // --- Rot6d blending using normalized weights (no double-cover issues, simple weighted average then normalize) ---
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


        // Other joints (1+): choose blend mode
        switch (cc->cursorBlendMode)
        {
        case CursorBlendMode::LookaheadDragging:
        {
            // Lookahead dragging: lerp running pose towards extrapolated future
           // with alpha = dt / lookaheadTime so animation tracks perfectly when not transitioning

           // Special handling for hip (joint 0): we need yaw-stripped versions for blending
           // because each cursor's hip rotation is in different animation space
            Rot6d blendedLookaheadHipNoYaw = Rot6dIdentity();
            blendedLookaheadHipNoYaw.ax = 0.0f;  // zero it out (identity has ax=1)

            // Re-accumulate lookahead rotations for hip with yaw stripped
            for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
            {
                const BlendCursor& cur = cc->cursors[ci];
                if (!cur.active) continue;
                const float w = cur.normalizedWeight;
                if (w <= 1e-6f) continue;

                // For hip: strip yaw from lookahead before accumulating
                Rot6d laHipNoYaw = cur.lookaheadRotations6d[0];
                Rot6dRemoveYComponent(laHipNoYaw, laHipNoYaw);

                Rot6dScaledAdd(w, laHipNoYaw, blendedLookaheadHipNoYaw);
            }


            // Normalize the blended hip lookahead (yaw-stripped)
            const float lenLA = sqrtf(blendedLookaheadHipNoYaw.ax * blendedLookaheadHipNoYaw.ax +
                blendedLookaheadHipNoYaw.ay * blendedLookaheadHipNoYaw.ay +
                blendedLookaheadHipNoYaw.az * blendedLookaheadHipNoYaw.az);
            if (lenLA > 1e-6f)
            {
                Rot6dNormalize(blendedLookaheadHipNoYaw);
            }
            else
            {
                blendedLookaheadHipNoYaw = Rot6dIdentity();
            }

            if (!cc->lookaheadDragInitialized)
            {
                // Initialize hip with yaw-stripped version
                cc->lookaheadDragPose6d[0] = blendedRot6d[0];  // current blended (already yaw-stripped)

                // Initialize other joints normally
                for (int j = 1; j < jc; ++j)
                {
                    cc->lookaheadDragPose6d[j] = blendedRot6d[j];
                }
                cc->lookaheadDragInitialized = true;
            }

            const float lookaheadTime = Max(dt, config.poseDragLookaheadTime);
            if (lookaheadTime > 1e-6f)
            {
                const float alpha = dt / lookaheadTime;

                // Hip: lerp towards yaw-stripped lookahead target
                Rot6dLerp(cc->lookaheadDragPose6d[0], blendedLookaheadHipNoYaw, alpha, cc->lookaheadDragPose6d[0]);

                // Other joints: use normal lookahead targets
                for (int j = 1; j < jc; ++j)
                {
                    Rot6dLerp(cc->lookaheadDragPose6d[j], blendedLookaheadRot6d[j], alpha, cc->lookaheadDragPose6d[j]);
                }
            }
            else
            {
                cc->lookaheadDragPose6d[0] = blendedRot6d[0];  // yaw-stripped current
                for (int j = 1; j < jc; ++j)
                {
                    cc->lookaheadDragPose6d[j] = blendedRot6d[j];
                }
            }

            // Convert to quaternions (hip already yaw-stripped, so convert directly)
            for (int j = 0; j < jc; ++j)
            {
                Rot6dToQuaternion(cc->lookaheadDragPose6d[j], cc->xformData.localRotations[j]);
            }
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

    // Update virtual toe positions
    // Intermediate: cursor velocity + viscous return (NO speed clamp) - represents natural motion
    // Final: intermediate + speed clamp + unlock clamp - constrained for IK
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

        // Blend toe position from cursors using normalized weights
        float blendedToeX = 0.0f;
        float blendedToeY = 0.0f;
        float blendedToeZ = 0.0f;
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
        }
        const Vector3 blendedToePos = Vector3{ blendedToeX, blendedToeY, blendedToeZ };

        if (!cc->virtualToeInitialized)
        {
            cc->intermediateVirtualToePos[side] = blendedToePos;
            cc->virtualToePos[side] = blendedToePos;
            TraceLog(LOG_INFO, "Virtual toe: initialized %s toe at (%.2f, %.2f, %.2f)",
                StringFromSide(side), cc->virtualToePos[side].x, cc->virtualToePos[side].y, cc->virtualToePos[side].z);
        }
        else
        {
            // Update intermediate virtual toe (no speed clamp)
                // Step 1: Primary motion from blended cursor velocity (XYZ)
                const Vector3 blendedToeVel = cc->toeBlendedVelocity[side];
                const Vector3 velocityDisplacement = Vector3Scale(blendedToeVel, dt);

                // Step 2: Viscous return force toward FK blended position
                const Vector3 intermediatePlusVel = Vector3Add(cc->intermediateVirtualToePos[side], velocityDisplacement);
                const Vector3 toTarget = Vector3Subtract(blendedToePos, intermediatePlusVel);

                const float returnHalflife = 0.1f;
                const float returnAlpha = 1.0f - powf(0.5f, dt / returnHalflife);
                const Vector3 returnDisplacement = Vector3Scale(toTarget, returnAlpha);

                // Step 3: Combine (NO speed clamp for intermediate)
                const Vector3 intermediateDisplacement = Vector3Add(velocityDisplacement, returnDisplacement);
                cc->intermediateVirtualToePos[side] = Vector3Add(cc->intermediateVirtualToePos[side], intermediateDisplacement);

                // Update final virtual toe with speed clamp
                // Start from current final position and apply same forces
                const Vector3 finalPlusVel = Vector3Add(cc->virtualToePos[side], velocityDisplacement);
                const Vector3 toTargetFinal = Vector3Subtract(blendedToePos, finalPlusVel);
                const Vector3 returnDisplacementFinal = Vector3Scale(toTargetFinal, returnAlpha);
                Vector3 finalDisplacement = Vector3Add(velocityDisplacement, returnDisplacementFinal);

                // Step 4: Speed clamp (XZ only) for final virtual toe
                const float distXZ = Vector3Length2D(finalDisplacement);

                if (distXZ > 1e-6f)
                {
                    const float blendedSpeedXZ = Vector3Length2D(blendedToeVel);

                    const float lowSpeed = 0.5f;
                    const float highSpeed = 2.0f;
                    const float lowMult = 1.2f;
                    const float highMult = 1.7f;
                    const float howFast = ClampedInvLerp(lowSpeed, highSpeed, blendedSpeedXZ);
                    const float speedMultiplier = SmoothLerp(lowMult, highMult, howFast);
                    const float maxSpeed = blendedSpeedXZ * speedMultiplier;
                    const float maxDistXZ = maxSpeed * dt;

                    if (distXZ > maxDistXZ)
                    {
                        const float scale = maxDistXZ / distXZ;
                        finalDisplacement.x *= scale;
                        finalDisplacement.z *= scale;
                    }
                }

                cc->virtualToePos[side] = Vector3Add(cc->virtualToePos[side], finalDisplacement);

                // Check unlock condition: distance between final and intermediate (XZ only)
                const Vector3 unlockDelta = Vector3Subtract(cc->intermediateVirtualToePos[side], cc->virtualToePos[side]);
                const float distXZUnlock = Vector3Length2D(unlockDelta);

                // Trigger unlock when final can't keep up with intermediate
                // BUT only if the other foot is not currently unlocked
                const int otherSide = OtherSideInt(side);
                const bool otherFootUnlocked = (cc->virtualToeUnlockTimer[otherSide] >= 0.0f);

                if (config.enableTimedUnlocking &&
                    distXZUnlock > config.unlockDistance &&
                    cc->virtualToeUnlockTimer[side] < 0.0f &&
                    !otherFootUnlocked)
                {
                    cc->virtualToeUnlockTimer[side] = config.unlockDuration;
                    cc->virtualToeUnlockStartDistance[side] = distXZUnlock;  // REMEMBER actual distance at unlock
                    //TraceLog(LOG_INFO, "Virtual toe %s unlocked: final-intermediate dist %.3f > %.3f (start dist %.3f)",
                    //    StringFromSide(side), distXZUnlock, config.unlockDistance, distXZUnlock);
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

                // If unlocked, pull final toward intermediate (shrinking sphere constraint)
                if (config.enableTimedUnlocking && cc->virtualToeUnlockTimer[side] >= 0.0f)
                {
                    const float unlockProgress = cc->virtualToeUnlockTimer[side] / config.unlockDuration;
                    const float smoothUnlockProgress = SmoothStep(unlockProgress);
                    // Shrink from actual unlock distance to 0, not from unlockDistance
                    const float maxClampDist = cc->virtualToeUnlockStartDistance[side] * smoothUnlockProgress;

                    // Store for debug visualization
                    cc->virtualToeUnlockClampRadius[side] = maxClampDist;

                    const Vector3 clampDelta = Vector3Subtract(cc->intermediateVirtualToePos[side], cc->virtualToePos[side]);
                    const float distXZClamp = Vector3Length2D(clampDelta);

                    if (distXZClamp > maxClampDist)
                    {
                        const float clampScale = maxClampDist / distXZClamp;
                        cc->virtualToePos[side].x = cc->intermediateVirtualToePos[side].x - clampDelta.x * clampScale;
                        cc->virtualToePos[side].z = cc->intermediateVirtualToePos[side].z - clampDelta.z * clampScale;
                    }
                }
                else
                {
                    cc->virtualToeUnlockClampRadius[side] = 0.0f;
                }
        }
    }

    cc->virtualToeInitialized = true;   
    


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
