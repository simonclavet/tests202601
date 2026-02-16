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
#include "networks.h"





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

    // Initialize lookahead debug transform buffer
    TransformDataInit(&cc->xformLookahead);
    TransformDataResize(&cc->xformLookahead, skeleton);

    // Ensure blend cursors have storage sized to joint count
    for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i) {

        cc->cursors[i].localRotations6d.resize(cc->xformData.jointCount);
        cc->cursors[i].lookaheadRotations6d.resize(cc->xformData.jointCount);
        cc->cursors[i].rootLocalPosition = Vector3Zero();

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
// Copies a segment of poseGenFeatures into the cursor so it can play back without touching the db.
// animTime is the time within the clip (used to find the global frame to start copying from).
// After this call, cursor->segmentAnimTime is reset to 0 (relative to segment start).
// If testSegmentAE is true and the segment AE exists, the segment is passed through the
// autoencoder (normalize -> encode -> decode -> denormalize) before being stored.
static void SpawnBlendCursor(
    ControlledCharacter* cc,
    const AnimDatabase* db,
    NetworkState* networkState,
    int animIndex,
    float animTime,
    float blendTime,
    bool immediate,
    const AppConfig* config)
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

    // copy a segment of poseGenFeatures into the cursor
    const float frameTime = db->animFrameTime[animIndex];
    const int clipStart = db->clipStartFrame[animIndex];
    const int clipEnd = db->clipEndFrame[animIndex];
    const int clipFrameCount = clipEnd - clipStart;

    // global frame where this cursor starts
    int localFrame = (int)floorf(animTime / frameTime);
    if (localFrame < 0) localFrame = 0;
    if (localFrame >= clipFrameCount) localFrame = clipFrameCount - 1;
    const int globalFrame = clipStart + localFrame;

    // how many frames fit in the segment (clamped to not exceed clip end)
    const int segmentLengthInFrames = (int)ceilf(db->poseGenFeaturesSegmentLength / frameTime) + 1;
    const int framesAvailable = clipEnd - globalFrame;
    const int segFrameCount = (segmentLengthInFrames < framesAvailable) ? segmentLengthInFrames : framesAvailable;
    assert(segFrameCount >= 1);

    const int dim = db->poseGenFeaturesComputeDim;
    cursor->segment.resize(segFrameCount, dim);
    cursor->segmentFrameTime = frameTime;

    // copy rows from raw poseGenFeatures
    for (int f = 0; f < segFrameCount; ++f)
    {
        std::span<const float> src = db->poseGenFeatures.row_view(globalFrame + f);
        std::span<float> dst = cursor->segment.row_view(f);
        for (int d = 0; d < dim; ++d)
        {
            dst[d] = src[d];
        }
    }

    // optionally pass through the segment autoencoder to test reconstruction quality
    if (config->testSegmentAutoEncoder)
    {
        NetworkApplySegmentAE(networkState, db, &cursor->segment);
    }

    if (config->testPoseAutoEncoder)
    {
        const int segFrames = cursor->segment.rows();
        //const int poseDim = cursor->segment.cols();
        for (int f = 0; f < segFrames; ++f)
        {
            std::span<float> pose = cursor->segment.row_view(f);
            NetworkApplyPoseAE(networkState, db, pose);
        }
    }

    // animTime is now relative to segment start
    cursor->segmentAnimTime = 0.0f;

    cc->animIndex = animIndex;
    cc->animTime = animTime;
}

// Spawn a blend cursor whose segment comes from the
// latent predictor pipeline instead of the database.
static void SpawnBlendCursorFromPredictor(
    ControlledCharacter* cc,
    const AnimDatabase* db,
    NetworkState* networkState,
    const std::vector<float>& query,
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
    cursor->animIndex = -1;
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
    cursor->segmentFrameTime = db->animFrameTime[0];

    // predict segment from features
    if (!NetworkPredictSegment(
        networkState, db, query, cursor->segment))
    {
        cursor->active = false;
        return;
    }

    cursor->segmentAnimTime = 0.0f;
}

// same thing but using the flow matching model for
// diverse sampling instead of the average predictor
static void SpawnBlendCursorFromFlow(
    ControlledCharacter* cc,
    const AnimDatabase* db,
    NetworkState* networkState,
    const std::vector<float>& query,
    float blendTime,
    bool immediate)
{
    for (int i = 0;
        i < ControlledCharacter::MAX_BLEND_CURSORS;
        ++i)
    {
        if (cc->cursors[i].active)
            cc->cursors[i].targetWeight = 0.0f;
    }

    BlendCursor* cursor = FindAvailableCursor(cc);
    assert(cursor != nullptr);
    if (!cursor) return;

    cursor->active = true;
    cursor->animIndex = -1;
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
    cursor->segmentFrameTime = db->animFrameTime[0];

    if (!NetworkPredictSegmentFlow(
        networkState, db, query, cursor->segment))
    {
        cursor->active = false;
        return;
    }

    cursor->segmentAnimTime = 0.0f;
}

static void SpawnBlendCursorFromFullFlow(
    ControlledCharacter* cc,
    const AnimDatabase* db,
    NetworkState* networkState,
    const std::vector<float>& query,
    float blendTime,
    bool immediate)
{
    for (int i = 0;
        i < ControlledCharacter::MAX_BLEND_CURSORS;
        ++i)
    {
        if (cc->cursors[i].active)
            cc->cursors[i].targetWeight = 0.0f;
    }

    BlendCursor* cursor = FindAvailableCursor(cc);
    assert(cursor != nullptr);
    if (!cursor) return;

    cursor->active = true;
    cursor->animIndex = -1;
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
    cursor->segmentFrameTime = db->animFrameTime[0];

    if (!NetworkPredictFullFlow(
        networkState, db, query, cursor->segment))
    {
        cursor->active = false;
        return;
    }

    cursor->segmentAnimTime = 0.0f;
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
    NetworkState* networkState)
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

    // input decided search: detect sudden input changes
    // and bring the next search forward
    if (cc->animMode == AnimationMode::MotionMatching
        || cc->animMode == AnimationMode::AverageLatentPredictor
        || cc->animMode == AnimationMode::FlowSampled
        || cc->animMode == AnimationMode::FullFlowSampled
        || cc->animMode == AnimationMode::FridayFlow
        || cc->animMode == AnimationMode::SinglePosePredictor
        || cc->animMode == AnimationMode::UnconditionedAdvance
        || cc->animMode == AnimationMode::MondayPredictor)
    {
        cc->inputDecidedSearchCooldown -= dt;
        const Vector3 velDiff = Vector3Subtract(
            cc->playerInput.desiredVelocity,
            cc->prevDesiredVelocity);
        const Vector3 aimDiff = Vector3Subtract(
            cc->playerInput.desiredAimDirection,
            cc->prevDesiredAimDirection);
        const bool bigInputChange =
            Vector3Length(velDiff) > 1.0f
            || Vector3Length(aimDiff) > 0.5f;
        if (bigInputChange
            && cc->inputDecidedSearchCooldown <= 0.0f)
        {
            cc->mmSearchTimer =
                fminf(cc->mmSearchTimer, 0.02f);
            cc->inputDecidedSearchCooldown =
                config.inputDecidedSearchPeriod;
        }

        // update prevs at fixed rate so slomo doesn't
        // accumulate tiny deltas into a false trigger
        cc->prevInputUpdateTimer -= dt;
        if (cc->prevInputUpdateTimer <= 0.0f)
        {
            cc->prevInputUpdateTimer = 0.015f;
            cc->prevDesiredVelocity =
                cc->playerInput.desiredVelocity;
            cc->prevDesiredAimDirection =
                cc->playerInput.desiredAimDirection;
        }
    }

    if (firstFrame)
    {
        // spawn initial cursor at random position
        const int newAnim = 0;// rand() % characterData->count;
        //const BVHData* newBvh = &characterData->bvhData[newAnim];
        //const float newMaxTime = (newBvh->frameCount - 1) * newBvh->frameTime;
        const float startTime = 0.1f;
        cc->lookaheadDragInitialized = false;
        SpawnBlendCursor(cc, db, networkState, 
            newAnim, startTime, 
            config.defaultBlendTime, true, &config);
        cc->switchTimer = config.switchInterval;
    }
    else if (cc->animMode == AnimationMode::RandomSwitch)
    {
        cc->switchTimer -= dt;
        if (cc->switchTimer <= 0.0f)
        {
            assert(!db->legalStartFrames.empty()); // Legal start frames should always be populated by AnimDatabaseBuild

            const int randomIndex = RandomInt((int)db->legalStartFrames.size());
            const int selectedGlobalFrame = db->legalStartFrames[randomIndex];
            
            const int newAnimIdx = FindClipForMotionFrame(db, selectedGlobalFrame);
            assert(newAnimIdx != -1); // Should always find a clip for a legal frame

            const int clipStart = db->clipStartFrame[newAnimIdx];
            const int localFrame = selectedGlobalFrame - clipStart;
            const float startTime = localFrame * db->animFrameTime[newAnimIdx];

            SpawnBlendCursor(cc, db, networkState, newAnimIdx, startTime, 
                config.defaultBlendTime, false, &config);
            
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
                cc->mmQuery,
                true, 1.0f);

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

                    bool spawnNewCursor = true;
                    const bool preventBlendingCloseToExisting = false;
                    if (preventBlendingCloseToExisting)
                    {
                        // Don't spawn new cursor if an existing cursor is already playing the same anim at nearby time
                        const float minTimeDiff = 0.2f;  // seconds
                        bool tooCloseToExisting = false;
                        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
                        {
                            const BlendCursor& cur = cc->cursors[ci];
                            if (cur.targetWeight == 1.0f &&
                                cur.active &&
                                cur.animIndex == clipIdx)
                            {
                                const float timeDiff = fabsf(cur.segmentAnimTime - targetTime);
                                if (timeDiff < minTimeDiff)
                                {
                                    tooCloseToExisting = true;
                                    spawnNewCursor = false;
                                    break;
                                }
                            }
                        }
                    }


                    if (spawnNewCursor)
                    {
                        SpawnBlendCursor(cc, db, networkState, clipIdx, 
                            targetTime, config.defaultBlendTime, 
                            false, &config);
                    }
                }
            }
        }
    }
    else if (cc->animMode == AnimationMode::AverageLatentPredictor)
    {
        cc->mmSearchTimer -= dt;
        if (cc->mmSearchTimer <= 0.0f)
        {
            ComputeMotionFeatures(
                db, cc, cc->mmQuery, true, 1.0f);
            cc->mmSearchTimer = config.mmSearchPeriod;

            SpawnBlendCursorFromPredictor(
                cc, db, networkState, cc->mmQuery,
                config.defaultBlendTime, false);
        }
    }
    else if (cc->animMode == AnimationMode::FlowSampled)
    {
        // same timer pattern as AverageLatentPredictor but uses the flow model
        // to sample diverse poses instead of just the average
        cc->mmSearchTimer -= dt;
        if (cc->mmSearchTimer <= 0.0f)
        {
            ComputeMotionFeatures(db, cc, cc->mmQuery, true, 1.0f);
            cc->mmSearchTimer = config.mmSearchPeriod;

            SpawnBlendCursorFromFlow(
                cc, db, networkState, cc->mmQuery,
                config.defaultBlendTime, false);
        }
    }
    else if (cc->animMode == AnimationMode::FullFlowSampled)
    {
        cc->mmSearchTimer -= dt;
        if (cc->mmSearchTimer <= 0.0f)
        {
            ComputeMotionFeatures(db, cc, cc->mmQuery, true, 1.0f);
            cc->mmSearchTimer = config.mmSearchPeriod;

            SpawnBlendCursorFromFullFlow(
                cc, db, networkState, cc->mmQuery,
                config.defaultBlendTime, false);
        }
    }
    else if (cc->animMode == AnimationMode::FridayFlow)
    {
        // frame-by-frame flow in pose latent space
        // we write decoded poses into a 1-row cursor segment so the rest of the
        // cursor pipeline (blending, virtual toes, IK) just works
        // pose features update every frame, future features only at search rate
        cc->mmSearchTimer -= dt;
        const bool updateFuture = cc->mmSearchTimer <= 0.0f;
        if (updateFuture) cc->mmSearchTimer = config.mmSearchPeriod;

        const float poseAlpha = Min(1.0f, dt / db->animFrameTime[0]);
        ComputeMotionFeatures(db, cc, cc->mmQuery, updateFuture, poseAlpha);

        const int poseDim = PoseFeatures::GetDim(jc);

        // seed latent from database frame 0 on first use
        if (!cc->fridayFlowInitialized)
        {
            std::span<const float> frame0 = db->poseGenFeatures.row_view(0);
            std::vector<float> rawPose(frame0.begin(), frame0.end());
            NetworkEncodePoseToLatent(networkState, db, rawPose, /*out*/ cc->fridayFlowLatent);
            cc->fridayFlowInitialized = true;
        }

        // predict next latent directly via flow matching
        // the network was trained on consecutive frames, so the prediction is one
        // database frame step ahead. Lerp to handle varying runtime dt:
        // alpha < 1 interpolates (dt shorter than frame), alpha > 1 extrapolates
        std::vector<float> predictedNextLatent;
        const bool predicted = NetworkPredictFridayFlow(
            networkState, db, cc->mmQuery, cc->fridayFlowLatent, /*out*/ predictedNextLatent);

        if (predicted)
        {
            const float alpha = Max(1.0f, dt / db->animFrameTime[0]); // don't extrapolate
            for (int d = 0; d < (int)cc->fridayFlowLatent.size(); ++d)
            {
                cc->fridayFlowLatent[d] += alpha * (predictedNextLatent[d] - cc->fridayFlowLatent[d]);
            }
        }

        // decode latent to raw poseGenFeatures, write into cursor segment
        std::vector<float> rawPose;
        const bool decoded = NetworkDecodeFridayFlowLatent(
            networkState, db, cc->fridayFlowLatent, /*out*/ rawPose);

        if (decoded)
        {
            // kill existing cursors
            for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i)
            {
                if (cc->cursors[i].active)
                {
                    cc->cursors[i].active = false;
                }
            }

            // find or init our single cursor
            BlendCursor& cursor = cc->cursors[0];
            cursor.active = true;
            cursor.animIndex = -1;
            cursor.weightSpring = { 1.0f, 1.0f };
            cursor.fastWeightSpring = { 1.0f, 1.0f };
            cursor.targetWeight = 1.0f;
            cursor.blendTime = 0.01f;
            cursor.segmentFrameTime = db->animFrameTime[0];            

            // overwrite segment with a single decoded frame
            cursor.segment.resize(1, poseDim);
            std::span<float> row = cursor.segment.row_view(0);
            for (int d = 0; d < poseDim; ++d)
                row[d] = rawPose[d];
            cursor.segmentAnimTime = 0.0f;
        }
    }
    else if (cc->animMode == AnimationMode::SinglePosePredictor)
    {
        // deterministic single-pose prediction: features -> pose latent -> decoded pose
        // pose features update every frame, future features only at search rate
        cc->mmSearchTimer -= dt;
        const bool updateFuture = cc->mmSearchTimer <= 0.0f;
        if (updateFuture)
        {
            cc->mmSearchTimer = config.mmSearchPeriod;
        }
        const float poseAlpha = Min(1.0f, dt / db->animFrameTime[0]);
        ComputeMotionFeatures(db, cc, cc->mmQuery, updateFuture, poseAlpha);
        const int poseDim = PoseFeatures::GetDim(jc);

        std::vector<float> rawPose;
        const bool decoded = NetworkPredictSinglePose(
            networkState, db, cc->mmQuery, /*out*/ rawPose);

        if (decoded)
        {
            for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i)
            {
                if (cc->cursors[i].active)
                {
                    cc->cursors[i].active = false;
                }
            }

            BlendCursor& cursor = cc->cursors[0];
            cursor.active = true;
            cursor.animIndex = -1;
            cursor.weightSpring = { 1.0f, 1.0f };
            cursor.fastWeightSpring = { 1.0f, 1.0f };
            cursor.targetWeight = 1.0f;
            cursor.blendTime = 0.01f;
            cursor.segmentFrameTime = db->animFrameTime[0];

            cursor.segment.resize(1, poseDim);
            std::span<float> row = cursor.segment.row_view(0);
            for (int d = 0; d < poseDim; ++d)
            {
                row[d] = rawPose[d];
            }
            cursor.segmentAnimTime = 0.0f;
        }
    }
    else if (cc->animMode == AnimationMode::UnconditionedAdvance)
    {
        // hybrid: conditioned search (SinglePosePredictor) on search frames,
        // unconditioned advance (pose latent → next pose latent) between searches
        cc->mmSearchTimer -= dt;
        const bool searchThisFrame = cc->mmSearchTimer <= 0.0f;
        if (searchThisFrame)
        {
            cc->mmSearchTimer = config.mmSearchPeriod;
        }

        const float poseAlpha = Min(1.0f, dt / db->animFrameTime[0]);
        ComputeMotionFeatures(db, cc, cc->mmQuery, searchThisFrame, poseAlpha);

        const int poseDim = PoseFeatures::GetDim(jc);

        // seed latent from database frame 0 on first use
        if (!cc->uncondAdvanceInitialized)
        {
            std::span<const float> frame0 = db->poseGenFeatures.row_view(0);
            std::vector<float> rawPose(frame0.begin(), frame0.end());
            NetworkEncodePoseToLatent(networkState, db, rawPose, /*out*/ cc->uncondAdvanceLatent);
            cc->uncondAdvanceInitialized = true;
        }

        if (searchThisFrame)
        {
            // conditioned path: use SinglePosePredictor to anchor pose to current input
            std::vector<float> conditionedLatent;
            const bool gotLatent = NetworkPredictSinglePoseLatent(
                networkState, db, cc->mmQuery, /*out*/ conditionedLatent);
            if (gotLatent)
            {
                cc->uncondAdvanceLatent = conditionedLatent;
            }
        }
        else
        {
            // unconditioned path: advance pose latent without feature conditioning
            std::vector<float> predictedNext;
            const bool predicted = NetworkPredictUncondAdvance(
                networkState, cc->uncondAdvanceLatent, /*out*/ predictedNext);
            if (predicted)
            {
                const float alpha = Min(1.0f, dt / db->animFrameTime[0]);
                for (int d = 0; d < (int)cc->uncondAdvanceLatent.size(); ++d)
                {
                    cc->uncondAdvanceLatent[d] +=
                        alpha * (predictedNext[d] - cc->uncondAdvanceLatent[d]);
                }
            }
        }

        // decode latent to raw pose and write into cursor
        std::vector<float> rawPose;
        const bool decoded = NetworkDecodeFridayFlowLatent(
            networkState, db, cc->uncondAdvanceLatent, /*out*/ rawPose);

        if (decoded)
        {
            for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i)
            {
                if (cc->cursors[i].active)
                {
                    cc->cursors[i].active = false;
                }
            }

            BlendCursor& cursor = cc->cursors[0];
            cursor.active = true;
            cursor.animIndex = -1;
            cursor.weightSpring = { 1.0f, 1.0f };
            cursor.fastWeightSpring = { 1.0f, 1.0f };
            cursor.targetWeight = 1.0f;
            cursor.blendTime = 0.01f;
            cursor.segmentFrameTime = db->animFrameTime[0];

            cursor.segment.resize(1, poseDim);
            std::span<float> row = cursor.segment.row_view(0);
            for (int d = 0; d < poseDim; ++d)
            {
                row[d] = rawPose[d];
            }
            cursor.segmentAnimTime = 0.0f;
        }
    }
    else if (cc->animMode == AnimationMode::MondayPredictor)
    {
        // conditioned delta advance: features + pose latent → delta every frame
        cc->mmSearchTimer -= dt;
        const bool updateFuture = cc->mmSearchTimer <= 0.0f;
        if (updateFuture)
        {
            cc->mmSearchTimer = config.mmSearchPeriod;
        }

        const float poseAlpha = Min(1.0f, dt / db->animFrameTime[0]);
        ComputeMotionFeatures(db, cc, cc->mmQuery, updateFuture, poseAlpha);

        const int poseDim = PoseFeatures::GetDim(jc);

        // seed latent from database frame 0 on first use
        if (!cc->mondayInitialized)
        {
            std::span<const float> frame0 = db->poseGenFeatures.row_view(0);
            std::vector<float> rawPose(frame0.begin(), frame0.end());
            NetworkEncodePoseToLatent(networkState, db, rawPose, /*out*/ cc->mondayLatent);
            cc->mondayInitialized = true;
        }

        // predict delta and advance latent
        std::vector<float> delta;
        const bool predicted = NetworkPredictMonday(
            networkState, db, cc->mmQuery, cc->mondayLatent, /*out*/ delta);

        if (predicted)
        {
            const float alpha = Min(1.0f, dt / db->animFrameTime[0]);
            for (int d = 0; d < (int)cc->mondayLatent.size(); ++d)
            {
                cc->mondayLatent[d] += alpha * delta[d];
            }
        }

        // decode latent to raw pose, then re-encode back to snap onto the AE manifold
        std::vector<float> rawPose;
        const bool decoded = NetworkDecodeFridayFlowLatent(
            networkState, db, cc->mondayLatent, /*out*/ rawPose);

        if (decoded)
        {
            NetworkEncodePoseToLatent(networkState, db, rawPose, /*out*/ cc->mondayLatent);

            for (int i = 0; i < ControlledCharacter::MAX_BLEND_CURSORS; ++i)
            {
                if (cc->cursors[i].active)
                {
                    cc->cursors[i].active = false;
                }
            }

            BlendCursor& cursor = cc->cursors[0];
            cursor.active = true;
            cursor.animIndex = -1;
            cursor.weightSpring = { 1.0f, 1.0f };
            cursor.fastWeightSpring = { 1.0f, 1.0f };
            cursor.targetWeight = 1.0f;
            cursor.blendTime = 0.01f;
            cursor.segmentFrameTime = db->animFrameTime[0];

            cursor.segment.resize(1, poseDim);
            std::span<float> row = cursor.segment.row_view(0);
            for (int d = 0; d < poseDim; ++d)
            {
                row[d] = rawPose[d];
            }
            cursor.segmentAnimTime = 0.0f;
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

            // Advance cursor time and clamp to segment length
            cur.segmentAnimTime += dt;
            // old: const BVHData* cbvh = &characterData->bvhData[cur.animIndex];
            // old: const float clipMax = (cbvh->frameCount - 1) * cbvh->frameTime;
            const float segMax = (cur.segment.rows() - 1) * cur.segmentFrameTime;
            if (cur.segmentAnimTime > segMax)
            {
                cur.segmentAnimTime = segMax;
                //static int lastWarnedCursor = -1;
                //static float lastWarnedTime = -1.0f;
                //if (lastWarnedCursor != ci || fabsf(lastWarnedTime - segMax) > 0.01f)
                //{
                //    TraceLog(LOG_WARNING, "Cursor %d reached end of segment (anim %d) at time %.2f - stopped advancing",
                //        ci, cur.animIndex, segMax);
                //    lastWarnedCursor = ci;
                //    lastWarnedTime = segMax;
                //}
            }

            // Update weight via spring integrator
            DoubleSpringDamper(cur.weightSpring, cur.targetWeight, cur.blendTime, dt);
            // Update fast weight via spring integrator 
            DoubleSpringDamper(cur.fastWeightSpring, cur.targetWeight, 0.07f, dt);

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
            if (cur.normalizedWeight <= 1e-3f && cur.fastNormalizedWeight <= 1e-3f && cur.targetWeight <= 1e-3f)
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
    Vector3 nextToeBlendedVelocityRootSpace[SIDES_COUNT] = { Vector3Zero(), Vector3Zero() };
    Vector3 toeBlendedPosDiffRootSpace = Vector3Zero();
    Vector3 nextToeBlendedPosDiffRootSpace = Vector3Zero();
    float blendedToeSpeedDiff = 0.0f;
    
    Vector3 blendedRootVelocityRootSpace = Vector3Zero();
    Vector3 blendedLookaheadRootVelocityRootSpace = Vector3Zero();
    float blendedYawRate = 0.0f;
    float blendedLookaheadYawRate = 0.0f;

    // Two PoseFeatures temporaries for interpolation between adjacent segment frames
    PoseFeatures poseFeat0;
    PoseFeatures poseFeat1;
    poseFeat0.Resize(jc);
    poseFeat1.Resize(jc);

    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        BlendCursor& cur = cc->cursors[ci];
        if (!cur.active) continue;
        const float w = cur.normalizedWeight;

        // --- sample from cursor's local segment instead of db ---

        // compute f0, f1, alpha within the segment
        int sf0 = 0;
        int sf1 = 0;
        float sAlpha = 0.0f;
        if (cur.segmentFrameTime > 0.0f && cur.segment.rows() > 0)
        {
            const float maxFrame = (float)(cur.segment.rows() - 1);
            float frameF = cur.segmentAnimTime / cur.segmentFrameTime;
            if (frameF < 0.0f) frameF = 0.0f;
            if (frameF > maxFrame) frameF = maxFrame;
            sf0 = (int)floorf(frameF);
            sf1 = sf0 + 1;
            if (sf1 >= cur.segment.rows()) sf1 = cur.segment.rows() - 1;
            sAlpha = frameF - (float)sf0;
        }

        // deserialize the two bounding frames
        std::span<const float> segRow0 = cur.segment.row_view(sf0);
        std::span<const float> segRow1 = cur.segment.row_view(sf1);
        poseFeat0.DeserializeFrom(segRow0, jc);
        poseFeat1.DeserializeFrom(segRow1, jc);

        // interpolate into cursor workspace (lookahead data = primary data now)
        // Root bone (j=0) position from deserialized pose features
        cur.rootLocalPosition = Vector3Lerp(poseFeat0.rootLocalPosition, poseFeat1.rootLocalPosition, sAlpha);

        // lookahead rotations are for all bones
        for (int j = 0; j < jc; ++j)
        {
            const Rot6d r = Rot6dLerp(poseFeat0.lookaheadLocalRotations[j], poseFeat1.lookaheadLocalRotations[j], sAlpha);
            cur.localRotations6d[j] = r;
            cur.lookaheadRotations6d[j] = r;
        }

        // root velocity (lookahead only, same for both old fields)
        const Vector3 rootVel = Vector3Lerp(poseFeat0.lookaheadRootVelocity, poseFeat1.lookaheadRootVelocity, sAlpha);
        cur.sampledRootVelocityRootSpace = rootVel;
        cur.sampledLookaheadRootVelocityRootSpace = rootVel;
        blendedRootVelocityRootSpace = Vector3Add(blendedRootVelocityRootSpace, Vector3Scale(rootVel, w));
        blendedLookaheadRootVelocityRootSpace = Vector3Add(blendedLookaheadRootVelocityRootSpace, Vector3Scale(rootVel, w));

        // yaw rate
        const float yawRate = Lerp(poseFeat0.rootYawRate, poseFeat1.rootYawRate, sAlpha);
        cur.sampledRootYawRate = yawRate;
        cur.sampledLookaheadRootYawRate = yawRate;
        blendedYawRate += yawRate * cur.fastNormalizedWeight;
        blendedLookaheadYawRate += yawRate * cur.fastNormalizedWeight;

        // display velocity
        cur.rootVelocityWorldForDisplayOnly = Vector3RotateByQuaternion(rootVel, cc->worldRotation);

        // toe positions and velocities from segment
        for (int side : sides)
        {
            const Vector3 toePos = Vector3Lerp(
                poseFeat0.lookaheadToePositionsRootSpace[side],
                poseFeat1.lookaheadToePositionsRootSpace[side], sAlpha);
            // lookahead toe = current toe for our purposes
            toeBlendedPositionRootSpace[side] = Vector3Add(toeBlendedPositionRootSpace[side], Vector3Scale(toePos, w));
            toeBlendedLookaheadPositionRootSpace[side] = Vector3Add(toeBlendedLookaheadPositionRootSpace[side], Vector3Scale(toePos, w));

            const Vector3 toeVel = Vector3Lerp(
                poseFeat0.toeVelocitiesRootSpace[side],
                poseFeat1.toeVelocitiesRootSpace[side], sAlpha);
            toeBlendedVelocityRootSpace[side] = Vector3Add(toeBlendedVelocityRootSpace[side], Vector3Scale(toeVel, w));

            const Vector3 nextToeVel = Vector3Lerp(
                poseFeat0.nextToeVelocitiesRootSpace[side],
                poseFeat1.nextToeVelocitiesRootSpace[side], sAlpha);
            nextToeBlendedVelocityRootSpace[side] = Vector3Add(nextToeBlendedVelocityRootSpace[side], Vector3Scale(nextToeVel, w));
        }

        // toe position difference (left - right) in root space
        const Vector3 toePosDiff = Vector3Lerp(
            poseFeat0.toePosDiffRootSpace, poseFeat1.toePosDiffRootSpace, sAlpha);
        toeBlendedPosDiffRootSpace = Vector3Add(toeBlendedPosDiffRootSpace, Vector3Scale(toePosDiff, w));

        const Vector3 nextToePosDiff = Vector3Lerp(
            poseFeat0.nextToePosDiffRootSpace, poseFeat1.nextToePosDiffRootSpace, sAlpha);
        nextToeBlendedPosDiffRootSpace = Vector3Add(nextToeBlendedPosDiffRootSpace, Vector3Scale(nextToePosDiff, w));

        // toe speed difference magnitude (always positive)
        const float speedDiff = Lerp(poseFeat0.toeSpeedDiff, poseFeat1.toeSpeedDiff, sAlpha);
        blendedToeSpeedDiff += speedDiff * w;

    }


    for (int side : sides)
    {
        // toes velocity in world space from blended toe velocities in root space
        cc->toeBlendedVelocityWorld[side] = Vector3RotateByQuaternion(toeBlendedVelocityRootSpace[side], cc->worldRotation);
        cc->nextToeBlendedVelocityWorld[side] = Vector3RotateByQuaternion(nextToeBlendedVelocityRootSpace[side], cc->worldRotation);
        // toes position in world space from blended toe positions in root space
        cc->toeBlendedPositionWorld[side] = Vector3Add(
            Vector3RotateByQuaternion(toeBlendedPositionRootSpace[side], cc->worldRotation),
            cc->worldPosition);
    }
    cc->toeBlendedPosDiffRootSpace = toeBlendedPosDiffRootSpace;
    cc->nextToeBlendedPosDiffRootSpace = nextToeBlendedPosDiffRootSpace;
    cc->blendedToeSpeedDiff = blendedToeSpeedDiff;

    // --- Rot6d blending using normalized weights
    std::vector<Vector3> blendedLocalPositions(jc); // only root will be blended, others are static offsets
    blendedLocalPositions[0] = Vector3Zero();

    std::vector<Rot6d> blendedLocalRotations(jc, Rot6dZero());
    std::vector<Rot6d> blendedLookaheadLocalRotations(jc, Rot6dZero());
       
    //std::vector<Vector3> angVelAccum(jc, Vector3Zero());

    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        const BlendCursor& cur = cc->cursors[ci];
        if (!cur.active) continue;
        const float w = cur.normalizedWeight;

        // Blending for root position (index 0)
        blendedLocalPositions[0] = Vector3Add(
            blendedLocalPositions[0], Vector3Scale(cur.rootLocalPosition, w));

        for (int j = 0; j < jc; ++j)
        {
           // weighted accumulation of Rot6d using helper
            Rot6dScaledAdd(w, cur.localRotations6d[j], blendedLocalRotations[j]);
        }
        for (int j = 0; j < jc; ++j)
        {
            Rot6dScaledAdd(w, cur.lookaheadRotations6d[j], blendedLookaheadLocalRotations[j]);
        }
    } // End of for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)

    // Initialize non-root blendedLocalPositions with static skeleton offsets
    for (int j = 1; j < jc; ++j)
    {
        blendedLocalPositions[j] = Vector3Scale(cc->skeleton->joints[j].offset, cc->scale);
    }

    // normalize blended Rot6d to get target rotations
    for (int j = 0; j < jc; ++j)
    {
        // normalize current rotations
        Rot6d blended = blendedLocalRotations[j];
        const float lenA = sqrtf(blended.ax * blended.ax + blended.ay * blended.ay + blended.az * blended.az);

        if (lenA < 1e-6f)
        {
            blendedLocalRotations[j] = Rot6dIdentity();
            //assertEvenInRelease(false);
        }

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

    // Save lookahead pose for debug visualization
    if (config.drawLookaheadPose)
    {
        for (int j = 0; j < jc; ++j)
        {
            if (j == 0)
            {
                cc->xformLookahead.localPositions[j] = blendedLocalPositions[0];
            }
            else
            {
                cc->xformLookahead.localPositions[j] = Vector3Scale(cc->skeleton->joints[j].offset, cc->scale);
            }
            Rot6d lookaheadRot = blendedLookaheadLocalRotations[j];
            Rot6dNormalize(lookaheadRot);
            Rot6dToQuaternion(lookaheadRot, cc->xformLookahead.localRotations[j]);
        }
        // FK
        TransformDataForwardKinematics(&cc->xformLookahead);
        // Transform to world space
        for (int i = 0; i < jc; ++i)
        {
            cc->xformLookahead.globalPositions[i] = Vector3Add(
                Vector3RotateByQuaternion(cc->xformLookahead.globalPositions[i], cc->worldRotation),
                cc->worldPosition);
            cc->xformLookahead.globalRotations[i] = QuaternionMultiply(
                cc->worldRotation, cc->xformLookahead.globalRotations[i]);
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
                if (j == 0)
                {
                    cc->lookaheadDragLocalPositions[j] = blendedLocalPositions[0];
                }
                else
                {
                    cc->lookaheadDragLocalPositions[j] = Vector3Scale(cc->skeleton->joints[j].offset, cc->scale);
                }
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

        // Lerp for root position (j=0)
        cc->lookaheadDragLocalPositions[0] =
            Vector3Lerp(cc->lookaheadDragLocalPositions[0], blendedLocalPositions[0], lookaheadProjectionAlpha);

        // For other bones (j > 0), positions are static and do not lerp.

        // Apply dragged rotations (convert from Rot6d to quaternions)
        for (int j = 0; j < jc; ++j)
        {
            Rot6dToQuaternion(cc->lookaheadDragLocalRotations6d[j], cc->xformData.localRotations[j]);
        }

        // Apply dragged positions
        cc->xformData.localPositions[0] = cc->lookaheadDragLocalPositions[0];
        for (int j = 1; j < jc; ++j)
        {
            cc->xformData.localPositions[j] = Vector3Scale(cc->skeleton->joints[j].offset, cc->scale);
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
            if (j == 0)
            {
                cc->xformData.localPositions[j] = blendedLocalPositions[0];
            }
            else
            {
                cc->xformData.localPositions[j] = Vector3Scale(cc->skeleton->joints[j].offset, cc->scale);
            }
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

    // per-cursor FK for debug visualization
    for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
    {
        BlendCursor& cur = cc->cursors[ci];
        if (!cur.active) continue;

        for (int j = 0; j < jc; ++j)
        {
            if (j == 0)
                cur.globalPositions[j] = cur.rootLocalPosition;
            else
                cur.globalPositions[j] = Vector3Scale(cc->skeleton->joints[j].offset, cc->scale);
            Rot6dToQuaternion(cur.localRotations6d[j], cur.globalRotations[j]);
        }

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
    // Update virtual toe positions using lookahead dragging
    // lookaheadDragToePos: drags toward blended lookahead target (unconstrained, for unlock detection)
    // virtualToePos: speed-clamped for IK (constrained)
    const float lookaheadTime = Max(dt, db->featuresConfig.poseDragLookaheadTime);
    const float toeAlpha = dt / lookaheadTime;
    assert(toeAlpha <= 1.0f);

    // determine which foot should be faster from blended velocities
    const float leftBlendedSpeed = Vector3Length2D(cc->toeBlendedVelocityWorld[SIDE_LEFT]);
    const float rightBlendedSpeed = Vector3Length2D(cc->toeBlendedVelocityWorld[SIDE_RIGHT]);
    const int fastSide = (leftBlendedSpeed >= rightBlendedSpeed) ? SIDE_LEFT : SIDE_RIGHT;

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

                    // speed floor: fast foot must move at least toeSpeedDiff faster than slow foot
                    // since slow foot is typically near-stationary, floor ≈ toeSpeedDiff
                    constexpr bool doFastFootVelFloor = false;
                    if (doFastFootVelFloor && side == fastSide && cc->blendedToeSpeedDiff > 0.1f)
                    {
                        const Vector3 postClampDisp = Vector3Subtract(newVirtualToePos, prevVirtualToePos);
                        const float postClampDistXZ = Vector3Length2D(postClampDisp);
                        const float minDistXZ = cc->blendedToeSpeedDiff * dt;
                        if (postClampDistXZ < minDistXZ)
                        {
                            // scale in original target direction to enforce minimum speed
                            const float scale = minDistXZ / distXZ;
                            newVirtualToePos.x = prevVirtualToePos.x + displacement.x * scale;
                            newVirtualToePos.z = prevVirtualToePos.z + displacement.z * scale;
                        }
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


// TODO: remove the vertical component of root motion
// velocity in poseGenFeatures


