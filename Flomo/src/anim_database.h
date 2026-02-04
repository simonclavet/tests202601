#pragma once

#include <string>
#include <vector>
#include <span>
#include <cmath>

#include "raylib.h"
#include "raymath.h"
#include "math_utils.h"
#include "utils.h"
#include "bvh_parser.h"
#include "transform_data.h"
#include "character_data.h"
#include "app_config.h"


static inline int FindClipForMotionFrame(const AnimDatabase* db, int frame)
{
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        if (frame >= db->clipStartFrame[c] && frame < db->clipEndFrame[c]) return c;
    }
    return -1;
}

// Updated AnimDatabaseRebuild: require all animations to match canonical skeleton.
// Populate localJointPositions/localJointRotations6d as well as global arrays.
// If any clip mismatches jointCount we invalidate the DB (db->valid = false).
static void AnimDatabaseRebuild(AnimDatabase* db, const CharacterData* characterData)
{
    using std::vector;
    using std::string;
    using std::span;

    LOG_PROFILE_START(AnimDatabaseRebuild);

    AnimDatabaseFree(db);

    db->animCount = characterData->count;
    db->animStartFrame.resize(db->animCount);
    db->animFrameCount.resize(db->animCount);
    db->animFrameTime.resize(db->animCount);
    db->motionFrameCount = 0;
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

    if (db->animCount == 0 || db->totalFrames == 0)
    {
        TraceLog(LOG_INFO, "AnimDatabase: no animations available for motion DB");
        return;
    }

    const BVHData* canonBvh = &characterData->bvhData[0];
    db->jointCount = canonBvh->jointCount;

    // STRICT: require every clip to have the same jointCount as the canonical skeleton.
    for (int a = 0; a < db->animCount; ++a)
    {
        const BVHData* bvh = &characterData->bvhData[a];
        if (bvh->jointCount != db->jointCount)
        {
            TraceLog(LOG_WARNING, "AnimDatabase: incompatible anim %d (%s) - jointCount mismatch (%d != %d). Aborting DB build.",
                a, characterData->filePaths[a].c_str(), bvh->jointCount, db->jointCount);
            db->motionFrameCount = 0;
            db->valid = false;
            return;
        }
    }

    // Compute total frames (all clips included)
    int includedFrames = 0;
    for (int a = 0; a < db->animCount; ++a)
    {
        includedFrames += characterData->bvhData[a].frameCount;
    }

    if (includedFrames == 0)
    {
        TraceLog(LOG_WARNING, "AnimDatabase: no compatible animations for motion DB (jointCount=%d)", db->jointCount);
        db->motionFrameCount = 0;
        db->valid = false;
        return;
    }

    vector<string> jointNames;
    jointNames.reserve((size_t)canonBvh->jointCount);
    for (int j = 0; j < canonBvh->jointCount; ++j)
    {
        // BVHJointData::name is now string
        jointNames.push_back(canonBvh->joints[j].name);
    }

    // Reset indices
    db->hipJointIndex = -1;
    for (int side : sides)
    {
        db->toeIndices[side] = -1;
        db->footIndices[side] = -1;
        db->lowlegIndices[side] = -1;
        db->uplegIndices[side] = -1;
    }

    // HIP candidates (lowercase)
    vector<string> hipCandidates = { "hips", "hip", "pelvis", "root" };

    // Leg chain candidates (lowercase) - lafan1, mixamo, unity humanoid, blender conventions
    const vector<string> leftToeCandidates = { "lefttoebase", "lefttoe", "left_toe", "l_toe", "toe.l", "toe_l" };
    const vector<string> rightToeCandidates = { "righttoebase", "righttoe", "right_toe", "r_toe", "toe.r", "toe_r" };
    const vector<vector<string>> toeCandidates = { leftToeCandidates, rightToeCandidates };

    const vector<string> leftFootCandidates = { "leftfoot", "left_foot", "l_foot", "foot.l", "foot_l", "leftankle" };
    const vector<string> rightFootCandidates = { "rightfoot", "right_foot", "r_foot", "foot.r", "foot_r", "rightankle" };
    const vector<vector<string>> footCandidates = { leftFootCandidates, rightFootCandidates };

    const vector<string> leftLowlegCandidates = { "leftleg", "left_leg", "l_leg", "shin.l", "shin_l", "leftlowerleg", "left_shin", "leftcalf", "calf.l" };
    const vector<string> rightLowlegCandidates = { "rightleg", "right_leg", "r_leg", "shin.r", "shin_r", "rightlowerleg", "right_shin", "rightcalf", "calf.r" };
    const vector<vector<string>> lowlegCandidates = { leftLowlegCandidates, rightLowlegCandidates };

    const vector<string> leftUplegCandidates = { "leftupleg", "left_upleg", "l_upleg", "thigh.l", "thigh_l", "leftupperleg", "left_thigh", "leftthigh" };
    const vector<string> rightUplegCandidates = { "rightupleg", "right_upleg", "r_upleg", "thigh.r", "thigh_r", "rightupperleg", "right_thigh", "rightthigh" };
    const vector<vector<string>> uplegCandidates = { leftUplegCandidates, rightUplegCandidates };

    // Use helper to find hip
    db->hipJointIndex = FindJointIndexByNames(canonBvh, hipCandidates);

    if (db->hipJointIndex == -1)
    {
        TraceLog(LOG_WARNING, "can't find hip joint: aborting animdatabase building");
        return;
    }

    // Find leg chain joints for each side
    for (int side : sides)
    {
        db->toeIndices[side] = FindJointIndexByNames(canonBvh, toeCandidates[side]);
        db->footIndices[side] = FindJointIndexByNames(canonBvh, footCandidates[side]);
        db->lowlegIndices[side] = FindJointIndexByNames(canonBvh, lowlegCandidates[side]);
        db->uplegIndices[side] = FindJointIndexByNames(canonBvh, uplegCandidates[side]);

        const char* sideName = (side == SIDE_LEFT) ? "left" : "right";

        if (db->toeIndices[side] < 0)
        {
            TraceLog(LOG_WARNING, "AnimDatabase: %s toe not found, aborting", sideName);
            return;
        }
        if (db->footIndices[side] < 0)
        {
            TraceLog(LOG_WARNING, "AnimDatabase: %s foot not found, aborting", sideName);
            return;
        }
        if (db->lowlegIndices[side] < 0)
        {
            TraceLog(LOG_WARNING, "AnimDatabase: %s lowleg (shin) not found, aborting", sideName);
            return;
        }
        if (db->uplegIndices[side] < 0)
        {
            TraceLog(LOG_WARNING, "AnimDatabase: %s upleg (thigh) not found, aborting", sideName);
            return;
        }

        // Verify parent chain: toe->foot->lowleg->upleg->hip
        {
            const int toeParent = canonBvh->joints[db->toeIndices[side]].parent;
            const int footParent = canonBvh->joints[db->footIndices[side]].parent;
            const int lowlegParent = canonBvh->joints[db->lowlegIndices[side]].parent;
            const int uplegParent = canonBvh->joints[db->uplegIndices[side]].parent;

            if (toeParent != db->footIndices[side])
            {
                TraceLog(LOG_WARNING, "AnimDatabase: %s toe parent (%d) != foot (%d), chain broken",
                    sideName, toeParent, db->footIndices[side]);
            }
            if (footParent != db->lowlegIndices[side])
            {
                TraceLog(LOG_WARNING, "AnimDatabase: %s foot parent (%d) != lowleg (%d), chain broken",
                    sideName, footParent, db->lowlegIndices[side]);
            }
            if (lowlegParent != db->uplegIndices[side])
            {
                TraceLog(LOG_WARNING, "AnimDatabase: %s lowleg parent (%d) != upleg (%d), chain broken",
                    sideName, lowlegParent, db->uplegIndices[side]);
            }
            if (uplegParent != db->hipJointIndex)
            {
                TraceLog(LOG_WARNING, "AnimDatabase: %s upleg parent (%d) != hip (%d), chain broken",
                    sideName, uplegParent, db->hipJointIndex);
            }

            TraceLog(LOG_INFO, "AnimDatabase: %s leg chain: hip(%d)->upleg(%d)->lowleg(%d)->foot(%d)->toe(%d)",
                sideName, db->hipJointIndex, db->uplegIndices[side], db->lowlegIndices[side],
                db->footIndices[side], db->toeIndices[side]);
        }
    }

    // Log resolved feature indices
    TraceLog(LOG_INFO, "AnimDatabase: hip=%d", db->hipJointIndex);


    // allocate compact storage [motionFrameCount x jointCount]
    db->motionFrameCount = includedFrames;
    db->jointPositionsAnimSpace.resize(db->motionFrameCount, db->jointCount);
    db->jointRotationsAnimSpace.resize(db->motionFrameCount, db->jointCount);
    db->jointVelocitiesRootSpace.resize(db->motionFrameCount, db->jointCount);
    db->jointAccelerationsRootSpace.resize(db->motionFrameCount, db->jointCount);
    db->localJointPositions.resize(db->motionFrameCount, db->jointCount);
    db->localJointRotations6d.resize(db->motionFrameCount, db->jointCount);
    db->localJointAngularVelocities.resize(db->motionFrameCount, db->jointCount);
    db->lookaheadLocalRotations6d.resize(db->motionFrameCount, db->jointCount);

    // sample each compatible clip frame and fill the flat arrays
    TransformData tmpXform;
    TransformDataInit(&tmpXform);
    TransformDataResize(&tmpXform, canonBvh); // sized to canonical skeleton

    int motionFrameIdx = 0;
    for (int a = 0; a < db->animCount; ++a)
    {
        const BVHData* bvh = &characterData->bvhData[a];
        db->clipStartFrame.push_back(motionFrameIdx);

        for (int f = 0; f < bvh->frameCount; ++f)
        {
            TransformDataSampleFrame(&tmpXform, bvh, f, characterData->scales[a]);
            TransformDataForwardKinematics(&tmpXform);

            span<Vector3> globalPos = db->jointPositionsAnimSpace.row_view(motionFrameIdx);
            span<Quaternion> globalRot = db->jointRotationsAnimSpace.row_view(motionFrameIdx);
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

    // Compute velocities for each joint at each frame, then transform to root space
    // Velocity at frame i is defined at midpoint between frame i and i+1: v = (pos[i+1] - pos[i]) / frameTime
    // Transformed by inverse root yaw at frame i (start of interval)
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float invFrameTime = 1.0f / frameTime;
        const int rootIdx = 0;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);
            const int nextF = isLastFrame ? f : (f + 1);
            const int prevF = isLastFrame ? (f - 1) : f;

            span<Vector3> velRow = db->jointVelocitiesRootSpace.row_view(f);

            // handle edge case: single-frame clip
            if (clipEnd - clipStart <= 1)
            {
                for (int j = 0; j < db->jointCount; ++j)
                {
                    velRow[j] = Vector3Zero();
                }
                continue;
            }

            span<const Vector3> pos0Row = db->jointPositionsAnimSpace.row_view(prevF);
            span<const Vector3> pos1Row = db->jointPositionsAnimSpace.row_view(nextF);

            // Get root yaw at frame f (start of interval) for transforming to root space
            span<const Rot6d> rot0 = db->localJointRotations6d.row_view(f);
            const float rootYaw = Rot6dGetYaw(rot0[rootIdx]);
            const Rot6d invYawRot = Rot6dFromYaw(-rootYaw);

            for (int j = 0; j < db->jointCount; ++j)
            {
                Vector3 velAnimSpace = Vector3Scale(Vector3Subtract(pos1Row[j], pos0Row[j]), invFrameTime);
                // Transform to root space (heading-relative)
                Vector3 velRootSpace;
                Rot6dTransformVector(invYawRot, velAnimSpace, velRootSpace);
                velRow[j] = velRootSpace;
            }
        }
    }

    // Compute accelerations for each joint at each frame (also in root space)
    // Acceleration at frame i: a = (vel[i+1] - vel[i]) / frameTime
    // Since velocities are in root space, accelerations are too
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

            span<Vector3> accRow = db->jointAccelerationsRootSpace.row_view(f);

            // handle edge case: single-frame or two-frame clip
            if (clipEnd - clipStart <= 2)
            {
                for (int j = 0; j < db->jointCount; ++j)
                {
                    accRow[j] = Vector3Zero();
                }
                continue;
            }

            span<const Vector3> vel0Row = db->jointVelocitiesRootSpace.row_view(prevF);
            span<const Vector3> vel1Row = db->jointVelocitiesRootSpace.row_view(nextF);

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

    // Compute lookahead poses: pose[f] + n * (pose[f+1] - pose[f]) = n*pose[f+1] - (n-1)*pose[f]
    // where n = lookaheadTime / frameTime (extrapolates lookaheadTime seconds ahead)
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float n = db->poseDragLookaheadTime / frameTime;
        const float nextWeight = n;
        const float currWeight = 1.0f - n;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);

            span<Rot6d> lookaheadRow = db->lookaheadLocalRotations6d.row_view(f);
            span<const Rot6d> currRow = db->localJointRotations6d.row_view(f);

            if (isLastFrame || clipEnd - clipStart <= 1)
            {
                // no next frame to extrapolate from, just copy current
                for (int j = 0; j < db->jointCount; ++j)
                {
                    lookaheadRow[j] = currRow[j];
                }
            }
            else
            {
                span<const Rot6d> nextRow = db->localJointRotations6d.row_view(f + 1);

                for (int j = 0; j < db->jointCount; ++j)
                {
                    Rot6d curr = currRow[j];
                    Rot6d next = nextRow[j];

                    // For hip (joint 0): strip yaw BEFORE extrapolation
                    // This prevents yaw changes from affecting pitch/roll during extrapolation
                    if (j == 0)
                    {
                        Rot6dRemoveYComponent(curr, curr);
                        Rot6dRemoveYComponent(next, next);
                    }

                    // lookahead = (1-n)*curr + n*next = curr + n*(next - curr)
                    Rot6d result;
                    result.ax = currWeight * curr.ax + nextWeight * next.ax;
                    result.ay = currWeight * curr.ay + nextWeight * next.ay;
                    result.az = currWeight * curr.az + nextWeight * next.az;
                    result.bx = currWeight * curr.bx + nextWeight * next.bx;
                    result.by = currWeight * curr.by + nextWeight * next.by;
                    result.bz = currWeight * curr.bz + nextWeight * next.bz;
                    Rot6dNormalize(result);
                    lookaheadRow[j] = result;
                }
            }
        }
    }

    // Compute root motion velocities in LOCAL space (heading-relative, XZ only)
    // Velocity at frame f represents movement during [f, f+1], transformed by yaw at frame f
    // This makes velocities independent of animation facing direction, suitable for NN features
    db->rootMotionVelocitiesRootSpace.resize(db->motionFrameCount);
    db->rootMotionYawRates.resize(db->motionFrameCount);

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float invFrameTime = 1.0f / frameTime;
        const int rootIdx = 0;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);

            if (clipEnd - clipStart <= 1)
            {
                // Single frame clip - zero velocity
                db->rootMotionVelocitiesRootSpace[f] = Vector3Zero();
                db->rootMotionYawRates[f] = 0.0f;
                continue;
            }

            if (isLastFrame)
            {
                // Last frame - copy from previous
                db->rootMotionVelocitiesRootSpace[f] = db->rootMotionVelocitiesRootSpace[f - 1];
                db->rootMotionYawRates[f] = db->rootMotionYawRates[f - 1];
            }
            else
            {
                // Compute XZ linear velocity from local root positions (animation space)
                span<const Vector3> pos0 = db->localJointPositions.row_view(f);
                span<const Vector3> pos1 = db->localJointPositions.row_view(f + 1);

                Vector3 velAnim = Vector3Subtract(pos1[rootIdx], pos0[rootIdx]);
                velAnim.y = 0.0f;  // XZ only
                velAnim = Vector3Scale(velAnim, invFrameTime);

                // Get yaw at frame f (start of interval) and transform velocity to local space
                span<const Rot6d> rot0 = db->localJointRotations6d.row_view(f);
                const float yaw0 = Rot6dGetYaw(rot0[rootIdx]);
                const Rot6d invYawRot = Rot6dFromYaw(-yaw0);

                Vector3 velLocal;
                Rot6dTransformVector(invYawRot, velAnim, velLocal);
                db->rootMotionVelocitiesRootSpace[f] = velLocal;

                // Compute yaw angular velocity from local root rotations
                span<const Rot6d> rot1 = db->localJointRotations6d.row_view(f + 1);
                const float yaw1 = Rot6dGetYaw(rot1[rootIdx]);

                const float deltaYaw = WrapAngleToPi(yaw1 - yaw0);
                db->rootMotionYawRates[f] = deltaYaw * invFrameTime;
            }
        }
    }

    // Compute lookahead root motion velocities (extrapolated for smooth anticipation)
    // lookahead = vel[f] + n*(vel[f+1] - vel[f]) where n = lookaheadTime / frameTime
    // These are also in local space since they're derived from rootMotionVelocitiesRootSpace
    db->lookaheadRootMotionVelocitiesRootSpace.resize(db->motionFrameCount);
    db->lookaheadRootMotionYawRates.resize(db->motionFrameCount);

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float n = db->poseDragLookaheadTime / frameTime;
        const float nextWeight = n;
        const float currWeight = 1.0f - n;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);

            if (isLastFrame || clipEnd - clipStart <= 1)
            {
                // no next frame to extrapolate from, just copy current
                db->lookaheadRootMotionVelocitiesRootSpace[f] = db->rootMotionVelocitiesRootSpace[f];
                db->lookaheadRootMotionYawRates[f] = db->rootMotionYawRates[f];
            }
            else
            {
                // extrapolate: lookahead = curr + n*(next - curr) = (1-n)*curr + n*next
                const Vector3 currVel = db->rootMotionVelocitiesRootSpace[f];
                const Vector3 nextVel = db->rootMotionVelocitiesRootSpace[f + 1];
                db->lookaheadRootMotionVelocitiesRootSpace[f] = Vector3Add(
                    Vector3Scale(currVel, currWeight),
                    Vector3Scale(nextVel, nextWeight));

                const float currYawRate = db->rootMotionYawRates[f];
                const float nextYawRate = db->rootMotionYawRates[f + 1];
                db->lookaheadRootMotionYawRates[f] = currWeight * currYawRate + nextWeight * nextYawRate;
            }
        }
    }

    // Compute lookahead hips heights (extrapolated Y position of hip joint)
    // lookahead = (1-n)*curr + n*next where n = lookaheadTime / frameTime
    db->lookaheadHipsHeights.resize(db->motionFrameCount);

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float n = db->poseDragLookaheadTime / frameTime;
        const float nextWeight = n;
        const float currWeight = 1.0f - n;

        const int hipIdx = 0;  // root/hip is joint 0

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);

            if (isLastFrame || clipEnd - clipStart <= 1)
            {
                // no next frame to extrapolate from, just use current
                std::span<const Vector3> currPos = db->localJointPositions.row_view(f);
                db->lookaheadHipsHeights[f] = currPos[hipIdx].y;
            }
            else
            {
                // extrapolate: lookahead = (1-n)*curr + n*next
                std::span<const Vector3> currPos = db->localJointPositions.row_view(f);
                std::span<const Vector3> nextPos = db->localJointPositions.row_view(f + 1);

                const float currHeight = currPos[hipIdx].y;
                const float nextHeight = nextPos[hipIdx].y;

                db->lookaheadHipsHeights[f] = currWeight * currHeight + nextWeight * nextHeight;
            }
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: built motion DB with %d frames and %d joints",
        db->motionFrameCount, db->jointCount);

    // Compute yaw-free hip rotation track (separate from the full rotation, for clean dragging)
    db->hipRotationYawFree.resize(db->motionFrameCount);
    db->lookaheadHipRotationYawFree.resize(db->motionFrameCount);

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float n = db->poseDragLookaheadTime / frameTime;
        const float nextWeight = n;
        const float currWeight = 1.0f - n;

        const int hipIdx = 0;  // root/hip is joint 0

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);

            // Get hip rotation and strip yaw
            span<const Rot6d> rotRow = db->localJointRotations6d.row_view(f);
            Rot6d hipNoYaw;
            Rot6dRemoveYComponent(rotRow[hipIdx], hipNoYaw);
            db->hipRotationYawFree[f] = hipNoYaw;

            if (isLastFrame || clipEnd - clipStart <= 1)
            {
                // no next frame to extrapolate from, just use current
                db->lookaheadHipRotationYawFree[f] = hipNoYaw;
            }
            else
            {
                // Get next frame's yaw-free hip rotation
                span<const Rot6d> nextRotRow = db->localJointRotations6d.row_view(f + 1);
                Rot6d nextHipNoYaw;
                Rot6dRemoveYComponent(nextRotRow[hipIdx], nextHipNoYaw);

                // Extrapolate using Rot6d lerp: lookahead = (1-n)*curr + n*next
                // We blend the raw Rot6d values then normalize
                Rot6d lookahead;
                lookahead.ax = currWeight * hipNoYaw.ax + nextWeight * nextHipNoYaw.ax;
                lookahead.ay = currWeight * hipNoYaw.ay + nextWeight * nextHipNoYaw.ay;
                lookahead.az = currWeight * hipNoYaw.az + nextWeight * nextHipNoYaw.az;
                lookahead.bx = currWeight * hipNoYaw.bx + nextWeight * nextHipNoYaw.bx;
                lookahead.by = currWeight * hipNoYaw.by + nextWeight * nextHipNoYaw.by;
                lookahead.bz = currWeight * hipNoYaw.bz + nextWeight * nextHipNoYaw.bz;
                Rot6dNormalize(lookahead);
                db->lookaheadHipRotationYawFree[f] = lookahead;
            }
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: computed yaw-free hip rotation track");

    // Compute toe positions in root space (relative to hip, heading-aligned)
    // and lookahead toe positions (extrapolated)
    for (int side : sides)
    {
        db->toePositionsRootSpace[side].resize(db->motionFrameCount);
        db->lookaheadToePositionsRootSpace[side].resize(db->motionFrameCount);
    }

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float n = db->poseDragLookaheadTime / frameTime;
        const float nextWeight = n;
        const float currWeight = 1.0f - n;
        const int hipIdx = db->hipJointIndex;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);
            span<const Vector3> posRow = db->jointPositionsAnimSpace.row_view(f);
            span<const Quaternion> rotRow = db->jointRotationsAnimSpace.row_view(f);

            // Get hip position and yaw for transforming to root space
            const Vector3 hipPos = posRow[hipIdx];
            const Quaternion hipYaw = QuaternionYComponent(rotRow[hipIdx]);
            const Quaternion invHipYaw = QuaternionInvert(hipYaw);

            for (int side : sides)
            {
                const int toeIdx = db->toeIndices[side];

                // Current toe position in root space
                const Vector3 toePos = posRow[toeIdx];
                //const Vector3 hipToToe = Vector3Subtract(toePos, hipPos);
                
                // Project hip to ground (XZ from hip, Y=0)
                const Vector3 groundRootPos = Vector3{ hipPos.x, 0.0f, hipPos.z };

                // Offset from ground-projected root
                const Vector3 groundRootToToe = Vector3Subtract(toePos, groundRootPos);

                // Transform to heading-aligned space
                const Vector3 toePosRootSpace = Vector3RotateByQuaternion(groundRootToToe, invHipYaw);
                db->toePositionsRootSpace[side][f] = toePosRootSpace;
                
                // Lookahead toe position
                if (isLastFrame || clipEnd - clipStart <= 1)
                {
                    db->lookaheadToePositionsRootSpace[side][f] = toePosRootSpace;
                }
                else
                {
                    // Get next frame's toe and hip positions
                    span<const Vector3> nextPosRow = db->jointPositionsAnimSpace.row_view(f + 1);
                    const Vector3 nextHipPos = nextPosRow[hipIdx];
                    const Vector3 nextToePos = nextPosRow[toeIdx];

                    //const Vector3 nextGroundRootPos = Vector3{ nextHipPos.x, 0.0f, nextHipPos.z };

                    // Compute leg vector (toe relative to hip) at next frame
                    //const Vector3 groundRootToNextToe = Vector3Subtract(nextToePos, nextGroundRootPos);
                    const Vector3 groundRootToNextToe = Vector3Subtract(nextToePos, groundRootPos);

                    //span<const Quaternion> nextRotRow = db->jointRotationsAnimSpace.row_view(f + 1);
                    //const Quaternion nextHipYaw = (hipIdx >= 0) ? QuaternionYComponent(nextRotRow[hipIdx]) : QuaternionIdentity();
                    //const Quaternion invNextHipYaw = QuaternionInvert(nextHipYaw);

                    // Rotate into current frame's heading space (same rotational frame as toePosRootSpace)
                    const Vector3 nextToePosRootSpace = Vector3RotateByQuaternion(groundRootToNextToe, invHipYaw);
                    //const Vector3 nextToePosRootSpace = Vector3RotateByQuaternion(groundRootToNextToe, invNextHipYaw);

                    // Extrapolate: lookahead = (1-n)*curr + n*next
                    db->lookaheadToePositionsRootSpace[side][f] = Vector3Add(
                        Vector3Scale(toePosRootSpace, currWeight),
                        Vector3Scale(nextToePosRootSpace, nextWeight));
                }
            }
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: computed toe positions in root space");

    const MotionMatchingFeaturesConfig& cfg = db->featuresConfig;

    db->featureDim = 0;
    if (cfg.IsFeatureEnabled(FeatureType::ToePos)) db->featureDim += 4;
    if (cfg.IsFeatureEnabled(FeatureType::ToeVel)) db->featureDim += 4;
    if (cfg.IsFeatureEnabled(FeatureType::ToeDiff)) db->featureDim += 2;
    if (cfg.IsFeatureEnabled(FeatureType::FutureVel)) db->featureDim += (int)cfg.futureTrajPointTimes.size() * 2;
    if (cfg.IsFeatureEnabled(FeatureType::FutureVelClamped)) db->featureDim += (int)cfg.futureTrajPointTimes.size() * 2;
    if (cfg.IsFeatureEnabled(FeatureType::FutureSpeed)) db->featureDim += (int)cfg.futureTrajPointTimes.size();
    if (cfg.IsFeatureEnabled(FeatureType::PastPosition)) db->featureDim += 2;

    db->features.clear();
    db->featureNames.clear();

    db->features.resize(db->motionFrameCount, db->featureDim);
    db->features.fill(0.0f);

    if (db->featureDim == 0)
    {
        TraceLog(LOG_WARNING, "AnimDatabase: no features enabled in configuration");
        return;
    }

    // Populate features from jointPositions and jointRotations
    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        const bool isFirstFrame = f == 0;
        const int clipIdx = FindClipForMotionFrame(db, f);
        assert(clipIdx != -1);

        const int clipStart = db->clipStartFrame[clipIdx];
        const int clipEnd = db->clipEndFrame[clipIdx];
        const float dt = db->animFrameTime[clipIdx];
        assert(dt > 1e-8f);

        span<const Vector3> posRow = db->jointPositionsAnimSpace.row_view(f);
        span<const Quaternion> rotRow = db->jointRotationsAnimSpace.row_view(f);
        span<float> featRow = db->features.row_view(f);

        Vector3 hipPos = { 0.0f, 0.0f, 0.0f };
        Vector3 leftPos = { 0.0f, 0.0f, 0.0f };
        Vector3 rightPos = { 0.0f, 0.0f, 0.0f };

        if (db->hipJointIndex >= 0)
        {
            hipPos = posRow[db->hipJointIndex];
        }

        const int leftIdx = db->toeIndices[SIDE_LEFT];
        const int rightIdx = db->toeIndices[SIDE_RIGHT];

        if (leftIdx >= 0)
        {
            leftPos = posRow[leftIdx];
        }
        if (rightIdx >= 0)
        {
            rightPos = posRow[rightIdx];
        }

        // Extract hip yaw (if available) once per frame
        Quaternion hipYaw = QuaternionIdentity();
        if (db->hipJointIndex >= 0)
        {
            hipYaw = QuaternionYComponent(rotRow[db->hipJointIndex]);
        }
        const Quaternion invHipYaw = QuaternionInvert(hipYaw);

        int currentFeature = 0;

        // Precompute local toe positions (hip horizontal frame) - used by pos and diff
        const Vector3 hipToLeft = Vector3Subtract(leftPos, hipPos);
        const Vector3 localLeftPos = Vector3RotateByQuaternion(hipToLeft, invHipYaw);

        const Vector3 hipToRight = Vector3Subtract(rightPos, hipPos);
        const Vector3 localRightPos = Vector3RotateByQuaternion(hipToRight, invHipYaw);

        // POSITION: toePos->Left(X, Z), Right(X, Z)
        if (cfg.IsFeatureEnabled(FeatureType::ToePos))
        {
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
                db->featureTypes.push_back(FeatureType::ToePos);
                db->featureTypes.push_back(FeatureType::ToePos);
                db->featureTypes.push_back(FeatureType::ToePos);
                db->featureTypes.push_back(FeatureType::ToePos);
            }
        }

        // VELOCITY: toeVel -> compute world finite-difference then rotate into hip frame
        if (cfg.IsFeatureEnabled(FeatureType::ToeVel))
        {
            Vector3 localLeftVel = Vector3Zero();
            Vector3 localRightVel = Vector3Zero();

            if (f > clipStart && dt > 0.0f)
            {
                span<const Vector3> posPrevRow = db->jointPositionsAnimSpace.row_view(f - 1);

                if (leftIdx >= 0)
                {
                    const Vector3 deltaLeft = Vector3Subtract(leftPos, posPrevRow[leftIdx]);
                    const Vector3 velLeftWorld = Vector3Scale(deltaLeft, 1.0f / dt);
                    localLeftVel = Vector3RotateByQuaternion(velLeftWorld, invHipYaw);
                }

                if (rightIdx >= 0)
                {
                    const Vector3 deltaRight = Vector3Subtract(rightPos, posPrevRow[rightIdx]);
                    const Vector3 velRightWorld = Vector3Scale(deltaRight, 1.0f / dt);
                    localRightVel = Vector3RotateByQuaternion(velRightWorld, invHipYaw);
                }
            }


            featRow[currentFeature++] = localLeftVel.x;
            featRow[currentFeature++] = localLeftVel.z;
            featRow[currentFeature++] = localRightVel.x;
            featRow[currentFeature++] = localRightVel.z;

            if (isFirstFrame)
            {
                db->featureNames.push_back(string("LeftToeVelX"));
                db->featureNames.push_back(string("LeftToeVelZ"));
                db->featureNames.push_back(string("RightToeVelX"));
                db->featureNames.push_back(string("RightToeVelZ"));
                db->featureTypes.push_back(FeatureType::ToeVel);
                db->featureTypes.push_back(FeatureType::ToeVel);
                db->featureTypes.push_back(FeatureType::ToeVel);
                db->featureTypes.push_back(FeatureType::ToeVel);
            }
        }

        // DIFFERENCE: toeDifference = Left - Right (in hip horizontal frame) => (dx, dz)
        if (cfg.IsFeatureEnabled(FeatureType::ToeDiff))
        {
            featRow[currentFeature++] = localLeftPos.x - localRightPos.x;
            featRow[currentFeature++] = localLeftPos.z - localRightPos.z;

            if (isFirstFrame)
            {
                db->featureNames.push_back(string("ToeDiffX"));
                db->featureNames.push_back(string("ToeDiffZ"));
                db->featureTypes.push_back(FeatureType::ToeDiff);
                db->featureTypes.push_back(FeatureType::ToeDiff);
            }
        }        

        // FUTURE TRAJECTORY: future root velocity direction (XZ) at sample points
        // Compute velocity in animSpace at future frame, then transform to rootSpace relative to current frame
        if (cfg.IsFeatureEnabled(FeatureType::FutureVel))
        {
            const float frameTime = db->animFrameTime[clipIdx];
            const float invFrameTime = 1.0f / frameTime;
            const int rootIdx = 0;

            // Get current root yaw for transforming to root space
            const Quaternion currRootRot = rotRow[rootIdx];
            const Quaternion currRootYaw = QuaternionYComponent(currRootRot);
            const Quaternion invCurrRootYaw = QuaternionInvert(currRootYaw);

            for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
            {
                const float futureTime = cfg.futureTrajPointTimes[p];
                const int futureFrameOffset = (int)(futureTime / frameTime + 0.5f);
                const int futureFrame = f + futureFrameOffset;

                Vector3 futureVelRootSpace = Vector3Zero();

                // Check if future frame and next frame are within the same clip
                if (futureFrame >= clipStart && futureFrame < clipEnd - 1)
                {
                    // Compute velocity in animSpace at future frame: (pos[f+1] - pos[f]) / dt
                    span<const Vector3> futurePosRow0 = db->jointPositionsAnimSpace.row_view(futureFrame);
                    span<const Vector3> futurePosRow1 = db->jointPositionsAnimSpace.row_view(futureFrame + 1);
                    Vector3 futureVelAnimSpace = Vector3Scale(
                        Vector3Subtract(futurePosRow1[rootIdx], futurePosRow0[rootIdx]), invFrameTime);
                    futureVelAnimSpace.y = 0.0f;  // XZ only

                    // Transform to rootSpace relative to current frame's yaw
                    futureVelRootSpace = Vector3RotateByQuaternion(futureVelAnimSpace, invCurrRootYaw);
                }
                // else: future frame outside clip or at last frame, leave as zero

                featRow[currentFeature++] = futureVelRootSpace.x;
                featRow[currentFeature++] = futureVelRootSpace.z;

                if (isFirstFrame)
                {
                    char nameBufX[64];
                    char nameBufZ[64];
                    snprintf(nameBufX, sizeof(nameBufX), "FutureVelX_%.2fs", futureTime);
                    snprintf(nameBufZ, sizeof(nameBufZ), "FutureVelZ_%.2fs", futureTime);
                    db->featureNames.push_back(string(nameBufX));
                    db->featureNames.push_back(string(nameBufZ));
                    db->featureTypes.push_back(FeatureType::FutureVel);
                    db->featureTypes.push_back(FeatureType::FutureVel);
                }
            }
        }

        // FUTURE VELOCITY CLAMPED: future velocity clamped to max magnitude
        if (cfg.IsFeatureEnabled(FeatureType::FutureVelClamped))
        {
            constexpr float MaxFutureVelClampedMag = 1.0f;

            const float frameTime = db->animFrameTime[clipIdx];
            const float invFrameTime = 1.0f / frameTime;
            const int rootIdx = 0;

            const Quaternion currRootRot = rotRow[rootIdx];
            const Quaternion currRootYaw = QuaternionYComponent(currRootRot);
            const Quaternion invCurrRootYaw = QuaternionInvert(currRootYaw);

            for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
            {
                const float futureTime = cfg.futureTrajPointTimes[p];
                const int futureFrameOffset = (int)(futureTime / frameTime + 0.5f);
                const int futureFrame = f + futureFrameOffset;

                Vector3 futureVelRootSpace = Vector3Zero();

                if (futureFrame >= clipStart && futureFrame < clipEnd - 1)
                {
                    span<const Vector3> futurePosRow0 = db->jointPositionsAnimSpace.row_view(futureFrame);
                    span<const Vector3> futurePosRow1 = db->jointPositionsAnimSpace.row_view(futureFrame + 1);
                    Vector3 futureVelAnimSpace = Vector3Scale(
                        Vector3Subtract(futurePosRow1[rootIdx], futurePosRow0[rootIdx]), invFrameTime);
                    futureVelAnimSpace.y = 0.0f;  // XZ only

                    futureVelRootSpace = Vector3RotateByQuaternion(futureVelAnimSpace, invCurrRootYaw);

                    // Clamp to max magnitude
                    const float mag = Vector3Length(futureVelRootSpace);
                    if (mag > MaxFutureVelClampedMag)
                    {
                        futureVelRootSpace = Vector3Scale(futureVelRootSpace, MaxFutureVelClampedMag / mag);
                    }
                }

                featRow[currentFeature++] = futureVelRootSpace.x;
                featRow[currentFeature++] = futureVelRootSpace.z;

                if (isFirstFrame)
                {
                    char nameBufX[64];
                    char nameBufZ[64];
                    snprintf(nameBufX, sizeof(nameBufX), "FutureVelClampedX_%.2fs", futureTime);
                    snprintf(nameBufZ, sizeof(nameBufZ), "FutureVelClampedZ_%.2fs", futureTime);
                    db->featureNames.push_back(string(nameBufX));
                    db->featureNames.push_back(string(nameBufZ));
                    db->featureTypes.push_back(FeatureType::FutureVelClamped);
                    db->featureTypes.push_back(FeatureType::FutureVelClamped);
                }
            }
        }

        // FUTURE SPEED: scalar speed at future sample points
        // Computed from same root velocity used in FutureVel, just take magnitude
        if (cfg.IsFeatureEnabled(FeatureType::FutureSpeed))
        {
            const float frameTime = db->animFrameTime[clipIdx];
            const float invFrameTime = 1.0f / frameTime;
            const int rootIdx = 0;

            // Get current root yaw (not needed for speed, but kept for consistency)
            const Quaternion currRootRot = rotRow[rootIdx];
            const Quaternion currRootYaw = QuaternionYComponent(currRootRot);
            const Quaternion invCurrRootYaw = QuaternionInvert(currRootYaw);

            for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
            {
                const float futureTime = cfg.futureTrajPointTimes[p];
                const int futureFrameOffset = (int)(futureTime / frameTime + 0.5f);
                const int futureFrame = f + futureFrameOffset;

                float futureSpeed = 0.0f;

                // Check if future frame and next frame are within the same clip
                if (futureFrame >= clipStart && futureFrame < clipEnd - 1)
                {
                    // Compute velocity in animSpace at future frame: (pos[f+1] - pos[f]) / dt
                    span<const Vector3> futurePosRow0 = db->jointPositionsAnimSpace.row_view(futureFrame);
                    span<const Vector3> futurePosRow1 = db->jointPositionsAnimSpace.row_view(futureFrame + 1);
                    Vector3 futureVelAnimSpace = Vector3Scale(
                        Vector3Subtract(futurePosRow1[rootIdx], futurePosRow0[rootIdx]), invFrameTime);
                    futureVelAnimSpace.y = 0.0f;  // XZ only

                    // Compute speed (magnitude, no need for rotation transform)
                    futureSpeed = Vector3Length(futureVelAnimSpace);
                }
                // else: future frame outside clip or at last frame, leave as zero

                featRow[currentFeature++] = futureSpeed;

                if (isFirstFrame)
                {
                    char nameBuf[64];
                    snprintf(nameBuf, sizeof(nameBuf), "FutureSpeed_%.2fs", futureTime);
                    db->featureNames.push_back(string(nameBuf));
                    db->featureTypes.push_back(FeatureType::FutureSpeed);
                }
            }
        }

        // PAST POSITION: past hip position in current hip horizontal frame (XZ)
        if (cfg.IsFeatureEnabled(FeatureType::PastPosition))
        {
            Vector3 pastPosLocal = Vector3Zero();

            assert(db->hipJointIndex != -1);

            const float frameTime = db->animFrameTime[clipIdx];
            const float pastTime = cfg.pastTimeOffset;
            const int pastFrameOffset = (int)(pastTime / frameTime + 0.5f);
            const int pastFrame = f - pastFrameOffset;

            // Check if past frame is within the same clip
            if (pastFrame >= clipStart && pastFrame < clipEnd)
            {
                // Get past hip position
                span<const Vector3> pastPosRow = db->jointPositionsAnimSpace.row_view(pastFrame);
                const Vector3 pastHipPos = pastPosRow[db->hipJointIndex];

                // Compute vector from current hip to past hip
                const Vector3 hipToPastHip = Vector3Subtract(pastHipPos, hipPos);

                // Transform to current hip horizontal frame
                pastPosLocal = Vector3RotateByQuaternion(hipToPastHip, invHipYaw);
            }

            // Store only XZ components (horizontal position)
            featRow[currentFeature++] = pastPosLocal.x;
            featRow[currentFeature++] = pastPosLocal.z;

            if (isFirstFrame)
            {
                db->featureNames.push_back(string("PastPosX"));
                db->featureNames.push_back(string("PastPosZ"));
                db->featureTypes.push_back(FeatureType::PastPosition);
                db->featureTypes.push_back(FeatureType::PastPosition);
            }
        }

        assert(currentFeature == db->featureDim);
    }


    // Allocate mean and std vectors
    db->featuresMean.resize(db->featureDim, 0.0f);
    db->featuresStd.resize(db->featureDim, 0.0f);

    // Compute mean for each feature dimension
    for (int d = 0; d < db->featureDim; ++d)
    {
        double sum = 0.0;
        for (int f = 0; f < db->motionFrameCount; ++f)
        {
            sum += db->features.at(f, d);
        }
        db->featuresMean[d] = (float)(sum / db->motionFrameCount);
    }

    // Compute standard deviation for each feature dimension
    for (int d = 0; d < db->featureDim; ++d)
    {
        double sumSquaredDiff = 0.0;
        for (int f = 0; f < db->motionFrameCount; ++f)
        {
            const double diff = db->features.at(f, d) - db->featuresMean[d];
            sumSquaredDiff += diff * diff;
        }
        const double variance = sumSquaredDiff / db->motionFrameCount;
        db->featuresStd[d] = (float)std::sqrt(variance);

        // Avoid division by zero - use 1.0 if std is too small
        if (db->featuresStd[d] < 1e-8f)
        {
            db->featuresStd[d] = 1.0f;
        }
    }

    // Compute normalized features: (x - mean) / std
    db->normalizedFeatures.resize(db->motionFrameCount, db->featureDim);
    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        span<const float> featRow = db->features.row_view(f);
        span<float> normRow = db->normalizedFeatures.row_view(f);

        for (int d = 0; d < db->featureDim; ++d)
        {
            normRow[d] = (featRow[d] - db->featuresMean[d]) / db->featuresStd[d];
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: computed feature normalization (mean/std for %d dimensions)", db->featureDim);

    // Apply feature type weights to normalized features
    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        span<float> normRow = db->normalizedFeatures.row_view(f);

        for (int d = 0; d < db->featureDim; ++d)
        {
            const FeatureType featureType = db->featureTypes[d];
            const float weight = cfg.GetFeatureWeight(featureType);
            normRow[d] *= weight;
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: applied feature type weights to normalized features");

    // set db->valid true now that we completed full build
    db->valid = true;

    LOG_PROFILE_END(AnimDatabaseRebuild);

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

// Sample interpolated pose and motion data from AnimDatabase at animTime.
// Outputs: joint-local positions/rotations, angular velocities, lookahead rotations,
//          root position/rotation, root linear velocity (root space), root yaw rate.
// velocityTimeOffset: offset added to animTime when sampling velocities (use -dt/2 for midpoint sampling)
static inline void SamplePoseAndMotion(
    const AnimDatabase* db,
    int animIndex,
    float animTime,
    float velocityTimeOffset,
    std::vector<Vector3>& outPositions,
    std::vector<Rot6d>& outRotations6d,
    std::vector<Vector3>* outAngularVelocities,
    std::vector<Rot6d>* outLookaheadRotations6d,
    Vector3* outRootPos,
    Rot6d* outRootRot6d,
    Vector3* outRootVelocityRootSpace,            // root linear velocity (XZ, root space)
    float* outRootYawRate,                        // root yaw rate (rad/s)
    Vector3* outLookaheadRootVelocityRootSpace,   // lookahead root velocity (XZ, root space)
    float* outLookaheadRootYawRate,               // lookahead root yaw rate (rad/s)
    float* outLookaheadHipsHeight,                // lookahead hips Y position
    Rot6d* outHipRotationYawFree,                 // current hip rotation (yaw-free)
    Rot6d* outLookaheadHipRotationYawFree)        // lookahead hip rotation (yaw-free)
{
    using std::span;

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

    // sample lookahead rotations (extrapolated poses for lookahead dragging)
    if (outLookaheadRotations6d)
    {
        span<const Rot6d> laRow0 = db->lookaheadLocalRotations6d.row_view(clipStart + f0);
        span<const Rot6d> laRow1 = db->lookaheadLocalRotations6d.row_view(clipStart + f1);
        for (int j = 0; j < jointCount; ++j)
        {
            Rot6dLerp(laRow0[j], laRow1[j], alpha, (*outLookaheadRotations6d)[j]);
        }
    }

    if (outRootPos) *outRootPos = outPositions[0];
    if (outRootRot6d) *outRootRot6d = outRotations6d[0];

    // Sample root motion velocity at midpoint (animTime + velocityTimeOffset)
    // Velocities are in local space (heading-relative)
    if (outRootVelocityRootSpace || outRootYawRate)
    {
        int vf0, vf1;
        float vAlpha;
        GetInterFrameAlpha(db, animIndex, animTime + velocityTimeOffset, vf0, vf1, vAlpha);

        if (outRootVelocityRootSpace)
        {
            const Vector3 vel0 = db->rootMotionVelocitiesRootSpace[clipStart + vf0];
            const Vector3 vel1 = db->rootMotionVelocitiesRootSpace[clipStart + vf1];
            *outRootVelocityRootSpace = Vector3Lerp(vel0, vel1, vAlpha);
        }

        if (outRootYawRate)
        {
            const float rate0 = db->rootMotionYawRates[clipStart + vf0];
            const float rate1 = db->rootMotionYawRates[clipStart + vf1];
            *outRootYawRate = Lerp(rate0, rate1, vAlpha);
        }
    }

    // Sample lookahead root motion velocity (extrapolated for anticipation, also local space)
    if (outLookaheadRootVelocityRootSpace || outLookaheadRootYawRate)
    {
        int lf0, lf1;
        float lAlpha;
        GetInterFrameAlpha(db, animIndex, animTime, lf0, lf1, lAlpha);

        if (outLookaheadRootVelocityRootSpace)
        {
            const Vector3 vel0 = db->lookaheadRootMotionVelocitiesRootSpace[clipStart + lf0];
            const Vector3 vel1 = db->lookaheadRootMotionVelocitiesRootSpace[clipStart + lf1];
            *outLookaheadRootVelocityRootSpace = Vector3Lerp(vel0, vel1, lAlpha);
        }

        if (outLookaheadRootYawRate)
        {
            const float rate0 = db->lookaheadRootMotionYawRates[clipStart + lf0];
            const float rate1 = db->lookaheadRootMotionYawRates[clipStart + lf1];
            *outLookaheadRootYawRate = Lerp(rate0, rate1, lAlpha);
        }
    }

    // Sample lookahead hips height (extrapolated Y position)
    if (outLookaheadHipsHeight)
    {
        const float height0 = db->lookaheadHipsHeights[clipStart + f0];
        const float height1 = db->lookaheadHipsHeights[clipStart + f1];
        *outLookaheadHipsHeight = Lerp(height0, height1, alpha);
    }

    // Sample current hip rotation (yaw-free)
    if (outHipRotationYawFree)
    {
        const Rot6d rot0 = db->hipRotationYawFree[clipStart + f0];
        const Rot6d rot1 = db->hipRotationYawFree[clipStart + f1];
        Rot6dLerp(rot0, rot1, alpha, *outHipRotationYawFree);
    }

    // Sample lookahead hip rotation (yaw-free, for clean dragging)
    if (outLookaheadHipRotationYawFree)
    {
        const Rot6d rot0 = db->lookaheadHipRotationYawFree[clipStart + f0];
        const Rot6d rot1 = db->lookaheadHipRotationYawFree[clipStart + f1];
        Rot6dLerp(rot0, rot1, alpha, *outLookaheadHipRotationYawFree);
    }
}

// Sample toe velocity in root space from db->jointVelocitiesRootSpace at animTime
// Used for foot IK velocity blending
static inline void SampleToeVelocityRootSpace(
    const AnimDatabase* db,
    int animIndex,
    float animTime,
    /*out*/ Vector3 outToeVelocityRootSpace[SIDES_COUNT])
{
    const int clipStart = db->clipStartFrame[animIndex];

    // sample pose at animTime
    int f0, f1;
    float alpha;
    GetInterFrameAlpha(db, animIndex, animTime, f0, f1, alpha);

    std::span<const Vector3> velRow0 = db->jointVelocitiesRootSpace.row_view(clipStart + f0);
    std::span<const Vector3> velRow1 = db->jointVelocitiesRootSpace.row_view(clipStart + f1);

    for (int side : sides)
    {
        const int toeIdx = db->toeIndices[side];
        if (toeIdx >= 0)
        {
            outToeVelocityRootSpace[side] = Vector3Lerp(velRow0[toeIdx], velRow1[toeIdx], alpha);
        }
        else
        {
            outToeVelocityRootSpace[side] = Vector3Zero();
        }
    }
}

// Sample lookahead toe positions in root space at animTime
// Used for predictive foot IK
static inline void SampleLookaheadToePosRootSpace(
    const AnimDatabase* db,
    int animIndex,
    float animTime,
    /*out*/ Vector3 outLookaheadToePosRootSpace[SIDES_COUNT])
{
    const int clipStart = db->clipStartFrame[animIndex];

    int f0, f1;
    float alpha;
    GetInterFrameAlpha(db, animIndex, animTime, f0, f1, alpha);

    for (int side : sides)
    {
        if (db->toeIndices[side] >= 0 && !db->lookaheadToePositionsRootSpace[side].empty())
        {
            const Vector3 pos0 = db->lookaheadToePositionsRootSpace[side][clipStart + f0];
            const Vector3 pos1 = db->lookaheadToePositionsRootSpace[side][clipStart + f1];
            outLookaheadToePosRootSpace[side] = Vector3Lerp(pos0, pos1, alpha);
        }
        else
        {
            outLookaheadToePosRootSpace[side] = Vector3Zero();
        }
    }
}
