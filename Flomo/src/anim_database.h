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
// Populate localJointPositions/localJointRotations as well as global arrays.
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

    // Hand candidates for Magic anchor
    const vector<string> leftHandCandidates = { "lefthand", "left_hand", "l_hand", "hand.l", "hand_l" };
    const vector<string> rightHandCandidates = { "righthand", "right_hand", "r_hand", "hand.r", "hand_r" };
    const vector<vector<string>> handCandidates = { leftHandCandidates, rightHandCandidates };

    // Spine3 and head candidates for Magic anchor
    const vector<string> spine3Candidates = { "spine3", "spine2", "chest", "upperchest", "upper_chest" };
    const vector<string> headCandidates = { "head" };

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

    // Find hand indices
    for (int side : sides)
    {
        db->handIndices[side] = FindJointIndexByNames(canonBvh, handCandidates[side]);
        const char* sideName = (side == SIDE_LEFT) ? "left" : "right";
        if (db->handIndices[side] < 0)
        {
            TraceLog(LOG_WARNING, "AnimDatabase: %s hand not found, aborting", sideName);
            return;
        }
    }

    // Find spine3 and head for Magic anchor
    db->spine3Index = FindJointIndexByNames(canonBvh, spine3Candidates);
    db->headIndex = FindJointIndexByNames(canonBvh, headCandidates);

    if (db->spine3Index < 0)
    {
        TraceLog(LOG_WARNING, "AnimDatabase: spine3/chest not found, aborting");
        return;
    }
    if (db->headIndex < 0)
    {
        TraceLog(LOG_WARNING, "AnimDatabase: head not found, aborting");
        return;
    }

    // Log resolved feature indices
    TraceLog(LOG_INFO, "AnimDatabase: hip=%d, spine3=%d, head=%d, leftHand=%d, rightHand=%d",
        db->hipJointIndex, db->spine3Index, db->headIndex, db->handIndices[SIDE_LEFT], db->handIndices[SIDE_RIGHT]);

    // allocate compact storage [motionFrameCount x jointCount]
    db->motionFrameCount = includedFrames;
    db->jointPositionsAnimSpace.resize(db->motionFrameCount, db->jointCount);
    db->jointRotationsAnimSpace.resize(db->motionFrameCount, db->jointCount);
    db->jointVelocitiesRootSpace.resize(db->motionFrameCount, db->jointCount);
    db->jointAccelerationsRootSpace.resize(db->motionFrameCount, db->jointCount);
    db->localJointPositions.resize(db->motionFrameCount, db->jointCount);
    db->localJointRotations.resize(db->motionFrameCount, db->jointCount);
    db->localJointAngularVelocities.resize(db->motionFrameCount, db->jointCount);
    db->lookaheadLocalRotations.resize(db->motionFrameCount, db->jointCount);
    db->lookaheadLocalPositions.resize(db->motionFrameCount, db->jointCount);

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
            span<Rot6d> globalRot = db->jointRotationsAnimSpace.row_view(motionFrameIdx);
            span<Vector3> localPos = db->localJointPositions.row_view(motionFrameIdx);
            span<Rot6d> localRot = db->localJointRotations.row_view(motionFrameIdx);

            for (int j = 0; j < db->jointCount; ++j)
            {
                globalPos[j] = tmpXform.globalPositions[j];
                Rot6dFromQuaternion(tmpXform.globalRotations[j], globalRot[j]);
                localPos[j] = tmpXform.localPositions[j];
                Rot6dFromQuaternion(tmpXform.localRotations[j], localRot[j]);
            }

            ++motionFrameIdx;
        }

        db->clipEndFrame.push_back(motionFrameIdx);
    }

    // Compute magic anchor position and yaw for all frames FIRST
    // This is needed before joint velocities since root space = magic space
    db->magicPosition.resize(db->motionFrameCount);
    db->magicYaw.resize(db->motionFrameCount);

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];

        for (int f = clipStart; f < clipEnd; ++f)
        {
            span<const Vector3> posRow = db->jointPositionsAnimSpace.row_view(f);

            // Magic position = spine3 projected to ground
            const Vector3 spine3Pos = posRow[db->spine3Index];
            db->magicPosition[f] = Vector3{ spine3Pos.x, 0.0f, spine3Pos.z };

            // Magic yaw = direction from head to right hand projected onto XZ plane
            const Vector3 headPos = posRow[db->headIndex];
            const Vector3 handPos = posRow[db->handIndices[SIDE_RIGHT]];
            const Vector3 headToHand = Vector3Subtract(handPos, headPos);
            db->magicYaw[f] = atan2f(headToHand.x, headToHand.z);
        }
    }

    // Compute velocities for each joint at each frame, in root space (= magic space)
    // Velocity at frame i is defined at midpoint between frame i and i+1: v = (pos[i+1] - pos[i]) / frameTime
    // Transformed by average magic yaw between frames i and i+1 (midpoint rotation)
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

            // Use midpoint magic yaw between the two frames (correctly handling angle wrapping)
            const float magicYaw0 = db->magicYaw[prevF];
            const float magicYaw1 = db->magicYaw[nextF];
            const float midpointMagicYaw = LerpAngle(magicYaw0, magicYaw1, 0.5f);
            const Rot6d invMidpointMagicYawRot = Rot6dFromYaw(-midpointMagicYaw);

            for (int j = 0; j < db->jointCount; ++j)
            {
                // Compute velocity in anim space
                Vector3 velAnimSpace = Vector3Scale(Vector3Subtract(pos1Row[j], pos0Row[j]), invFrameTime);
                // Transform to root space using midpoint magic yaw (magic-heading-relative at midpoint)
                velRow[j] = Vector3RotateByRot6d(velAnimSpace, invMidpointMagicYawRot);
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

            span<const Rot6d> rot0Row = db->localJointRotations.row_view(prevF);
            span<const Rot6d> rot1Row = db->localJointRotations.row_view(nextF);

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

            span<Rot6d> lookaheadRow = db->lookaheadLocalRotations.row_view(f);
            span<const Rot6d> currRow = db->localJointRotations.row_view(f);

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
                span<const Rot6d> nextRow = db->localJointRotations.row_view(f + 1);

                for (int j = 0; j < db->jointCount; ++j)
                {
                    Rot6d curr = currRow[j];
                    Rot6d next = nextRow[j];

                    // NOTE: We no longer strip yaw from joint 0 here because:
                    // - localJointRotations[0] will be overwritten later to be relative to Magic anchor
                    // - lookaheadLocalRotations[0] will also be overwritten with lookaheadHipRotationInMagicSpace
                    // The yaw-free hip rotation is now handled by the Magic anchor system

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

    // Compute lookahead positions: pos[f] + n * (pos[f+1] - pos[f])
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

            span<Vector3> lookaheadPosRow = db->lookaheadLocalPositions.row_view(f);
            span<const Vector3> currPosRow = db->localJointPositions.row_view(f);

            if (isLastFrame || clipEnd - clipStart <= 1)
            {
                // no next frame to extrapolate from, just copy current
                for (int j = 0; j < db->jointCount; ++j)
                {
                    lookaheadPosRow[j] = currPosRow[j];
                }
            }
            else
            {
                span<const Vector3> nextPosRow = db->localJointPositions.row_view(f + 1);

                for (int j = 0; j < db->jointCount; ++j)
                {
                    Vector3 curr = currPosRow[j];
                    Vector3 next = nextPosRow[j];

                    // lookahead = (1-n)*curr + n*next = curr + n*(next - curr)
                    Vector3 result;
                    result.x = currWeight * curr.x + nextWeight * next.x;
                    result.y = currWeight * curr.y + nextWeight * next.y;
                    result.z = currWeight * curr.z + nextWeight * next.z;
                    lookaheadPosRow[j] = result;
                }
            }
        }
    }

    // Compute Magic anchor velocities and yaw rates FIRST
    // (magicPosition and magicYaw were already computed earlier for joint velocities)
    db->magicVelocityRootSpace.resize(db->motionFrameCount);
    db->magicYawRate.resize(db->motionFrameCount);

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float invFrameTime = 1.0f / frameTime;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);

            // Compute magic velocities and yaw rates
            if (isLastFrame || clipEnd - clipStart <= 1)
            {
                if (f > clipStart)
                {
                    db->magicVelocityRootSpace[f] = db->magicVelocityRootSpace[f - 1];
                    db->magicYawRate[f] = db->magicYawRate[f - 1];
                }
                else
                {
                    db->magicVelocityRootSpace[f] = Vector3Zero();
                    db->magicYawRate[f] = 0.0f;
                }
            }
            else
            {
                // Use already-computed magic positions (don't recompute from bones)
                const Vector3 nextMagicPos = db->magicPosition[f + 1];
                Vector3 velAnim = Vector3Scale(Vector3Subtract(nextMagicPos, db->magicPosition[f]), invFrameTime);

                // Transform to magic space (heading-relative)
                // Rotate by -magicYaw to go from anim space to magic-local space
                const float magicYaw = db->magicYaw[f];
                const float cosY = cosf(magicYaw);
                const float sinY = sinf(magicYaw);
                db->magicVelocityRootSpace[f] = Vector3{
                    velAnim.x * cosY - velAnim.z * sinY,
                    0.0f,
                    velAnim.x * sinY + velAnim.z * cosY
                };

                // Use already-computed magic yaw (don't recompute from bones)
                const float nextMagicYaw = db->magicYaw[f + 1];
                db->magicYawRate[f] = WrapAngleToPi(nextMagicYaw - magicYaw) * invFrameTime;
            }
        }
    }

    // Now compute root motion velocities by copying from magic velocities
    // (These arrays are kept separate for now for compatibility, will clean up later)
    db->rootMotionVelocitiesRootSpace.resize(db->motionFrameCount);
    db->rootMotionYawRates.resize(db->motionFrameCount);

    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        db->rootMotionVelocitiesRootSpace[f] = db->magicVelocityRootSpace[f];
        db->rootMotionYawRates[f] = db->magicYawRate[f];
    }

    // Compute lookahead magic velocities (extrapolated for smooth anticipation)
    db->lookaheadMagicVelocity.resize(db->motionFrameCount);
    db->lookaheadMagicYawRate.resize(db->motionFrameCount);

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
                db->lookaheadMagicVelocity[f] = db->magicVelocityRootSpace[f];
                db->lookaheadMagicYawRate[f] = db->magicYawRate[f];
            }
            else
            {
                // Extrapolate: lookahead = (1-n)*curr + n*next
                const Vector3 currVel = db->magicVelocityRootSpace[f];
                const Vector3 nextVel = db->magicVelocityRootSpace[f + 1];
                db->lookaheadMagicVelocity[f] = Vector3Add(
                    Vector3Scale(currVel, currWeight),
                    Vector3Scale(nextVel, nextWeight));

                const float currYawRate = db->magicYawRate[f];
                const float nextYawRate = db->magicYawRate[f + 1];
                db->lookaheadMagicYawRate[f] = currWeight * currYawRate + nextWeight * nextYawRate;
            }
        }
    }

    // Now compute lookahead root motion velocities by copying from lookahead magic
    // (These arrays are kept separate for now for compatibility, will clean up later)
    db->lookaheadRootMotionVelocitiesRootSpace.resize(db->motionFrameCount);
    db->lookaheadRootMotionYawRates.resize(db->motionFrameCount);

    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        db->lookaheadRootMotionVelocitiesRootSpace[f] = db->lookaheadMagicVelocity[f];
        db->lookaheadRootMotionYawRates[f] = db->lookaheadMagicYawRate[f];
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
            span<const Rot6d> rotRow = db->localJointRotations.row_view(f);
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
                span<const Rot6d> nextRotRow = db->localJointRotations.row_view(f + 1);
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
    TraceLog(LOG_INFO, "AnimDatabase: computed Magic anchor tracks");

    // Compute hip transform relative to Magic anchor (for placing skeleton when using Magic root motion)
    db->hipPositionInMagicSpace.resize(db->motionFrameCount);
    db->hipRotationInMagicSpace.resize(db->motionFrameCount);

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const int hipIdx = db->hipJointIndex;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            span<const Vector3> posRow = db->jointPositionsAnimSpace.row_view(f);
            span<const Rot6d> rotRow = db->localJointRotations.row_view(f);

            // Get hip position and rotation in animation space
            const Vector3 hipPos = posRow[hipIdx];
            const Rot6d hipRot = rotRow[hipIdx];

            // Get Magic position and yaw
            const Vector3 magicPos = db->magicPosition[f];
            const float magicYaw = db->magicYaw[f];

            // Compute hip position relative to magic, in magic-heading space
            // hip_in_magic = invMagicYaw * (hipPos - magicPos)
            const Vector3 magicToHip = Vector3Subtract(hipPos, magicPos);
            const float cosY = cosf(-magicYaw);
            const float sinY = sinf(-magicYaw);
            db->hipPositionInMagicSpace[f] = Vector3{
                magicToHip.x * cosY - magicToHip.z * sinY,
                magicToHip.y,  // keep Y (hip height)
                magicToHip.x * sinY + magicToHip.z * cosY
            };

            // Compute hip rotation relative to magic yaw: hipRotInMagic = invMagicYaw * hipRot
            const Rot6d invMagicYawRot = Rot6dFromYaw(-magicYaw);
            Rot6dMultiply(invMagicYawRot, hipRot, db->hipRotationInMagicSpace[f]);
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: computed hip-in-magic-space tracks");

    // IMPORTANT: Overwrite localJointPositions[0] and localJointRotations[0] to be relative to Magic
    // Also compute lookahead hip transform on-the-fly
    // This makes hip "just another bone" parented to the Magic anchor
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        //const int hipIdx = db->hipJointIndex;
        const float frameTime = db->animFrameTime[c];
        const float n = db->poseDragLookaheadTime / frameTime;
        const float nextWeight = n;
        const float currWeight = 1.0f - n;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);
            span<Vector3> localPos = db->localJointPositions.row_view(f);
            span<Rot6d> localRot = db->localJointRotations.row_view(f);
            span<Vector3> lookaheadPos = db->lookaheadLocalPositions.row_view(f);
            span<Rot6d> lookaheadRot = db->lookaheadLocalRotations.row_view(f);

            // Set current hip transform relative to magic
            localPos[0] = db->hipPositionInMagicSpace[f];
            localRot[0] = db->hipRotationInMagicSpace[f];

            // Compute and set lookahead hip transform
            if (isLastFrame || clipEnd - clipStart <= 1)
            {
                // No next frame to extrapolate from, just copy current
                lookaheadPos[0] = db->hipPositionInMagicSpace[f];
                lookaheadRot[0] = db->hipRotationInMagicSpace[f];
            }
            else
            {
                // Extrapolate hip position: lookahead = (1-n)*curr + n*next
                const Vector3& currHipPos = db->hipPositionInMagicSpace[f];
                const Vector3& nextHipPos = db->hipPositionInMagicSpace[f + 1];
                lookaheadPos[0] = Vector3{
                    currWeight * currHipPos.x + nextWeight * nextHipPos.x,
                    currWeight * currHipPos.y + nextWeight * nextHipPos.y,
                    currWeight * currHipPos.z + nextWeight * nextHipPos.z
                };

                // Extrapolate hip rotation: lookahead = (1-n)*curr + n*next
                const Rot6d& currHipRot = db->hipRotationInMagicSpace[f];
                const Rot6d& nextHipRot = db->hipRotationInMagicSpace[f + 1];
                lookaheadRot[0].ax = currWeight * currHipRot.ax + nextWeight * nextHipRot.ax;
                lookaheadRot[0].ay = currWeight * currHipRot.ay + nextWeight * nextHipRot.ay;
                lookaheadRot[0].az = currWeight * currHipRot.az + nextWeight * nextHipRot.az;
                lookaheadRot[0].bx = currWeight * currHipRot.bx + nextWeight * nextHipRot.bx;
                lookaheadRot[0].by = currWeight * currHipRot.by + nextWeight * nextHipRot.by;
                lookaheadRot[0].bz = currWeight * currHipRot.bz + nextWeight * nextHipRot.bz;
                Rot6dNormalize(lookaheadRot[0]);
            }
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: hip (joint 0) now stored relative to Magic anchor");

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

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const bool isLastFrame = (f == clipEnd - 1);
            span<const Vector3> posRow = db->jointPositionsAnimSpace.row_view(f);

            // Use magic anchor as root (root space = magic space)
            const Vector3 magicPos = db->magicPosition[f];
            const float magicYaw = db->magicYaw[f];
            const Rot6d invMagicYawRot = Rot6dFromYaw(-magicYaw);

            for (int side : sides)
            {
                const int toeIdx = db->toeIndices[side];

                // Current toe position in root space (magic space)
                const Vector3 toePos = posRow[toeIdx];

                // Offset from magic anchor (already at Y=0)
                const Vector3 magicToToe = Vector3Subtract(toePos, magicPos);

                // Transform to magic-heading-aligned space
                const Vector3 toePosRootSpace = Vector3RotateByRot6d(magicToToe, invMagicYawRot);
                db->toePositionsRootSpace[side][f] = toePosRootSpace;

                // Lookahead toe position
                if (isLastFrame || clipEnd - clipStart <= 1)
                {
                    db->lookaheadToePositionsRootSpace[side][f] = toePosRootSpace;
                }
                else
                {
                    // Get next frame's toe position in next frame's root space
                    span<const Vector3> nextPosRow = db->jointPositionsAnimSpace.row_view(f + 1);
                    const Vector3 nextToeAnimSpace = nextPosRow[toeIdx];
                    const Vector3 nextMagicPos = db->magicPosition[f + 1];
                    const float nextMagicYaw = db->magicYaw[f + 1];
                    const Rot6d nextInvMagicYawRot = Rot6dFromYaw(-nextMagicYaw);
                    const Vector3 nextMagicToToe = Vector3Subtract(nextToeAnimSpace, nextMagicPos);
                    const Vector3 nextToePosRootSpace = Vector3RotateByRot6d(nextMagicToToe, nextInvMagicYawRot);

                    // Extrapolate between current and next toe positions in their respective root spaces
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
    if (cfg.IsFeatureEnabled(FeatureType::AimDirection)) db->featureDim += (int)cfg.futureTrajPointTimes.size() * 2;

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
        span<const Rot6d> rotRow = db->jointRotationsAnimSpace.row_view(f);
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

        // Use magic yaw instead of hip yaw for feature extraction (magic space is more stable for blending)
        const float magicYaw = db->magicYaw[f];  // already computed earlier in rebuild
        const float invMagicYaw = -magicYaw;
        const Rot6d magicYawRot = Rot6dFromYaw(magicYaw);
        const Rot6d invMagicYawRot = Rot6dFromYaw(invMagicYaw);

        // Get magic position for reference frame origin
        const Vector3 magicPos = db->magicPosition[f];

        int currentFeature = 0;

        // Precompute local toe positions (magic horizontal frame) - used by pos and diff
        const Vector3 magicToLeft = Vector3Subtract(leftPos, magicPos);
        Vector3 localLeftPos = Vector3RotateByRot6d(magicToLeft, invMagicYawRot);

        const Vector3 magicToRight = Vector3Subtract(rightPos, magicPos);
        const Vector3 localRightPos = Vector3RotateByRot6d(magicToRight, invMagicYawRot);

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

        // VELOCITY: toeVel -> compute world finite-difference then rotate into magic frame
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
                    localLeftVel = Vector3RotateByRot6d(velLeftWorld, invMagicYawRot);
                }

                if (rightIdx >= 0)
                {
                    const Vector3 deltaRight = Vector3Subtract(rightPos, posPrevRow[rightIdx]);
                    const Vector3 velRightWorld = Vector3Scale(deltaRight, 1.0f / dt);
                    localRightVel = Vector3RotateByRot6d(velRightWorld, invMagicYawRot);
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

        // DIFFERENCE: toeDifference = Left - Right (in magic horizontal frame) => (dx, dz)
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
        // Compute velocity in animSpace at future frame, then transform to magicSpace relative to current frame
        if (cfg.IsFeatureEnabled(FeatureType::FutureVel))
        {
            const float frameTime = db->animFrameTime[clipIdx];
            const float invFrameTime = 1.0f / frameTime;
            const int rootIdx = 0;

            for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
            {
                const float futureTime = cfg.futureTrajPointTimes[p];
                const int futureFrameOffset = (int)(futureTime / frameTime + 0.5f);
                const int futureFrame = f + futureFrameOffset;

                Vector3 futureVelMagicSpace = Vector3Zero();

                // Check if future frame and next frame are within the same clip
                if (futureFrame >= clipStart && futureFrame < clipEnd - 1)
                {
                    // Compute velocity in animSpace at future frame: (pos[f+1] - pos[f]) / dt
                    span<const Vector3> futurePosRow0 = db->jointPositionsAnimSpace.row_view(futureFrame);
                    span<const Vector3> futurePosRow1 = db->jointPositionsAnimSpace.row_view(futureFrame + 1);
                    Vector3 futureVelAnimSpace = Vector3Scale(
                        Vector3Subtract(futurePosRow1[rootIdx], futurePosRow0[rootIdx]), invFrameTime);
                    futureVelAnimSpace.y = 0.0f;  // XZ only

                    // Transform to magicSpace relative to current frame's magic yaw
                    futureVelMagicSpace = Vector3RotateByRot6d(futureVelAnimSpace, invMagicYawRot);
                }
                // else: future frame outside clip or at last frame, leave as zero

                featRow[currentFeature++] = futureVelMagicSpace.x;
                featRow[currentFeature++] = futureVelMagicSpace.z;

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

            for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
            {
                const float futureTime = cfg.futureTrajPointTimes[p];
                const int futureFrameOffset = (int)(futureTime / frameTime + 0.5f);
                const int futureFrame = f + futureFrameOffset;

                Vector3 futureVelMagicSpace = Vector3Zero();

                if (futureFrame >= clipStart && futureFrame < clipEnd - 1)
                {
                    span<const Vector3> futurePosRow0 = db->jointPositionsAnimSpace.row_view(futureFrame);
                    span<const Vector3> futurePosRow1 = db->jointPositionsAnimSpace.row_view(futureFrame + 1);
                    Vector3 futureVelAnimSpace = Vector3Scale(
                        Vector3Subtract(futurePosRow1[rootIdx], futurePosRow0[rootIdx]), invFrameTime);
                    futureVelAnimSpace.y = 0.0f;  // XZ only

                    futureVelMagicSpace = Vector3RotateByRot6d(futureVelAnimSpace, invMagicYawRot);

                    // Clamp to max magnitude
                    const float mag = Vector3Length(futureVelMagicSpace);
                    if (mag > MaxFutureVelClampedMag)
                    {
                        futureVelMagicSpace = Vector3Scale(futureVelMagicSpace, MaxFutureVelClampedMag / mag);
                    }
                }

                featRow[currentFeature++] = futureVelMagicSpace.x;
                featRow[currentFeature++] = futureVelMagicSpace.z;

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

        // PAST POSITION: past magic position in current magic horizontal frame (XZ)
        if (cfg.IsFeatureEnabled(FeatureType::PastPosition))
        {
            Vector3 pastPosLocal = Vector3Zero();

            const float frameTime = db->animFrameTime[clipIdx];
            const float pastTime = cfg.pastTimeOffset;
            const int pastFrameOffset = (int)(pastTime / frameTime + 0.5f);
            const int pastFrame = f - pastFrameOffset;

            // Check if past frame is within the same clip
            if (pastFrame >= clipStart && pastFrame < clipEnd)
            {
                // Get past magic position (already computed)
                const Vector3 pastMagicPos = db->magicPosition[pastFrame];

                // Compute vector from current magic to past magic
                const Vector3 magicToPastMagic = Vector3Subtract(pastMagicPos, magicPos);

                // Transform to current magic horizontal frame
                pastPosLocal = Vector3RotateByRot6d(magicToPastMagic, invMagicYawRot);
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

        // AIM DIRECTION: head→rightHand direction at future trajectory times, relative to current magic yaw
        if (cfg.IsFeatureEnabled(FeatureType::AimDirection))
        {
            const float frameTime = db->animFrameTime[clipIdx];

            for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
            {
                const float futureTime = cfg.futureTrajPointTimes[p];
                const int futureFrameOffset = (int)(futureTime / frameTime + 0.5f);
                const int futureFrame = f + futureFrameOffset;

                Vector3 aimDirLocal = { 0.0f, 0.0f, 1.0f };  // default: forward

                // Check if future frame is within the same clip
                if (futureFrame >= clipStart && futureFrame < clipEnd)
                {
                    span<const Vector3> futurePosRow = db->jointPositionsAnimSpace.row_view(futureFrame);

                    // Compute head→rightHand direction at future frame
                    const Vector3 futureHeadPos = futurePosRow[db->headIndex];
                    const Vector3 futureHandPos = futurePosRow[db->handIndices[SIDE_RIGHT]];
                    Vector3 aimDirWorld = Vector3Subtract(futureHandPos, futureHeadPos);
                    aimDirWorld.y = 0.0f;  // project to XZ plane

                    // Normalize to unit length
                    const float aimLen = Vector3Length(aimDirWorld);
                    if (aimLen > 1e-6f)
                    {
                        aimDirWorld = Vector3Scale(aimDirWorld, 1.0f / aimLen);
                    }
                    else
                    {
                        aimDirWorld = { 0.0f, 0.0f, 1.0f };
                    }

                    // Transform to current frame's magic space (relative to current magic yaw)
                    aimDirLocal = Vector3RotateByRot6d(aimDirWorld, invMagicYawRot);
                }

                featRow[currentFeature++] = aimDirLocal.x;
                featRow[currentFeature++] = aimDirLocal.z;

                if (isFirstFrame)
                {
                    char nameBufX[64];
                    char nameBufZ[64];
                    snprintf(nameBufX, sizeof(nameBufX), "AimDirX_%.2fs", futureTime);
                    snprintf(nameBufZ, sizeof(nameBufZ), "AimDirZ_%.2fs", futureTime);
                    db->featureNames.push_back(string(nameBufX));
                    db->featureNames.push_back(string(nameBufZ));
                    db->featureTypes.push_back(FeatureType::AimDirection);
                    db->featureTypes.push_back(FeatureType::AimDirection);
                }
            }
        }

        assert(currentFeature == db->featureDim);
    }


    // Allocate mean vector (per-dimension)
    db->featuresMean.resize(db->featureDim, 0.0f);

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

    // Compute standard deviation per feature TYPE (shared across all dimensions of the same type)
    // First pass: sum squared differences grouped by feature type
    double typesSumSquaredDiff[static_cast<int>(FeatureType::COUNT)] = {};
    int typesCount[static_cast<int>(FeatureType::COUNT)] = {};

    for (int d = 0; d < db->featureDim; ++d)
    {
        const int typeIdx = static_cast<int>(db->featureTypes[d]);
        for (int f = 0; f < db->motionFrameCount; ++f)
        {
            const double diff = db->features.at(f, d) - db->featuresMean[d];
            typesSumSquaredDiff[typeIdx] += diff * diff;
        }
        typesCount[typeIdx] += db->motionFrameCount;
    }

    // Compute std for each feature type
    for (int t = 0; t < static_cast<int>(FeatureType::COUNT); ++t)
    {
        if (typesCount[t] > 0)
        {
            const double variance = typesSumSquaredDiff[t] / typesCount[t];
            db->featureTypesStd[t] = (float)std::sqrt(variance);

            // avoid division by zero
            if (db->featureTypesStd[t] < 1e-8f)
            {
                db->featureTypesStd[t] = 1.0f;
            }
        }
        else
        {
            db->featureTypesStd[t] = 1.0f;
        }
    }

    // Compute normalized features: (x - mean) / typeStd
    db->normalizedFeatures.resize(db->motionFrameCount, db->featureDim);
    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        span<const float> featRow = db->features.row_view(f);
        span<float> normRow = db->normalizedFeatures.row_view(f);

        for (int d = 0; d < db->featureDim; ++d)
        {
            const int typeIdx = static_cast<int>(db->featureTypes[d]);
            normRow[d] = (featRow[d] - db->featuresMean[d]) / db->featureTypesStd[typeIdx];
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: computed feature normalization (mean per-dim, std per-type for %d dimensions)", db->featureDim);

    // Apply feature type weights to normalized features
    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        span<float> normRow = db->normalizedFeatures.row_view(f);

        for (int d = 0; d < db->featureDim; ++d)
        {
            const FeatureType featureType = db->featureTypes[d];
            const float weight = cfg.featureTypeWeights[(int)featureType];
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

// Sample toe velocity in root space from db->jointVelocitiesRootSpace at animTime
// Used for foot IK velocity blending
