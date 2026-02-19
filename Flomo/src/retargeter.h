
#pragma once

// retarget motion between two BVH skeletons.
// copies rotations from source to target with twist offset corrections
// to handle bone direction differences between rest poses.

#include "bvh_parser.h"
#include "transform_data.h"
#include "math_utils.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

// ============================================================
// joint name alias table for cross-skeleton matching
// ============================================================

struct RetargetAlias
{
    const char* names[8];
    int count;
};

static const RetargetAlias RETARGET_ALIASES[] = {
    { { "hips", "pelvis", "root" }, 3 },
    { { "spine", "spine0" }, 2 },
    { { "spine1" }, 1 },
    { { "spine2", "chest" }, 2 },
    { { "spine3", "upperchest", "upper_chest" }, 3 },
    { { "neck" }, 1 },
    { { "head" }, 1 },
    { { "leftshoulder", "left_shoulder" }, 2 },
    { { "leftarm", "leftupperarm", "left_arm" }, 3 },
    { { "leftforearm", "leftlowerarm", "left_forearm" }, 3 },
    { { "lefthand", "left_hand" }, 2 },
    { { "rightshoulder", "right_shoulder" }, 2 },
    { { "rightarm", "rightupperarm", "right_arm" }, 3 },
    { { "rightforearm", "rightlowerarm", "right_forearm" }, 3 },
    { { "righthand", "right_hand" }, 2 },
    { { "leftupleg", "leftthigh", "left_upleg" }, 3 },
    { { "leftleg", "leftshin", "leftlowerleg", "left_leg" }, 4 },
    { { "leftfoot", "leftankle", "left_foot" }, 3 },
    { { "lefttoebase", "lefttoe", "left_toe" }, 3 },
    { { "rightupleg", "rightthigh", "right_upleg" }, 3 },
    { { "rightleg", "rightshin", "rightlowerleg", "right_leg" }, 4 },
    { { "rightfoot", "rightankle", "right_foot" }, 3 },
    { { "righttoebase", "righttoe", "right_toe" }, 3 },
    // finger joints (only needed as target child references for hand twist)
    { { "lefthandmiddle1", "leftmiddleproximal" }, 2 },
    { { "righthandmiddle1", "rightmiddleproximal" }, 2 },
};
static const int RETARGET_ALIAS_COUNT = sizeof(RETARGET_ALIASES) / sizeof(RETARGET_ALIASES[0]);

// find which alias group a joint name belongs to (-1 if not found)
static int RetargetFindAliasGroup(const char* jointName)
{
    std::string lower = ToLowerCopy(jointName);
    for (int g = 0; g < RETARGET_ALIAS_COUNT; g++)
    {
        for (int n = 0; n < RETARGET_ALIASES[g].count; n++)
        {
            if (lower == RETARGET_ALIASES[g].names[n]) return g;
        }
    }
    return -1;
}

// find a joint in a BVH by alias group index
static int RetargetFindJointByAlias(const BVHData* bvh, int aliasGroup)
{
    if (aliasGroup < 0 || aliasGroup >= RETARGET_ALIAS_COUNT) return -1;
    const RetargetAlias& alias = RETARGET_ALIASES[aliasGroup];
    std::vector<std::string> candidates;
    for (int n = 0; n < alias.count; n++)
    {
        candidates.push_back(alias.names[n]);
    }
    return FindJointIndexByNames(bvh, candidates);
}

// find joint by exact name (case-insensitive)
static int RetargetFindJointByName(const BVHData* bvh, const char* name)
{
    std::string lower = ToLowerCopy(name);
    for (int j = 0; j < bvh->jointCount; j++)
    {
        if (!bvh->joints[j].endSite && ToLowerCopy(bvh->joints[j].name.c_str()) == lower)
        {
            return j;
        }
    }
    // fallback: alias lookup
    int group = RetargetFindAliasGroup(name);
    return (group >= 0) ? RetargetFindJointByAlias(bvh, group) : -1;
}

// max Y coordinate across all joints (skeleton height from FK global positions)
static float RetargetSkeletonHeight(const TransformData* xform)
{
    float maxY = 0.0f;
    for (int i = 0; i < xform->jointCount; i++)
    {
        if (xform->globalPositions[i].y > maxY) maxY = xform->globalPositions[i].y;
    }
    return maxY;
}

// ============================================================
// the main retargeting function
// ============================================================

static bool RetargetBVH(
    BVHData* outBvh,
    const BVHData* srcBvh,
    const BVHData* tgtBvh,
    char* errMsg,
    int errMsgSize)
{
    if (srcBvh->frameCount == 0)
    {
        snprintf(errMsg, errMsgSize, "Error: Source BVH has no frames");
        return false;
    }

    // ====================================================================
    // 1. build joint mapping (for each target joint, find the source joint)
    // ====================================================================

    std::vector<int> srcForTgt(tgtBvh->jointCount, -1);
    int mappedCount = 0;

    for (int t = 0; t < tgtBvh->jointCount; t++)
    {
        if (tgtBvh->joints[t].endSite) continue;

        // try alias-based matching first
        int tgtGroup = RetargetFindAliasGroup(tgtBvh->joints[t].name.c_str());
        if (tgtGroup >= 0)
        {
            int srcIdx = RetargetFindJointByAlias(srcBvh, tgtGroup);
            if (srcIdx >= 0)
            {
                srcForTgt[t] = srcIdx;
                mappedCount++;
                continue;
            }
        }

        // fallback: exact name match (case-insensitive)
        std::string tgtLower = ToLowerCopy(tgtBvh->joints[t].name.c_str());
        for (int s = 0; s < srcBvh->jointCount; s++)
        {
            if (!srcBvh->joints[s].endSite &&
                ToLowerCopy(srcBvh->joints[s].name.c_str()) == tgtLower)
            {
                srcForTgt[t] = s;
                mappedCount++;
                break;
            }
        }
    }

    printf("Joint mapping: %d target joints mapped to source\n", mappedCount);
    for (int t = 0; t < tgtBvh->jointCount; t++)
    {
        if (srcForTgt[t] >= 0)
        {
            printf("  %s -> %s\n",
                tgtBvh->joints[t].name.c_str(),
                srcBvh->joints[srcForTgt[t]].name.c_str());
        }
    }

    if (mappedCount == 0)
    {
        snprintf(errMsg, errMsgSize, "Error: No joints could be mapped between source and target");
        return false;
    }

    // find root joints (first non-end-site joint with parent == -1)
    int srcRoot = -1;
    int tgtRoot = -1;
    for (int i = 0; i < srcBvh->jointCount; i++)
    {
        if (srcBvh->joints[i].parent == -1 && !srcBvh->joints[i].endSite) { srcRoot = i; break; }
    }
    for (int i = 0; i < tgtBvh->jointCount; i++)
    {
        if (tgtBvh->joints[i].parent == -1 && !tgtBvh->joints[i].endSite) { tgtRoot = i; break; }
    }
    assert(srcRoot >= 0 && tgtRoot >= 0);

    // ====================================================================
    // 2. source T-pose (frame 0) — FK to get global positions/rotations
    // ====================================================================

    TransformData srcTpose;
    TransformDataResize(&srcTpose, srcBvh);
    TransformDataSampleFrame(&srcTpose, srcBvh, 0, 1.0f);
    TransformDataForwardKinematics(&srcTpose);

    // ====================================================================
    // 3. target rest pose (identity rotations + BVH offsets)
    // ====================================================================
    // the rest pose is what the skeleton looks like with all rotations at identity.
    // bone directions in this pose are used for twist offset computation.

    TransformData tgtRest;
    tgtRest.jointCount = tgtBvh->jointCount;
    tgtRest.parents.resize(tgtRest.jointCount);
    tgtRest.endSite.resize(tgtRest.jointCount);
    tgtRest.localPositions.resize(tgtRest.jointCount);
    tgtRest.localRotations.resize(tgtRest.jointCount);
    tgtRest.globalPositions.resize(tgtRest.jointCount);
    tgtRest.globalRotations.resize(tgtRest.jointCount);

    for (int i = 0; i < tgtRest.jointCount; i++)
    {
        tgtRest.parents[i] = tgtBvh->joints[i].parent;
        tgtRest.endSite[i] = tgtBvh->joints[i].endSite;
        tgtRest.localPositions[i] = tgtBvh->joints[i].offset;
        tgtRest.localRotations[i] = QuaternionIdentity();
    }
    TransformDataForwardKinematics(&tgtRest);

    // ====================================================================
    // 4. height ratio for root position scaling
    // ====================================================================

    const float srcHeight = RetargetSkeletonHeight(&srcTpose);
    const float tgtHeight = RetargetSkeletonHeight(&tgtRest);

    if (srcHeight < 0.001f || tgtHeight < 0.001f)
    {
        snprintf(errMsg, errMsgSize,
            "Error: Could not compute skeleton heights (src=%.4f, tgt=%.4f)",
            srcHeight, tgtHeight);
        return false;
    }

    const float heightRatio = tgtHeight / srcHeight;
    printf("Height: src=%.4f tgt=%.4f ratio=%.4f\n", srcHeight, tgtHeight, heightRatio);

    // ====================================================================
    // 5. twist offsets for override chains
    // ====================================================================
    //
    // when copying a rotation from source to target, the bone DIRECTION can
    // be wrong because child offsets differ. example: target arm bones point
    // up (+Y) in rest pose, source T-pose has them horizontal.
    //
    // for each chain joint:
    //   d_geno   = bone direction in target rest pose (offset to child)
    //   d_target = source T-pose bone direction in joint's local frame
    //   R_twist  = rotation from d_geno to d_target
    //
    // per-frame: q_corrected = inv(R_twist_parent) * q_src * R_twist_self
    // inv(R_twist_parent) prevents twist from accumulating through chains.

    // override chain definitions
    // for hands: source has no fingers, so we use a proxy direction
    //   (forearm->hand direction) and a proxy target child (MiddleFinger1)
    struct ChainDef
    {
        const char* joints[8];
        int count;
        int handIdx;                // which chain index is the hand (-1 = none)
        const char* handTgtChild;   // target child for d_geno
        const char* handSrcFrom;    // source "from" for d_src
        const char* handSrcTo;      // source "to" for d_src
    };

    const ChainDef chainDefs[] = {
        { { "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand" }, 4,
          3, "LeftHandMiddle1", "LeftForeArm", "LeftHand" },
        { { "RightShoulder", "RightArm", "RightForeArm", "RightHand" }, 4,
          3, "RightHandMiddle1", "RightForeArm", "RightHand" },
        { { "Neck", "Head" }, 2, -1, nullptr, nullptr, nullptr },
        { { "LeftFoot", "LeftToeBase" }, 2, -1, nullptr, nullptr, nullptr },
        { { "RightFoot", "RightToeBase" }, 2, -1, nullptr, nullptr, nullptr },
    };
    const int chainDefCount = sizeof(chainDefs) / sizeof(chainDefs[0]);

    struct TwistEntry
    {
        int tgtIdx;
        Quaternion twistSelf;
        Quaternion twistParentInv;
    };

    std::vector<TwistEntry> twists;
    std::vector<int> tgtTwistIdx(tgtBvh->jointCount, -1);

    printf("Twist offsets:\n");

    for (int ci = 0; ci < chainDefCount; ci++)
    {
        const ChainDef& chain = chainDefs[ci];
        Quaternion parentTwist = QuaternionIdentity();

        for (int ji = 0; ji < chain.count; ji++)
        {
            const char* name = chain.joints[ji];
            const int tgtIdx = RetargetFindJointByName(tgtBvh, name);
            const int srcIdx = (tgtIdx >= 0) ? srcForTgt[tgtIdx] : -1;

            if (tgtIdx < 0 || srcIdx < 0)
            {
                printf("  %s: skipped (not in both skeletons)\n", name);
                parentTwist = QuaternionIdentity();
                continue;
            }

            const Quaternion currentParentTwistInv = QuaternionInvert(parentTwist);

            // figure out child reference joints for bone direction
            int tgtChildIdx = -1;
            int srcFromIdx = srcIdx;
            int srcToIdx = -1;
            const bool isHand = (ji == chain.handIdx && chain.handTgtChild != nullptr);

            if (isHand)
            {
                tgtChildIdx = RetargetFindJointByName(tgtBvh, chain.handTgtChild);
                srcFromIdx = RetargetFindJointByName(srcBvh, chain.handSrcFrom);
                srcToIdx = RetargetFindJointByName(srcBvh, chain.handSrcTo);
            }
            else if (ji + 1 < chain.count)
            {
                tgtChildIdx = RetargetFindJointByName(tgtBvh, chain.joints[ji + 1]);
                srcToIdx = (tgtChildIdx >= 0) ? srcForTgt[tgtChildIdx] : -1;
                if (srcToIdx < 0)
                {
                    // source might not have next chain joint mapped via tgt,
                    // try finding it directly
                    srcToIdx = RetargetFindJointByName(srcBvh, chain.joints[ji + 1]);
                }
            }
            else
            {
                // leaf joint (Head, ToeBase): no child bone to align, twist = identity
                TwistEntry e;
                e.tgtIdx = tgtIdx;
                e.twistSelf = QuaternionIdentity();
                e.twistParentInv = currentParentTwistInv;
                tgtTwistIdx[tgtIdx] = (int)twists.size();
                twists.push_back(e);
                parentTwist = QuaternionIdentity();
                printf("  %s: 0.0 deg (leaf)\n", name);
                continue;
            }

            // need valid child references on both sides
            if (tgtChildIdx < 0 || srcFromIdx < 0 || srcToIdx < 0)
            {
                TwistEntry e;
                e.tgtIdx = tgtIdx;
                e.twistSelf = QuaternionIdentity();
                e.twistParentInv = currentParentTwistInv;
                tgtTwistIdx[tgtIdx] = (int)twists.size();
                twists.push_back(e);
                parentTwist = QuaternionIdentity();
                printf("  %s: 0.0 deg (missing child ref)\n", name);
                continue;
            }

            // d_geno: target rest-pose bone direction (joint → child)
            // with identity rotations, child's local offset IS the bone direction
            const Vector3 dGenoRaw = tgtRest.localPositions[tgtChildIdx];
            if (Vector3LengthSqr(dGenoRaw) < 1e-12f)
            {
                TwistEntry e;
                e.tgtIdx = tgtIdx;
                e.twistSelf = QuaternionIdentity();
                e.twistParentInv = currentParentTwistInv;
                tgtTwistIdx[tgtIdx] = (int)twists.size();
                twists.push_back(e);
                parentTwist = QuaternionIdentity();
                printf("  %s: 0.0 deg (zero bone)\n", name);
                continue;
            }
            const Vector3 dGeno = Vector3Normalize(dGenoRaw);

            // d_target: source T-pose bone direction in joint's local frame
            const Vector3 dSrcWorldRaw = Vector3Subtract(
                srcTpose.globalPositions[srcToIdx],
                srcTpose.globalPositions[srcFromIdx]);
            if (Vector3LengthSqr(dSrcWorldRaw) < 1e-12f)
            {
                TwistEntry e;
                e.tgtIdx = tgtIdx;
                e.twistSelf = QuaternionIdentity();
                e.twistParentInv = currentParentTwistInv;
                tgtTwistIdx[tgtIdx] = (int)twists.size();
                twists.push_back(e);
                parentTwist = QuaternionIdentity();
                printf("  %s: 0.0 deg (zero src bone)\n", name);
                continue;
            }
            const Vector3 dSrcWorld = Vector3Normalize(dSrcWorldRaw);

            // rotate world direction into joint's local frame using inverse of
            // the source joint's T-pose global rotation
            const Vector3 dTarget = Vector3RotateByQuaternion(
                dSrcWorld, QuaternionInvert(srcTpose.globalRotations[srcIdx]));

            // minimum-angle rotation from dGeno to dTarget
            // using QuaternionBetween which handles antiparallel vectors safely
            const Quaternion twistSelf = QuaternionBetween(dGeno, dTarget);

            const float angle = 2.0f * acosf(Clamp(fabsf(twistSelf.w), 0.0f, 1.0f)) * RAD2DEG;
            printf("  %s: %.1f deg\n", name, angle);

            TwistEntry e;
            e.tgtIdx = tgtIdx;
            e.twistSelf = twistSelf;
            e.twistParentInv = currentParentTwistInv;
            tgtTwistIdx[tgtIdx] = (int)twists.size();
            twists.push_back(e);

            parentTwist = twistSelf;
        }
    }

    // ====================================================================
    // 6. construct output BVH (target hierarchy + new motion data)
    // ====================================================================

    BVHDataInit(outBvh);
    outBvh->jointCount = tgtBvh->jointCount;
    outBvh->joints = tgtBvh->joints;
    outBvh->channelCount = tgtBvh->channelCount;
    outBvh->frameCount = srcBvh->frameCount;
    outBvh->frameTime = srcBvh->frameTime;
    outBvh->motionData.resize(outBvh->frameCount * outBvh->channelCount, 0.0f);

    // ====================================================================
    // 7. per-frame retargeting
    // ====================================================================

    TransformData srcFrame;
    TransformDataResize(&srcFrame, srcBvh);

    printf("Retargeting %d frames...\n", srcBvh->frameCount);

    for (int frame = 0; frame < srcBvh->frameCount; frame++)
    {
        TransformDataSampleFrame(&srcFrame, srcBvh, frame, 1.0f);
        TransformDataForwardKinematics(&srcFrame);

        // target local rotations for this frame (start at identity)
        std::vector<Quaternion> tgtRots(tgtBvh->jointCount, QuaternionIdentity());
        Vector3 tgtRootPos = tgtBvh->joints[tgtRoot].offset;

        // root: scale position, copy rotation
        if (srcForTgt[tgtRoot] >= 0)
        {
            tgtRootPos = Vector3Scale(srcFrame.localPositions[srcForTgt[tgtRoot]], heightRatio);
            tgtRots[tgtRoot] = srcFrame.localRotations[srcForTgt[tgtRoot]];
        }

        // all other mapped joints: copy source local rotation, apply twist if needed
        for (int t = 0; t < tgtBvh->jointCount; t++)
        {
            if (t == tgtRoot) continue;
            const int s = srcForTgt[t];
            if (s < 0) continue;

            // use source local rotation directly.
            // this works even when target has extra intermediate joints that aren't
            // in the source: those joints stay at identity and don't break the chain.
            Quaternion q = srcFrame.localRotations[s];

            // apply twist correction
            const int twIdx = tgtTwistIdx[t];
            if (twIdx >= 0)
            {
                const TwistEntry& tw = twists[twIdx];
                q = QuaternionMultiply(QuaternionMultiply(tw.twistParentInv, q), tw.twistSelf);
            }

            tgtRots[t] = q;
        }

        // write to output motion data
        int channelOffset = 0;
        for (int t = 0; t < tgtBvh->jointCount; t++)
        {
            const BVHJointData& joint = tgtBvh->joints[t];

            // decompose quaternion into euler angles matching this joint's channel order
            float rotDeg[3] = { 0.0f, 0.0f, 0.0f };
            QuaternionToChannelOrder(tgtRots[t], joint.channels, joint.channelCount, rotDeg);

            int rotIdx = 0;
            for (int c = 0; c < joint.channelCount; c++)
            {
                const int ch = joint.channels[c];
                float value = 0.0f;

                if (ch == CHANNEL_X_POSITION)       value = (t == tgtRoot) ? tgtRootPos.x : joint.offset.x;
                else if (ch == CHANNEL_Y_POSITION)  value = (t == tgtRoot) ? tgtRootPos.y : joint.offset.y;
                else if (ch == CHANNEL_Z_POSITION)  value = (t == tgtRoot) ? tgtRootPos.z : joint.offset.z;
                else                                value = rotDeg[rotIdx++];

                outBvh->motionData[frame * outBvh->channelCount + channelOffset + c] = value;
            }

            channelOffset += joint.channelCount;
        }

        if ((frame + 1) % 100 == 0 || frame == srcBvh->frameCount - 1)
        {
            printf("  frame %d / %d\n", frame + 1, srcBvh->frameCount);
        }
    }

    printf("Done: %d frames, %d joints, %.1f fps\n",
        outBvh->frameCount, outBvh->jointCount, 1.0f / outBvh->frameTime);
    return true;
}
