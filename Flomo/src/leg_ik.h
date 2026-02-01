
#pragma once

#include "raylib.h"
#include "raymath.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include "math_utils.h"
#include "transform_data.h"

// ==================================================================================
// UTILS
// ==================================================================================

static inline Vector3 RotateVec(Vector3 v, Quaternion q) {
    return Vector3RotateByQuaternion(v, q);
}

// Calculates rotation to align 'currentDir' to 'targetDir' 
// while preserving the 'twist' relative to the rotation.
Quaternion AlignBone(Quaternion currentRot, Vector3 currentDir, Vector3 targetDir) {
    if (Vector3LengthSqr(currentDir) < 0.0001f || Vector3LengthSqr(targetDir) < 0.0001f)
        return currentRot;

    currentDir = Vector3Normalize(currentDir);
    targetDir = Vector3Normalize(targetDir);

    // Delta rotation required to move A to B
    Quaternion delta = QuaternionFromVector3ToVector3(currentDir, targetDir);

    // Apply delta to the existing rotation
    return QuaternionMultiply(delta, currentRot);
}

// Smoothly clamps 'dist' so it approaches 'limit' as 'dist' approaches 'maxInput'.
// Maps [softStart, maxInput] -> [softStart, limit] using a sine-out curve.
float SoftClampDistance(float dist, float softStart, float limit, float maxInput) {
    if (dist <= softStart) return dist;
    if (dist >= maxInput) return limit;

    const float t = (dist - softStart) / (maxInput - softStart); // 0.0 to 1.0
    // Sine ease-out: sin(t * PI/2)
    const float ease = t;// std::sin(t * PI * 0.5f);

    return softStart + (limit - softStart) * ease;
}

// ==================================================================================
// DATA STRUCTURES
// ==================================================================================

struct Bone {
    Vector3 localPos;
    Quaternion localRot;
    float length;
};

struct LegChain {
    Bone upleg;   // Thigh
    Bone lowleg;  // Shin
    Bone foot;    // Ankle
    Bone toe;

    Vector3 parentGlobalPos;
    Quaternion parentGlobalRot;
};

struct LegChainIndices {
    int hip = -1;
    int upleg = -1;
    int lowleg = -1;
    int foot = -1;
    int toe = -1;

    bool IsValid() const {
        return hip >= 0 && upleg >= 0 && lowleg >= 0 && foot >= 0 && toe >= 0;
    }
};

// ==================================================================================
// CORE SOLVER LOGIC
// ==================================================================================

// Solves for the ideal knee position that satisfies bone lengths 
// and stays on the bending plane defined by the current animation.
Vector3 CalculateIdealKneePos(
    Vector3 hipToTarget,    // Vector from Hip to Target (in Hip Space)
    Vector3 hipToKneeCurrent, // Vector from Hip to Current Knee (in Hip Space)
    const Vector3 kneeToAnkleCurrent, // Shin Vector (in Hip Space)
    float lenThigh,
    float lenShin)
{
    const float distToTarget = Vector3Length(hipToTarget);

    // 1. Law of Cosines: Find angle at Hip (alpha)
    // cos(alpha) = (b^2 + c^2 - a^2) / 2bc
    // b = thigh, c = targetDist, a = shin
    const float cosAlpha = (lenThigh * lenThigh + distToTarget * distToTarget - lenShin * lenShin)
        / (2.0f * lenThigh * distToTarget);

    // Clamp is vital here in case target is slightly out of reach due to float errors
    const float alpha = std::acos(Clamp(cosAlpha, -1.0f, 1.0f));
    // 2. Define the Bend Plane
    // Preferred: The plane defined by the current Thigh and Shin vectors.
    // This captures the "Hinge Axis" of the knee.
    Vector3 planeNormal = Vector3Negate(Vector3CrossProduct(hipToKneeCurrent, kneeToAnkleCurrent));

    // Fallback 1: If Thigh and Shin are parallel (straight leg in anim),
    // try using the Hip->Target and Hip->Knee relation.
    if (Vector3LengthSqr(planeNormal) < 0.0001f) {
        planeNormal = Vector3CrossProduct(hipToTarget, hipToKneeCurrent);
    }

    // Fallback 2: If everything is collinear (Straight leg pointing exactly at target),
    // pick an arbitrary axis perpendicular to the target vector.
    if (Vector3LengthSqr(planeNormal) < 0.0001f) {
        const Vector3 absDir = { std::abs(hipToTarget.x), std::abs(hipToTarget.y), std::abs(hipToTarget.z) };
        const Vector3 helper = (absDir.x < absDir.y) ? Vector3{ 1,0,0 } : Vector3{ 0,1,0 };
        planeNormal = Vector3CrossProduct(hipToTarget, helper);
    }
    planeNormal = Vector3Normalize(planeNormal);

    // 3. Rotate the target vector by 'alpha' around the plane normal
    // We start with the normalized vector pointing at the target
    Vector3 thighDir = Vector3Normalize(hipToTarget);

    // Rotate it 'up' (away from target) by alpha to find the thigh direction
    const Quaternion rotToKnee = QuaternionFromAxisAngle(planeNormal, alpha);
    thighDir = RotateVec(thighDir, rotToKnee);

    return Vector3Scale(thighDir, lenThigh);
}

// ==================================================================================
// MAIN IK FUNCTION
// ==================================================================================

void SolveLegIK(LegChain& chain, const Vector3 targetToePosWorld)
{
    // 1. CAPTURE CONTEXT
    const Quaternion invParentRot = QuaternionInvert(chain.parentGlobalRot);
    const Vector3 hipGlobalPos = Vector3Add(chain.parentGlobalPos,
        RotateVec(chain.upleg.localPos, chain.parentGlobalRot));

    // 2. CALCULATE CURRENT GLOBAL ORIENTATIONS (From Animation)
    // We need these to maintain the foot's global rotation.
    const Quaternion uplegGlobalRot = QuaternionMultiply(chain.parentGlobalRot, chain.upleg.localRot);
    const Quaternion lowlegGlobalRot = QuaternionMultiply(uplegGlobalRot, chain.lowleg.localRot);
    const Quaternion footGlobalRot = QuaternionMultiply(lowlegGlobalRot, chain.foot.localRot); // This is what we preserve!
    const Quaternion toeGlobalRot = QuaternionMultiply(footGlobalRot, chain.toe.localRot);

    // 3. CALCULATE ANKLE TARGET
    // We want the Toe to hit 'targetToePosWorld'.
    // The Foot is rigid in world space (orientation from Anim).
    // AnklePos = TargetToe - (ToeOffset rotated by FootGlobalRot)

    // Vector from Ankle to Toe in World Space
    // toe.localPos is offset from foot to toe, so rotate by footGlobalRot (not toeGlobalRot)
    const Vector3 ankleToToeGlobal = RotateVec(chain.toe.localPos, footGlobalRot);
    Vector3 targetAnkleGlobal = Vector3Subtract(targetToePosWorld, ankleToToeGlobal);

    // 4. CLAMP ANKLE TARGET (Hyperextension Prevention)
    // We calculate the Hip->Ankle vector in World Space
    Vector3 hipToTargetAnkle = Vector3Subtract(targetAnkleGlobal, hipGlobalPos);
    float distToAnkle = Vector3Length(hipToTargetAnkle);

    // Distances
    const float lenFullLeg = chain.upleg.length + chain.lowleg.length;

    // We calculate the current extension in the animation frame to set our baseline.
    // Ideally this is calculated from world positions, but local rotations suffice here.
    // Current Knee in Hip Space
    const Vector3 currentKneeLocal = RotateVec(chain.lowleg.localPos, chain.upleg.localRot);
    // Current Ankle in Hip Space (Knee + Rotate(Shin))
    const Vector3 currentAnkleLocal = Vector3Add(currentKneeLocal,
        RotateVec(chain.foot.localPos, QuaternionMultiply(chain.upleg.localRot, chain.lowleg.localRot)));

    const float currentExtension = Vector3Length(currentAnkleLocal);

    // PARAMETERS FOR SOFT CLAMP
    // 1. MaxInput: The leg is physically straight.
    const float maxInput = lenFullLeg;

    // 2. Limit: The absolute max length we will allow the IK to output.
    // Formula: Allow 25% of the remaining range beyond current pose.
    // If current is 0.8 and max is 1.0, allowed = 0.8 + 0.25 * 0.2 = 0.85
    const float extensionSlack = (lenFullLeg - currentExtension) * 0.5f; // Giving 50% slack for smoother feel
    const float limit = currentExtension + extensionSlack;

    // 3. SoftStart: Where we begin to deviate from linear 1:1 behavior.
    const float softStart = currentExtension + 0.25f * extensionSlack;

    // Calculate clamped distance
    float clampedDist = distToAnkle;

    // Only apply clamp if we are actually extending (and leg isn't already hyper-extended/broken)
    if (distToAnkle > softStart && maxInput > softStart + 0.001f) {
        clampedDist = SoftClampDistance(distToAnkle, softStart, limit, maxInput);

        // Apply the new length to the vector
        // Note: We perform this scaling even if distToAnkle < maxInput, because soft clamp kicks in earlier.
        hipToTargetAnkle = Vector3Scale(Vector3Normalize(hipToTargetAnkle), clampedDist);

        // Recalculate global ankle target based on clamped vector
        targetAnkleGlobal = Vector3Add(hipGlobalPos, hipToTargetAnkle);
    }
    else if (distToAnkle > maxInput) {
        // Hard clamp for safety if logic falls through or leg is super short
        hipToTargetAnkle = Vector3Scale(Vector3Normalize(hipToTargetAnkle), maxInput);
        targetAnkleGlobal = Vector3Add(hipGlobalPos, hipToTargetAnkle);
    }


    // We use the current animation pose to define the "Safe Max Reach".
    // Or simpler: strictly Thigh + Shin length.
    // If we want to strictly respect the animation's "straightness", we could measure
    // the current Hip->Ankle distance, but (Thigh+Shin) is physically correct.
    //float maxReach = chain.upleg.length + chain.lowleg.length;

    //if (distToAnkle > maxReach) {
    //    hipToTargetAnkle = Vector3Scale(Vector3Normalize(hipToTargetAnkle), maxReach);
    //    distToAnkle = maxReach;
    //    // Recalculate the valid Ankle Target
    //    targetAnkleGlobal = Vector3Add(hipGlobalPos, hipToTargetAnkle);
    //}

    // 5. PREPARE VECTORS FOR SOLVER (IN HIP SPACE)
    // We work in Hip Local Space (Relative to Parent Global Rotation)
    // This removes the parent's rotation from the equation.

    // Target Ankle (Relative to Hip, rotated into Hip Space)
    const Vector3 targetAnkleLocal = RotateVec(hipToTargetAnkle, invParentRot);

    // Current Knee Position (Relative to Hip, rotated into Hip Space)
    // We assume the bone local positions in the struct are valid offsets
    // Hip -> Knee is just chain.lowleg.localPos relative to hip? 
    // Wait, struct Bone.localPos is offset *from parent*. 
    // So Upleg.localPos is Hip->ThighStart. Lowleg.localPos is Knee relative to Hip?
    // Usually Lowleg.localPos is the offset from Hip to Knee.
    // Let's assume standard hierarchy:
    // Upleg local pos = offset from Hip bone origin.
    // Lowleg local pos = offset from Upleg bone origin (Knee).

    // We need the vector representing the Thigh Bone in Hip Space.
    // Since we are solving FOR rotations, we need the vectors derived from the CURRENT rotations.
    //const Vector3 currentKneeLocal = RotateVec(chain.lowleg.localPos, chain.upleg.localRot);

    // We also need the Shin vector (Knee -> Ankle) in Hip Space to calculate rotation deltas later.
    // CurrentShinGlobal = Upleg * Lowleg * FootPos
    // We need it relative to the Knee though.
    // Let's just solve positions first.
    // Current Knee Position (Relative to Hip, rotated into Hip Space)
    //const Vector3 currentKneeLocal = RotateVec(chain.lowleg.localPos, chain.upleg.localRot);

    // NEW: Current Shin Vector (Relative to Knee, rotated into Hip Space)
    // We need the vector from Knee to Ankle, represented in Hip Space.
    // Rotation chain: UplegRot * LowlegRot
    const Quaternion currentShinRotHipSpace = QuaternionMultiply(chain.upleg.localRot, chain.lowleg.localRot);
    const Vector3 currentShinVectorHipSpace = RotateVec(chain.foot.localPos, currentShinRotHipSpace);

    // 6. SOLVE KNEE POSITION
    const Vector3 idealKneeLocal = CalculateIdealKneePos(
        targetAnkleLocal,
        currentKneeLocal,
        currentShinVectorHipSpace, // Pass the Shin Vector here
        chain.upleg.length,
        chain.lowleg.length
    );

    // 7. APPLY ROTATIONS

    // A. UPLEG (Thigh)
    // Rotate the Thigh vector (Hip->Knee) to align with IdealKneePos
    chain.upleg.localRot = AlignBone(chain.upleg.localRot, currentKneeLocal, idealKneeLocal);

    // B. LOWLEG (Shin)
    // The Shin must point from IdealKnee -> TargetAnkle.
    const Vector3 shinTargetDir = Vector3Subtract(targetAnkleLocal, idealKneeLocal);

    // We need the *current* Shin direction in Hip Space to calculate the delta.
    // CurrentShinGlobal = (NewUplegRot * OldLowlegRot) applied to ShinVector
    // ShinVector is chain.foot.localPos
    const Quaternion currentShinGlobalRot = QuaternionMultiply(chain.upleg.localRot, chain.lowleg.localRot);
    const Vector3 currentShinDir = RotateVec(chain.foot.localPos, currentShinGlobalRot);

    // Calculate global delta to align Shin
    const Quaternion shinDelta = QuaternionFromVector3ToVector3(Vector3Normalize(currentShinDir), Vector3Normalize(shinTargetDir));

    // Apply delta to local rotation
    const Quaternion newShinGlobalRot = QuaternionMultiply(shinDelta, currentShinGlobalRot);
    // Remove Upleg rotation to get local Lowleg rotation
    chain.lowleg.localRot = QuaternionMultiply(QuaternionInvert(chain.upleg.localRot), newShinGlobalRot);

    // C. FOOT (Ankle)
    // We want the foot to maintain 'footGlobalRot' (captured from anim).
    // Current Chain Rotation = Parent * NewUpleg * NewLowleg
    Quaternion chainRot = QuaternionMultiply(chain.parentGlobalRot, chain.upleg.localRot);
    chainRot = QuaternionMultiply(chainRot, chain.lowleg.localRot);

    // NewFootLocal = Inv(ChainRot) * DesiredFootGlobal
    chain.foot.localRot = QuaternionMultiply(QuaternionInvert(chainRot), footGlobalRot);

    // D. TOE
    // Toe stays rigid relative to foot, so we generally don't touch it, 
    // OR we explicitly set it to match global anim if desired. 
    // Since Foot matches global, and Toe is child of Foot, Toe automatically matches global 
    // IF we keep local rotation same.
    // chain.toe.localRot = chain.toe.localRot; // No change needed
}

// ==================================================================================
// INTEGRATION
// ==================================================================================

void ExtractLegChain(LegChain& chain, const TransformData* xform, const LegChainIndices& idx)
{
    if (!idx.IsValid()) return;
    chain.upleg.localPos = xform->localPositions[idx.upleg];
    chain.lowleg.localPos = xform->localPositions[idx.lowleg];
    chain.foot.localPos = xform->localPositions[idx.foot];
    chain.toe.localPos = xform->localPositions[idx.toe];

    chain.upleg.localRot = xform->localRotations[idx.upleg];
    chain.lowleg.localRot = xform->localRotations[idx.lowleg];
    chain.foot.localRot = xform->localRotations[idx.foot];
    chain.toe.localRot = xform->localRotations[idx.toe];

    // Calculate lengths based on offsets to children
    chain.upleg.length = Vector3Length(chain.lowleg.localPos);
    chain.lowleg.length = Vector3Length(chain.foot.localPos);

    chain.parentGlobalPos = xform->globalPositions[idx.hip];
    chain.parentGlobalRot = xform->globalRotations[idx.hip];
}

void ApplyLegChainToTransform(const LegChain& chain, TransformData* xform, const LegChainIndices& idx)
{
    if (!idx.IsValid()) return;
    xform->localRotations[idx.upleg] = chain.upleg.localRot;
    xform->localRotations[idx.lowleg] = chain.lowleg.localRot;
    xform->localRotations[idx.foot] = chain.foot.localRot;
    xform->localRotations[idx.toe] = chain.toe.localRot;
}

void RecomputeLegFK(TransformData* xform, const LegChainIndices& idx)
{
    if (!idx.IsValid()) return;

    // track which joints have been updated (for propagating to descendants)
    std::vector<bool> updated(xform->jointCount, false);

    // first update the 4 leg chain joints
    const int legJoints[] = { idx.upleg, idx.lowleg, idx.foot, idx.toe };
    for (int i = 0; i < 4; ++i)
    {
        const int j = legJoints[i];
        const int p = xform->parents[j];
        xform->globalPositions[j] = Vector3Add(xform->globalPositions[p], Vector3RotateByQuaternion(xform->localPositions[j], xform->globalRotations[p]));
        xform->globalRotations[j] = QuaternionMultiply(xform->globalRotations[p], xform->localRotations[j]);
        updated[j] = true;
    }

    // propagate to descendants - multiple passes until no more updates
    bool anyUpdated = true;
    while (anyUpdated)
    {
        anyUpdated = false;
        for (int j = 0; j < xform->jointCount; ++j)
        {
            if (updated[j]) continue; // already done
            const int p = xform->parents[j];
            if (p >= 0 && updated[p])
            {
                xform->globalPositions[j] = Vector3Add(xform->globalPositions[p], Vector3RotateByQuaternion(xform->localPositions[j], xform->globalRotations[p]));
                xform->globalRotations[j] = QuaternionMultiply(xform->globalRotations[p], xform->localRotations[j]);
                updated[j] = true;
                anyUpdated = true;
            }
        }
    }
}

bool SolveLegIKInPlace(TransformData* xform, const LegChainIndices& idx, const Vector3 targetToePos)
{
    if (!idx.IsValid()) return false;
    LegChain chain;
    ExtractLegChain(chain, xform, idx);
    SolveLegIK(chain, targetToePos);

    ApplyLegChainToTransform(chain, xform, idx);
    RecomputeLegFK(xform, idx);
    return true;
}