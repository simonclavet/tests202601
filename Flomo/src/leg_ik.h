//#pragma once
//
//#include "raylib.h"
//#include "raymath.h"
//#include <algorithm>
//#include <cstdio>
//#include <cmath>
//#include <vector>
//#include "math_utils.h"
//#include "transform_data.h"
//
//// ==================================================================================
//// DATA STRUCTURES
//// ==================================================================================
//
//struct Bone {
//    Vector3 localPos;
//    Quaternion localRot;
//    float length;
//};
//
//struct LegChain {
//    Bone upleg;
//    Bone lowleg;
//    Bone foot;
//    Bone toe;
//
//    Vector3 parentGlobalPos;
//    Quaternion parentGlobalRot;
//
//    // Captured at start of frame (FK state)
//    Quaternion initialFootGlobalRot;
//    Quaternion initialToeGlobalRot;
//};
//
//struct LegChainIndices {
//    int hip = -1;
//    int upleg = -1;
//    int lowleg = -1;
//    int foot = -1;
//    int toe = -1;
//
//    bool IsValid() const {
//        return hip >= 0 && upleg >= 0 && lowleg >= 0 && foot >= 0 && toe >= 0;
//    }
//};
//
//// ==================================================================================
//// UTILS
//// ==================================================================================
//
//// Helper: Rotates a bone to point 'currentDir' towards 'targetDir'
//// respecting the existing "roll" of the bone as much as possible.
//Quaternion RotateBoneToward(Quaternion currentRot, Vector3 currentDir, Vector3 targetDir) {
//    if (Vector3LengthSqr(currentDir) < 0.001f || Vector3LengthSqr(targetDir) < 0.001f) return currentRot;
//
//    currentDir = Vector3Normalize(currentDir);
//    targetDir = Vector3Normalize(targetDir);
//
//    // Calculate the rotation from current direction to target direction
//    Quaternion rotFromTo = QuaternionFromVector3ToVector3(currentDir, targetDir);
//
//    // Apply this delta to the current local rotation
//    return QuaternionMultiply(rotFromTo, currentRot);
//}
//
//// Helper: Standard 2-Bone IK Law of Cosines
//// Returns the BEND ANGLE (in radians) for the knee.
//float CalculateKneeAngle(float lenUpper, float lenLower, float distToTarget) {
//    // Law of Cosines
//    // c^2 = a^2 + b^2 - 2ab cos(C)
//    // We want the internal angle C, so: cos(C) = (a^2 + b^2 - c^2) / 2ab
//    // BUT: The "knee bend" is (PI - C) because 0 is straight.
//
//    float cosKnee = (lenUpper * lenUpper + lenLower * lenLower - distToTarget * distToTarget)
//        / (2.0f * lenUpper * lenLower);
//
//    // Clamp to avoid NaN
//    cosKnee = Clamp(cosKnee, -1.0f, 1.0f);
//
//    // Internal angle
//    float angleInternal = SafeAcos(cosKnee);
//
//    // Bend angle (0 = straight, PI = folded back)
//    return PI - angleInternal;
//}
//
//// ==================================================================================
//// MAIN IK FUNCTION
//// ==================================================================================
//
//void SolveLegIK(LegChain& chain, const Vector3 targetToePos, float allowedFootRotationRatio)
//{
//    allowedFootRotationRatio = Clamp(allowedFootRotationRatio, 0.0f, 1.0f);
//
//    // 1. GLOBAL CONTEXT
//    Quaternion invParentRot = QuaternionInvert(chain.parentGlobalRot);
//    Vector3 hipGlobalPos = Vector3Add(chain.parentGlobalPos,
//        Vector3RotateByQuaternion(chain.upleg.localPos, chain.parentGlobalRot));
//
//    // 2. ANIMATION POSE DATA
//    // We need the vectors representing the bones in the CURRENT animation pose.
//    // This defines our "Preferred" bending plane (Pole Vector is implicit here).
//
//    // Upleg Local Vector (Hip -> Knee)
//    Vector3 kneeLocalPos = chain.lowleg.localPos;
//
//    // Lowleg Local Vector (Knee -> Ankle)
//    Vector3 ankleLocalPos = chain.foot.localPos;
//
//    float lenUpper = chain.upleg.length;
//    float lenLower = chain.lowleg.length;
//    float maxReach = lenUpper + lenLower;
//
//    // Output Storage
//    Quaternion uplegRotA, lowlegRotA, footRotA, toeRotA; // Strategy A
//    Quaternion uplegRotB, lowlegRotB, footRotB, toeRotB; // Strategy B
//
//    // ==============================================================================
//    // STRATEGY A: Tip-Based (Swimming/Pointed)
//    // The "End Effector" is the TOE. The whole leg points to target.
//    // ==============================================================================
//    {
//        // 1. Define the "End Effector" in local space of the thigh
//        // This is Hip->Knee + Knee->Ankle + Ankle->Toe
//        // We approximate the leg as a 2-bone chain where the "Shin" includes the foot offset.
//
//        Vector3 kneeToToe = Vector3Add(chain.foot.localPos, chain.toe.localPos);
//        float lenShinPlusFoot = Vector3Length(kneeToToe);
//
//        // Target relative to Hip
//        Vector3 targetDiff = Vector3Subtract(targetToePos, hipGlobalPos);
//        float distToTarget = Vector3Length(targetDiff);
//
//        // CLAMP: Do not extend further than the animation allows (or physical limit)
//        // We use the sum of lengths as the hard physical limit.
//        float currentReach = lenUpper + lenShinPlusFoot;
//        if (distToTarget > currentReach) {
//            targetDiff = Vector3Scale(Vector3Normalize(targetDiff), currentReach);
//            distToTarget = currentReach;
//        }
//
//        Vector3 targetRelToHip = Vector3RotateByQuaternion(targetDiff, invParentRot);
//
//        // 2. Solve Rotation for Thigh (Aim Hip->Toe vector at Target)
//        // Current Vector: Hip -> Toe (in current animation frame)
//        Vector3 currentEndEffectorPos = Vector3Add(chain.lowleg.localPos, kneeToToe);
//        uplegRotA = RotateBoneToward(chain.upleg.localRot, currentEndEffectorPos, targetRelToHip);
//
//        // 3. Solve Knee Bend (Simple distance adjustment)
//        // We just bend the knee to fit the distance.
//        // Note: For Strategy A, since we are treating (Shin+Foot) as one bone, accurate knee bending 
//        // is tricky without IK iterations. A simple approach is to keep the knee bend from animation
//        // and only rotate the thigh. 
//        // *Correction*: We will apply the knee bend based on the distance Hip->Toe.
//
//        float kneeAngle = CalculateKneeAngle(lenUpper, lenShinPlusFoot, distToTarget);
//
//        // Apply bend on the X-axis (assuming X is the bend axis in local space)
//        // Better: Apply bend around the local "Right" vector derived from the animation.
//        Vector3 bendAxis = Vector3Normalize(Vector3CrossProduct(chain.lowleg.localPos, kneeToToe));
//        if (Vector3LengthSqr(bendAxis) < 0.01f) bendAxis = { 1, 0, 0 }; // Fallback X
//
//        // Mix the solved bend with the animation's existing local rotation
//        // For simplicity in Strategy A, we often just use the LookAt for the thigh 
//        // and keep the knee relatively stiff or purely distance-based.
//        lowlegRotA = QuaternionFromAxisAngle(bendAxis, -kneeAngle); // Negative usually for knee
//
//        // Strategy A keeps foot/toe rigid relative to shin
//        footRotA = chain.foot.localRot;
//        toeRotA = chain.toe.localRot;
//    }
//
//    // ==============================================================================
//    // STRATEGY B: Ankle-Based (Walking/Flat Foot)
//    // The "End Effector" is the ANKLE. The Foot/Toe compensate to stay flat.
//    // ==============================================================================
//    {
//        // 1. Calculate where the Ankle needs to be.
//        // Target is Toe. Foot orientation is fixed to Initial Global.
//        // AnklePos = TargetToePos - (ToeOffset * InitialFootRotation)
//
//        // We want the foot to maintain the rotation it had at the start of the frame (FK).
//        Vector3 toeOffsetGlobal = Vector3RotateByQuaternion(chain.toe.localPos, chain.initialToeGlobalRot);
//        Vector3 targetAnkleGlobal = Vector3Subtract(targetToePos, toeOffsetGlobal);
//        Vector3 targetAnkleRelHipGlobal = Vector3Subtract(targetAnkleGlobal, hipGlobalPos);
//
//        float distToAnkle = Vector3Length(targetAnkleRelHipGlobal);
//
//        // CLAMP: Ensure Ankle never goes further than max leg extension
//        if (distToAnkle > maxReach) {
//            targetAnkleRelHipGlobal = Vector3Scale(Vector3Normalize(targetAnkleRelHipGlobal), maxReach);
//            distToAnkle = maxReach;
//        }
//
//        Vector3 targetAnkleRelHip = Vector3RotateByQuaternion(targetAnkleRelHipGlobal, invParentRot);
//
//        // 2. Solve Thigh Rotation (Aim Hip->Ankle vector at TargetAnkle)
//        // Use the current animation's knee position to define the plane.
//
//        // Current Vector: Hip -> Ankle (FK)
//        Vector3 currentAnklePos = Vector3Add(chain.lowleg.localPos, chain.foot.localPos);
//        uplegRotB = RotateBoneToward(chain.upleg.localRot, currentAnklePos, targetAnkleRelHip);
//
//        // 3. Solve Knee Bend
//        // Calculate required bend angle to hit the exact distance
//        float kneeAngle = CalculateKneeAngle(lenUpper, lenLower, distToAnkle);
//
//        // Find the local bend axis (Perpendicular to Thigh and Shin in current pose)
//        Vector3 bendAxis = Vector3Normalize(Vector3CrossProduct(chain.lowleg.localPos, chain.foot.localPos));
//        if (Vector3LengthSqr(bendAxis) < 0.01f) bendAxis = { 1, 0, 0 }; // Fallback
//
//        // Apply bend. We use the calculated angle, replacing the animation's knee bend.
//        // Ideally, we'd add delta, but for distance solving we need exact angles.
//        lowlegRotB = QuaternionFromAxisAngle(bendAxis, -kneeAngle);
//
//        // 4. Counter-Rotate Foot
//        // We need the foot to be at chain.initialFootGlobalRot
//        // Current Chain Rotation = Parent * NewUpleg * NewLowleg
//        Quaternion chainGlobal = QuaternionMultiply(chain.parentGlobalRot, uplegRotB);
//        chainGlobal = QuaternionMultiply(chainGlobal, lowlegRotB);
//
//        Quaternion invChainGlobal = QuaternionInvert(chainGlobal);
//        footRotB = QuaternionMultiply(invChainGlobal, chain.initialFootGlobalRot);
//
//        // Toe matches initial global too
//        // FootGlobal = ChainGlobal * FootRotB
//        Quaternion footGlobal = QuaternionMultiply(chainGlobal, footRotB);
//        toeRotB = QuaternionMultiply(QuaternionInvert(footGlobal), chain.initialToeGlobalRot);
//    }
//
//    // ==============================================================================
//    // BLEND
//    // ==============================================================================
//    chain.upleg.localRot = QuaternionSlerp(uplegRotB, uplegRotA, allowedFootRotationRatio);
//    chain.lowleg.localRot = QuaternionSlerp(lowlegRotB, lowlegRotA, allowedFootRotationRatio);
//    chain.foot.localRot = QuaternionSlerp(footRotB, footRotA, allowedFootRotationRatio);
//    chain.toe.localRot = QuaternionSlerp(toeRotB, toeRotA, allowedFootRotationRatio);
//}
//
//// ==================================================================================
//// INTEGRATION & CAPTURE
//// ==================================================================================
//
//// Must be called BEFORE SolveLegIK every frame to capture the animation pose
//void CaptureFKPose(LegChain& chain) {
//    Vector3 globalPos = chain.parentGlobalPos;
//    Quaternion globalRot = chain.parentGlobalRot;
//
//    // Hip
//    // (Assume parent transforms already applied to context)
//
//    // Upleg Global
//    globalPos = Vector3Add(globalPos, Vector3RotateByQuaternion(chain.upleg.localPos, globalRot));
//    globalRot = QuaternionMultiply(globalRot, chain.upleg.localRot);
//
//    // Lowleg Global
//    globalPos = Vector3Add(globalPos, Vector3RotateByQuaternion(chain.lowleg.localPos, globalRot));
//    Quaternion lowlegRot = QuaternionMultiply(globalRot, chain.lowleg.localRot);
//
//    // Foot Global
//    chain.initialFootGlobalRot = QuaternionMultiply(lowlegRot, chain.foot.localRot);
//
//    // Toe Global
//    chain.initialToeGlobalRot = QuaternionMultiply(chain.initialFootGlobalRot, chain.toe.localRot);
//}
//

//#pragma once
//
//#include "raylib.h"
//#include "raymath.h"
//#include <algorithm>
//#include <cstdio>
//#include <cmath>
//#include <vector>
//#include "math_utils.h"
//#include "transform_data.h"
//
//// ==================================================================================
//// UTILS
//// ==================================================================================
//
//// Robustly rotates a vector by a quaternion
//static inline Vector3 RotateVec(Vector3 v, Quaternion q) {
//    return Vector3RotateByQuaternion(v, q);
//}
//
//// Calculates the rotation required to align 'currentDir' to 'targetDir'
//// and applies it to 'currentRot'.
//Quaternion AlignBone(Quaternion currentRot, Vector3 currentDir, Vector3 targetDir) {
//    if (Vector3LengthSqr(currentDir) < 0.0001f || Vector3LengthSqr(targetDir) < 0.0001f)
//        return currentRot;
//
//    currentDir = Vector3Normalize(currentDir);
//    targetDir = Vector3Normalize(targetDir);
//
//    // Get the rotation from A to B
//    Quaternion delta = QuaternionFromVector3ToVector3(currentDir, targetDir);
//
//    // Apply delta: NewRot = Delta * OldRot
//    return QuaternionMultiply(delta, currentRot);
//}
//
//
//struct Bone {
//    Vector3 localPos;
//    Quaternion localRot;
//    float length;
//};
//
//struct LegChain {
//    Bone upleg;
//    Bone lowleg;
//    Bone foot;
//    Bone toe;
//
//    Vector3 parentGlobalPos;
//    Quaternion parentGlobalRot;
//
//    // Captured at start of frame (FK state)
//    Quaternion initialFootGlobalRot;
//    Quaternion initialToeGlobalRot;
//};
//
//struct LegChainIndices {
//    int hip = -1;
//    int upleg = -1;
//    int lowleg = -1;
//    int foot = -1;
//    int toe = -1;
//
//    bool IsValid() const {
//        return hip >= 0 && upleg >= 0 && lowleg >= 0 && foot >= 0 && toe >= 0;
//    }
//};
//
//// ==================================================================================
//// UTILS
//// ==================================================================================
//
//// Helper: Rotates a bone to point 'currentDir' towards 'targetDir'
//// respecting the existing "roll" of the bone as much as possible.
//Quaternion RotateBoneToward(Quaternion currentRot, Vector3 currentDir, Vector3 targetDir) {
//    if (Vector3LengthSqr(currentDir) < 0.001f || Vector3LengthSqr(targetDir) < 0.001f) return currentRot;
//
//    currentDir = Vector3Normalize(currentDir);
//    targetDir = Vector3Normalize(targetDir);
//
//    // Calculate the rotation from current direction to target direction
//    Quaternion rotFromTo = QuaternionFromVector3ToVector3(currentDir, targetDir);
//
//    // Apply this delta to the current local rotation
//    return QuaternionMultiply(rotFromTo, currentRot);
//}
//
//// Helper: Standard 2-Bone IK Law of Cosines
//// Returns the BEND ANGLE (in radians) for the knee.
//float CalculateKneeAngle(float lenUpper, float lenLower, float distToTarget) {
//    // Law of Cosines
//    // c^2 = a^2 + b^2 - 2ab cos(C)
//    // We want the internal angle C, so: cos(C) = (a^2 + b^2 - c^2) / 2ab
//    // BUT: The "knee bend" is (PI - C) because 0 is straight.
//
//    float cosKnee = (lenUpper * lenUpper + lenLower * lenLower - distToTarget * distToTarget)
//        / (2.0f * lenUpper * lenLower);
//
//    // Clamp to avoid NaN
//    cosKnee = Clamp(cosKnee, -1.0f, 1.0f);
//
//    // Internal angle
//    float angleInternal = SafeAcos(cosKnee);
//
//    // Bend angle (0 = straight, PI = folded back)
//    return PI - angleInternal;
//}
//
//// ==================================================================================
//// SOLVER LOGIC
//// ==================================================================================
//
//// Solves the position of the Knee in Hip-Local space.
//// It tries to keep the new knee as close to the 'currentKneePos' plane as possible.
//Vector3 CalculateKneePosition(
//    Vector3 targetPos,     // Target in Hip Space
//    Vector3 currentKneePos,// Current Knee in Hip Space (defines bend plane)
//    float lenThigh,
//    float lenShin)
//{
//    float distToTarget = Vector3Length(targetPos);
//
//    // 1. Clamp distance (Reach limit)
//    float maxReach = lenThigh + lenShin;
//    float minReach = std::abs(lenThigh - lenShin);
//
//    // Soften or hard clamp
//    if (distToTarget > maxReach) {
//        targetPos = Vector3Scale(Vector3Normalize(targetPos), maxReach);
//        distToTarget = maxReach;
//    }
//    else if (distToTarget < minReach) {
//        targetPos = Vector3Scale(Vector3Normalize(targetPos), minReach);
//        distToTarget = minReach;
//    }
//
//    // 2. Law of Cosines to find the angle of the Thigh relative to the Target Vector
//    // cos(alpha) = (b^2 + c^2 - a^2) / 2bc
//    // Thigh is 'b', TargetDist is 'c', Shin is 'a'
//    float cosAlpha = (lenThigh * lenThigh + distToTarget * distToTarget - lenShin * lenShin)
//        / (2.0f * lenThigh * distToTarget);
//
//    float alpha = std::acos(Clamp(cosAlpha, -1.0f, 1.0f));
//
//    // 3. Construct the Bend Plane
//    // We need a "Right" vector perpendicular to the [Hip -> Target] line.
//    // We use the Current Knee position to define this plane.
//    Vector3 hipToTargetDir = Vector3Normalize(targetPos);
//
//    // Plane Normal = Cross(HipToTarget, HipToCurrentKnee)
//    Vector3 planeNormal = Vector3CrossProduct(hipToTargetDir, currentKneePos);
//
//    // FALLBACK: If leg is perfectly straight (Singularity), we can't find a plane.
//    // Use the Thigh's initial direction or a generic axis to force a bend.
//    if (Vector3LengthSqr(planeNormal) < 0.001f) {
//        // Try getting an orthogonal vector to the target
//        Vector3 absDir = { std::abs(hipToTargetDir.x), std::abs(hipToTargetDir.y), std::abs(hipToTargetDir.z) };
//        Vector3 helper = (absDir.x < absDir.y) ? Vector3{ 1,0,0 } : Vector3{ 0,1,0 };
//        planeNormal = Vector3CrossProduct(hipToTargetDir, helper);
//    }
//    planeNormal = Vector3Normalize(planeNormal);
//
//    // 4. Calculate Knee Position
//    // Rotate the TargetVector away from the target by 'alpha' around the PlaneNormal.
//    // However, the TargetVector points AT the target. The Thigh starts at Hip.
//    // We actually want to rotate the vector (Hip->Target) by 'alpha' to get (Hip->Knee).
//
//    Quaternion rotToKnee = QuaternionFromAxisAngle(planeNormal, alpha);
//    Vector3 thighDir = RotateVec(hipToTargetDir, rotToKnee);
//
//    return Vector3Scale(thighDir, lenThigh);
//}
//
//// ==================================================================================
//// MAIN FUNCTION
//// ==================================================================================
//
//void SolveLegIK(LegChain& chain, const Vector3 targetToePos, float allowedFootRotationRatio)
//{
//    allowedFootRotationRatio = Clamp(allowedFootRotationRatio, 0.0f, 1.0f);
//
//    // CONTEXT TRANSFORMS
//    Quaternion invParentRot = QuaternionInvert(chain.parentGlobalRot);
//    Vector3 hipGlobalPos = Vector3Add(chain.parentGlobalPos,
//        RotateVec(chain.upleg.localPos, chain.parentGlobalRot));
//
//    // BONE VECTORS (Current FK Pose)
//    // We use these to define the bone axes without assuming "Y is down" etc.
//    // Note: These are vectors relative to their parent bone's origin.
//    Vector3 vThighBone = chain.lowleg.localPos; // Vector from Hip to Knee (in Hip local space)
//
//    // For Strategy A, we combine Shin + Foot + Toe into one "Shin" vector
//    // We need this vector in the Knee's local space.
//    // ShinVec = AnklePos + Rotate(AnkleRot, ToePos)
//    Vector3 vShinBoneComposite = Vector3Add(chain.foot.localPos,
//        RotateVec(chain.toe.localPos, chain.foot.localRot));
//
//    float lenThigh = chain.upleg.length;
//    float lenShinComposite = Vector3Length(vShinBoneComposite);
//
//    // Output Storage
//    Quaternion uplegRotA, lowlegRotA;
//
//    // ==============================================================================
//    // STRATEGY A: Tip-Based (Geometric Plane Solver)
//    // ==============================================================================
//    {
//        // 1. Target in Hip Space
//        Vector3 targetGlobalDiff = Vector3Subtract(targetToePos, hipGlobalPos);
//        Vector3 targetLocal = RotateVec(targetGlobalDiff, invParentRot);
//
//        // 2. Current Knee in Hip Space
//        // We calculate where the knee is right now in the animation.
//        // This is crucial for the "Pole Vector" implicit logic.
//        Vector3 currentKneeLocal = RotateVec(vThighBone, chain.upleg.localRot);
//
//        // 3. Solve for EXACT Knee Position
//        // This function guarantees the knee length is preserved and it bends 
//        // on the natural plane defined by the current animation.
//        Vector3 idealKneePos = CalculateKneePosition(targetLocal, currentKneeLocal, lenThigh, lenShinComposite);
//
//        // 4. Align Thigh
//        // Rotate the Thigh Bone Vector (vThighBone) to point at IdealKneePos
//        uplegRotA = AlignBone(chain.upleg.localRot, currentKneeLocal, idealKneePos);
//
//        // 5. Align Shin
//        // The Shin needs to point from [IdealKnee] to [Target]
//        Vector3 shinTargetDir = Vector3Subtract(targetLocal, idealKneePos); // Global direction for shin
//
//        // We need the Current Shin Direction in Hip Space to calculate the delta.
//        // CurrentShinGlobal = UplegRot * LowlegRot * ShinBoneVec
//        Quaternion currentShinGlobalRot = QuaternionMultiply(uplegRotA, chain.lowleg.localRot);
//        Vector3 currentShinDir = RotateVec(vShinBoneComposite, currentShinGlobalRot);
//
//        // Calculate Delta Rotation in Global (Hip) Space
//        Quaternion shinDelta = QuaternionFromVector3ToVector3(Vector3Normalize(currentShinDir), Vector3Normalize(shinTargetDir));
//
//        // Apply delta to the Shin's Local Rotation
//        // NewGlobal = Delta * OldGlobal
//        // NewLocal = InvParent * NewGlobal
//        Quaternion newShinGlobalRot = QuaternionMultiply(shinDelta, currentShinGlobalRot);
//        lowlegRotA = QuaternionMultiply(QuaternionInvert(uplegRotA), newShinGlobalRot);
//    }
//
//    // Apply (ignoring Strategy B blending for now as requested, just using A)
//    chain.upleg.localRot = uplegRotA;
//    chain.lowleg.localRot = lowlegRotA;
//
//    // Foot/Toe stay rigid in Strategy A
//    // (If you want to blend later, re-add the lerp logic here)
//}
//
//// ... (ExtractLegChain, ApplyLegChain, InitLegChain, etc. remain the same)
//// ==================================================================================
//// INTEGRATION
//// ==================================================================================
//
//void ExtractLegChain(LegChain& chain, const TransformData* xform, const LegChainIndices& idx)
//{
//    if (!idx.IsValid()) return;
//    chain.upleg.localPos = xform->localPositions[idx.upleg];
//    chain.lowleg.localPos = xform->localPositions[idx.lowleg];
//    chain.foot.localPos = xform->localPositions[idx.foot];
//    chain.toe.localPos = xform->localPositions[idx.toe];
//
//    chain.upleg.localRot = xform->localRotations[idx.upleg];
//    chain.lowleg.localRot = xform->localRotations[idx.lowleg];
//    chain.foot.localRot = xform->localRotations[idx.foot];
//    chain.toe.localRot = xform->localRotations[idx.toe];
//
//    chain.upleg.length = Vector3Length(chain.lowleg.localPos);
//    chain.lowleg.length = Vector3Length(chain.foot.localPos);
//
//    chain.parentGlobalPos = xform->globalPositions[idx.hip];
//    chain.parentGlobalRot = xform->globalRotations[idx.hip];
//}
//
//void ApplyLegChainToTransform(const LegChain& chain, TransformData* xform, const LegChainIndices& idx)
//{
//    if (!idx.IsValid()) return;
//    xform->localRotations[idx.upleg] = chain.upleg.localRot;
//    xform->localRotations[idx.lowleg] = chain.lowleg.localRot;
//    xform->localRotations[idx.foot] = chain.foot.localRot;
//    xform->localRotations[idx.toe] = chain.toe.localRot;
//}
//
//void RecomputeLegFK(TransformData* xform, const LegChainIndices& idx)
//{
//    if (!idx.IsValid()) return;
//    const int joints[] = { idx.upleg, idx.lowleg, idx.foot, idx.toe };
//    for (int i = 0; i < 4; ++i) {
//        const int j = joints[i];
//        const int p = xform->parents[j];
//        xform->globalPositions[j] = Vector3Add(xform->globalPositions[p], Vector3RotateByQuaternion(xform->localPositions[j], xform->globalRotations[p]));
//        xform->globalRotations[j] = QuaternionMultiply(xform->globalRotations[p], xform->localRotations[j]);
//    }
//}
//
//bool SolveLegIKInPlace(TransformData* xform, const LegChainIndices& idx, const Vector3 targetToePos, const float allowedFootRotationRatio)
//{
//    if (!idx.IsValid()) return false;
//    LegChain chain;
//    ExtractLegChain(chain, xform, idx);
//    //CaptureFKPose(chain);
//    SolveLegIK(chain, targetToePos, allowedFootRotationRatio);
//    ApplyLegChainToTransform(chain, xform, idx);
//    RecomputeLegFK(xform, idx);
//    return true;
//}
//
//
//
//
//
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
    const Vector3 ankleToToeGlobal = RotateVec(chain.toe.localPos, toeGlobalRot);
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
    const float extensionSlack = (lenFullLeg - currentExtension) * 0.9f; // Giving 50% slack for smoother feel
    const float limit = currentExtension + extensionSlack;

    // 3. SoftStart: Where we begin to deviate from linear 1:1 behavior.
    // Let's start smoothing at the current extension point? 
    // Or slightly further out? Let's say we soft clamp the "slack" region.
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
    const int joints[] = { idx.upleg, idx.lowleg, idx.foot, idx.toe };
    for (int i = 0; i < 4; ++i) {
        const int j = joints[i];
        const int p = xform->parents[j];
        xform->globalPositions[j] = Vector3Add(xform->globalPositions[p], Vector3RotateByQuaternion(xform->localPositions[j], xform->globalRotations[p]));
        xform->globalRotations[j] = QuaternionMultiply(xform->globalRotations[p], xform->localRotations[j]);
    }
}

bool SolveLegIKInPlace(TransformData* xform, const LegChainIndices& idx, const Vector3 targetToePos, const float allowedFootRotationRatio)
{
    if (!idx.IsValid()) return false;
    LegChain chain;
    ExtractLegChain(chain, xform, idx);

    // We ignore 'allowedFootRotationRatio' now as we are doing pure Strategy B
    SolveLegIK(chain, targetToePos);

    ApplyLegChainToTransform(chain, xform, idx);
    RecomputeLegFK(xform, idx);
    return true;
}