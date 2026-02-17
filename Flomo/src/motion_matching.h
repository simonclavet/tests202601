#pragma once

#include <vector>
#include <span>
#include <cassert>
#include "raylib.h"
#include "anim_database.h"
#include "transform_data.h"
#include "app_config.h"
#include "utils.h"
#include "definitions.h"

// Compute future velocities at given trajectory times
// Uses acceleration-based extrapolation from current towards desired
// currentVel: current velocity (XZ, world space)
// desiredVel: desired velocity (XZ, world space)
// maxAcceleration: maximum acceleration constraint (m/s^2)
// futureTimes: array of future time offsets (seconds)
// outVelocities: output array (must be sized to match futureTimes)
static inline void ComputeFutureVelocities(
    const Vector3& currentVel,
    const Vector3& desiredVel,
    float maxAcceleration,
    const std::vector<float>& futureTimes,
    std::vector<Vector3>& outVelocities)
{
    outVelocities.resize(futureTimes.size());

    for (int p = 0; p < (int)futureTimes.size(); ++p)
    {
        const float futureTime = futureTimes[p];

        // How much velocity can change in this time?
        const float maxDeltaVelMag = maxAcceleration * futureTime;

        // Compute velocity at futureTime: move from current towards desired, clamped by max change
        const Vector3 velDelta = Vector3Subtract(desiredVel, currentVel);
        const float velDeltaMag = Vector3Length(velDelta);

        Vector3 futureVel;
        if (velDeltaMag <= maxDeltaVelMag)
        {
            // Can reach desired velocity within this time
            futureVel = desiredVel;
        }
        else if (velDeltaMag > 1e-6f)
        {
            // Clamp to max achievable change
            const Vector3 velDeltaDir = Vector3Scale(velDelta, 1.0f / velDeltaMag);
            futureVel = Vector3Add(currentVel, Vector3Scale(velDeltaDir, maxDeltaVelMag));
        }
        else
        {
            futureVel = currentVel;
        }

        outVelocities[p] = futureVel;
    }
}

// max angular velocity for aim direction smoothing (180 degrees in 0.5 seconds)
constexpr float MAX_AIM_ANGULAR_VELOCITY = 3.14159f / 0.5f;

struct FutureAimPoint
{
    Vector3 direction;
    float angularVelocity;  // rad/s around Y
};

// compute planned future aim directions, same idea as ComputeFutureVelocities but for angles.
// the smoothed aim chases the desired aim at a capped angular velocity.
// for each future time point we compute where the aim will be and how fast it's turning.
static inline void ComputeFutureAimDirections(
    const Vector3& currentAimDir,
    const Vector3& desiredAimDir,
    float maxAngularVelocity,
    const std::vector<float>& futureTimes,
    std::vector<FutureAimPoint>& outPoints)
{
    outPoints.resize(futureTimes.size());

    const float angleDiff = SignedAngleY(currentAimDir, desiredAimDir);
    const float sign = (angleDiff >= 0.0f) ? 1.0f : -1.0f;
    const float absAngle = fabsf(angleDiff);

    for (int p = 0; p < (int)futureTimes.size(); ++p)
    {
        const float t = futureTimes[p];
        const float maxTurn = maxAngularVelocity * t;

        if (absAngle <= maxTurn)
        {
            // we'll have reached the desired direction by this time
            outPoints[p].direction = desiredAimDir;
            outPoints[p].angularVelocity = 0.0f;
        }
        else
        {
            // still turning: rotate by maxTurn in the right direction
            outPoints[p].direction = RotateXZByAngle(currentAimDir, sign * maxTurn);
            outPoints[p].angularVelocity = sign * maxAngularVelocity;
        }
    }
}

//----------------------------------------------------------------------------------
// Motion Matching - Feature Extraction and Search
//----------------------------------------------------------------------------------
// Extract motion features from current character state
// This produces a feature vector in the same format as AnimDatabase features
// Used by motion matching to build query vector from runtime character state
// poseAlpha: how far into the next database frame we are (dt / dbFrameTime, clamped to [0,1]).
// 1.0 = full database frame step (use "next" values as-is), 0.5 = half step (lerp halfway).
// This matters for ToeVel, ToePosDiff, HeadToSlowestToe which use next-frame quantities.
static void ComputeMotionFeatures(
    const AnimDatabase* db,
    const ControlledCharacter* cc,
    std::vector<float>& outFeatures,
    bool updateFutureFeatures,
    float poseAlpha)
{
    if (db == nullptr || !db->valid) return;
    if (cc == nullptr) return;

    const TransformData* xform = &cc->xformBeforeIK;
    const PlayerControlInput* input = &cc->playerInput;

    const MotionMatchingFeaturesConfig& cfg = db->featuresConfig;

    // Resize output if needed
    if ((int)outFeatures.size() != db->featureDim)
    {
        outFeatures.resize(db->featureDim, 0.0f);
    }

    // Use the already-computed magic anchor from controlled character (no need to recompute from arms)
    const Vector3 magicPos = cc->worldPosition;
    const Quaternion magicWorldRot = cc->worldRotation;

    // Extract yaw from magic world rotation for transforming to magic-local frame
    const Quaternion invMagicWorldRot = QuaternionInvert(magicWorldRot);

    // Get toe positions from pose
    const int leftToeIdx = db->toeIndices[SIDE_LEFT];
    const int rightToeIdx = db->toeIndices[SIDE_RIGHT];

    Vector3 leftToePos = Vector3Zero();
    Vector3 rightToePos = Vector3Zero();

    if (leftToeIdx >= 0 && leftToeIdx < xform->jointCount)
    {
        leftToePos = xform->globalPositions[leftToeIdx];
    }
    if (rightToeIdx >= 0 && rightToeIdx < xform->jointCount)
    {
        rightToePos = xform->globalPositions[rightToeIdx];
    }

    // Compute toe positions in magic-local frame
    const Vector3 magicToLeft = Vector3Subtract(leftToePos, magicPos);
    const Vector3 localLeftPos = Vector3RotateByQuaternion(magicToLeft, invMagicWorldRot);

    const Vector3 magicToRight = Vector3Subtract(rightToePos, magicPos);
    const Vector3 localRightPos = Vector3RotateByQuaternion(magicToRight, invMagicWorldRot);

    // Compute toe velocities in magic-local frame, lerped between current and next
    // based on poseAlpha to handle runtime framerate != database framerate
    const Vector3 leftVelWorld = Vector3Lerp(
        cc->toeBlendedVelocityWorld[SIDE_LEFT],
        cc->nextToeBlendedVelocityWorld[SIDE_LEFT], poseAlpha);
    const Vector3 rightVelWorld = Vector3Lerp(
        cc->toeBlendedVelocityWorld[SIDE_RIGHT],
        cc->nextToeBlendedVelocityWorld[SIDE_RIGHT], poseAlpha);
    const Vector3 localLeftVel = Vector3RotateByQuaternion(leftVelWorld, invMagicWorldRot);
    const Vector3 localRightVel = Vector3RotateByQuaternion(rightVelWorld, invMagicWorldRot);

    // Fill the feature vector in the same order as database features
    int fi = 0;

    // ToePos: left(X,Z), right(X,Z)
    if (cfg.IsFeatureEnabled(FeatureType::ToePos))
    {
        outFeatures[fi++] = localLeftPos.x;
        outFeatures[fi++] = localLeftPos.z;
        outFeatures[fi++] = localRightPos.x;
        outFeatures[fi++] = localRightPos.z;
    }

    // ToeVel: left(X,Z), right(X,Z)
    if (cfg.IsFeatureEnabled(FeatureType::ToeVel))
    {
        outFeatures[fi++] = localLeftVel.x;
        outFeatures[fi++] = localLeftVel.z;
        outFeatures[fi++] = localRightVel.x;
        outFeatures[fi++] = localRightVel.z;
    }

    // ToePosDiff: (left - right) toe pos diff, lerped between current and next frame
    if (cfg.IsFeatureEnabled(FeatureType::ToePosDiff))
    {
        const Vector3 posDiff = Vector3Lerp(
            cc->toeBlendedPosDiffRootSpace,
            cc->nextToeBlendedPosDiffRootSpace, poseAlpha);
        outFeatures[fi++] = posDiff.x;
        outFeatures[fi++] = posDiff.z;
    }


    // Compute future velocities for velocity-based features (FutureVel, FutureSpeed, FutureVelClamped, FutureAccelClamped)
    // Only computed when updateFutureFeatures is true; otherwise old values in outFeatures persist
    Vector3 currentVelWorld = Vector3Zero();
    Vector3 desiredVelWorld = Vector3Zero();
    const float maxAcceleration = 4.0f;
    std::vector<Vector3> futureVelocities;

    if (updateFutureFeatures)
    {
        currentVelWorld = cc->virtualControlSmoothedVelocity;
        desiredVelWorld = input->desiredVelocity;
        ComputeFutureVelocities(
            currentVelWorld, desiredVelWorld, maxAcceleration,
            cfg.futureTrajPointTimes, futureVelocities);
    }

    // FutureVel: predicted velocity at future sample times (XZ components in magic space)
    if (cfg.IsFeatureEnabled(FeatureType::FutureVel))
    {
        const int count = 2 * (int)cfg.futureTrajPointTimes.size();
        if (updateFutureFeatures)
        {
            for (int p = 0; p < (int)futureVelocities.size(); ++p)
            {
                // Transform to magic space for feature output
                const Vector3 futureVelMagicSpace =
                    Vector3RotateByQuaternion(futureVelocities[p], invMagicWorldRot);
                outFeatures[fi++] = futureVelMagicSpace.x;
                outFeatures[fi++] = futureVelMagicSpace.z;
            }
        }
        else
        {
            fi += count;
        }
    }

    // FutureVelClamped: predicted velocity clamped to max magnitude (XZ in magic space)
    if (cfg.IsFeatureEnabled(FeatureType::FutureVelClamped))
    {
        const int count = 2 * (int)cfg.futureTrajPointTimes.size();
        if (updateFutureFeatures)
        {
            constexpr float MaxFutureVelClampedMag = 1.0f;

            for (int p = 0; p < (int)futureVelocities.size(); ++p)
            {
                const Vector3 futureVelMagicSpace =
                    Vector3RotateByQuaternion(futureVelocities[p], invMagicWorldRot);

                // Clamp to max magnitude
                Vector3 clampedVel = futureVelMagicSpace;
                const float mag = Vector3Length(futureVelMagicSpace);
                if (mag > MaxFutureVelClampedMag)
                {
                    clampedVel = Vector3Scale(
                        futureVelMagicSpace, MaxFutureVelClampedMag / mag);
                }

                outFeatures[fi++] = clampedVel.x;
                outFeatures[fi++] = clampedVel.z;
            }
        }
        else
        {
            fi += count;
        }
    }

    // FutureSpeed: predicted scalar speed at future sample times
    if (cfg.IsFeatureEnabled(FeatureType::FutureSpeed))
    {
        const int count = 1 * (int)cfg.futureTrajPointTimes.size();
        if (updateFutureFeatures)
        {
            for (int p = 0; p < (int)futureVelocities.size(); ++p)
            {
                outFeatures[fi++] = Vector3Length(futureVelocities[p]);
            }
        }
        else
        {
            fi += count;
        }
    }


    // PastPosition: past magic position in current magic horizontal frame (XZ)
    if (cfg.IsFeatureEnabled(FeatureType::PastPosition))
    {
        Vector3 pastPosLocal = Vector3Zero();

        if (!cc->positionHistory.empty())
        {
            const float currentTime = (float)GetTime();
            const float targetPastTime = currentTime - cfg.pastTimeOffset;

            // Find the history point closest to targetPastTime
            // History is ordered from oldest (front) to newest (back)
            int bestIdx = -1;
            float bestTimeDiff = FLT_MAX;

            for (int i = 0; i < (int)cc->positionHistory.size(); ++i)
            {
                const float timeDiff = (float)abs(cc->positionHistory[i].timestamp - (double)targetPastTime);
                if (timeDiff < bestTimeDiff)
                {
                    bestTimeDiff = timeDiff;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0)
            {
                // Past position is stored as world magic anchor position
                const Vector3 pastMagicPos = cc->positionHistory[bestIdx].position;

                // Current world magic position
                const Vector3 currentWorldMagicPos = magicPos;

                // Compute vector from current magic to past magic
                const Vector3 magicToPastMagic = Vector3Subtract(pastMagicPos, currentWorldMagicPos);

                // Transform to current magic horizontal frame
                pastPosLocal = Vector3RotateByQuaternion(magicToPastMagic, invMagicWorldRot);
            }
            // else: no history available yet, leave as zero
        }

        // Store only XZ components (horizontal position)
        outFeatures[fi++] = pastPosLocal.x;
        outFeatures[fi++] = pastPosLocal.z;
    }

    // FutureAimDirection + FutureAimVelocity: planned aim trajectory at future times
    // the starting point is the actual current aim direction read from the live bones,
    // not a smoothed state. unlike velocity (where we want the acceleration boost right away),
    // for rotation we want to start from where the character actually is.
    {
        const bool needDir = cfg.IsFeatureEnabled(FeatureType::FutureAimDirection);
        const bool needVel = cfg.IsFeatureEnabled(FeatureType::FutureAimVelocity);
        const int numPoints = (int)cfg.futureTrajPointTimes.size();

        if ((needDir || needVel) && updateFutureFeatures)
        {
            // compute current aim direction from the live bones
            Vector3 currentAimWorld = { 0.0f, 0.0f, 1.0f };
            switch (cfg.aimDirectionMode)
            {
            case AimDirectionMode::HeadToRightHand:
            {
                if (db->headIndex >= 0 && db->handIndices[SIDE_RIGHT] >= 0)
                {
                    currentAimWorld = Vector3Subtract(
                        xform->globalPositions[db->handIndices[SIDE_RIGHT]],
                        xform->globalPositions[db->headIndex]);
                }
                break;
            }
            case AimDirectionMode::HeadDirection:
            {
                if (db->headIndex >= 0)
                {
                    currentAimWorld = Vector3RotateByQuaternion(
                        Vector3{ 0.0f, 0.0f, 1.0f },
                        xform->globalRotations[db->headIndex]);
                }
                break;
            }
            case AimDirectionMode::HipsDirection:
            {
                if (db->hipJointIndex >= 0)
                {
                    currentAimWorld = Vector3RotateByQuaternion(
                        Vector3{ 0.0f, 0.0f, 1.0f },
                        xform->globalRotations[db->hipJointIndex]);
                }
                break;
            }
            default: break;
            }
            currentAimWorld.y = 0.0f;
            const float aimLen = Vector3Length(currentAimWorld);
            if (aimLen > 1e-6f)
            {
                currentAimWorld = Vector3Scale(currentAimWorld, 1.0f / aimLen);
            }
            else
            {
                currentAimWorld = { 0.0f, 0.0f, 1.0f };
            }

            std::vector<FutureAimPoint> aimPoints;
            ComputeFutureAimDirections(
                currentAimWorld,
                input->desiredAimDirection,
                MAX_AIM_ANGULAR_VELOCITY,
                cfg.futureTrajPointTimes,
                aimPoints);

            if (needDir)
            {
                for (int p = 0; p < numPoints; ++p)
                {
                    const Vector3 aimLocal =
                        Vector3RotateByQuaternion(aimPoints[p].direction, invMagicWorldRot);
                    outFeatures[fi++] = aimLocal.x;
                    outFeatures[fi++] = aimLocal.z;
                }
            }
            if (needVel)
            {
                for (int p = 0; p < numPoints; ++p)
                {
                    outFeatures[fi++] = aimPoints[p].angularVelocity;
                }
            }
        }
        else
        {
            if (needDir) { fi += numPoints * 2; }
            if (needVel) { fi += numPoints; }
        }
    }

    // HeadToSlowestToe: vector from head to a speed-weighted average of toe positions (in magic space)
    if (cfg.IsFeatureEnabled(FeatureType::HeadToSlowestToe))
    {
        const int headIdx = db->headIndex;
        Vector3 headPos = Vector3Zero();
        if (headIdx >= 0 && headIdx < xform->jointCount)
        {
            headPos = xform->globalPositions[headIdx];
        }

        // Get positions in magic frame
        const Vector3 magicToHead = Vector3Subtract(headPos, magicPos);
        const Vector3 localHeadPos = Vector3RotateByQuaternion(magicToHead, invMagicWorldRot);

        // Get toe speeds from lerped velocity (same lerp as ToeVel feature)
        const float leftSpeed = Vector3Length(leftVelWorld);
        const float rightSpeed = Vector3Length(rightVelWorld);

        float wLeft, wRight;
        float totalSpeed = leftSpeed + rightSpeed;
        if (totalSpeed < 1e-6f)
        {
            wLeft = 0.5f;
            wRight = 0.5f;
        }
        else
        {
            wLeft = rightSpeed / totalSpeed;
            wRight = leftSpeed / totalSpeed;
        }

        // P_slowest = wLeft*P_left + wRight*P_right (already in magic space)
        const Vector3 localSlowestToePos = Vector3Add(
            Vector3Scale(localLeftPos, wLeft),
            Vector3Scale(localRightPos, wRight));

        const Vector3 headToSlowest = Vector3Subtract(localSlowestToePos, localHeadPos);

        outFeatures[fi++] = headToSlowest.x;
        outFeatures[fi++] = headToSlowest.z;
    }

    if (cfg.IsFeatureEnabled(FeatureType::HeadToToeAverage))
    {
        const int headIdx = db->headIndex;
        Vector3 headPos = Vector3Zero();
        if (headIdx >= 0 && headIdx < xform->jointCount)
        {
            headPos = xform->globalPositions[headIdx];
        }

       // Get positions in magic frame
        const Vector3 magicToHead = Vector3Subtract(headPos, magicPos);
        const Vector3 localHeadPos = Vector3RotateByQuaternion(magicToHead, invMagicWorldRot);

        const Vector3 avgToePos = Vector3Lerp(localLeftPos, localRightPos, 0.5f);

        const Vector3 headToAvg = Vector3Subtract(avgToePos, localHeadPos);

        outFeatures[fi++] = headToAvg.x;
        outFeatures[fi++] = headToAvg.z;
    }

    // FutureAccelClamped: predicted acceleration with dead zone and cap
    // Same remapping as database: mag < 1 → 0, [1,3] → [0,3], > 3 → 3
    if (cfg.IsFeatureEnabled(FeatureType::FutureAccelClamped))
    {
        const int count = 2 * (int)cfg.futureTrajPointTimes.size();
        if (updateFutureFeatures)
        {
            constexpr float accelDeadZone = 1.0f;
            constexpr float accelMaxMag = 3.0f;
            constexpr float accelRemapScale =
                accelMaxMag / (accelMaxMag - accelDeadZone);

            // Compute the constant control acceleration
            const Vector3 velDelta =
                Vector3Subtract(desiredVelWorld, currentVelWorld);
            const float velDeltaMag = Vector3Length(velDelta);

            Vector3 controlAccelWorld = Vector3Zero();
            if (velDeltaMag > 1e-6f)
            {
                const Vector3 velDeltaDir =
                    Vector3Scale(velDelta, 1.0f / velDeltaMag);
                controlAccelWorld =
                    Vector3Scale(velDeltaDir, maxAcceleration);
            }

            for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
            {
                const float futureTime = cfg.futureTrajPointTimes[p];
                Vector3 futureAccelWorld = controlAccelWorld;

                const float maxDeltaVelMag = maxAcceleration * futureTime;
                if (velDeltaMag <= maxDeltaVelMag)
                {
                    futureAccelWorld = Vector3Zero();
                }

                // Transform to magic space
                Vector3 accelMagic =
                    Vector3RotateByQuaternion(futureAccelWorld, invMagicWorldRot);

                // Apply dead zone + remap + clamp
                const float mag = Vector3Length(accelMagic);
                if (mag > accelDeadZone)
                {
                    float remappedMag =
                        (mag - accelDeadZone) * accelRemapScale;
                    if (remappedMag > accelMaxMag) remappedMag = accelMaxMag;
                    accelMagic = Vector3Scale(accelMagic, remappedMag / mag);
                }
                else
                {
                    accelMagic = Vector3Zero();
                }

                outFeatures[fi++] = accelMagic.x;
                outFeatures[fi++] = accelMagic.z;
            }
        }
        else
        {
            fi += count;
        }
    }

    assert(fi == db->featureDim);
}
// Compute squared L2 distance between two feature vectors
static inline float ComputeFeatureDistance(
    const std::vector<float>& query,
    std::span<const float> dbFeatures)
{
    assert(query.size() == dbFeatures.size());
    const int dim = (int)query.size();

    float cost = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        const float diff = query[i] - dbFeatures[i];
        cost += diff * diff;
    }
    return cost;
}

// Search the database for the best matching frame using brute-force L2 distance
// Returns the motion frame index with lowest cost, or -1 if no valid frame found
// Query is automatically normalized and weighted to match database's normalized features
static int MotionMatchingSearch(
    const AnimDatabase* db,
    const std::vector<float>& query,
    int skipBoundaryFrames,
    float* outCost,
    const AppConfig& config,
    NetworkState* networkState = nullptr)
{
    if (!db || !db->valid || query.empty()) return -1;
    if ((int)query.size() != db->featureDim) return -1;

    // Normalize and weight-scale the query to match the normalized features
    std::vector<float> normalizedQuery(db->featureDim);
    for (int d = 0; d < db->featureDim; ++d)
    {
        // Step 1: Normalize using per-dimension mean and per-type std
        const FeatureType featureType = db->featureTypes[d];
        const int typeIdx = static_cast<int>(featureType);
        const float normalized = (query[d] - db->featuresMean[d]) / db->featureTypesStd[typeIdx];

        // Step 2: Apply feature type weight (from config)
        const float weight = db->featuresConfig.featureTypeWeights[typeIdx];

        normalizedQuery[d] = normalized * weight;
    }

    // Pass the features through the autoencoder if enabled and available
    if (config.useMMFeatureDenoiser && networkState && networkState->featuresAutoEncoder)
    {
        torch::NoGradGuard no_grad;
        networkState->featuresAutoEncoder->eval();

        try {
            // Move query to model device
            torch::Tensor queryTensor = torch::from_blob(normalizedQuery.data(), { 1, db->featureDim }).clone().to(networkState->device);
            torch::Tensor denoisedTensor = networkState->featuresAutoEncoder->forward(queryTensor);
            
            // Move back to CPU before copying data back to vector
            torch::Tensor resultHost = denoisedTensor.to(torch::kCPU);
            std::copy(resultHost.data_ptr<float>(), resultHost.data_ptr<float>() + db->featureDim, normalizedQuery.begin());
        }
        catch (const std::exception& e) {
            TraceLog(LOG_ERROR, "AE Inference failed: %s", e.what());
        }
    }

    // Brute-force search through all frames (skip clip boundaries for stability)
    int bestFrame = -1;
    float bestCost = FLT_MAX;

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];

        const int searchStart = clipStart + skipBoundaryFrames;
        const int searchEnd = clipEnd - skipBoundaryFrames;

        for (int f = searchStart; f < searchEnd; ++f)
        {
            std::span<const float> dbFeatures = db->normalizedFeatures.row_view(f);
            const float cost = ComputeFeatureDistance(normalizedQuery, dbFeatures);

            if (cost < bestCost)
            {
                bestCost = cost;
                bestFrame = f;
            }
        }
    }

    if (outCost) *outCost = bestCost;
    return bestFrame;
}