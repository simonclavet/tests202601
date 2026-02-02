#pragma once

#include <vector>
#include <span>
#include <cassert>
#include "raylib.h"
#include "anim_database.h"
#include "transform_data.h"
#include "app_config.h"
#include "utils.h"


//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
// Player Control Input
//----------------------------------------------------------------------------------

struct PlayerControlInput {
    Vector3 desiredVelocity = Vector3Zero();  // Desired velocity in world space (XZ plane)
    float maxSpeed = 2.0f;                     // Maximum movement speed (m/s)
};

//----------------------------------------------------------------------------------
// Motion Matching - Feature Extraction and Search
//----------------------------------------------------------------------------------

// Extract motion features from current character state
// This produces a feature vector in the same format as AnimDatabase features
// Used by motion matching to build query vector from runtime character state
static void ComputeMotionFeatures(
    const AnimDatabase* db,
    const TransformData* xform,
    const Vector3* toeVelocity,        // [SIDES_COUNT] - post-IK toe velocities
    const PlayerControlInput* input,
    std::vector<float>& outFeatures)
{
    if (db == nullptr || !db->valid) return;
    if (xform == nullptr || toeVelocity == nullptr || input == nullptr) return;

    const MotionMatchingFeaturesConfig& cfg = db->featuresConfig;

    // Resize output if needed
    if ((int)outFeatures.size() != db->featureDim)
    {
        outFeatures.resize(db->featureDim, 0.0f);
    }

    // Get hip position and yaw from pose
    const int hipIdx = db->hipJointIndex;
    if (hipIdx < 0 || hipIdx >= xform->jointCount) return;

    const Vector3 hipPos = xform->globalPositions[hipIdx];
    const Quaternion hipRot = xform->globalRotations[hipIdx];

    // Extract hip yaw and compute inverse for transforming to hip-local frame
    const Quaternion hipYaw = QuaternionYComponent(hipRot);
    const Quaternion invHipYaw = QuaternionInvert(hipYaw);

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

    // Compute toe positions in hip-local frame
    const Vector3 hipToLeft = Vector3Subtract(leftToePos, hipPos);
    const Vector3 localLeftPos = Vector3RotateByQuaternion(hipToLeft, invHipYaw);

    const Vector3 hipToRight = Vector3Subtract(rightToePos, hipPos);
    const Vector3 localRightPos = Vector3RotateByQuaternion(hipToRight, invHipYaw);

    // Compute toe velocities in hip-local frame
    Vector3 localLeftVel = Vector3RotateByQuaternion(toeVelocity[SIDE_LEFT], invHipYaw);
    Vector3 localRightVel = Vector3RotateByQuaternion(toeVelocity[SIDE_RIGHT], invHipYaw);

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

    // ToeDiff: (left - right) in hip frame
    if (cfg.IsFeatureEnabled(FeatureType::ToeDiff))
    {
        outFeatures[fi++] = localLeftPos.x - localRightPos.x;
        outFeatures[fi++] = localLeftPos.z - localRightPos.z;
    }

    // FutureVel: desired velocity at future sample times
    // Key difference: we use player's desired velocity instead of animation's future trajectory
    if (cfg.IsFeatureEnabled(FeatureType::FutureVel))
    {
        // Transform desired velocity to character's heading frame
        const Vector3 desiredVel = input->desiredVelocity;
        const Vector3 localDesiredVel = Vector3RotateByQuaternion(desiredVel, invHipYaw);

        // Assume player holds direction constant at all future sample times
        for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
        {
            outFeatures[fi++] = localDesiredVel.x;
            outFeatures[fi++] = localDesiredVel.z;
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
    float* outCost)
{
    if (!db || !db->valid || query.empty()) return -1;
    if ((int)query.size() != db->featureDim) return -1;

    // Normalize and weight-scale the query to match the normalized features
    std::vector<float> normalizedQuery(db->featureDim);
    for (int d = 0; d < db->featureDim; ++d)
    {
        // Step 1: Normalize using mean and std (z-score normalization)
        const float normalized = (query[d] - db->featuresMean[d]) / db->featuresStd[d];

        // Step 2: Apply feature type weight (from config)
        const FeatureType featureType = db->featureTypes[d];
        const float weight = db->featuresConfig.GetFeatureWeight(featureType);

        normalizedQuery[d] = normalized * weight;
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