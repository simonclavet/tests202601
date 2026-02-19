#pragma once

#include <string>
#include <vector>
#include <span>
#include <cmath>
#include <unordered_map>

#include "raylib.h"
#include "raymath.h"
#include "math_utils.h"
#include "utils.h"
#include "bvh_parser.h"
#include "transform_data.h"
#include "character_data.h"
#include "app_config.h"
#include "kmeans_clustering.h"


// augmentation parameters sampled per training example.
// passed to feature computation and target construction so everything is consistent.
struct MotionTrainingAugmentations
{
    float timescale = 1.0f;     // playback speed multiplier (1.0 = original)
    float aimAngleOffset = 0.0f; // random aim direction rotation (radians)
};

static inline MotionTrainingAugmentations SampleTrainingAugmentations()
{
    MotionTrainingAugmentations aug;
    aug.timescale = 1.0f + RandomGaussian() * 0.1f;
    aug.aimAngleOffset = RandomGaussian() * (10.0f * DEG2RAD);
    return aug;
}

static inline int FindClipForMotionFrame(const AnimDatabase* db, int frame)
{
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        if (frame >= db->clipStartFrame[c] && frame < db->clipEndFrame[c]) return c;
    }
    return -1;
}

// normalize a raw MM feature query: clamp to data bounds, then (raw - mean) / typeStd * typeWeight
// writes featureDim floats into dest
static inline void NormalizeFeatureQuery(
    const AnimDatabase* db, const float* raw, float* dest)
{
    for (int d = 0; d < db->featureDim; ++d)
    {
        const float clamped = std::clamp(raw[d], db->featuresMin[d], db->featuresMax[d]);
        const int ti = static_cast<int>(db->featureTypes[d]);
        const float norm = (clamped - db->featuresMean[d]) / db->featureTypesStd[ti];
        dest[d] = norm * db->featuresConfig.featureTypeWeights[ti];
    }
}

// normalize raw features without clamping: (raw - mean) / typeStd * typeWeight
// suitable for augmented training features that may exceed data bounds
static inline void NormalizeFeatureRaw(
    const AnimDatabase* db, const float* raw, float* dest)
{
    for (int d = 0; d < db->featureDim; ++d)
    {
        const int ti = static_cast<int>(db->featureTypes[d]);
        const float norm = (raw[d] - db->featuresMean[d]) / db->featureTypesStd[ti];
        dest[d] = norm * db->featuresConfig.featureTypeWeights[ti];
    }
}

// project a flat normalized segment [flatDim] to PCA coefficients [K]
// coeffs[k] = dot(basis[k], segment - mean)
static inline void PcaProjectSegment(
    const AnimDatabase* db, const float* normalizedSegment, float* coeffs)
{
    const int K = db->pcaSegmentK;
    const int flatDim = db->poseGenSegmentFlatDim;
    const float* mean = db->pcaSegmentMean.data();
    const float* basis = db->pcaSegmentBasis.data();

    for (int k = 0; k < K; ++k)
    {
        float dot = 0.0f;
        const float* row = basis + k * flatDim;
        for (int d = 0; d < flatDim; ++d)
        {
            dot += row[d] * (normalizedSegment[d] - mean[d]);
        }
        coeffs[k] = dot;
    }
}

// reconstruct a flat normalized segment [flatDim] from PCA coefficients [K]
// segment[d] = mean[d] + sum_k(coeffs[k] * basis[k][d])
static inline void PcaReconstructSegment(
    const AnimDatabase* db, const float* coeffs, float* normalizedSegment)
{
    const int K = db->pcaSegmentK;
    const int flatDim = db->poseGenSegmentFlatDim;
    const float* mean = db->pcaSegmentMean.data();
    const float* basis = db->pcaSegmentBasis.data();

    for (int d = 0; d < flatDim; ++d)
    {
        normalizedSegment[d] = mean[d];
    }
    for (int k = 0; k < K; ++k)
    {
        const float c = coeffs[k];
        const float* row = basis + k * flatDim;
        for (int d = 0; d < flatDim; ++d)
        {
            normalizedSegment[d] += c * row[d];
        }
    }
}

// project concatenated glimpse poses [GLIMPSE_POSE_COUNT * pgDim] to PCA coefficients [K]
// project 16 raw glimpse toe dims to K PCA coefficients
static inline void PcaProjectGlimpseToe(
    const AnimDatabase* db, const float* rawToe, float* coeffs)
{
    const int K = db->pcaGlimpseToeK;
    const float* mean = db->pcaGlimpseToeMean.data();
    const float* basis = db->pcaGlimpseToeBasis.data();

    for (int k = 0; k < K; ++k)
    {
        float dot = 0.0f;
        const float* row = basis + k * GLIMPSE_TOE_RAW_DIM;
        for (int d = 0; d < GLIMPSE_TOE_RAW_DIM; ++d)
        {
            dot += row[d] * (rawToe[d] - mean[d]);
        }
        coeffs[k] = dot;
    }
}

// reconstruct 16 raw glimpse toe dims from K PCA coefficients
static inline void PcaReconstructGlimpseToe(
    const AnimDatabase* db, const float* coeffs, float* rawToe)
{
    const int K = db->pcaGlimpseToeK;
    const float* mean = db->pcaGlimpseToeMean.data();
    const float* basis = db->pcaGlimpseToeBasis.data();

    for (int d = 0; d < GLIMPSE_TOE_RAW_DIM; ++d)
    {
        rawToe[d] = mean[d];
    }
    for (int k = 0; k < K; ++k)
    {
        const float c = coeffs[k];
        const float* row = basis + k * GLIMPSE_TOE_RAW_DIM;
        for (int d = 0; d < GLIMPSE_TOE_RAW_DIM; ++d)
        {
            rawToe[d] += c * row[d];
        }
    }
}

// build 16 raw toe dims for a given frame: 2 future times x 2 toes x (pos_xz + vel_xz)
// positions are world toe positions transformed to current root space
// velocities are rotated from future root space to current root space
static inline void BuildRawGlimpseToe(
    const AnimDatabase* db, int frame, const int* futureFrames, float* raw16)
{
    const float curYaw = db->magicYaw[frame];
    const float cosC = cosf(-curYaw);
    const float sinC = sinf(-curYaw);
    int idx = 0;
    for (int t = 0; t < GLIMPSE_POSE_COUNT; ++t)
    {
        const int futureF = frame + futureFrames[t];
        const float deltaYaw = db->magicYaw[futureF] - curYaw;
        const float cosD = cosf(deltaYaw);
        const float sinD = sinf(deltaYaw);
        for (int side : sides)
        {
            // position: world toe -> current root space (XZ only)
            const Vector3 toeWorld =
                db->jointPositionsAnimSpace.row_view(futureF)[db->toeIndices[side]];
            const float dx = toeWorld.x - db->magicPosition[frame].x;
            const float dz = toeWorld.z - db->magicPosition[frame].z;
            raw16[idx++] = dx * cosC - dz * sinC;
            raw16[idx++] = dx * sinC + dz * cosC;
            // velocity: future root space -> current root space (XZ only)
            const Vector3 vel =
                db->jointVelocitiesRootSpace.row_view(futureF)[db->toeIndices[side]];
            raw16[idx++] = vel.x * cosD - vel.z * sinD;
            raw16[idx++] = vel.x * sinD + vel.z * cosD;
        }
    }
    assert(idx == GLIMPSE_TOE_RAW_DIM);
}

// build 16 raw toe dims with augmentation: future offsets become fractional,
// positions are interpolated, velocities are interpolated and scaled by timescale.
static inline void BuildRawGlimpseToeAugmented(
    const AnimDatabase* db,
    int frame,
    const MotionTrainingAugmentations& aug,
    float animDt,
    int clipStart,
    int clipEnd,
    float* raw16)
{
    const float curYaw = db->magicYaw[frame];
    const float cosC = cosf(-curYaw);
    const float sinC = sinf(-curYaw);
    int idx = 0;
    for (int t = 0; t < GLIMPSE_POSE_COUNT; ++t)
    {
        const float srcFrac = (float)frame + GLIMPSE_POSE_TIMES[t] * aug.timescale / animDt;
        const float srcClamped = std::clamp(srcFrac, (float)clipStart, (float)(clipEnd - 1) - 0.001f);
        const int f0 = (int)srcClamped;
        const int f1 = std::min(f0 + 1, clipEnd - 1);
        const float alpha = srcClamped - (float)f0;

        // interpolate magicYaw at the future time for velocity rotation
        const float futureYaw = db->magicYaw[f0] * (1.0f - alpha) + db->magicYaw[f1] * alpha;
        const float deltaYaw = futureYaw - curYaw;
        const float cosD = cosf(deltaYaw);
        const float sinD = sinf(deltaYaw);

        for (int side : sides)
        {
            const int toeIdx = db->toeIndices[side];

            // position: interpolate world toe positions, then transform to current root space
            const Vector3 toe0 = db->jointPositionsAnimSpace.row_view(f0)[toeIdx];
            const Vector3 toe1 = db->jointPositionsAnimSpace.row_view(f1)[toeIdx];
            const float px = toe0.x + alpha * (toe1.x - toe0.x) - db->magicPosition[frame].x;
            const float pz = toe0.z + alpha * (toe1.z - toe0.z) - db->magicPosition[frame].z;
            raw16[idx++] = px * cosC - pz * sinC;
            raw16[idx++] = px * sinC + pz * cosC;

            // velocity: interpolate, rotate to current root space, scale by timescale
            const Vector3 vel0 = db->jointVelocitiesRootSpace.row_view(f0)[toeIdx];
            const Vector3 vel1 = db->jointVelocitiesRootSpace.row_view(f1)[toeIdx];
            const float vx = vel0.x + alpha * (vel1.x - vel0.x);
            const float vz = vel0.z + alpha * (vel1.z - vel0.z);
            raw16[idx++] = (vx * cosD - vz * sinD) * aug.timescale;
            raw16[idx++] = (vx * sinD + vz * cosD) * aug.timescale;
        }
    }
    assert(idx == GLIMPSE_TOE_RAW_DIM);
}

// returns true if dimension d of poseGenFeatures is a velocity/rate
// that should be scaled by timescale during augmentation.
// layout from PoseFeatures::SerializeTo (definitions.h):
//   [0, jc*6):        rotations              — no
//   [jc*6, jc*6+3):   root position          — no
//   [jc*6+3, jc*6+5): root velocity XZ       — yes
//   [jc*6+5]:          root yaw rate          — yes
//   [jc*6+6, jc*6+12): toe positions (2x3)   — no
//   [jc*6+12, jc*6+18): toe velocities (2x3) — yes
//   [jc*6+18, jc*6+20): toe pos diff XZ      — no
//   [jc*6+20]:         toe speed diff         — yes
static inline bool IsPoseGenVelocityDim(int d, int jointCount)
{
    const int jc6 = jointCount * 6;
    if (d >= jc6 + 3 && d <= jc6 + 5) return true;   // root vel XZ + yaw rate
    if (d >= jc6 + 12 && d <= jc6 + 17) return true;  // toe velocities
    if (d == jc6 + 20) return true;                    // toe speed diff
    return false;
}

// build an augmented normalized segment for decompressor training.
// for each output frame i, interpolates raw poseGenFeatures at
// source frame (frame + i * aug.timescale), scales velocity dims, normalizes.
// outNormFlat must have room for segFrameCount * pgDim floats.
static inline void BuildAugmentedNormalizedSegment(
    const AnimDatabase* db,
    int frame,
    const MotionTrainingAugmentations& aug,
    int clipStart,
    int clipEnd,
    float* outNormFlat)
{
    const int segFrames = db->poseGenSegmentFrameCount;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int jc = db->jointCount;

    for (int i = 0; i < segFrames; ++i)
    {
        // playback frame i maps to i * timescale original frames ahead
        const float srcFrac = (float)frame + (float)i * aug.timescale;
        const float srcClamped = std::clamp(srcFrac, (float)clipStart, (float)(clipEnd - 1) - 0.001f);
        const int f0 = (int)srcClamped;
        const int f1 = std::min(f0 + 1, clipEnd - 1);
        const float alpha = srcClamped - (float)f0;

        std::span<const float> row0 = db->poseGenFeatures.row_view(f0);
        std::span<const float> row1 = db->poseGenFeatures.row_view(f1);

        for (int d = 0; d < pgDim; ++d)
        {
            float rawInterp = row0[d] + alpha * (row1[d] - row0[d]);
            if (IsPoseGenVelocityDim(d, jc))
            {
                rawInterp *= aug.timescale;
            }
            outNormFlat[i * pgDim + d] =
                (rawInterp - db->poseGenFeaturesMean[d])
                / db->poseGenFeaturesStd[d]
                * db->poseGenFeaturesWeight[d];
        }
    }
}

// project normalized MM feature vector [featureDim] to PCA coefficients [pcaFeatureK]
static inline void PcaProjectFeature(
    const AnimDatabase* db, const float* normalizedFeature, float* coeffs)
{
    const int K = db->pcaFeatureK;
    const int dim = db->featureDim;
    const float* mean = db->pcaFeatureMean.data();
    const float* basis = db->pcaFeatureBasis.data();

    for (int k = 0; k < K; ++k)
    {
        float dot = 0.0f;
        const float* row = basis + k * dim;
        for (int d = 0; d < dim; ++d)
        {
            dot += row[d] * (normalizedFeature[d] - mean[d]);
        }
        coeffs[k] = dot;
    }
}

constexpr int KMEANS_CLUSTER_COUNT = 500;
constexpr double KMEANS_TIME_BUDGET_SECONDS = 10.0;

// k-means clustering on normalizedFeatures for the legal start frames.
// after this, db->clusterFrames[c] has the global frame indices for cluster c.
// we use this for stratified sampling during training so rare motions
// get proportionally more representation in each batch.
static void AnimDatabaseClusterFeatures(AnimDatabase* db)
{
    const int numFrames = (int)db->legalStartFrames.size();
    const int dim = db->featureDim;

    if (numFrames == 0 || dim <= 0 || db->normalizedFeatures.empty())
    {
        db->clusterCount = 0;
        db->clusterFrames.clear();
        return;
    }

    const int k = (numFrames < KMEANS_CLUSTER_COUNT)
        ? numFrames : KMEANS_CLUSTER_COUNT;

    // centroids: k vectors of dim floats
    std::vector<float> centroids(k * dim);

    // seed centroids from k random legal frames (shuffle-pick to avoid dupes)
    {
        std::vector<int> indices(numFrames);
        for (int i = 0; i < numFrames; ++i) indices[i] = i;

        // fisher-yates for the first k elements
        for (int i = 0; i < k; ++i)
        {
            const int j = i + RandomInt(numFrames - i);
            std::swap(indices[i], indices[j]);
        }
        for (int c = 0; c < k; ++c)
        {
            const int frame = db->legalStartFrames[indices[c]];
            std::span<const float> row = db->normalizedFeatures.row_view(frame);
            memcpy(&centroids[c * dim], row.data(), dim * sizeof(float));
        }
    }

    // assignments: which cluster each legal frame belongs to
    std::vector<int> assignments(numFrames, 0);

    const Clock::time_point start = Clock::now();
    int iter = 0;
    while (ElapsedSeconds(start) < KMEANS_TIME_BUDGET_SECONDS)
    {
        iter++;
        // assign each frame to nearest centroid
        for (int i = 0; i < numFrames; ++i)
        {
            const int frame = db->legalStartFrames[i];
            std::span<const float> row = db->normalizedFeatures.row_view(frame);

            float bestDist = FLT_MAX;
            int bestCluster = 0;
            for (int c = 0; c < k; ++c)
            {
                float dist = 0.0f;
                const float* ctr = &centroids[c * dim];
                for (int d = 0; d < dim; ++d)
                {
                    const float diff = row[d] - ctr[d];
                    dist += diff * diff;
                }
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestCluster = c;
                }
            }
            assignments[i] = bestCluster;
        }

        // recompute centroids as mean of assigned frames
        std::vector<float> sums(k * dim, 0.0f);
        std::vector<int> counts(k, 0);

        for (int i = 0; i < numFrames; ++i)
        {
            const int c = assignments[i];
            const int frame = db->legalStartFrames[i];
            std::span<const float> row = db->normalizedFeatures.row_view(frame);
            float* dst = &sums[c * dim];
            for (int d = 0; d < dim; ++d)
                dst[d] += row[d];
            counts[c]++;
        }

        for (int c = 0; c < k; ++c)
        {
            if (counts[c] == 0)
            {
                // empty cluster: re-seed from a random legal frame
                const int ri = RandomInt(numFrames);
                const int frame = db->legalStartFrames[ri];
                std::span<const float> row = db->normalizedFeatures.row_view(frame);
                memcpy(&centroids[c * dim], row.data(), dim * sizeof(float));
            }
            else
            {
                const float inv = 1.0f / (float)counts[c];
                for (int d = 0; d < dim; ++d)
                    centroids[c * dim + d] = sums[c * dim + d] * inv;
            }
        }
    }

    // build the per-cluster frame lists
    db->clusterCount = k;
    db->clusterFrames.resize(k);
    for (int c = 0; c < k; ++c)
        db->clusterFrames[c].clear();

    for (int i = 0; i < numFrames; ++i)
    {
        const int c = assignments[i];
        const int frame = db->legalStartFrames[i];
        db->clusterFrames[c].push_back(frame);
    }

    // log some stats
    int minSize = numFrames;
    int maxSize = 0;
    int nonEmpty = 0;
    for (int c = 0; c < k; ++c)
    {
        const int sz = (int)db->clusterFrames[c].size();
        if (sz > 0)
        {
            nonEmpty++;
            if (sz < minSize) minSize = sz;
            if (sz > maxSize) maxSize = sz;
        }
    }
    TraceLog(LOG_INFO,
        "AnimDatabase: k-means %d clusters (%d non-empty) in %d iters (%.1fs), sizes min=%d max=%d avg=%d",
        k, nonEmpty, iter, ElapsedSeconds(start),
        minSize, maxSize, numFrames / (nonEmpty > 0 ? nonEmpty : 1));
}

// wrapper: gather the normalized features for legal start frames into a contiguous array,
// run bisection clustering, then map indices back to global frame indices.
static void AnimDatabaseClusterFeatures2(AnimDatabase* db)
{
    const int numFrames = (int)db->legalStartFrames.size();
    const int dim = db->featureDim;

    if (numFrames == 0 || dim <= 0 || db->normalizedFeatures.empty())
    {
        db->clusterCount = 0;
        db->clusterFrames.clear();
        return;
    }

    const int targetK = (numFrames < KMEANS_CLUSTER_COUNT)
        ? numFrames : KMEANS_CLUSTER_COUNT;

    const Clock::time_point start = Clock::now();

    // gather features into a contiguous array [numFrames x dim]
    std::vector<float> points(numFrames * dim);
    for (int i = 0; i < numFrames; ++i)
    {
        const int frame = db->legalStartFrames[i];
        std::span<const float> row = db->normalizedFeatures.row_view(frame);
        memcpy(&points[i * dim], row.data(), dim * sizeof(float));
    }

    // run generic bisection clustering
    std::vector<std::vector<int>> clusterIndices;
    BisectionCluster_Run(points.data(), numFrames, dim, targetK, clusterIndices);

    // map point indices back to global frame indices
    const int k = (int)clusterIndices.size();
    db->clusterCount = k;
    db->clusterFrames.resize(k);
    for (int c = 0; c < k; ++c)
    {
        const int sz = (int)clusterIndices[c].size();
        db->clusterFrames[c].resize(sz);
        for (int i = 0; i < sz; ++i)
        {
            db->clusterFrames[c][i] = db->legalStartFrames[clusterIndices[c][i]];
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: bisection clustering done (%.1fs)", ElapsedSeconds(start));
}

// pick a random legal frame using cluster-stratified sampling if available,
// otherwise fall back to plain uniform
static inline int SampleLegalSegmentStartFrame(const AnimDatabase* db)
{
    assert(db->clusterCount > 0);
    if (db->clusterCount <= 0)
        return db->legalStartFrames[RandomInt((int)db->legalStartFrames.size())];

    const int c = RandomInt(db->clusterCount);
    const std::vector<int>& frames = db->clusterFrames[c];
    return frames[RandomInt((int)frames.size())];
}

// PCA on flat normalized segments using cluster-weighted sampling.
// must be called after clustering (AnimDatabaseClusterFeatures).
// each cluster contributes equally, so rare motions are well represented in the basis.
static void AnimDatabaseComputeSegmentPCA(AnimDatabase* db)
{
    if (db->pcaSegmentK > 0) return;  // already computed
    if (db->poseGenSegmentFlatDim <= 0) return;

    const int segFrames = db->poseGenSegmentFrameCount;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int flatDim = db->poseGenSegmentFlatDim;
    const int K = PCA_SEGMENT_K;

    // count valid segment starts to size our sample
    int numValidStarts = 0;
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int room = db->clipEndFrame[c] - db->clipStartFrame[c] - segFrames;
        if (room >= 0) numValidStarts += room + 1;
    }

    // sample 2x that using cluster-weighted sampling so rare motions are well represented
    const int numSamples = numValidStarts * 2;
    TraceLog(LOG_INFO, "PCA: sampling %d segments cluster-weighted (flatDim=%d, K=%d)", numSamples, flatDim, K);

    torch::Tensor dataMat = torch::empty({numSamples, flatDim});
    float* dPtr = dataMat.data_ptr<float>();
    for (int i = 0; i < numSamples; ++i)
    {
        const int frame = SampleLegalSegmentStartFrame(db);

        // clamp to valid segment start (must fit segFrames consecutive frames in its clip)
        int segStart = frame;
        for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
        {
            if (frame >= db->clipStartFrame[c] && frame < db->clipEndFrame[c])
            {
                segStart = (frame < db->clipEndFrame[c] - segFrames) ? frame : db->clipEndFrame[c] - segFrames;
                break;
            }
        }

        const float* src = db->normalizedPoseGenFeatures.data() + segStart * pgDim;
        memcpy(dPtr + i * flatDim, src, flatDim * sizeof(float));
    }

    // compute and subtract mean
    torch::Tensor meanTensor = dataMat.mean(0);  // [flatDim]
    dataMat = dataMat - meanTensor.unsqueeze(0);

    // SVD: dataMat = U * S * Vt, we want top K rows of Vt as our basis
    auto [U, S, Vt] = torch::linalg::svd(dataMat, false, std::nullopt);

    // variance explained
    torch::Tensor totalVar = S.square().sum();
    torch::Tensor topKVar = S.slice(0, 0, K).square().sum();
    const float varianceExplained = topKVar.item<float>() / totalVar.item<float>() * 100.0f;
    TraceLog(LOG_INFO, "PCA: top %d components explain %.2f%% of variance", K, varianceExplained);

    torch::Tensor basisTensor = Vt.slice(0, 0, K).contiguous();  // [K x flatDim]

    db->pcaSegmentK = K;
    db->pcaSegmentMean.resize(flatDim);
    memcpy(db->pcaSegmentMean.data(), meanTensor.data_ptr<float>(), flatDim * sizeof(float));
    db->pcaSegmentBasis.resize(K * flatDim);
    memcpy(db->pcaSegmentBasis.data(), basisTensor.data_ptr<float>(), K * flatDim * sizeof(float));

    TraceLog(LOG_INFO, "PCA segment: stored basis [%d x %d]", K, flatDim);
}

// joint PCA on concatenated glimpse poses.
// for each sampled frame, concatenates the normalized poses at all future offsets
// into one [GLIMPSE_POSE_COUNT * pgDim] vector, then runs SVD on that.
static void AnimDatabaseComputeGlimpseToePCA(
    AnimDatabase* db, const int* futureFrames, int futureFrameCount)
{
    if (db->pcaGlimpseToeK > 0) return;  // already computed
    if (db->jointPositionsAnimSpace.empty()) return;
    if (db->jointVelocitiesRootSpace.empty()) return;
    if (db->toeIndices[0] < 0 || db->toeIndices[1] < 0) return;
    assert(futureFrameCount == GLIMPSE_POSE_COUNT);

    const int K = GLIMPSE_TOE_PCA_K;

    // find the largest future offset to determine valid starts
    int maxFutureFrame = 0;
    for (int i = 0; i < futureFrameCount; ++i)
    {
        if (futureFrames[i] > maxFutureFrame) maxFutureFrame = futureFrames[i];
    }

    // count valid starts (frames where frame + maxFutureFrame is in same clip)
    int numValidStarts = 0;
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int room = db->clipEndFrame[c] - db->clipStartFrame[c] - maxFutureFrame;
        if (room >= 0) numValidStarts += room + 1;
    }

    const int numSamples = numValidStarts * 2;
    TraceLog(LOG_INFO, "PCA glimpse toe: sampling %d frames (rawDim=%d, K=%d)",
        numSamples, GLIMPSE_TOE_RAW_DIM, K);

    torch::Tensor dataMat = torch::empty({numSamples, GLIMPSE_TOE_RAW_DIM});
    float* dPtr = dataMat.data_ptr<float>();
    for (int i = 0; i < numSamples; ++i)
    {
        const int frame = SampleLegalSegmentStartFrame(db);
        BuildRawGlimpseToe(db, frame, futureFrames, dPtr + i * GLIMPSE_TOE_RAW_DIM);
    }

    // compute and subtract mean
    torch::Tensor meanTensor = dataMat.mean(0);  // [GLIMPSE_TOE_RAW_DIM]
    dataMat = dataMat - meanTensor.unsqueeze(0);

    // SVD
    auto [U, S, Vt] = torch::linalg::svd(dataMat, false, std::nullopt);

    torch::Tensor totalVar = S.square().sum();
    torch::Tensor topKVar = S.slice(0, 0, K).square().sum();
    const float varianceExplained = topKVar.item<float>() / totalVar.item<float>() * 100.0f;
    TraceLog(LOG_INFO, "PCA glimpse toe: top %d of %d components explain %.2f%% of variance",
        K, GLIMPSE_TOE_RAW_DIM, varianceExplained);

    torch::Tensor basisTensor = Vt.slice(0, 0, K).contiguous();  // [K x GLIMPSE_TOE_RAW_DIM]

    db->pcaGlimpseToeK = K;
    db->pcaGlimpseToeMean.resize(GLIMPSE_TOE_RAW_DIM);
    memcpy(db->pcaGlimpseToeMean.data(), meanTensor.data_ptr<float>(),
        GLIMPSE_TOE_RAW_DIM * sizeof(float));
    db->pcaGlimpseToeBasis.resize(K * GLIMPSE_TOE_RAW_DIM);
    memcpy(db->pcaGlimpseToeBasis.data(), basisTensor.data_ptr<float>(),
        K * GLIMPSE_TOE_RAW_DIM * sizeof(float));

    TraceLog(LOG_INFO, "PCA glimpse toe: stored basis [%d x %d]", K, GLIMPSE_TOE_RAW_DIM);
}

// PCA on normalized MM features for compact glimpse conditioning
static void AnimDatabaseComputeFeaturePCA(AnimDatabase* db)
{
    if (db->pcaFeatureK > 0) return;  // already computed
    if (db->normalizedFeatures.empty()) return;
    if (db->featureDim <= 0) return;

    const int dim = db->featureDim;
    const int K = PCA_FEATURE_K;

    int numValidStarts = 0;
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        numValidStarts += db->clipEndFrame[c] - db->clipStartFrame[c];
    }

    const int numSamples = numValidStarts * 2;
    TraceLog(LOG_INFO, "PCA feature: sampling %d frames (featureDim=%d, K=%d)",
        numSamples, dim, K);

    torch::Tensor dataMat = torch::empty({numSamples, dim});
    float* dPtr = dataMat.data_ptr<float>();
    for (int i = 0; i < numSamples; ++i)
    {
        const int frame = SampleLegalSegmentStartFrame(db);
        std::span<const float> row = db->normalizedFeatures.row_view(frame);
        memcpy(dPtr + i * dim, row.data(), dim * sizeof(float));
    }

    torch::Tensor meanTensor = dataMat.mean(0);
    dataMat = dataMat - meanTensor.unsqueeze(0);

    auto [U, S, Vt] = torch::linalg::svd(dataMat, false, std::nullopt);

    torch::Tensor totalVar = S.square().sum();
    torch::Tensor topKVar = S.slice(0, 0, K).square().sum();
    const float varianceExplained = topKVar.item<float>() / totalVar.item<float>() * 100.0f;
    TraceLog(LOG_INFO, "PCA feature: top %d of %d components explain %.2f%% of variance",
        K, dim, varianceExplained);

    torch::Tensor basisTensor = Vt.slice(0, 0, K).contiguous();

    db->pcaFeatureK = K;
    db->pcaFeatureMean.resize(dim);
    memcpy(db->pcaFeatureMean.data(), meanTensor.data_ptr<float>(), dim * sizeof(float));
    db->pcaFeatureBasis.resize(K * dim);
    memcpy(db->pcaFeatureBasis.data(), basisTensor.data_ptr<float>(), K * dim * sizeof(float));

    TraceLog(LOG_INFO, "PCA feature: stored basis [%d x %d]", K, dim);
}

// save clusters + PCA to a binary file so they don't need to be recomputed
static void AnimDatabaseSaveDerived(const AnimDatabase* db, const std::string& folderPath)
{
    const std::string path = folderPath + "/database_derived.bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
    {
        TraceLog(LOG_ERROR, "Failed to save database derived data: %s", path.c_str());
        return;
    }

    // clusters
    fwrite(&db->clusterCount, sizeof(int), 1, f);
    for (int c = 0; c < db->clusterCount; ++c)
    {
        const int sz = (int)db->clusterFrames[c].size();
        fwrite(&sz, sizeof(int), 1, f);
        fwrite(db->clusterFrames[c].data(), sizeof(int), sz, f);
    }

    // segment PCA
    fwrite(&db->pcaSegmentK, sizeof(int), 1, f);
    const int flatDim = db->poseGenSegmentFlatDim;
    fwrite(&flatDim, sizeof(int), 1, f);
    if (db->pcaSegmentK > 0)
    {
        fwrite(db->pcaSegmentMean.data(), sizeof(float), flatDim, f);
        fwrite(db->pcaSegmentBasis.data(), sizeof(float), db->pcaSegmentK * flatDim, f);
    }

    // glimpse toe PCA
    fwrite(&db->pcaGlimpseToeK, sizeof(int), 1, f);
    const int glimpseToeDim = GLIMPSE_TOE_RAW_DIM;
    fwrite(&glimpseToeDim, sizeof(int), 1, f);
    if (db->pcaGlimpseToeK > 0)
    {
        fwrite(db->pcaGlimpseToeMean.data(), sizeof(float), glimpseToeDim, f);
        fwrite(db->pcaGlimpseToeBasis.data(), sizeof(float),
            db->pcaGlimpseToeK * glimpseToeDim, f);
    }

    // feature PCA (for compact glimpse conditioning)
    fwrite(&db->pcaFeatureK, sizeof(int), 1, f);
    fwrite(&db->featureDim, sizeof(int), 1, f);
    if (db->pcaFeatureK > 0)
    {
        fwrite(db->pcaFeatureMean.data(), sizeof(float), db->featureDim, f);
        fwrite(db->pcaFeatureBasis.data(), sizeof(float),
            db->pcaFeatureK * db->featureDim, f);
    }

    fclose(f);
    TraceLog(LOG_INFO,
        "Saved clusters(%d) segPCA(K=%d) toePCA(K=%d) featPCA(K=%d) to: %s",
        db->clusterCount, db->pcaSegmentK, db->pcaGlimpseToeK,
        db->pcaFeatureK, path.c_str());
}

// load clusters + PCA from file. returns true if successful.
static bool AnimDatabaseLoadDerived(AnimDatabase* db, const std::string& folderPath)
{
    const std::string path = folderPath + "/database_derived.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // clusters
    int clusterCount = 0;
    fread(&clusterCount, sizeof(int), 1, f);
    db->clusterCount = clusterCount;
    db->clusterFrames.resize(clusterCount);
    for (int c = 0; c < clusterCount; ++c)
    {
        int sz = 0;
        fread(&sz, sizeof(int), 1, f);
        db->clusterFrames[c].resize(sz);
        fread(db->clusterFrames[c].data(), sizeof(int), sz, f);
    }

    // segment PCA
    int K = 0;
    int flatDim = 0;
    fread(&K, sizeof(int), 1, f);
    fread(&flatDim, sizeof(int), 1, f);

    if (K > 0 && flatDim == db->poseGenSegmentFlatDim)
    {
        db->pcaSegmentK = K;
        db->pcaSegmentMean.resize(flatDim);
        fread(db->pcaSegmentMean.data(), sizeof(float), flatDim, f);
        db->pcaSegmentBasis.resize(K * flatDim);
        fread(db->pcaSegmentBasis.data(), sizeof(float), K * flatDim, f);
    }

    // glimpse toe PCA (may not exist in older files — check for EOF)
    int toeK = 0;
    int toeDim = 0;
    if (fread(&toeK, sizeof(int), 1, f) == 1 && fread(&toeDim, sizeof(int), 1, f) == 1)
    {
        if (toeK > 0 && toeDim == GLIMPSE_TOE_RAW_DIM)
        {
            db->pcaGlimpseToeK = toeK;
            db->pcaGlimpseToeMean.resize(toeDim);
            fread(db->pcaGlimpseToeMean.data(), sizeof(float), toeDim, f);
            db->pcaGlimpseToeBasis.resize(toeK * toeDim);
            fread(db->pcaGlimpseToeBasis.data(), sizeof(float), toeK * toeDim, f);
        }
    }

    // feature PCA (may not exist in older files)
    int featK = 0;
    int featDim = 0;
    if (fread(&featK, sizeof(int), 1, f) == 1 && fread(&featDim, sizeof(int), 1, f) == 1)
    {
        if (featK > 0 && featDim == db->featureDim)
        {
            db->pcaFeatureK = featK;
            db->pcaFeatureMean.resize(featDim);
            fread(db->pcaFeatureMean.data(), sizeof(float), featDim, f);
            db->pcaFeatureBasis.resize(featK * featDim);
            fread(db->pcaFeatureBasis.data(), sizeof(float), featK * featDim, f);
        }
    }

    fclose(f);
    TraceLog(LOG_INFO,
        "Loaded clusters(%d) segPCA(K=%d) toePCA(K=%d) featPCA(K=%d) from: %s",
        db->clusterCount, db->pcaSegmentK, db->pcaGlimpseToeK,
        db->pcaFeatureK, path.c_str());
    return true;
}

// Compute raw (un-normalized) MM features for a single frame.
// If outNames/outTypes are non-null, pushes feature names and types (call once on first frame).
// If aug is non-null, applies timescale and noise augmentation to features.
static inline void ComputeRawFeaturesForFrame(
    const AnimDatabase* db,
    const MotionMatchingFeaturesConfig& cfg,
    int frame,
    float* outFeatures,
    std::vector<std::string>* outNames,
    std::vector<FeatureType>* outTypes,
    const MotionTrainingAugmentations* aug)
{
    using std::string;
    using std::span;

    assert((outNames == nullptr) == (outTypes == nullptr));
    const bool populateNames = (outNames != nullptr);

    const int f = frame;
    const int clipIdx = FindClipForMotionFrame(db, f);
    assert(clipIdx != -1);

    const int clipStart = db->clipStartFrame[clipIdx];
    const int clipEnd = db->clipEndFrame[clipIdx];
    const float dt = db->animFrameTime[clipIdx];
    assert(dt > 1e-8f);

    span<const Vector3> posRow = db->jointPositionsAnimSpace.row_view(f);

    Vector3 leftPos = { 0.0f, 0.0f, 0.0f };
    Vector3 rightPos = { 0.0f, 0.0f, 0.0f };

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

    const float invMagicYaw = -db->magicYaw[f];
    const Rot6d invMagicYawRot = Rot6dFromYaw(invMagicYaw);
    const Vector3 magicPos = db->magicPosition[f];

    int currentFeature = 0;

    // precompute local toe positions (magic horizontal frame)
    const Vector3 magicToLeft = Vector3Subtract(leftPos, magicPos);
    const Vector3 localLeftPos = Vector3RotateByRot6d(magicToLeft, invMagicYawRot);

    const Vector3 magicToRight = Vector3Subtract(rightPos, magicPos);
    const Vector3 localRightPos = Vector3RotateByRot6d(magicToRight, invMagicYawRot);

    // augmentation parameters come from the caller (nullptr = no augmentation)
    const float timescale = (aug != nullptr) ? aug->timescale : 1.0f;
    const float aimAugAngle = (aug != nullptr) ? aug->aimAngleOffset : 0.0f;
    const Rot6d aimAugRot = Rot6dFromYaw(aimAugAngle);

    // compute a future/past frame offset, clamped to clip bounds
    const float frameTime = db->animFrameTime[clipIdx];
    auto futureFrameClamped = [&](float seconds) -> int
    {
        const int offset = (int)(seconds * timescale / frameTime + 0.5f);
        return std::clamp(f + offset, clipStart, clipEnd - 1);
    };
    auto pastFrameClamped = [&](float seconds) -> int
    {
        const int offset = (int)(seconds * timescale / frameTime + 0.5f);
        return std::clamp(f - offset, clipStart, clipEnd - 1);
    };

    // ToePos
    if (cfg.IsFeatureEnabled(FeatureType::ToePos))
    {
        outFeatures[currentFeature++] = localLeftPos.x;
        outFeatures[currentFeature++] = localLeftPos.z;
        outFeatures[currentFeature++] = localRightPos.x;
        outFeatures[currentFeature++] = localRightPos.z;

        if (populateNames)
        {
            outNames->push_back(string("LeftToePosX"));
            outNames->push_back(string("LeftToePosZ"));
            outNames->push_back(string("RightToePosX"));
            outNames->push_back(string("RightToePosZ"));
            outTypes->push_back(FeatureType::ToePos);
            outTypes->push_back(FeatureType::ToePos);
            outTypes->push_back(FeatureType::ToePos);
            outTypes->push_back(FeatureType::ToePos);
        }
    }

    // ToeVel
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

        outFeatures[currentFeature++] = localLeftVel.x * timescale;
        outFeatures[currentFeature++] = localLeftVel.z * timescale;
        outFeatures[currentFeature++] = localRightVel.x * timescale;
        outFeatures[currentFeature++] = localRightVel.z * timescale;

        if (populateNames)
        {
            outNames->push_back(string("LeftToeVelX"));
            outNames->push_back(string("LeftToeVelZ"));
            outNames->push_back(string("RightToeVelX"));
            outNames->push_back(string("RightToeVelZ"));
            outTypes->push_back(FeatureType::ToeVel);
            outTypes->push_back(FeatureType::ToeVel);
            outTypes->push_back(FeatureType::ToeVel);
            outTypes->push_back(FeatureType::ToeVel);
        }
    }

    // ToePosDiff
    if (cfg.IsFeatureEnabled(FeatureType::ToePosDiff))
    {
        outFeatures[currentFeature++] = localLeftPos.x - localRightPos.x;
        outFeatures[currentFeature++] = localLeftPos.z - localRightPos.z;

        if (populateNames)
        {
            outNames->push_back(string("ToePosDiffX"));
            outNames->push_back(string("ToePosDiffZ"));
            outTypes->push_back(FeatureType::ToePosDiff);
            outTypes->push_back(FeatureType::ToePosDiff);
        }
    }

    // FutureVel
    if (cfg.IsFeatureEnabled(FeatureType::FutureVel))
    {
        for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
        {
            const float futureTime = cfg.futureTrajPointTimes[p];
            const int ff = futureFrameClamped(futureTime);

            Vector3 futureVelAnimSpace = db->magicSmoothedVelocityAnimSpace[ff];
            futureVelAnimSpace.y = 0.0f;
            Vector3 futureVelMagicSpace = Vector3RotateByRot6d(futureVelAnimSpace, invMagicYawRot);

            outFeatures[currentFeature++] = futureVelMagicSpace.x * timescale;
            outFeatures[currentFeature++] = futureVelMagicSpace.z * timescale;

            if (populateNames)
            {
                char nameBufX[64];
                char nameBufZ[64];
                snprintf(nameBufX, sizeof(nameBufX), "FutureVelX_%.2fs", futureTime);
                snprintf(nameBufZ, sizeof(nameBufZ), "FutureVelZ_%.2fs", futureTime);
                outNames->push_back(string(nameBufX));
                outNames->push_back(string(nameBufZ));
                outTypes->push_back(FeatureType::FutureVel);
                outTypes->push_back(FeatureType::FutureVel);
            }
        }
    }

    // FutureVelClamped
    if (cfg.IsFeatureEnabled(FeatureType::FutureVelClamped))
    {
        constexpr float MaxFutureVelClampedMag = 1.0f;

        for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
        {
            const float futureTime = cfg.futureTrajPointTimes[p];
            const int ff = futureFrameClamped(futureTime);

            Vector3 futureVelAnimSpace = db->magicSmoothedVelocityAnimSpace[ff];
            futureVelAnimSpace.y = 0.0f;
            Vector3 futureVelMagicSpace = Vector3RotateByRot6d(futureVelAnimSpace, invMagicYawRot);
            futureVelMagicSpace = Vector3Scale(futureVelMagicSpace, timescale);

            const float mag = Vector3Length(futureVelMagicSpace);
            if (mag > MaxFutureVelClampedMag)
            {
                futureVelMagicSpace = Vector3Scale(futureVelMagicSpace, MaxFutureVelClampedMag / mag);
            }

            outFeatures[currentFeature++] = futureVelMagicSpace.x;
            outFeatures[currentFeature++] = futureVelMagicSpace.z;

            if (populateNames)
            {
                char nameBufX[64];
                char nameBufZ[64];
                snprintf(nameBufX, sizeof(nameBufX), "FutureVelClampedX_%.2fs", futureTime);
                snprintf(nameBufZ, sizeof(nameBufZ), "FutureVelClampedZ_%.2fs", futureTime);
                outNames->push_back(string(nameBufX));
                outNames->push_back(string(nameBufZ));
                outTypes->push_back(FeatureType::FutureVelClamped);
                outTypes->push_back(FeatureType::FutureVelClamped);
            }
        }
    }

    // FutureSpeed
    if (cfg.IsFeatureEnabled(FeatureType::FutureSpeed))
    {
        for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
        {
            const float futureTime = cfg.futureTrajPointTimes[p];
            const int ff = futureFrameClamped(futureTime);

            Vector3 futureVelAnimSpace = db->magicSmoothedVelocityAnimSpace[ff];
            futureVelAnimSpace.y = 0.0f;
            const float futureSpeed = Vector3Length(futureVelAnimSpace) * timescale;

            outFeatures[currentFeature++] = futureSpeed;

            if (populateNames)
            {
                char nameBuf[64];
                snprintf(nameBuf, sizeof(nameBuf), "FutureSpeed_%.2fs", futureTime);
                outNames->push_back(string(nameBuf));
                outTypes->push_back(FeatureType::FutureSpeed);
            }
        }
    }

    // PastPosition
    if (cfg.IsFeatureEnabled(FeatureType::PastPosition))
    {
        const int pf = pastFrameClamped(cfg.pastTimeOffset);
        const Vector3 pastMagicPos = db->magicPosition[pf];
        const Vector3 magicToPastMagic = Vector3Subtract(pastMagicPos, magicPos);
        const Vector3 pastPosLocal = Vector3RotateByRot6d(magicToPastMagic, invMagicYawRot);

        outFeatures[currentFeature++] = pastPosLocal.x;
        outFeatures[currentFeature++] = pastPosLocal.z;

        if (populateNames)
        {
            outNames->push_back(string("PastPosX"));
            outNames->push_back(string("PastPosZ"));
            outTypes->push_back(FeatureType::PastPosition);
            outTypes->push_back(FeatureType::PastPosition);
        }
    }

    // FutureAimDirection: read from precomputed aimDirectionAnimSpace
    if (cfg.IsFeatureEnabled(FeatureType::FutureAimDirection))
    {
        for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
        {
            const float futureTime = cfg.futureTrajPointTimes[p];
            const int ff = futureFrameClamped(futureTime);

            Vector3 aimDirWorld = db->aimDirectionAnimSpace[ff];
            Vector3 aimDirLocal = Vector3RotateByRot6d(aimDirWorld, invMagicYawRot);

            // augmentation: rotate aim direction by a random angle
            if (aug != nullptr)
            {
                aimDirLocal = Vector3RotateByRot6d(aimDirLocal, aimAugRot);
            }

            outFeatures[currentFeature++] = aimDirLocal.x;
            outFeatures[currentFeature++] = aimDirLocal.z;

            if (populateNames)
            {
                char nameBufX[64];
                char nameBufZ[64];
                snprintf(nameBufX, sizeof(nameBufX), "FutureAimDirX_%.2fs", futureTime);
                snprintf(nameBufZ, sizeof(nameBufZ), "FutureAimDirZ_%.2fs", futureTime);
                outNames->push_back(string(nameBufX));
                outNames->push_back(string(nameBufZ));
                outTypes->push_back(FeatureType::FutureAimDirection);
                outTypes->push_back(FeatureType::FutureAimDirection);
            }
        }
    }

    // FutureAimVelocity: angular velocity of aim direction around Y, rad/s
    // augmentation angle doesn't affect yaw rate (it's a constant offset)
    // timescale augmentation scales it proportionally
    if (cfg.IsFeatureEnabled(FeatureType::FutureAimVelocity))
    {
        for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
        {
            const float futureTime = cfg.futureTrajPointTimes[p];
            const int ff = futureFrameClamped(futureTime);

            outFeatures[currentFeature++] = db->aimYawRate[ff] * timescale;

            if (populateNames)
            {
                char nameBuf[64];
                snprintf(nameBuf, sizeof(nameBuf), "FutureAimVel_%.2fs", futureTime);
                outNames->push_back(string(nameBuf));
                outTypes->push_back(FeatureType::FutureAimVelocity);
            }
        }
    }

    // HeadToSlowestToe
    if (cfg.IsFeatureEnabled(FeatureType::HeadToSlowestToe))
    {
        const int headIdx = db->headIndex;
        const Vector3 magicToHead = Vector3Subtract(posRow[headIdx], magicPos);
        const Vector3 localHeadPos = Vector3RotateByRot6d(magicToHead, invMagicYawRot);

        float leftSpeed = 0.0f;
        float rightSpeed = 0.0f;

        if (f > clipStart && dt > 0.0f)
        {
            span<const Vector3> posPrevRow = db->jointPositionsAnimSpace.row_view(f - 1);
            const Vector3 velLeftWorld = Vector3Scale(
                Vector3Subtract(posRow[leftIdx], posPrevRow[leftIdx]), 1.0f / dt);
            const Vector3 velRightWorld = Vector3Scale(
                Vector3Subtract(posRow[rightIdx], posPrevRow[rightIdx]), 1.0f / dt);
            leftSpeed = Vector3Length(velLeftWorld);
            rightSpeed = Vector3Length(velRightWorld);
        }

        float wLeft, wRight;
        const float totalSpeed = leftSpeed + rightSpeed;
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

        const Vector3 localSlowestToePos = Vector3Add(
            Vector3Scale(localLeftPos, wLeft),
            Vector3Scale(localRightPos, wRight));
        const Vector3 headToSlowest = Vector3Subtract(localSlowestToePos, localHeadPos);

        outFeatures[currentFeature++] = headToSlowest.x;
        outFeatures[currentFeature++] = headToSlowest.z;

        if (populateNames)
        {
            outNames->push_back(string("HeadToSlowestToeX"));
            outNames->push_back(string("HeadToSlowestToeZ"));
            outTypes->push_back(FeatureType::HeadToSlowestToe);
            outTypes->push_back(FeatureType::HeadToSlowestToe);
        }
    }

    // HeadToToeAverage
    if (cfg.IsFeatureEnabled(FeatureType::HeadToToeAverage))
    {
        const int headIdx = db->headIndex;
        const Vector3 magicToHead = Vector3Subtract(posRow[headIdx], magicPos);
        const Vector3 localHeadPos = Vector3RotateByRot6d(magicToHead, invMagicYawRot);

        const Vector3 avgToePos = {
            (localLeftPos.x + localRightPos.x) * 0.5f,
            (localLeftPos.y + localRightPos.y) * 0.5f,
            (localLeftPos.z + localRightPos.z) * 0.5f,
        };
        const Vector3 headToAvg = Vector3Subtract(avgToePos, localHeadPos);

        outFeatures[currentFeature++] = headToAvg.x;
        outFeatures[currentFeature++] = headToAvg.z;

        if (populateNames)
        {
            outNames->push_back(string("HeadToToeAvgX"));
            outNames->push_back(string("HeadToToeAvgZ"));
            outTypes->push_back(FeatureType::HeadToToeAverage);
            outTypes->push_back(FeatureType::HeadToToeAverage);
        }
    }

    // FutureAccelClamped (accel scales by timescale^2)
    if (cfg.IsFeatureEnabled(FeatureType::FutureAccelClamped))
    {
        constexpr float accelDeadZone = 1.0f;
        constexpr float accelMaxMag = 3.0f;
        constexpr float accelRemapScale = accelMaxMag / (accelMaxMag - accelDeadZone);
        const float timescaleSq = timescale * timescale;

        for (int p = 0; p < (int)cfg.futureTrajPointTimes.size(); ++p)
        {
            const float futureTime = cfg.futureTrajPointTimes[p];
            const int ff = futureFrameClamped(futureTime);

            Vector3 accelAnimSpace = db->magicSmoothedAccelerationAnimSpace[ff];
            accelAnimSpace.y = 0.0f;
            Vector3 accelMagic = Vector3RotateByRot6d(accelAnimSpace, invMagicYawRot);
            accelMagic = Vector3Scale(accelMagic, timescaleSq);

            Vector3 futureAccelMagicSpace = Vector3Zero();
            const float mag = Vector3Length(accelMagic);
            if (mag > accelDeadZone)
            {
                float remappedMag = (mag - accelDeadZone) * accelRemapScale;
                if (remappedMag > accelMaxMag) remappedMag = accelMaxMag;
                futureAccelMagicSpace = Vector3Scale(accelMagic, remappedMag / mag);
            }

            outFeatures[currentFeature++] = futureAccelMagicSpace.x;
            outFeatures[currentFeature++] = futureAccelMagicSpace.z;

            if (populateNames)
            {
                char nameBufX[64];
                char nameBufZ[64];
                snprintf(nameBufX, sizeof(nameBufX), "FutureAccelClampedX_%.2fs", futureTime);
                snprintf(nameBufZ, sizeof(nameBufZ), "FutureAccelClampedZ_%.2fs", futureTime);
                outNames->push_back(string(nameBufX));
                outNames->push_back(string(nameBufZ));
                outTypes->push_back(FeatureType::FutureAccelClamped);
                outTypes->push_back(FeatureType::FutureAccelClamped);
            }
        }
    }

    assert(currentFeature == db->featureDim);
}

// Updated AnimDatabaseRebuild: require all animations to match canonical skeleton.
// Populate localJointPositions/localJointRotations as well as global arrays.
// If any clip mismatches jointCount we invalidate the DB (db->valid = false).
static void AnimDatabaseRebuild(AnimDatabase* db, const CharacterData* characterData)
{
    using std::vector;
    using std::string;
    using std::span;

    LOG_PROFILE_SCOPE(AnimDatabaseRebuild);

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
    db->motionFrameCount = globalFrame;

    // Use scale from first animation
    if (db->animCount > 0)
    {
        db->scale = characterData->scales[0];
    }

    printf("AnimDatabase: Rebuilt with %d anims, %d total frames\n", db->animCount, db->motionFrameCount);

    // -------------------------
    // Build compact Motion Database
    // -------------------------
    // pick canonical skeleton (first animation) if available

    if (db->animCount == 0 || db->motionFrameCount == 0)
    {
        TraceLog(LOG_INFO, "AnimDatabase: no animations available for motion DB");
        return;
    }

    const BVHData* canonBvh = &characterData->bvhData[0];
    const float animDt = db->animFrameTime[0];
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

    const vector<string> spine3Candidates = { "spine3", "spine2", "chest", "upperchest", "upper_chest" };
    const vector<string> spine1Candidates = { "spine1" };
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
    db->spine1Index = FindJointIndexByNames(canonBvh, spine1Candidates);
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
                globalRot[j] = QuaternionToRot6d(tmpXform.globalRotations[j]);
                localPos[j] = tmpXform.localPositions[j];
                localRot[j] = QuaternionToRot6d(tmpXform.localRotations[j]);
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
            span<const Rot6d> rotRow = db->jointRotationsAnimSpace.row_view(f);

            // Magic position = spine projected to ground
            const Vector3 spinePos = posRow[db->spine1Index];
            db->magicPosition[f] = Vector3{ spinePos.x, 0.0f, spinePos.z };

            float hipYaw = 0.0f;

            // Magic yaw = hip forward direction projected onto XZ plane
            const Rot6d hipRot = rotRow[db->hipJointIndex];
            // Rotate Z-axis (0,0,1) by hip rotation to get forward direction
            const Vector3 hipForward = Vector3RotateByRot6d(Vector3{ 0.0f, 0.0f, 1.0f }, hipRot);
            // Project to XZ plane and normalize
            Vector3 hipForwardHorizontal = Vector3{ hipForward.x, 0.0f, hipForward.z };
            const float len = Vector3Length(hipForwardHorizontal);
            if (len > 1e-6f)
            {
                hipForwardHorizontal = Vector3Scale(hipForwardHorizontal, 1.0f / len);
                hipYaw = atan2f(hipForwardHorizontal.x, hipForwardHorizontal.z);
            }
            


            // Magic yaw depends on blend root mode rotation setting
            switch (db->featuresConfig.blendRootModeRotation)
            {
            case BlendRootModeRotation::HeadToRightHand:
            {
                const float howMuchMagicFollowsArm = 0.5f;

                // Magic yaw = direction from head to right hand projected onto XZ plane
                const Vector3 headPos = posRow[db->headIndex];
                const Vector3 handPos = posRow[db->handIndices[SIDE_RIGHT]];
                const Vector3 headToHand = Vector3Subtract(handPos, headPos);

                const float armYaw = atan2f(headToHand.x, headToHand.z);

                db->magicYaw[f] = LerpAngle(hipYaw, armYaw, howMuchMagicFollowsArm);
                break;
            }
            case BlendRootModeRotation::Hips:
            {
                db->magicYaw[f] = hipYaw;
                break;
            }
            default:
                db->magicYaw[f] = 0.0f;
                break;
            }
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



    // Compute lookahead poses: pose[f] + n * (pose[f+1] - pose[f]) = n*pose[f+1] - (n-1)*pose[f]
    // where n = lookaheadTime / frameTime (extrapolates lookaheadTime seconds ahead)
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float n = db->featuresConfig.poseDragLookaheadTime / frameTime;
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
        const float n = db->featuresConfig.poseDragLookaheadTime / frameTime;
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
    // Raw velocities go into a temp array, then we gaussian-smooth into magicSmoothedVelocityAnimSpace
    std::vector<Vector3> rawMagicVelocityAnimSpace(db->motionFrameCount);
    db->magicSmoothedVelocityAnimSpace.resize(db->motionFrameCount);
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

            if (isLastFrame || clipEnd - clipStart <= 1)
            {
                if (f > clipStart)
                {
                    rawMagicVelocityAnimSpace[f] = rawMagicVelocityAnimSpace[f - 1];
                    db->magicYawRate[f] = db->magicYawRate[f - 1];
                }
                else
                {
                    rawMagicVelocityAnimSpace[f] = Vector3Zero();
                    db->magicYawRate[f] = 0.0f;
                }
            }
            else
            {
                const Vector3 nextMagicPos = db->magicPosition[f + 1];
                const Vector3 currMagicPos = db->magicPosition[f];
                const Vector3 velAnim = Vector3Scale(Vector3Subtract(nextMagicPos, currMagicPos), invFrameTime);
                rawMagicVelocityAnimSpace[f] = velAnim;

                const float magicYaw = db->magicYaw[f];
                const float nextMagicYaw = db->magicYaw[f + 1];
                db->magicYawRate[f] = WrapAngleToPi(nextMagicYaw - magicYaw) * invFrameTime;
            }
        }
    }

    // Gaussian-smooth raw velocities per clip (0.2s window)
    constexpr float smoothVelocityWindowSeconds = 0.2f;
    constexpr float smoothAccelerationWindowSeconds = 0.4f;

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];

        // gaussian sigma: window covers ~2*sigma, so sigma = windowSeconds / 4
        // kernel radius in frames
        const float sigma = smoothVelocityWindowSeconds / 4.0f;
        const int radiusFrames = (int)ceilf(smoothVelocityWindowSeconds / (2.0f * frameTime));

        for (int f = clipStart; f < clipEnd; ++f)
        {
            Vector3 sum = Vector3Zero();
            float weightSum = 0.0f;

            for (int k = -radiusFrames; k <= radiusFrames; ++k)
            {
                const int sf = f + k;
                if (sf < clipStart || sf >= clipEnd) continue;

                const float t = (float)k * frameTime;
                const float w = expf(-(t * t) / (2.0f * sigma * sigma));
                sum = Vector3Add(sum, Vector3Scale(rawMagicVelocityAnimSpace[sf], w));
                weightSum += w;
            }

            if (weightSum > 1e-8f)
                db->magicSmoothedVelocityAnimSpace[f] = Vector3Scale(sum, 1.0f / weightSum);
            else
                db->magicSmoothedVelocityAnimSpace[f] = rawMagicVelocityAnimSpace[f];
        }
    }

    // Compute root-space velocities from smoothed anim-space velocities
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const Vector3 velAnim = db->magicSmoothedVelocityAnimSpace[f];
            const float magicYaw = db->magicYaw[f];
            const float cosY = cosf(magicYaw);
            const float sinY = sinf(magicYaw);
            db->magicVelocityRootSpace[f] = Vector3{
                velAnim.x * cosY - velAnim.z * sinY,
                0.0f,
                velAnim.x * sinY + velAnim.z * cosY
            };
        }
    }

    // Compute Magic anchor accelerations: differentiate smoothed velocity, then gaussian-smooth again
    std::vector<Vector3> rawMagicAccelerationAnimSpace(db->motionFrameCount);
    db->magicSmoothedAccelerationAnimSpace.resize(db->motionFrameCount);

    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];
        const float invFrameTime = 1.0f / frameTime;

        for (int f = clipStart; f < clipEnd; ++f)
        {
            if (f > clipStart && f < clipEnd - 1)
            {
                rawMagicAccelerationAnimSpace[f] = Vector3Scale(
                    Vector3Subtract(db->magicSmoothedVelocityAnimSpace[f], db->magicSmoothedVelocityAnimSpace[f - 1]),
                    invFrameTime);
            }
            else if (f == clipStart)
            {
                if (clipEnd - clipStart > 1)
                    rawMagicAccelerationAnimSpace[f] = rawMagicAccelerationAnimSpace[f + 1];
                else
                    rawMagicAccelerationAnimSpace[f] = Vector3Zero();
            }
            else // f == clipEnd - 1
            {
                if (clipEnd - clipStart > 1)
                    rawMagicAccelerationAnimSpace[f] = rawMagicAccelerationAnimSpace[f - 1];
                else
                    rawMagicAccelerationAnimSpace[f] = Vector3Zero();
            }
        }
    }

    // Gaussian-smooth raw accelerations per clip (0.2s window, same as velocity)
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];

        const float accelSigma = smoothAccelerationWindowSeconds / 4.0f;
        const int accelRadiusFrames = (int)ceilf(smoothAccelerationWindowSeconds / (2.0f * frameTime));

        for (int f = clipStart; f < clipEnd; ++f)
        {
            Vector3 sum = Vector3Zero();
            float weightSum = 0.0f;

            for (int k = -accelRadiusFrames; k <= accelRadiusFrames; ++k)
            {
                const int sf = f + k;
                if (sf < clipStart || sf >= clipEnd) continue;

                const float t = (float)k * frameTime;
                const float w = expf(-(t * t) / (2.0f * accelSigma * accelSigma));
                sum = Vector3Add(sum, Vector3Scale(rawMagicAccelerationAnimSpace[sf], w));
                weightSum += w;
            }

            if (weightSum > 1e-8f)
                db->magicSmoothedAccelerationAnimSpace[f] = Vector3Scale(sum, 1.0f / weightSum);
            else
                db->magicSmoothedAccelerationAnimSpace[f] = rawMagicAccelerationAnimSpace[f];
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
        const float n = db->featuresConfig.poseDragLookaheadTime / frameTime;
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

    // precompute aim direction per frame (unit vector in anim-world XZ plane)
    // and aim yaw rate (angular velocity around Y, rad/s) via finite difference
    db->aimDirectionAnimSpace.resize(db->motionFrameCount);
    db->aimYawRate.resize(db->motionFrameCount);

    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        Vector3 aimDir = { 0.0f, 0.0f, 1.0f };

        switch (db->featuresConfig.aimDirectionMode)
        {
        case AimDirectionMode::HeadToRightHand:
        {
            span<const Vector3> posRow = db->jointPositionsAnimSpace.row_view(f);
            aimDir = Vector3Subtract(
                posRow[db->handIndices[SIDE_RIGHT]], posRow[db->headIndex]);
            break;
        }
        case AimDirectionMode::HeadDirection:
        {
            span<const Rot6d> rotRow = db->jointRotationsAnimSpace.row_view(f);
            aimDir = Vector3RotateByRot6d(
                Vector3{ 0.0f, 0.0f, 1.0f }, rotRow[db->headIndex]);
            break;
        }
        case AimDirectionMode::HipsDirection:
        {
            span<const Rot6d> rotRow = db->jointRotationsAnimSpace.row_view(f);
            aimDir = Vector3RotateByRot6d(
                Vector3{ 0.0f, 0.0f, 1.0f }, rotRow[db->hipJointIndex]);
            break;
        }
        default: break;
        }

        aimDir.y = 0.0f;
        const float aimLen = Vector3Length(aimDir);
        if (aimLen > 1e-6f)
        {
            aimDir = Vector3Scale(aimDir, 1.0f / aimLen);
        }
        else
        {
            aimDir = { 0.0f, 0.0f, 1.0f };
        }
        db->aimDirectionAnimSpace[f] = aimDir;
    }

    // compute yaw rate via forward difference, clamped at clip boundaries
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        for (int f = clipStart; f < clipEnd; ++f)
        {
            if (f + 1 < clipEnd)
            {
                const float angle = SignedAngleY(
                    db->aimDirectionAnimSpace[f],
                    db->aimDirectionAnimSpace[f + 1]);
                db->aimYawRate[f] = angle / animDt;
            }
            else
            {
                // last frame of clip: copy previous rate (or zero for single-frame clips)
                db->aimYawRate[f] = (f > clipStart) ? db->aimYawRate[f - 1] : 0.0f;
            }
        }
    }

    TraceLog(LOG_INFO, "AnimDatabase: built motion DB with %d frames and %d joints",
        db->motionFrameCount, db->jointCount);


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
            db->hipRotationInMagicSpace[f] = Rot6dMultiply(invMagicYawRot, hipRot);
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
        const float n = db->featuresConfig.poseDragLookaheadTime / frameTime;
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
        const float n = db->featuresConfig.poseDragLookaheadTime / frameTime;
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

    // Populate legalStartFrames
    db->legalStartFrames.clear();
    for (int c = 0; c < (int)db->clipStartFrame.size(); ++c)
    {
        const int clipStart = db->clipStartFrame[c];
        const int clipEnd = db->clipEndFrame[c];
        const float frameTime = db->animFrameTime[c];

        const int minLegalFrameOffset = (int)ceilf(0.1f / frameTime);
        const int maxLegalFrameOffset = (int)ceilf(1.0f / frameTime);

        for (int f = clipStart; f < clipEnd; ++f)
        {
            const int framesFromStart = f - clipStart;
            const int framesToEnd = clipEnd - 1 - f;

            if (framesFromStart >= minLegalFrameOffset && framesToEnd >= maxLegalFrameOffset)
            {
                db->legalStartFrames.push_back(f);
            }
        }
    }
    TraceLog(LOG_INFO, "AnimDatabase: populated %d legal start frames", (int)db->legalStartFrames.size());

    const MotionMatchingFeaturesConfig& cfg = db->featuresConfig;

    db->featureDim = 0;
    if (cfg.IsFeatureEnabled(FeatureType::ToePos)) db->featureDim += 4;
    if (cfg.IsFeatureEnabled(FeatureType::ToeVel)) db->featureDim += 4;
    if (cfg.IsFeatureEnabled(FeatureType::ToePosDiff)) db->featureDim += 2;
    if (cfg.IsFeatureEnabled(FeatureType::FutureVel)) db->featureDim += (int)cfg.futureTrajPointTimes.size() * 2;
    if (cfg.IsFeatureEnabled(FeatureType::FutureVelClamped)) db->featureDim += (int)cfg.futureTrajPointTimes.size() * 2;
    if (cfg.IsFeatureEnabled(FeatureType::FutureSpeed)) db->featureDim += (int)cfg.futureTrajPointTimes.size();
    if (cfg.IsFeatureEnabled(FeatureType::PastPosition)) db->featureDim += 2;
    if (cfg.IsFeatureEnabled(FeatureType::FutureAimDirection)) db->featureDim += (int)cfg.futureTrajPointTimes.size() * 2;
    if (cfg.IsFeatureEnabled(FeatureType::FutureAimVelocity)) db->featureDim += (int)cfg.futureTrajPointTimes.size();
    if (cfg.IsFeatureEnabled(FeatureType::HeadToSlowestToe)) db->featureDim += 2;
    if (cfg.IsFeatureEnabled(FeatureType::HeadToToeAverage)) db->featureDim += 2;
    if (cfg.IsFeatureEnabled(FeatureType::FutureAccelClamped)) db->featureDim += (int)cfg.futureTrajPointTimes.size() * 2;

    db->features.clear();
    db->featureNames.clear();

    db->features.resize(db->motionFrameCount, db->featureDim);
    db->features.fill(0.0f);

    if (db->featureDim == 0)
    {
        TraceLog(LOG_WARNING, "AnimDatabase: no features enabled in configuration");
        return;
    }
    // Populate features using the shared per-frame function
    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        float* featRow = db->features.row_view(f).data();
        const bool isFirstFrame = (f == 0);
        ComputeRawFeaturesForFrame(
            db, cfg, f, featRow,
            isFirstFrame ? &db->featureNames : nullptr,
            isFirstFrame ? &db->featureTypes : nullptr,
            nullptr);
    }



    // Allocate mean/min/max vectors (per-dimension)
    db->featuresMean.resize(db->featureDim, 0.0f);
    db->featuresMin.resize(db->featureDim, FLT_MAX);
    db->featuresMax.resize(db->featureDim, -FLT_MAX);

    // Compute mean, min, max for each feature dimension
    for (int d = 0; d < db->featureDim; ++d)
    {
        double sum = 0.0;
        float dMin = FLT_MAX;
        float dMax = -FLT_MAX;
        for (int f = 0; f < db->motionFrameCount; ++f)
        {
            const float v = db->features.at(f, d);
            sum += v;
            if (v < dMin) dMin = v;
            if (v > dMax) dMax = v;
        }
        db->featuresMean[d] = (float)(sum / db->motionFrameCount);
        db->featuresMin[d] = dMin;
        db->featuresMax[d] = dMax;
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




    // Compute pose generation features (neural network training targets)
    // Using PoseFeatures struct for clean serialization
    db->poseGenFeaturesComputeDim = PoseFeatures::GetDim(db->jointCount);
    db->poseGenFeatures.resize(db->motionFrameCount, db->poseGenFeaturesComputeDim);
    db->poseGenFeatures.fill(0.0f);

    PoseFeatures tempPose;
    tempPose.Resize(db->jointCount);

    for (int f = 0; f < db->motionFrameCount; ++f)
    {
        // Copy lookahead local rotations
        span<const Rot6d> lookaheadRots = db->lookaheadLocalRotations.row_view(f);
        for (int j = 0; j < db->jointCount; ++j)
        {
            tempPose.lookaheadLocalRotations[j] = lookaheadRots[j];
        }

        // Copy bone 0 (root) local position only — other bones are static skeleton offsets
        span<const Vector3> lookaheadPos = db->lookaheadLocalPositions.row_view(f);
        tempPose.rootLocalPosition = lookaheadPos[0];

        // Copy lookahead root velocity
        tempPose.lookaheadRootVelocity = db->lookaheadRootMotionVelocitiesRootSpace[f];

        // Copy current root yaw rate (not lookahead)
        tempPose.rootYawRate = db->rootMotionYawRates[f];

        // Copy lookahead toe positions
        for (int side : sides)
        {
            tempPose.lookaheadToePositionsRootSpace[side] = db->lookaheadToePositionsRootSpace[side][f];
        }

        // Copy current toe velocities (not lookahead)
        span<const Vector3> jointVelRootSpaceRow = db->jointVelocitiesRootSpace.row_view(f);
        for (int side : sides)
        {
            const int toeIdx = db->toeIndices[side];
            assertEvenInRelease(toeIdx >= 0);
            tempPose.toeVelocitiesRootSpace[side] = jointVelRootSpaceRow[toeIdx];
        }

        // current-frame 2D toe position difference (crisp, directly from world positions)
        const Vector3 leftToe = db->toePositionsRootSpace[SIDE_LEFT][f];
        const Vector3 rightToe = db->toePositionsRootSpace[SIDE_RIGHT][f];
        tempPose.toePosDiffRootSpace = { leftToe.x - rightToe.x, 0.0f, leftToe.z - rightToe.z };

        // toe speed difference magnitude (always positive — can't regress to zero)
        const float leftSpeedXZ = Vector3Length2D(jointVelRootSpaceRow[db->toeIndices[SIDE_LEFT]]);
        const float rightSpeedXZ = Vector3Length2D(jointVelRootSpaceRow[db->toeIndices[SIDE_RIGHT]]);
        tempPose.toeSpeedDiff = fabsf(leftSpeedXZ - rightSpeedXZ);

        // Serialize to flat array
        span<float> poseRow = db->poseGenFeatures.row_view(f);
        tempPose.SerializeTo(poseRow);
    }

    TraceLog(LOG_INFO, "AnimDatabase: computed pose generation features (dim=%d) for neural network training",
        db->poseGenFeaturesComputeDim);

    // normalize poseGenFeatures for segment autoencoder training
    {
        const int pgDim = db->poseGenFeaturesComputeDim;
        const int N = db->motionFrameCount;
        const int jc = db->jointCount;

        // per-dim mean, min, max
        db->poseGenFeaturesMean.resize(pgDim, 0.0f);
        db->poseGenFeaturesMin.resize(pgDim, FLT_MAX);
        db->poseGenFeaturesMax.resize(pgDim, -FLT_MAX);
        for (int d = 0; d < pgDim; ++d)
        {
            float sum = 0.0f;
            float dMin = FLT_MAX;
            float dMax = -FLT_MAX;
            for (int f = 0; f < N; ++f)
            {
                const float v = db->poseGenFeatures.at(f, d);
                sum += v;
                if (v < dMin) { dMin = v; }
                if (v > dMax) { dMax = v; }
            }
            db->poseGenFeaturesMean[d] = sum / N;
            db->poseGenFeaturesMin[d] = dMin;
            db->poseGenFeaturesMax[d] = dMax;
        }

        // per-dim std
        db->poseGenFeaturesStd.resize(pgDim, 1.0f);
        for (int d = 0; d < pgDim; ++d)
        {
            float sumSqDiff = 0.0f;
            const float mean = db->poseGenFeaturesMean[d];
            for (int f = 0; f < N; ++f)
            {
                const float diff = db->poseGenFeatures.at(f, d) - mean;
                sumSqDiff += diff * diff;
            }
            const float s = sqrtf(sumSqDiff / N);
            db->poseGenFeaturesStd[d] = (s > 1e-8f) ? s : 1.0f;
        }

        // per-dim bone weight — we build a PoseFeatures where each component
        // holds the weight we want for that dim, then serialize it. this way the
        // layout is defined in one place (PoseFeatures::SerializeTo) and we don't
        // have to manually track index offsets here.
        static const std::unordered_map<std::string, float> boneWeights = {
            {"Hips",0.27089f},{"Spine",0.12777f},{"Spine1",0.10730f},
            {"Spine2",0.08734f},{"Spine3",0.07508f},{"Neck",0.00839f},{"Neck1",0.00640f},
            {"Head",0.00515f},{"HeadEnd",0.00063f},
            {"RightShoulder",0.02654f},{"RightArm",0.02061f},{"RightForeArm",0.00826f},
            {"RightHand",0.00213f},
            {"LeftShoulder",0.02739f},{"LeftArm",0.02113f},{"LeftForeArm",0.00850f},
            {"LeftHand",0.00211f},
            {"RightUpLeg",0.15690f},{"RightLeg",0.04044f},{"RightFoot",0.02f},
            //{"RightUpLeg",0.05690f},{"RightLeg",0.02044f},{"RightFoot",0.02f},
            {"RightToeBase",0.005f},{"RightToeBaseEnd",0.00063f},
            {"LeftUpLeg",0.15668f},{"LeftLeg",0.04034f},{"LeftFoot",0.02f},
            //{"LeftUpLeg",0.05668f},{"LeftLeg",0.02034f},{"LeftFoot",0.00289f},
            {"LeftToeBase",0.005f},{"LeftToeBaseEnd",0.00063f},
        };
        constexpr float defaultBoneWeight = 0.01f;
        constexpr float notSoBigWeight = 0.4f;
        constexpr float footPosVelWeight = 1.0f;

        const BVHData* skeleton = &characterData->bvhData[0];

        // fill a PoseFeatures with weights instead of actual values,
        // then serialize — each component becomes the weight for that dim
        PoseFeatures weightPose;
        weightPose.Resize(jc);

        for (int j = 0; j < jc; ++j)
        {
            const std::string& name = skeleton->joints[j].name;
            auto it = boneWeights.find(name);
            float w = (it != boneWeights.end()) ? it->second : defaultBoneWeight;
            w *= 5.0f; // scale up weights so the most important bones have weight around 1.0f, to avoid too small values after normalization
            weightPose.lookaheadLocalRotations[j] = { w, w, w, w, w, w };
        }
        weightPose.rootLocalPosition = { notSoBigWeight, notSoBigWeight, notSoBigWeight };
        weightPose.lookaheadRootVelocity = { notSoBigWeight, notSoBigWeight, notSoBigWeight };
        weightPose.rootYawRate = 3.0f;// this one is very important (?)
        for (int side : sides)
        {
            weightPose.lookaheadToePositionsRootSpace[side] = { footPosVelWeight, footPosVelWeight, footPosVelWeight };
            weightPose.toeVelocitiesRootSpace[side] = { footPosVelWeight, footPosVelWeight, footPosVelWeight };
        }
        weightPose.toePosDiffRootSpace = { footPosVelWeight, 0.0f, footPosVelWeight };
        weightPose.toeSpeedDiff = footPosVelWeight;

        db->poseGenFeaturesWeight.resize(pgDim);
        weightPose.SerializeTo(db->poseGenFeaturesWeight);

        // compute normalizedPoseGenFeatures = (raw - mean) / std * weight
        db->normalizedPoseGenFeatures.resize(N, pgDim);
        for (int f = 0; f < N; ++f)
        {
            span<const float> raw = db->poseGenFeatures.row_view(f);
            span<float> norm = db->normalizedPoseGenFeatures.row_view(f);
            for (int d = 0; d < pgDim; ++d)
            {
                norm[d] = (raw[d] - db->poseGenFeaturesMean[d]) / db->poseGenFeaturesStd[d] * db->poseGenFeaturesWeight[d];
            }
        }

        // compute segment sizing for the autoencoder
        // use frame time of first clip (they're usually all the same)
        const float frameTime = db->animFrameTime[0];
        db->poseGenSegmentFrameCount = (int)ceilf(db->poseGenFeaturesSegmentLength / frameTime) + 1;
        db->poseGenSegmentFlatDim = db->poseGenSegmentFrameCount * pgDim;

        TraceLog(LOG_INFO, "AnimDatabase: normalized poseGenFeatures (dim=%d, segFrames=%d, segFlatDim=%d)",
            pgDim, db->poseGenSegmentFrameCount, db->poseGenSegmentFlatDim);
    }

    // set db->valid true now that we completed full build
    db->valid = true;


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
