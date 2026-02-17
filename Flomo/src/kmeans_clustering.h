#pragma once

#include "utils.h"

// hierarchical bisection clustering: builds clusters of roughly equal radius.
// starts with all points in one cluster, then repeatedly splits the largest-radius
// cluster in two using k-means++ seeded bisection. stops at the target cluster count.
//
// this file is self-contained and does not depend on AnimDatabase.
// the thin wrapper AnimDatabaseClusterFeatures2 at the bottom adapts it for our use.

constexpr int BISECTION_MAX_SPLIT_ITERS = 20;

struct BisectionCluster
{
    std::vector<int> members;     // indices into the input point array
    std::vector<float> centroid;  // [dim]
    float radius = 0.0f;         // max distance from centroid to any member
};

// squared euclidean distance between two float vectors
static inline float FeatureDistSquared(const float* a, const float* b, int dim)
{
    float d2 = 0.0f;
    for (int d = 0; d < dim; ++d)
    {
        const float diff = a[d] - b[d];
        d2 += diff * diff;
    }
    return d2;
}

// get pointer to the i-th point (row-major, each row is dim floats)
static inline const float* PointAt(const float* points, int i, int dim)
{
    return points + i * dim;
}

// compute centroid (mean of members) and radius (max distance to any member)
static void ComputeClusterCentroidAndRadius(
    /*inout*/ BisectionCluster& cluster,
    const float* points,
    int dim)
{
    const int n = (int)cluster.members.size();
    cluster.centroid.assign(dim, 0.0f);
    if (n == 0)
    {
        cluster.radius = 0.0f;
        return;
    }

    // sum all member features
    for (int i = 0; i < n; ++i)
    {
        const float* p = PointAt(points, cluster.members[i], dim);
        for (int d = 0; d < dim; ++d)
        {
            cluster.centroid[d] += p[d];
        }
    }

    // mean
    const float inv = 1.0f / (float)n;
    for (int d = 0; d < dim; ++d)
    {
        cluster.centroid[d] *= inv;
    }

    // radius = max distance from centroid to any member
    float maxDist2 = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        const float* p = PointAt(points, cluster.members[i], dim);
        const float dist2 = FeatureDistSquared(p, cluster.centroid.data(), dim);
        if (dist2 > maxDist2)
        {
            maxDist2 = dist2;
        }
    }
    cluster.radius = sqrtf(maxDist2);
}

// split one cluster into two using k=2 k-means with k-means++ seeding.
// childA and childB should be empty on input.
static void SplitClusterBisection(
    const BisectionCluster& parent,
    /*out*/ BisectionCluster& childA,
    /*out*/ BisectionCluster& childB,
    const float* points,
    int dim)
{
    const int n = (int)parent.members.size();
    assert(n >= 2);

    std::vector<float> cA(dim);
    std::vector<float> cB(dim);

    // k-means++ seed 1: random member
    {
        const float* p = PointAt(points, parent.members[RandomInt(n)], dim);
        memcpy(cA.data(), p, dim * sizeof(float));
    }

    // k-means++ seed 2: pick with probability proportional to dist^2 from first seed.
    // this spreads the two seeds far apart for a good initial split.
    {
        std::vector<float> dists(n);
        float totalDist = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            const float* p = PointAt(points, parent.members[i], dim);
            dists[i] = FeatureDistSquared(p, cA.data(), dim);
            totalDist += dists[i];
        }

        float r = ((float)RandomInt(1000000) / 1000000.0f) * totalDist;
        int picked = n - 1;
        for (int i = 0; i < n; ++i)
        {
            r -= dists[i];
            if (r <= 0.0f) { picked = i; break; }
        }

        const float* p = PointAt(points, parent.members[picked], dim);
        memcpy(cB.data(), p, dim * sizeof(float));
    }

    // iterate k=2 k-means until convergence
    std::vector<int> assignments(n, 0);
    for (int iter = 0; iter < BISECTION_MAX_SPLIT_ITERS; ++iter)
    {
        // assign each member to nearest centroid
        bool changed = false;
        for (int i = 0; i < n; ++i)
        {
            const float* p = PointAt(points, parent.members[i], dim);
            const float dA = FeatureDistSquared(p, cA.data(), dim);
            const float dB = FeatureDistSquared(p, cB.data(), dim);
            const int newAssign = (dA <= dB) ? 0 : 1;
            if (newAssign != assignments[i])
            {
                changed = true;
            }
            assignments[i] = newAssign;
        }
        if (!changed) break;

        // recompute centroids
        std::vector<float> sumsA(dim, 0.0f);
        std::vector<float> sumsB(dim, 0.0f);
        int countA = 0;
        int countB = 0;

        for (int i = 0; i < n; ++i)
        {
            const float* p = PointAt(points, parent.members[i], dim);
            if (assignments[i] == 0)
            {
                for (int d = 0; d < dim; ++d) { sumsA[d] += p[d]; }
                countA++;
            }
            else
            {
                for (int d = 0; d < dim; ++d) { sumsB[d] += p[d]; }
                countB++;
            }
        }

        if (countA > 0)
        {
            const float inv = 1.0f / (float)countA;
            for (int d = 0; d < dim; ++d) { cA[d] = sumsA[d] * inv; }
        }
        if (countB > 0)
        {
            const float inv = 1.0f / (float)countB;
            for (int d = 0; d < dim; ++d) { cB[d] = sumsB[d] * inv; }
        }
    }

    // distribute members into children
    for (int i = 0; i < n; ++i)
    {
        if (assignments[i] == 0)
        {
            childA.members.push_back(parent.members[i]);
        }
        else
        {
            childB.members.push_back(parent.members[i]);
        }
    }

    ComputeClusterCentroidAndRadius(childA, points, dim);
    ComputeClusterCentroidAndRadius(childB, points, dim);
}

// generic bisection clustering.
// points: row-major float array, numPoints rows of dim floats each.
// targetK: desired number of clusters.
// outClusters: on return, outClusters[c] contains the point indices for cluster c.
static void BisectionCluster_Run(
    const float* points,
    int numPoints,
    int dim,
    int targetK,
    /*out*/ std::vector<std::vector<int>>& outClusters)
{
    if (numPoints == 0 || dim <= 0)
    {
        outClusters.clear();
        return;
    }

    // start with one cluster containing all points
    std::vector<BisectionCluster> clusters(1);
    clusters[0].members.resize(numPoints);
    for (int i = 0; i < numPoints; ++i)
    {
        clusters[0].members[i] = i;
    }
    ComputeClusterCentroidAndRadius(clusters[0], points, dim);

    // repeatedly split the largest-radius cluster
    while ((int)clusters.size() < targetK)
    {
        // find the splittable cluster with the largest radius
        int worstIdx = -1;
        float worstRadius = -1.0f;
        for (int i = 0; i < (int)clusters.size(); ++i)
        {
            if ((int)clusters[i].members.size() >= 2 && clusters[i].radius > worstRadius)
            {
                worstRadius = clusters[i].radius;
                worstIdx = i;
            }
        }

        // nothing left to split (all clusters are singletons)
        if (worstIdx < 0) break;

        BisectionCluster childA, childB;
        SplitClusterBisection(clusters[worstIdx], childA, childB, points, dim);

        // replace parent with one child, append the other
        clusters[worstIdx] = std::move(childA);
        clusters.push_back(std::move(childB));
    }

    // copy results out
    const int k = (int)clusters.size();
    outClusters.resize(k);
    for (int c = 0; c < k; ++c)
    {
        outClusters[c] = std::move(clusters[c].members);
    }

    // log stats
    float minRadius = FLT_MAX, maxRadius = 0.0f, avgRadius = 0.0f;
    int minSize = numPoints, maxSize = 0;
    for (int c = 0; c < k; ++c)
    {
        const int sz = (int)outClusters[c].size();
        if (sz < minSize) minSize = sz;
        if (sz > maxSize) maxSize = sz;
        avgRadius += clusters[c].radius;
        if (clusters[c].radius < minRadius) minRadius = clusters[c].radius;
        if (clusters[c].radius > maxRadius) maxRadius = clusters[c].radius;
    }
    avgRadius /= (float)k;
    TraceLog(LOG_INFO,
        "Bisection clustering: %d clusters, "
        "sizes min=%d max=%d avg=%d, radius min=%.3f max=%.3f avg=%.3f",
        k, minSize, maxSize, numPoints / k,
        minRadius, maxRadius, avgRadius);
}
