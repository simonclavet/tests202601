
// notes for ai reading this code, and asked to make modifications:
// some of these conventions are not really respected everywhere in this code, but try to follow them as much as possible for any new code you write.
// Function names are CamelCase, variable and member names are camelCase. constexpr for CONSTANTS
// don't use auto unless absolutely necessary
// don't use lambdas
// be const correct for functions and function parameters (except simple parameter values). 
// be const correct for local variable: if a variable is not modified after initialization, make it const.
// don't create unused variables
// When an argument is modified as the result, prefix the argument with /*out*/ if it is not obvious.
// never modify non-const value arguments of functions.
// don't use pairs and tuples, make small structs instead
// don't use smart pointers
// don't use complicated oop concepts like inheritance and polymorphism, unless absolutely necessary
// don't use exceptions, use assertions defensively. Don't use the keyword noexcept, catch, try, unless absolutely necessary
// use linebreaks before opening braces for functions and control blocks, always use braces for if/for/while, except when it is very simple single-line statements
// when writing comments, be casual, no need for things like ---- and other heading decorations like numbers or letters for steps
// no need for private and public. Use structs only
// don't remove these comments
// ask questions to the human if unsure about anything. Don't assume things, ask instead.

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <random>
#include <queue>
#include <cassert>
#include <chrono>
using timepoint = std::chrono::high_resolution_clock::time_point;

#include <limits>
#include <fstream>
#include <unordered_map>
#include <numeric>
#include <span> 


template<typename T>
class Array2D
{
private:
    int rows_;
    int cols_;
    std::vector<T> data_;

public:
    // Constructor
    Array2D(int rows, int cols)
        : rows_(rows), cols_(cols), data_(rows* cols)
    {
        assert(rows >= 0 && cols >= 0 && "Array2D dimensions must be non-negative");
    }

    // Constructor with initial value
    Array2D(int rows, int cols, const T& initial_value)
        : rows_(rows), cols_(cols), data_(rows* cols, initial_value)
    {
        assert(rows >= 0 && cols >= 0 && "Array2D dimensions must be non-negative");
    }

    // Non-const element access with bounds checking
    T& at(int row_view, int col)
    {
        assert(row_view >= 0 && row_view < rows_ && col >= 0 && col < cols_
            && "Array2D index out of bounds");
        return data_[row_view * cols_ + col];
    }

    // Const element access with bounds checking
    const T& at(int row_view, int col) const
    {
        assert(row_view >= 0 && row_view < rows_ && col >= 0 && col < cols_
            && "Array2D index out of bounds");
        return data_[row_view * cols_ + col];
    }

    // Non-const row view
    std::span<T> row_view(int row_idx)
    {
        assert(row_idx >= 0 && row_idx < rows_ && "Array2D row index out of bounds");
        return std::span<T>(data_.data() + row_idx * cols_, cols_);
    }

    // Const row view
    std::span<const T> row_view(int row_idx) const
    {
        assert(row_idx >= 0 && row_idx < rows_ && "Array2D row index out of bounds");
        return std::span<const T>(data_.data() + row_idx * cols_, cols_);
    }

    // Fill all elements with the same value
    void fill(const T& value)
    {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Get dimensions
    int rows() const noexcept
    {
        return rows_;
    }

    int cols() const noexcept
    {
        return cols_;
    }

    int size() const noexcept
    {
        return rows_ * cols_;
    }

    // Raw data access
    T* data() noexcept
    {
        return data_.data();
    }

    const T* data() const noexcept
    {
        return data_.data();
    }

    // Non-const operator[] for unchecked access
    T& operator()(int row_view, int col) noexcept
    {
        return data_[row_view * cols_ + col];
    }

    // Const operator[] for unchecked access
    const T& operator()(int row_view, int col) const noexcept
    {
        return data_[row_view * cols_ + col];
    }

    // Iterator support
    typename std::vector<T>::iterator begin() noexcept
    {
        return data_.begin();
    }

    typename std::vector<T>::iterator end() noexcept
    {
        return data_.end();
    }

    typename std::vector<T>::const_iterator begin() const noexcept
    {
        return data_.begin();
    }

    typename std::vector<T>::const_iterator end() const noexcept
    {
        return data_.end();
    }

    typename std::vector<T>::const_iterator cbegin() const noexcept
    {
        return data_.cbegin();
    }

    typename std::vector<T>::const_iterator cend() const noexcept
    {
        return data_.cend();
    }
};


// Helper for managing index lists
inline void SwapRemoveAt(std::vector<int>& v, size_t index)
{
    if (index >= v.size())
        return;
    std::swap(v[index], v.back());
    v.pop_back();
}


// Node struct using SoA layout
struct Node
{
    int repPointsStartIndex;                   // Starting row in the nodeSystemRepPoints Array2D
    std::vector<int> representativePointIndices; // Indices in the original dataset
    std::vector<int> nodePointers;             // Index in node array, -1 if leaf
    std::vector<float> radii;                  // Max distance
    int childCount;

    Node(int childrenNum)
        : repPointsStartIndex(-1),
        childCount(0),
        representativePointIndices(childrenNum, -1),
        nodePointers(childrenNum, -1),
        radii(childrenNum, 0.0f)
    {
    }
};


// Build task structure
struct BuildTask
{
    int nodeIndex;                // Index of the node to process
    std::vector<int> pointIndices; // Indices of points assigned to this node

    BuildTask(int idx, std::vector<int> indices)
        : nodeIndex(idx), pointIndices(std::move(indices))
    {
    }
};


// Container to hold the tree structure
struct BallTree
{
    std::vector<Node> nodes;
    Array2D<float> nodeSystemRepPoints; // 2D array for representative points
    int dim;
    int childrenNum;

    BallTree(int d, int childrenNum)
        : dim(d),
        childrenNum(childrenNum),
        nodeSystemRepPoints(0, d)
    {
    }
};


struct ScheduledNode
{
    float distanceLowerBound;
    int nodeIndex;

    bool operator<(const ScheduledNode& other) const
    {
        return distanceLowerBound > other.distanceLowerBound;
    }
};


struct SearchStats
{
    int nodesVisited = 0;
    int pointsChecked = 0;
    int nodesVisitedWhenTrueWinnerFound = 0;
    int nodesVisitedWhenBestFound = 0;
    bool terminatedEarly = false;
    std::string terminationReason;
    float finalBestDist = std::numeric_limits<float>::max();
    float linearSearchDist = std::numeric_limits<float>::max();
    int linearSearchIndex = -1;
    double searchTimeMs = 0.0;
    std::vector<std::pair<int, float>> improvementHistory;

    void reset()
    {
        nodesVisited = 0;
        pointsChecked = 0;
        nodesVisitedWhenTrueWinnerFound = 0;
        nodesVisitedWhenBestFound = 0;
        terminatedEarly = false;
        terminationReason = "";
        finalBestDist = std::numeric_limits<float>::max();
        linearSearchDist = std::numeric_limits<float>::max();
        linearSearchIndex = -1;
        searchTimeMs = 0.0;
        improvementHistory.clear();
    }
};


// --- Helper Functions using Array2D ---

// Calculate squared Euclidean distance between two spans
float DistanceSquared(std::span<const float> a, std::span<const float> b)
{
    const size_t d = a.size();
    float dist = 0.0f;

    for (size_t i = 0; i < d; ++i)
    {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }

    return dist;
}


// Compute Mean of a subset of points - fills result vector
void ComputeMean(const Array2D<float>& data,
    const std::vector<int>& indices,
    int dim,
    std::vector<float>& result)
{
    result.assign(dim, 0.0f);

    if (indices.empty())
        return;

    for (const int idx : indices)
    {
        const std::span<const float> p = data.row_view(idx);

        for (int i = 0; i < dim; ++i)
        {
            result[i] += p[i];
        }
    }

    const float invN = 1.0f / static_cast<float>(indices.size());

    for (int i = 0; i < dim; ++i)
    {
        result[i] *= invN;
    }
}


int FindClosestPointToMean(const Array2D<float>& data,
    const std::vector<int>& indices,
    std::span<const float> mean,
    int dim,
    /*out*/int& indexInIndices)
{

    if (indices.empty())
        return -1;

    int bestIdx = indices[0];
    indexInIndices = 0;

    const std::span<const float> p0 = data.row_view(bestIdx);
    float bestDist = DistanceSquared(p0, mean);

    for (size_t i = 1; i < indices.size(); ++i)
    {
        const int idx = indices[i];
        const std::span<const float> p = data.row_view(idx);
        const float dist = DistanceSquared(p, mean);

        if (dist < bestDist)
        {
            bestDist = dist;
            bestIdx = idx;
            indexInIndices = static_cast<int>(i);
        }
    }

    return bestIdx;
}


// Generate random point into Array2D
void GenerateRandomPoint(std::span<float> row_view,
    std::mt19937& gen,
    std::uniform_real_distribution<float>& distribution)
{

    const int dim = static_cast<int>(row_view.size());

    // Basic fill
    for (int d = 0; d < dim; ++d)
    {
        row_view[d] = distribution(gen);
    }

    // Apply specific transformations if dimension allows
    if (dim >= 20)
    {
        row_view[8] = std::sin(row_view[5] + row_view[1]);
        row_view[9] = std::sin(row_view[0] + row_view[6]);
        row_view[10] = std::sin(row_view[0] + row_view[1]);
        row_view[11] = std::cos(row_view[2]);
        row_view[12] = std::sin(row_view[3] * row_view[4]);
        row_view[13] = std::cos(row_view[5] - row_view[6]);
        row_view[14] = std::sin(row_view[7]) * std::cos(row_view[8]);
        row_view[15] = std::sqrt(std::fabs(row_view[9]));
        row_view[16] = row_view[0] * row_view[1] + row_view[2] * row_view[3];
        row_view[17] = row_view[4] * row_view[5] - row_view[6] * row_view[7];
        row_view[18] = std::sin(row_view[8]) + std::cos(row_view[9]);
        row_view[19] = row_view[0] * row_view[2] + row_view[4] * row_view[6];
    }
}


// Linear search
int LinearSearchNearestNeighbor(const Array2D<float>& data,
    std::span<const float> target,
    float& bestDist)
{
    const int numPoints = data.rows();

    if (numPoints == 0)
    {
        bestDist = std::numeric_limits<float>::max();
        return -1;
    }

    int bestIdx = 0;
    const std::span<const float> p0 = data.row_view(0);
    bestDist = DistanceSquared(p0, target);

    for (int i = 1; i < numPoints; ++i)
    {
        const std::span<const float> p = data.row_view(i);
        const float dist = DistanceSquared(p, target);

        if (dist < bestDist)
        {
            bestDist = dist;
            bestIdx = i;
        }
    }

    return bestIdx;
}


// K-NN Linear Search
std::vector<int> LinearSearchKNearestNeighbors(
    const Array2D<float>& data,
    std::span<const float> target,
    int k,
    std::vector<float>* outDists = nullptr)
{
    std::vector<int> result;
    const int numPoints = data.rows();

    if (numPoints == 0 || k <= 0)
        return result;

    k = std::min(k, numPoints);

    using Pair = std::pair<float, int>;
    std::priority_queue<Pair> pq;

    for (int i = 0; i < numPoints; ++i)
    {
        const std::span<const float> p = data.row_view(i);
        const float distSq = DistanceSquared(p, target);

        if (static_cast<int>(pq.size()) < k)
        {
            pq.emplace(distSq, i);
        }
        else if (distSq < pq.top().first)
        {
            pq.pop();
            pq.emplace(distSq, i);
        }
    }

    std::vector<Pair> temp;
    temp.reserve(pq.size());

    while (!pq.empty())
    {
        temp.push_back(pq.top());
        pq.pop();
    }

    std::sort(temp.begin(), temp.end(), [](const Pair& a, const Pair& b)
        {
            return a.first < b.first;
        });

    result.reserve(temp.size());

    if (outDists)
    {
        outDists->clear();
        outDists->reserve(temp.size());
    }

    for (const auto& p : temp)
    {
        result.push_back(p.second);

        if (outDists)
            outDists->push_back(p.first);
    }

    return result;
}


// BallTree Search
int BallTreeSearchNearestNeighbor(const BallTree& tree,
    const Array2D<float>& data,
    std::span<const float> target,
    SearchStats* stats,
    int maxPointsToVisit)
{

    if (tree.nodes.empty())
        return -1;

    std::priority_queue<ScheduledNode> pq;

    ScheduledNode rootNode;
    rootNode.distanceLowerBound = 0.0f;
    rootNode.nodeIndex = 0;
    pq.push(rootNode);

    int totalNodesVisited = 0;
    int totalPointsVisited = 0;
    int bestIdx = -1;
    float bestDist = std::numeric_limits<float>::max();
    bool trueWinnerFound = false;

    while (!pq.empty())
    {
        const ScheduledNode current = pq.top();
        pq.pop();

        if (maxPointsToVisit > 0 && totalPointsVisited >= maxPointsToVisit)
        {
            if (stats)
            {
                stats->terminatedEarly = true;
                stats->terminationReason = "Max nodes reached";
            }
            break;
        }

        if (current.distanceLowerBound >= bestDist)
        {
            continue;
        }

        const Node& node = tree.nodes[current.nodeIndex];
        totalNodesVisited++;

        if (stats)
            stats->nodesVisited++;

        // Access representative points from the Array2D
        const int baseRepRow = node.repPointsStartIndex;

        for (int childIdx = 0; childIdx < node.childCount; ++childIdx)
        {
            totalPointsVisited++;

            const std::span<const float> repPoint = tree.nodeSystemRepPoints.row_view(baseRepRow + childIdx);
            const int repIdx = node.representativePointIndices[childIdx];
            const int nodePtr = node.nodePointers[childIdx];
            const float radius = node.radii[childIdx];

            const float distToRep = DistanceSquared(repPoint, target);

            if (stats)
                stats->pointsChecked++;

            if (distToRep < bestDist)
            {
                bestDist = distToRep;
                bestIdx = repIdx;

                if (stats)
                {
                    stats->nodesVisitedWhenBestFound = totalNodesVisited;
                    stats->improvementHistory.emplace_back(totalNodesVisited, bestDist);

                    if (!trueWinnerFound && repIdx == stats->linearSearchIndex)
                    {
                        trueWinnerFound = true;
                        stats->nodesVisitedWhenTrueWinnerFound = totalNodesVisited;
                    }
                }
            }

            if (nodePtr == -1)
            {
                continue;
            }

            // Lower bound calc
            const float distToRepSqrt = std::sqrt(distToRep);
            const float radiusSqrt = std::sqrt(radius);
            float minPossibleDistSqrt = 0.0f;

            if (distToRepSqrt > radiusSqrt)
            {
                minPossibleDistSqrt = distToRepSqrt - radiusSqrt;
            }

            const float minPossibleDist = minPossibleDistSqrt * minPossibleDistSqrt;

            if (minPossibleDist < bestDist)
            {
                ScheduledNode childNode;
                childNode.distanceLowerBound = minPossibleDist;
                childNode.nodeIndex = nodePtr;
                pq.push(childNode);
            }
        }
    }

    if (stats)
        stats->finalBestDist = bestDist;

    return bestIdx;
}


// Build function
static void BuildBallTree(
    const Array2D<float>& data,
    int dim,
    int kmeansIterations,
    int childrenNum,
    std::mt19937& gen,
    /*out*/BallTree& tree)
{

    assert(tree.nodes.empty());
    const int numPoints = data.rows();

    // Estimate node count and reserve space. The tree is not balanced, so we over-allocate.
    const size_t estimatedMaxNodes = 100 * ((numPoints + childrenNum - 1) / childrenNum);

    // Pre-allocate all memory upfront
    tree.nodes.reserve(estimatedMaxNodes);
    tree.nodes.emplace_back(childrenNum);  // Create root node
    tree.nodeSystemRepPoints = Array2D<float>(estimatedMaxNodes * childrenNum, dim);

    // Track the next available node index
    size_t nextNodeIndex = 1; // 0 is used for root

    std::queue<BuildTask> buildQueue;

    // Initialize root node
    Node& root = tree.nodes[0];
    root.repPointsStartIndex = 0; // Root's rep points start at row 0

    std::vector<int> allIndices(numPoints);

    for (int i = 0; i < numPoints; ++i)
    {
        allIndices[i] = i;
    }

    buildQueue.push(BuildTask(0, std::move(allIndices)));

    // Temporary buffers for KMeans
    Array2D<float> centroids(childrenNum, dim);
    std::vector<int> centroidIndices(childrenNum);
    std::vector<std::vector<int>> clusters(childrenNum);
    std::vector<float> meanBuffer(dim); // Reusable buffer for mean computation

    while (!buildQueue.empty())
    {
        const BuildTask task = std::move(buildQueue.front());
        buildQueue.pop();

        Node& node = tree.nodes[task.nodeIndex];

        if (task.pointIndices.size() <= childrenNum)
        {
            node.childCount = static_cast<int>(task.pointIndices.size());

            for (int i = 0; i < node.childCount; ++i)
            {
                const int pIdx = task.pointIndices[i];
                const std::span<const float> src = data.row_view(pIdx);
                std::span<float> dest = tree.nodeSystemRepPoints.row_view(node.repPointsStartIndex + i);

                // Copy point data
                std::copy(src.begin(), src.end(), dest.begin());

                node.representativePointIndices[i] = pIdx;
                node.nodePointers[i] = -1;
                node.radii[i] = 0.0f;
            }
            continue;
        }

        // --- K-Means Initialization ---
        std::vector<int> shuffledIndices = task.pointIndices;
        std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), gen);

        for (int i = 0; i < childrenNum && i < static_cast<int>(shuffledIndices.size()); ++i)
        {
            centroidIndices[i] = shuffledIndices[i];
            const std::span<const float> src = data.row_view(centroidIndices[i]);
            std::span<float> dest = centroids.row_view(i);
            std::copy(src.begin(), src.end(), dest.begin());
        }

        // --- K-Means Loop ---
        for (int iter = 0; iter < kmeansIterations; ++iter)
        {
            for (int c = 0; c < childrenNum; ++c)
            {
                clusters[c].clear();
            }

            for (const int pIdx : task.pointIndices)
            {
                const std::span<const float> p = data.row_view(pIdx);
                int bestCluster = 0;
                const std::span<const float> c0 = centroids.row_view(0);
                float bestDist = DistanceSquared(p, c0);

                for (int c = 1; c < childrenNum; ++c)
                {
                    const std::span<const float> cc = centroids.row_view(c);
                    const float dist = DistanceSquared(p, cc);

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestCluster = c;
                    }
                }
                clusters[bestCluster].push_back(pIdx);
            }

            // Recompute centroids
            for (int c = 0; c < childrenNum; ++c)
            {
                if (!clusters[c].empty())
                {
                    ComputeMean(data, clusters[c], dim, meanBuffer);
                    std::span<float> dest = centroids.row_view(c);
                    std::copy(meanBuffer.begin(), meanBuffer.end(), dest.begin());
                }
            }
        }

        // --- Create Children ---
        node.childCount = 0;

        for (int c = 0; c < childrenNum; ++c)
        {
            if (!clusters[c].empty())
            {
                // Check if we have enough space for new nodes
                assert(nextNodeIndex < estimatedMaxNodes); // "Pre-allocated node capacity exceeded. Increase estimatedMaxNodes.
                
                ComputeMean(data, clusters[c], dim, meanBuffer);
                const std::span<const float> clusterMean(meanBuffer.data(), dim);

                int indexOfRepresentativeInCluster = -1;
                const int repIdx = FindClosestPointToMean(data, clusters[c], clusterMean,
                    dim, indexOfRepresentativeInCluster);

                // Calculate Radius
                const std::span<const float> repPoint = data.row_view(repIdx);
                float maxRadiusSq = 0.0f;

                for (const int clusterPointIdx : clusters[c])
                {
                    const std::span<const float> cp = data.row_view(clusterPointIdx);
                    const float distSq = DistanceSquared(cp, repPoint);

                    if (distSq > maxRadiusSq)
                    {
                        maxRadiusSq = distSq;
                    }
                }

                // Create child node
                tree.nodes.emplace_back(childrenNum);
                Node& childNode = tree.nodes[nextNodeIndex];
                childNode.repPointsStartIndex = static_cast<int>(nextNodeIndex * childrenNum);
                const int childNodeIdx = static_cast<int>(nextNodeIndex);
                nextNodeIndex++;

                // Store representative point
                std::span<float> dest = tree.nodeSystemRepPoints.row_view(node.repPointsStartIndex + node.childCount);
                std::copy(repPoint.begin(), repPoint.end(), dest.begin());

                node.representativePointIndices[node.childCount] = repIdx;
                node.nodePointers[node.childCount] = childNodeIdx;
                node.radii[node.childCount] = maxRadiusSq;

                SwapRemoveAt(clusters[c], indexOfRepresentativeInCluster);
                buildQueue.push(BuildTask(childNodeIdx, std::move(clusters[c])));
                node.childCount++;
            }
        }
    }

    // Resize representative points to actual used rows
    const int actualRepPointsRows = static_cast<int>(tree.nodes.size() * childrenNum);

    if (tree.nodeSystemRepPoints.rows() > actualRepPointsRows)
    {
        // Create properly sized Array2D
        Array2D<float> resized(actualRepPointsRows, dim);

        for (int i = 0; i < actualRepPointsRows; ++i)
        {
            const std::span<const float> src = tree.nodeSystemRepPoints.row_view(i);
            std::span<float> dest = resized.row_view(i);
            std::copy(src.begin(), src.end(), dest.begin());
        }

        tree.nodeSystemRepPoints = std::move(resized);
    }
}


static void TestBallTree()
{
    // Configurable parameters
    const int numPoints = 800000;
    const int dim = 20;
    const int kmeansIterations = 5;
    const int numTestQueries = 100;
    const int childrenNum = 64;// 32;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::cout << "Generating " << numPoints << " random points in " << dim << " dimensions" << std::endl;

    // Use Array2D for points
    Array2D<float> points(numPoints, dim);

    for (int i = 0; i < numPoints; ++i)
    {
        std::span<float> row_view = points.row_view(i);
        GenerateRandomPoint(row_view, gen, distribution);
    }

    std::cout << "Building BallTree..." << std::endl;
    timepoint buildStart = std::chrono::high_resolution_clock::now();

    BallTree tree(dim, childrenNum);
    BuildBallTree(points, dim, kmeansIterations, childrenNum, gen, tree);

    timepoint buildEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildTime = buildEnd - buildStart;

    std::cout << "\nBallTree built successfully!" << std::endl;
    std::cout << "Build time: " << buildTime.count() << " seconds" << std::endl;
    std::cout << "Total nodes: " << tree.nodes.size() << std::endl;
    std::cout << "Representative points stored: " << tree.nodeSystemRepPoints.rows() << std::endl;

    std::cout << "\nGenerating " << numTestQueries << " test queries..." << std::endl;
    Array2D<float> testQueries(numTestQueries, dim);

    for (int i = 0; i < numTestQueries; ++i)
    {
        std::span<float> row_view = testQueries.row_view(i);
        GenerateRandomPoint(row_view, gen, distribution);
    }

    std::cout << "\n=== Running Linear Search (for ground truth) ===" << std::endl;
    std::vector<int> linearResults(numTestQueries);
    std::vector<float> linearDistances(numTestQueries);
    std::vector<double> linearTimes(numTestQueries);

    for (int i = 0; i < numTestQueries; ++i)
    {
        timepoint start = std::chrono::high_resolution_clock::now();
        float dist;
        const std::span<const float> query = testQueries.row_view(i);
        linearResults[i] = LinearSearchNearestNeighbor(points, query, dist);
        linearDistances[i] = dist;
        timepoint end = std::chrono::high_resolution_clock::now();
        linearTimes[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double linearAvgTime = std::accumulate(linearTimes.begin(), linearTimes.end(), 0.0) / numTestQueries;
    std::sort(linearTimes.begin(), linearTimes.end());
    std::cout << "Linear search stats: Avg=" << linearAvgTime << "ms, Median=" << linearTimes[numTestQueries / 2] << "ms" << std::endl;

    std::cout << "\n=== Testing BallTree Search ===" << std::endl;

    std::vector<SearchStats> allStats(numTestQueries);
    std::vector<int> ballTreeResults(numTestQueries);
    std::vector<double> ballTreeTimes(numTestQueries);

    for (int i = 0; i < numTestQueries; ++i)
    {
        timepoint start = std::chrono::high_resolution_clock::now();
        allStats[i].linearSearchDist = linearDistances[i];
        allStats[i].linearSearchIndex = linearResults[i];

        const std::span<const float> query = testQueries.row_view(i);
        ballTreeResults[i] = BallTreeSearchNearestNeighbor(
            tree, points, query, &allStats[i], -1);

        auto end = std::chrono::high_resolution_clock::now();
        ballTreeTimes[i] = std::chrono::duration<double, std::milli>(end - start).count();
        allStats[i].searchTimeMs = ballTreeTimes[i];
    }

    const double ballTreeAvgTime = std::accumulate(ballTreeTimes.begin(), ballTreeTimes.end(), 0.0) / numTestQueries;
    std::sort(ballTreeTimes.begin(), ballTreeTimes.end());

    std::cout << "BallTree stats: Avg=" << ballTreeAvgTime << "ms, Median=" << ballTreeTimes[numTestQueries / 2] << "ms" << std::endl;
    std::cout << "Speedup: " << linearAvgTime / ballTreeAvgTime << "x" << std::endl;

    // Verify
    int mismatches = 0;

    for (int i = 0; i < numTestQueries; ++i)
    {
        if (ballTreeResults[i] != linearResults[i])
        {
            mismatches++;
        }
    }

    std::cout << (mismatches == 0 ? "All results match!" : "Found mismatches!") << std::endl;

    // k-NN Stats (k=3)
    {
        const int k = 3;
        double sumDist[3] = { 0.0, 0.0, 0.0 };
        int countDist[3] = { 0, 0, 0 };

        for (int i = 0; i < numTestQueries; ++i)
        {
            std::vector<float> distsSq;
            const std::span<const float> query = testQueries.row_view(i);
            std::vector<int> neighbors = LinearSearchKNearestNeighbors(points, query, k, &distsSq);

            for (size_t j = 0; j < neighbors.size() && j < 3; ++j)
            {
                sumDist[j] += std::sqrt(distsSq[j]);
                countDist[j] += 1;
            }
        }

        std::cout << "\n=== k-NN Stats (k=3) ===" << std::endl;

        for (int j = 0; j < 3; ++j)
        {
            if (countDist[j] > 0)
                std::cout << "Avg dist to neighbor " << (j + 1) << ": " << (sumDist[j] / countDist[j]) << std::endl;
        }
    }


    std::cout << "\n=== K-Means Iterations Performance Test ===" << std::endl;
    std::cout << "Testing with fixed dataset size: " << numPoints << " points, " << dim << " dimensions" << std::endl;
    std::cout << "Children per node: " << childrenNum << std::endl;

    std::vector<int> kmeansIterationsToTest = { 1, 2, 4, 8, 16, 32, 64 };

    std::cout << "\n";
    std::cout << "Iterations | Build Time (s) | Avg Exact Search (ms) | Speedup vs Linear | Accuracy (%) | Avg Approx Search (ms) | Avg Approx Dist Ratio\n";
    std::cout << "-----------|----------------|-----------------------|-------------------|--------------|------------------------|------------------------\n";

    for (const int testIterations : kmeansIterationsToTest)
    {
        std::cout << std::setw(10) << testIterations << " | ";

        // Build tree with testIterations
        timepoint iterationBuildStart = std::chrono::high_resolution_clock::now();
        BallTree iterationTree(dim, childrenNum);
        BuildBallTree(points, dim, testIterations, childrenNum, gen, iterationTree);
        timepoint iterationBuildEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iterationBuildTime = iterationBuildEnd - iterationBuildStart;

        std::cout << std::fixed << std::setprecision(3) << std::setw(14) << iterationBuildTime.count() << " | ";

        // Test exact search performance
        std::vector<double> exactSearchTimes(numTestQueries);
        int exactCorrectMatches = 0;
        std::vector<float> exactDistances(numTestQueries);

        for (int i = 0; i < numTestQueries; ++i)
        {
            timepoint start = std::chrono::high_resolution_clock::now();
            const std::span<const float> query = testQueries.row_view(i);
            int result = BallTreeSearchNearestNeighbor(iterationTree, points, query, nullptr, -1);
            timepoint end = std::chrono::high_resolution_clock::now();

            exactSearchTimes[i] = std::chrono::duration<double, std::milli>(end - start).count();

            if (result == linearResults[i])
            {
                exactCorrectMatches++;
            }

            // Store distance of found neighbor
            if (result >= 0)
            {
                const std::span<const float> foundPoint = points.row_view(result);
                exactDistances[i] = std::sqrt(DistanceSquared(foundPoint, query));
            }
            else
            {
                exactDistances[i] = std::numeric_limits<float>::max();
            }
        }

        double exactAvgSearchTime = std::accumulate(exactSearchTimes.begin(), exactSearchTimes.end(), 0.0) / numTestQueries;

        // Use average time for comparison
        std::cout << std::setw(21) << exactAvgSearchTime << " | ";

        double exactSpeedup = linearAvgTime / exactAvgSearchTime;
        std::cout << std::setw(17) << exactSpeedup << " | ";

        double exactAccuracyPercent = 100.0 * static_cast<double>(exactCorrectMatches) / numTestQueries;
        std::cout << std::setw(12) << exactAccuracyPercent << " | ";

        // Test approximate search 
        std::vector<double> approxSearchTimes(numTestQueries);
        std::vector<float> approxDistances(numTestQueries);
        double totalDistRatio = 0.0;
        int validApproxQueries = 0;

        for (int i = 0; i < numTestQueries; ++i)
        {
            timepoint start = std::chrono::high_resolution_clock::now();
            const std::span<const float> query = testQueries.row_view(i);
            int result = BallTreeSearchNearestNeighbor(iterationTree, points, query, nullptr, 1000);
            timepoint end = std::chrono::high_resolution_clock::now();

            approxSearchTimes[i] = std::chrono::duration<double, std::milli>(end - start).count();

            // Calculate distance of approximate neighbor
            if (result >= 0)
            {
                const std::span<const float> approxPoint = points.row_view(result);
                const float approxDistSq = DistanceSquared(approxPoint, query);
                approxDistances[i] = std::sqrt(approxDistSq);

                // Calculate ratio to true nearest distance
                const float trueDist = std::sqrt(linearDistances[i]);
                if (trueDist > 0.0f)
                {
                    const float distRatio = approxDistances[i] / trueDist;
                    totalDistRatio += distRatio;
                    validApproxQueries++;
                }
            }
            else
            {
                approxDistances[i] = std::numeric_limits<float>::max();
            }
        }

        double approxAvgSearchTime = std::accumulate(approxSearchTimes.begin(), approxSearchTimes.end(), 0.0) / numTestQueries;
        std::cout << std::setw(23) << approxAvgSearchTime << " | ";

        double avgDistRatio = 1.0;
        if (validApproxQueries > 0)
        {
            avgDistRatio = totalDistRatio / validApproxQueries;
        }
        std::cout << std::setw(22) << avgDistRatio << "\n";
    }

    std::cout << "\nNote: Linear search time (baseline): " << std::fixed << std::setprecision(3) << linearAvgTime << " ms" << std::endl;
    std::cout << "Accuracy is percentage of queries that match linear search ground truth." << std::endl;
    std::cout << "Avg Approx Dist Ratio: ratio of approximate neighbor distance to true nearest neighbor distance (Euclidean)." << std::endl;
    std::cout << "A ratio of 1.0 means perfect quality, higher values indicate worse approximation." << std::endl;
}