#pragma once

#include "raylib.h"
#include <string>
#include <vector>

// Forward declaration of ufbx types to avoid leaking dependencies in header
struct ufbx_node;
struct ufbx_scene;

struct ConverterConfig {
    std::string inputFile;
    std::string outputFile;
    float frameRate = 30.0f;
    // NOTE: ufbx converts node_to_world to meters, but node_to_parent stays in original units (cm)
    // So offsets need scaling, but animated root position doesn't
    float offsetScale = 0.01f;  // cm -> meters for offsets
};

struct Joint {
    std::string name;
    ufbx_node* ufbxNode = nullptr;
    int index = 0;
    int parentIndex = -1;
    Vector3 offset = {};
    std::vector<int> children;
    bool isEndSite = false;
    Matrix globalBindPose = {};
};

struct AnimationFrame {
    std::vector<Vector3> rotations;  // Euler ZXY in Degrees
    Vector3 rootPosition = {};
    double time = 0.0;
};

struct AnimationClip {
    std::string name;
    std::vector<AnimationFrame> frames;
    int frameCount = 0;
};

class FBXtoBVH {
private:
    const ConverterConfig config;
    ufbx_scene* scene = nullptr;
    std::vector<Joint> joints;
    std::vector<AnimationClip> animations;

    // Helpers
    void CollectNodes(ufbx_node* node, int parentIdx);
    void WriteHierarchy(std::ofstream& file, int idx, int depth) const;
    bool NodeHasTranslationAnimation(const ufbx_node* node) const;

    // Math Helpers (const - don't modify object state)
    Vector3 MatrixToEulerZXY(const Matrix& mat) const;
    Matrix UfbxToRaylibMatrix(const void* ufbxMat) const;

public:
    explicit FBXtoBVH(const ConverterConfig& cfg);
    ~FBXtoBVH();

    // Non-copyable (owns ufbx_scene pointer)
    FBXtoBVH(const FBXtoBVH&) = delete;
    FBXtoBVH& operator=(const FBXtoBVH&) = delete;

    bool Load();
    bool BuildSkeleton();
    bool ExtractAnimations();
    void SaveBVH() const;
};
