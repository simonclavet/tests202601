

#pragma once

// =============================================================================
// FBX Loader - Optimized for Real-time Animation Loading
// =============================================================================
//
// This loader reads FBX files using ufbx and populates the BVHData structure.
// It is designed to be a seamless replacement for standard BVH loading.
//
// OPTIMIZATION NOTES:
// 1. Scene Evaluation: We avoid `ufbx_evaluate_scene` inside the frame loop.
//    That function re-allocates the entire scene graph every frame. Instead,
//    we use `ufbx_evaluate_transform` on specific nodes, which is 0-allocation.
//
// 2. Unit Scaling: FBX is often in Centimeters. BVH is usually in Meters.
//    We apply an `offsetScale` (0.01) to translations to align them.
//
// 3. Static Nodes: We detect and skip "Reference" nodes often found in Mixamo
//    exports to ensure the skeleton structure matches standard BVH rigs.
//
// =============================================================================

#include "bvh_parser.h"
#include "raylib.h"
#include "raymath.h"
#include "ufbx.h"

#include <vector>
#include <string>
#include <cmath>
#include <functional>

// -----------------------------------------------------------------------------
// Profiling Macros
// -----------------------------------------------------------------------------
// Define FBX_PROFILING to enable console timing logs
#define FBX_PROFILING

#ifdef FBX_PROFILING
#include <stdio.h>
#define FBX_PROFILE_START(var_name) double var_name = GetTime()
#define FBX_PROFILE_END(var_name, section_name) \
    do { \
        double elapsed = GetTime() - var_name; \
        printf("[FBX Profiling] %s: %.3f ms\n", section_name, elapsed * 1000.0); \
    } while(0)
#else
#define FBX_PROFILE_START(var_name)
#define FBX_PROFILE_END(var_name, section_name)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// Internal Structures & Helpers
// -----------------------------------------------------------------------------

struct FBXJoint {
    std::string name;
    ufbx_node* node = nullptr;
    int index = 0;
    int parentIndex = -1;
    Vector3 offset = {};
    std::vector<int> children;
    bool isEndSite = false;
};

// Check if a node has translation animation.
// This is used to distinguish between actual root joints (hips) and static
// container nodes (like "Reference" or "World") often found in FBX exports.
static bool FBXNodeHasTranslationAnimation(const ufbx_scene* scene, const ufbx_node* node)
{
    for (size_t stack_idx = 0; stack_idx < scene->anim_stacks.count; ++stack_idx) {
        const ufbx_anim* anim = scene->anim_stacks.data[stack_idx]->anim;
        for (size_t layer_idx = 0; layer_idx < anim->layers.count; ++layer_idx) {
            const ufbx_anim_layer* layer = anim->layers.data[layer_idx];
            for (size_t prop_idx = 0; prop_idx < layer->anim_props.count; ++prop_idx) {
                const ufbx_anim_prop* aprop = &layer->anim_props.data[prop_idx];
                if (aprop->element == &node->element) {
                    if (aprop->prop_name.length >= 15 &&
                        strncmp(aprop->prop_name.data, "Lcl Translation", 15) == 0) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Extract ZXY Euler angles from rotation matrix (in degrees)
// BVH standard order is ZXY: R = Rz * Rx * Ry
static Vector3 FBXMatrixToEulerZXY(const Matrix& mat)
{
    // Normalize columns to remove scaling artifacts before rotation extraction
    const Vector3 col0 = Vector3Normalize(Vector3{ mat.m0, mat.m1, mat.m2 });
    const Vector3 col1 = Vector3Normalize(Vector3{ mat.m4, mat.m5, mat.m6 });
    const Vector3 col2 = Vector3Normalize(Vector3{ mat.m8, mat.m9, mat.m10 });

    // Extract elements
    const float r01 = col1.x; // M[0][1]
    const float r11 = col1.y; // M[1][1]
    const float r20 = col0.z; // M[2][0]
    const float r21 = col1.z; // M[2][1]
    const float r22 = col2.z; // M[2][2]

    // ZXY Formula: sin(X) = M[2][1]
    float ex = asinf(fmaxf(-1.0f, fminf(1.0f, r21)));
    float ey, ez;

    if (fabsf(r21) < 0.99999f) {
        ey = atan2f(-r20, r22);
        ez = atan2f(-r01, r11);
    }
    else {
        // Gimbal lock case
        const float r12 = col2.y;
        ey = 0.0f;
        ez = atan2f(r12, r22);
    }

    return Vector3{
        ex * (float)(180.0 / M_PI),
        ey * (float)(180.0 / M_PI),
        ez * (float)(180.0 / M_PI)
    };
}

// Recursively traverse ufbx nodes to build our flat FBXJoint list
static void FBXCollectNodes(
    ufbx_node* node,
    int parentIdx,
    std::vector<FBXJoint>& joints,
    float offsetScale)
{
    FBXJoint j;
    j.name = std::string(node->name.data, node->name.length);
    j.node = node;
    j.index = (int)joints.size();
    j.parentIndex = parentIdx;
    j.isEndSite = false;

    // Calculate static skeleton offsets
    if (parentIdx == -1) {
        // Root: ufbx converts root to meters based on load opts, but we apply scale 
        // to ensure it matches specific BVH expectations if needed.
        j.offset.x = (float)node->node_to_world.m03 * offsetScale;
        j.offset.y = (float)node->node_to_world.m13 * offsetScale;
        j.offset.z = (float)node->node_to_world.m23 * offsetScale;
    }
    else {
        // Child: ufbx leaves children in local units (often cm).
        // We scale this to meters (0.01).
        j.offset.x = (float)node->node_to_parent.m03 * offsetScale;
        j.offset.y = (float)node->node_to_parent.m13 * offsetScale;
        j.offset.z = (float)node->node_to_parent.m23 * offsetScale;
    }

    joints.push_back(j);
    const int currentIdx = j.index;

    if (parentIdx >= 0) {
        joints[parentIdx].children.push_back(currentIdx);
    }

    for (size_t i = 0; i < node->children.count; ++i) {
        FBXCollectNodes(node->children.data[i], currentIdx, joints, offsetScale);
    }
}

// Create a depth-first ordering for writing motion data
static void FBXCollectHierarchyOrder(
    const std::vector<FBXJoint>& joints,
    int idx,
    std::vector<int>& outOrder)
{
    const FBXJoint& j = joints[idx];
    if (j.isEndSite) return;

    outOrder.push_back(idx);
    for (int child : j.children) {
        FBXCollectHierarchyOrder(joints, child, outOrder);
    }
}

// -----------------------------------------------------------------------------
// Main Loader Function
// -----------------------------------------------------------------------------

static bool FBXDataLoad(BVHData* bvh, const char* filename, char* errMsg, int errMsgSize)
{
#ifdef FBX_PROFILING
    double total_start = GetTime();
    printf("\n=== FBX Loader Profiling ===\n");
    printf("Loading file: %s\n", filename);
#endif

    // 1. Load FBX File
    FBX_PROFILE_START(load_time);

    ufbx_load_opts opts = {};
    opts.target_axes = ufbx_axes_right_handed_y_up; // Force Y-up (Raylib/OpenGL standard)
    opts.target_unit_meters = 1.0f;                 // Request meters from ufbx

    ufbx_error error;
    ufbx_scene* scene = ufbx_load_file(filename, &opts, &error);

    FBX_PROFILE_END(load_time, "File Loading");

    if (!scene) {
        snprintf(errMsg, errMsgSize, "Error loading FBX: %s", error.description.data);
        return false;
    }

    // Settings
    const float frameRate = 30.0f;
    const float offsetScale = 0.01f; // Scale factor: Centimeters -> Meters

    // 2. Build Skeleton Hierarchy
    FBX_PROFILE_START(skeleton_build_time);
    std::vector<FBXJoint> joints;

    // Iterate root children to find the actual skeleton root
    for (size_t i = 0; i < scene->root_node->children.count; ++i) {
        ufbx_node* candidate = scene->root_node->children.data[i];

        // Heuristic: If it has no translation anim but has children, it's likely a container.
        // Dive one level deeper to find the hips.
        if (!FBXNodeHasTranslationAnimation(scene, candidate) && candidate->children.count > 0) {
            for (size_t j = 0; j < candidate->children.count; ++j) {
                FBXCollectNodes(candidate->children.data[j], -1, joints, offsetScale);
            }
        }
        else {
            FBXCollectNodes(candidate, -1, joints, offsetScale);
        }
    }
    FBX_PROFILE_END(skeleton_build_time, "Skeleton Building");

    if (joints.empty()) {
        snprintf(errMsg, errMsgSize, "Error: No skeleton found in FBX file");
        ufbx_free_scene(scene);
        return false;
    }

    // 3. Add End Sites (Leaf Nodes)
    // BVH requires End Sites for leaf joints to define bone length
    FBX_PROFILE_START(endsites_time);
    size_t jointCountBeforeEndSites = joints.size();
    for (size_t i = 0; i < jointCountBeforeEndSites; ++i) {
        if (joints[i].children.empty()) {
            FBXJoint end;
            end.name = joints[i].name + "_End";
            end.node = nullptr;
            end.index = (int)joints.size();
            end.parentIndex = (int)i;
            end.offset = Vector3{ 0.0f, 0.05f, 0.0f }; // Arbitrary small offset
            end.isEndSite = true;
            joints[i].children.push_back(end.index);
            joints.push_back(end);
        }
    }
    FBX_PROFILE_END(endsites_time, "End Sites Creation");

    // 4. Find Root & Prepare BVH Structure
    int rootIdx = -1;
    for (const FBXJoint& j : joints) {
        if (j.parentIndex == -1 && !j.isEndSite) {
            rootIdx = j.index;
            break;
        }
    }

    if (rootIdx < 0) {
        snprintf(errMsg, errMsgSize, "Error: No root joint found");
        ufbx_free_scene(scene);
        return false;
    }

    // 5. Hierarchy flattening
    FBX_PROFILE_START(hierarchy_time);
    std::vector<int> hierarchyOrder;
    FBXCollectHierarchyOrder(joints, rootIdx, hierarchyOrder);
    FBX_PROFILE_END(hierarchy_time, "Hierarchy Order Collection");

    // Initialize BVH Data
    BVHDataFree(bvh);
    BVHDataInit(bvh);

    // 6. Allocate & Map Joints
    FBX_PROFILE_START(joint_allocation_time);

    // Collect ALL joints including end sites for the linear buffer
    std::vector<int> fullHierarchyOrder;
    std::function<void(int)> collectAll = [&](int idx) {
        fullHierarchyOrder.push_back(idx);
        for (int child : joints[idx].children) collectAll(child);
        };
    collectAll(rootIdx);

    bvh->jointCount = (int)fullHierarchyOrder.size();
    bvh->joints.clear();
    bvh->joints.resize(bvh->jointCount);
    for (int i = 0; i < bvh->jointCount; ++i)
    {
        BVHJointDataInit(&bvh->joints[i]);
    }

    // Mapping from FBX list index to BVH array index
    std::vector<int> fbxToBvhIdx(joints.size(), -1);
    for (int i = 0; i < (int)fullHierarchyOrder.size(); ++i) {
        fbxToBvhIdx[fullHierarchyOrder[i]] = i;
    }
    FBX_PROFILE_END(joint_allocation_time, "Joint Allocation");

    // 7. Populate Static Joint Data
    FBX_PROFILE_START(joint_population_time);
    for (int i = 0; i < bvh->jointCount; ++i) {
        const FBXJoint& fj = joints[fullHierarchyOrder[i]];
        BVHJointData* bj = &bvh->joints[i];
        BVHJointDataInit(bj);

        BVHJointDataRename(bj, fj.name.c_str());
        bj->offset = fj.offset;
        bj->endSite = fj.isEndSite;
        bj->parent = (fj.parentIndex >= 0) ? fbxToBvhIdx[fj.parentIndex] : -1;

        if (fj.isEndSite) {
            bj->channelCount = 0;
        }
        else if (fj.parentIndex == -1) {
            // Root has 6DoF (Position + Rotation)
            bj->channelCount = 6;
            bj->channels[0] = CHANNEL_X_POSITION; bj->channels[1] = CHANNEL_Y_POSITION; bj->channels[2] = CHANNEL_Z_POSITION;
            bj->channels[3] = CHANNEL_Z_ROTATION; bj->channels[4] = CHANNEL_X_ROTATION; bj->channels[5] = CHANNEL_Y_ROTATION;
        }
        else {
            // Children have 3DoF (Rotation only)
            bj->channelCount = 3;
            bj->channels[0] = CHANNEL_Z_ROTATION; bj->channels[1] = CHANNEL_X_ROTATION; bj->channels[2] = CHANNEL_Y_ROTATION;
        }
    }
    FBX_PROFILE_END(joint_population_time, "Joint Data Population");

    // Calculate total channels for memory allocation
    bvh->channelCount = 0;
    for (int i = 0; i < bvh->jointCount; ++i) bvh->channelCount += bvh->joints[i].channelCount;

    // 8. Animation Extraction
    if (scene->anim_stacks.count == 0) {
        snprintf(errMsg, errMsgSize, "Error: No animation found in FBX file");
        ufbx_free_scene(scene);
        return false;
    }

    ufbx_anim_stack* stack = scene->anim_stacks.data[0];
    const double duration = stack->time_end - stack->time_begin;
    bvh->frameCount = (int)(duration * frameRate);
    if (bvh->frameCount < 1) bvh->frameCount = 1;
    bvh->frameTime = 1.0f / frameRate;

    bvh->motionData.clear();
    bvh->motionData.resize((size_t)bvh->frameCount* (size_t)bvh->channelCount);

    FBX_PROFILE_START(animation_extraction_time);

    int total_frames = bvh->frameCount;
    int processed_frames = 0;

    // --- ANIMATION LOOP OPTIMIZED ---
    // We strictly avoid ufbx_evaluate_scene() here. 
    // Instead we query local transforms per node, which is significantly faster.

    for (int f = 0; f < bvh->frameCount; ++f) {
        const double time = stack->time_begin + (double)f / frameRate;
        int channelOffset = 0;

        for (int hi = 0; hi < (int)hierarchyOrder.size(); ++hi) {
            const FBXJoint& fj = joints[hierarchyOrder[hi]];
            if (fj.isEndSite) continue;

            // Evaluate LOCAL transform for this specific node at this time
            const ufbx_transform xform = ufbx_evaluate_transform(stack->anim, fj.node, time);

            // Convert Rotation (Quaternion -> Matrix -> ZXY Euler)
            Quaternion q = { (float)xform.rotation.x, (float)xform.rotation.y, (float)xform.rotation.z, (float)xform.rotation.w };
            Matrix rotMat = QuaternionToMatrix(q);
            Vector3 euler = FBXMatrixToEulerZXY(rotMat);

            if (fj.parentIndex == -1) {
                // ROOT JOINT: Handle Position + Rotation
                // IMPORTANT: We use xform.translation (Local). For a root, Local == World 
                // (usually). We multiply by offsetScale to convert raw FBX units (cm) to Meters.
                bvh->motionData[f * bvh->channelCount + channelOffset + 0] = (float)xform.translation.x * offsetScale;
                bvh->motionData[f * bvh->channelCount + channelOffset + 1] = (float)xform.translation.y * offsetScale;
                bvh->motionData[f * bvh->channelCount + channelOffset + 2] = (float)xform.translation.z * offsetScale;

                bvh->motionData[f * bvh->channelCount + channelOffset + 3] = euler.z;
                bvh->motionData[f * bvh->channelCount + channelOffset + 4] = euler.x;
                bvh->motionData[f * bvh->channelCount + channelOffset + 5] = euler.y;
                channelOffset += 6;
            }
            else {
                // CHILD JOINTS: Rotation Only
                bvh->motionData[f * bvh->channelCount + channelOffset + 0] = euler.z;
                bvh->motionData[f * bvh->channelCount + channelOffset + 1] = euler.x;
                bvh->motionData[f * bvh->channelCount + channelOffset + 2] = euler.y;
                channelOffset += 3;
            }
        }
        processed_frames++;
    }

    FBX_PROFILE_END(animation_extraction_time, "Animation Extraction");

#ifdef FBX_PROFILING
    printf("Processed %d/%d frames\n", processed_frames, total_frames);
#endif

    // 9. Cleanup
    FBX_PROFILE_START(cleanup_time);
    ufbx_free_scene(scene);
    FBX_PROFILE_END(cleanup_time, "Scene Cleanup");

#ifdef FBX_PROFILING
    double total_elapsed = GetTime() - total_start;
    printf("Total FBX loading time: %.3f ms\n", total_elapsed * 1000.0);
    printf("=== End Profiling ===\n\n");
#endif

    return true;
}