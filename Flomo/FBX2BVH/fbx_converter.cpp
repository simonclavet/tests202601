// =============================================================================
// FBX to BVH Converter
// =============================================================================
//
// This converter reads FBX animation files using ufbx and outputs BVH format.
//
// KEY LESSONS LEARNED (what caused problems and how we fixed them):
//
// 1. STATIC REFERENCE NODES:
//    Many FBX files have a hierarchy like: Reference -> Hips -> Spine -> ...
//    The "Reference" node is often a static marker at the origin with NO animation.
//    If we use Reference as the BVH root, we get zero root motion!
//    FIX: Detect nodes without translation animation and skip them, using their
//    first animated child as the actual skeleton root.
//
// 2. UNIT SCALING CONFUSION:
//    - ufbx with target_unit_meters=1.0 converts node_to_world to METERS
//    - BUT node_to_parent and ufbx_evaluate_transform stay in ORIGINAL units (cm)
//    - We initially applied 0.01 scale to everything, causing 100x shrinkage
//    FIX: Only scale offsets (from node_to_parent), NOT root position (from node_to_world)
//
// 3. ROOT POSITION vs ROTATION:
//    - Root POSITION must be WORLD position (absolute coords in scene)
//    - Root ROTATION must be LOCAL rotation (same as all other joints)
//    - We tried using world rotation for root, causing weird wobble
//    FIX: Use node_to_world for position, ufbx_evaluate_transform for rotation
//
// 4. EULER ANGLE DECOMPOSITION:
//    BVH uses ZXY rotation order: R = Rz * Rx * Ry
//    The original decomposition formula was WRONG, causing wobble/foot sliding.
//    For ZXY order with matrix M[row][col]:
//      - sin(X) = M[2][1]  ->  X = asin(r21)
//      - Y = atan2(-M[2][0], M[2][2])  ->  atan2(-r20, r22)
//      - Z = atan2(-M[0][1], M[1][1])  ->  atan2(-r01, r11)
//
// =============================================================================

#include "fbx_converter.h"
#include "raymath.h"
#include "ufbx.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FBXtoBVH::FBXtoBVH(const ConverterConfig& cfg) : config(cfg) {}

FBXtoBVH::~FBXtoBVH() {
    if (scene) ufbx_free_scene(scene);
}

// =============================================================================
// Matrix Conversion
// =============================================================================

// Convert ufbx matrix to Raylib Matrix.
// Both use column-major storage, so we map directly.
// ufbx_matrix is 4x3 (no bottom row), Raylib Matrix is 4x4.
Matrix FBXtoBVH::UfbxToRaylibMatrix(const void* rawMat) const {
    const ufbx_matrix* const m = static_cast<const ufbx_matrix*>(rawMat);

    // ufbx column-major: m00,m10,m20 is first column, etc.
    // Raylib column-major: m0,m1,m2,m3 is first column, etc.
    Matrix result;
    result.m0  = (float)m->m00; result.m4  = (float)m->m01; result.m8  = (float)m->m02; result.m12 = (float)m->m03;
    result.m1  = (float)m->m10; result.m5  = (float)m->m11; result.m9  = (float)m->m12; result.m13 = (float)m->m13;
    result.m2  = (float)m->m20; result.m6  = (float)m->m21; result.m10 = (float)m->m22; result.m14 = (float)m->m23;
    result.m3  = 0.0f;          result.m7  = 0.0f;          result.m11 = 0.0f;          result.m15 = 1.0f;
    return result;
}

// =============================================================================
// Euler Angle Extraction - ZXY Order
// =============================================================================
//
// BVH rotation channels are specified as "Zrotation Xrotation Yrotation"
// This means rotations are applied in order: first Z, then X, then Y
// The combined rotation matrix is: R = Rz(z) * Rx(x) * Ry(y)
//
// DERIVATION:
// Given rotation matrices:
//   Rz = |cos(z) -sin(z) 0|    Rx = |1    0       0   |    Ry = |cos(y)  0 sin(y)|
//        |sin(z)  cos(z) 0|         |0  cos(x) -sin(x)|         |  0     1   0   |
//        |  0       0    1|         |0  sin(x)  cos(x)|         |-sin(y) 0 cos(y)|
//
// Computing R = Rz * Rx * Ry gives us matrix elements where:
//   r21 = sin(x)
//   r20 = -cos(x)*sin(y)
//   r22 = cos(x)*cos(y)
//   r01 = -sin(z)*cos(x)
//   r11 = cos(z)*cos(x)
//
// Therefore:
//   x = asin(r21)
//   y = atan2(-r20, r22)   [when cos(x) != 0]
//   z = atan2(-r01, r11)   [when cos(x) != 0]
//
// WHAT WENT WRONG BEFORE:
// The original code used different formulas (possibly for a different rotation order),
// which caused the skeleton to wobble and have foot sliding artifacts.
//
Vector3 FBXtoBVH::MatrixToEulerZXY(const Matrix& mat) const {
    // Raylib Matrix is column-major:
    //   Column 0: m0, m1, m2   (X basis vector)
    //   Column 1: m4, m5, m6   (Y basis vector)
    //   Column 2: m8, m9, m10  (Z basis vector)
    //   Column 3: m12,m13,m14  (Translation)
    //
    // In standard M[row][col] notation:
    //   r00=m0  r01=m4  r02=m8
    //   r10=m1  r11=m5  r12=m9
    //   r20=m2  r21=m6  r22=m10

    // Normalize columns to remove any scaling that might be present
    const Vector3 col0 = Vector3Normalize({ mat.m0, mat.m1, mat.m2 });
    const Vector3 col1 = Vector3Normalize({ mat.m4, mat.m5, mat.m6 });
    const Vector3 col2 = Vector3Normalize({ mat.m8, mat.m9, mat.m10 });

    // Extract the matrix elements we need
    const float r01 = col1.x;  // m4
    const float r11 = col1.y;  // m5
    const float r20 = col0.z;  // m2
    const float r21 = col1.z;  // m6  <-- This is sin(x) for ZXY order
    const float r22 = col2.z;  // m10
    const float r12 = col2.y;  // m9 (used for gimbal lock case)

    float ex, ey, ez;

    // Extract X rotation: sin(x) = r21
    ex = asin(std::clamp(r21, -1.0f, 1.0f));

    // Check for gimbal lock (when cos(x) approaches 0, i.e., x near ±90°)
    if (std::abs(r21) < 0.99999f) {
        // Normal case: cos(x) != 0, so we can compute y and z
        ey = atan2(-r20, r22);
        ez = atan2(-r01, r11);
    }
    else {
        // Gimbal lock: x = ±90°, y and z become coupled
        // We arbitrarily set y=0 and solve for z
        ey = 0.0f;
        ez = atan2(r12, r22);
    }

    // Return angles in degrees (BVH uses degrees)
    return { ex * RAD2DEG, ey * RAD2DEG, ez * RAD2DEG };
}

// =============================================================================
// FBX Loading
// =============================================================================

bool FBXtoBVH::Load() {
    ufbx_load_opts opts = { 0 };

    // Convert to right-handed Y-up coordinate system (standard for BVH)
    opts.target_axes = ufbx_axes_right_handed_y_up;

    // IMPORTANT: This makes ufbx convert node_to_world positions to meters.
    // However, node_to_parent and ufbx_evaluate_transform remain in original FBX units!
    // This asymmetry caused us confusion with scaling.
    opts.target_unit_meters = 1.0f;

    ufbx_error error;
    scene = ufbx_load_file(config.inputFile.c_str(), &opts, &error);

    if (!scene) {
        std::cerr << "[Error] " << error.description.data << std::endl;
        return false;
    }
    return true;
}

// =============================================================================
// Skeleton Building
// =============================================================================

// Check if a node has translation animation in any animation stack.
// Used to detect static "Reference" nodes that should be skipped.
bool FBXtoBVH::NodeHasTranslationAnimation(const ufbx_node* node) const {
    for (size_t stack_idx = 0; stack_idx < scene->anim_stacks.count; stack_idx++) {
        const ufbx_anim* const anim = scene->anim_stacks.data[stack_idx]->anim;
        for (size_t layer_idx = 0; layer_idx < anim->layers.count; layer_idx++) {
            const ufbx_anim_layer* const layer = anim->layers.data[layer_idx];
            for (size_t prop_idx = 0; prop_idx < layer->anim_props.count; prop_idx++) {
                const ufbx_anim_prop* const aprop = &layer->anim_props.data[prop_idx];
                if (aprop->element == &node->element) {
                    const std::string propName(aprop->prop_name.data, aprop->prop_name.length);
                    if (propName == "Lcl Translation") {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Recursively collect nodes into our joint list
void FBXtoBVH::CollectNodes(ufbx_node* node, int parentIdx) {
    Joint j;
    j.name = std::string(node->name.data, node->name.length);
    j.ufbxNode = node;
    j.index = (int)joints.size();
    j.parentIndex = parentIdx;
    j.isEndSite = false;
    j.globalBindPose = UfbxToRaylibMatrix(&node->node_to_world);

    joints.push_back(j);
    const int currentIdx = j.index;

    if (parentIdx >= 0) {
        joints[parentIdx].children.push_back(currentIdx);
    }

    for (size_t i = 0; i < node->children.count; ++i) {
        CollectNodes(node->children.data[i], currentIdx);
    }
}

// Build the skeleton hierarchy from the FBX scene.
//
// PROBLEM WE SOLVED:
// Many FBX files (especially from motion capture) have a structure like:
//   Scene Root
//     └─ Reference (static, identity transform, NO animation)
//         └─ Hips (actual skeleton root with translation + rotation animation)
//             └─ Spine, Legs, etc.
//
// If we naively use "Reference" as our BVH root, the root position is always (0,0,0)
// because Reference has no animation! The actual root motion is on "Hips".
//
// SOLUTION:
// Detect if a top-level node has no translation animation. If so, skip it and
// use its children as the skeleton roots instead. This makes "Hips" our BVH root.
//
bool FBXtoBVH::BuildSkeleton() {
    joints.clear();

    // Collect skeleton nodes, skipping static reference nodes
    for (size_t i = 0; i < scene->root_node->children.count; ++i) {
        ufbx_node* const candidate = scene->root_node->children.data[i];

        // Check if this is a static reference node (no translation animation)
        // If so, skip it and use its children as roots instead
        if (!NodeHasTranslationAnimation(candidate) && candidate->children.count > 0) {
            std::cout << "[Info] Skipping static reference node: " << candidate->name.data << "\n";
            for (size_t j = 0; j < candidate->children.count; ++j) {
                CollectNodes(candidate->children.data[j], -1);
            }
        } else {
            CollectNodes(candidate, -1);
        }
    }

    if (joints.empty()) return false;

    // =========================================================================
    // Calculate offsets for each joint (inlined from CalculateOffsets)
    // =========================================================================
    // SCALING GOTCHA:
    // We use ufbx's node_to_parent for local offsets. However, even though we set
    // target_unit_meters=1.0, node_to_parent stays in ORIGINAL FBX units (usually cm).
    // Only node_to_world gets converted to meters.
    //
    // So we must scale offsets by 0.01 to convert cm -> meters.
    // But we must NOT scale the root's animated world position (already in meters).
    //
    for (Joint& j : joints) {
        if (j.parentIndex == -1) {
            // Root: offset is the bind pose world position (already in meters from node_to_world)
            j.offset = { j.globalBindPose.m12, j.globalBindPose.m13, j.globalBindPose.m14 };
        }
        else {
            // Child joints: use node_to_parent for local offset (in original FBX units, typically cm)
            const Matrix localMat = UfbxToRaylibMatrix(&j.ufbxNode->node_to_parent);
            j.offset = { localMat.m12, localMat.m13, localMat.m14 };
        }

        // Scale offset from centimeters to meters
        j.offset = Vector3Scale(j.offset, config.offsetScale);
    }

    // =========================================================================
    // Add end sites to leaf joints (inlined from AddEndSites)
    // =========================================================================
    // BVH format requires end sites for leaf joints to define bone length
    //
    const size_t jointCount = joints.size();
    for (size_t i = 0; i < jointCount; ++i) {
        if (joints[i].children.empty()) {
            Joint end;
            end.name = joints[i].name + "_End";
            end.parentIndex = (int)i;
            end.isEndSite = true;
            end.offset = { 0.0f, 0.05f, 0.0f };  // 5cm end site pointing up

            joints[i].children.push_back((int)joints.size());
            joints.push_back(end);
        }
    }

    return true;
}

// =============================================================================
// Animation Extraction
// =============================================================================
//
// KEY INSIGHT: Root position vs rotation have different sources!
//
// For the ROOT joint:
//   - POSITION: Must be WORLD position (where the character is in the scene)
//               We get this from node_to_world which ufbx converts to meters.
//
//   - ROTATION: Must be LOCAL rotation (same as all other joints)
//               We get this from ufbx_evaluate_transform().
//
// WHAT WENT WRONG:
// We initially tried using WORLD rotation for the root (extracted from node_to_world).
// This caused strange wobbling because the world rotation includes the parent's rotation,
// but for a skeleton root whose parent is at identity, this should be the same...
// except ufbx may apply coordinate system adjustments that differ between
// node_to_world and ufbx_evaluate_transform.
//
// Using LOCAL rotation for the root (from ufbx_evaluate_transform) fixed the wobble.
//
bool FBXtoBVH::ExtractAnimations() {
    if (scene->anim_stacks.count == 0) return false;

    ufbx_anim_stack* const stack = scene->anim_stacks.data[0];

    AnimationClip clip;
    clip.name = std::string(stack->name.data, stack->name.length);

    const double duration = stack->time_end - stack->time_begin;
    clip.frameCount = (int)(duration * config.frameRate);
    if (clip.frameCount < 1) clip.frameCount = 1;

    // Find the root joint (first joint with no parent that isn't an end site)
    int rootJointIdx = -1;
    for (const Joint& j : joints) {
        if (j.parentIndex == -1 && !j.isEndSite) {
            rootJointIdx = j.index;
            break;
        }
    }

    // Extract animation frame by frame
    for (int f = 0; f < clip.frameCount; ++f) {
        const double time = stack->time_begin + (double)f * (1.0 / config.frameRate);

        AnimationFrame frame;
        frame.time = time;
        // Pre-size rotations vector so we can index by joint index directly.
        // This is important because SaveBVH outputs rotations in hierarchy order,
        // which may differ from joints vector order when there are multiple roots.
        frame.rotations.resize(joints.size());

        // Evaluate the entire scene at this time.
        // This gives us node_to_world matrices for getting world positions.
        ufbx_error error;
        ufbx_scene* const evalScene = ufbx_evaluate_scene(scene, stack->anim, time, nullptr, &error);
        if (!evalScene) {
            std::cerr << "[Error] Failed to evaluate scene at time " << time << std::endl;
            continue;
        }

        for (const Joint& j : joints) {
            if (j.isEndSite) continue;

            // Find the corresponding node in the evaluated scene
            // (nodes maintain the same typed_id index)
            const ufbx_node* const evalNode = evalScene->nodes.data[j.ufbxNode->typed_id];

            // =====================================================
            // ROTATION: Use LOCAL transform for ALL joints (including root!)
            // =====================================================
            // BVH expects local rotations - the rotation applied at this joint
            // relative to its parent. For the root, this is relative to world origin.
            //
            // We use ufbx_evaluate_transform which returns the animated local transform.
            // Then convert the quaternion to a rotation matrix for Euler extraction.
            //
            const ufbx_transform xform = ufbx_evaluate_transform(stack->anim, j.ufbxNode, time);
            const Quaternion q = {
                (float)xform.rotation.x,
                (float)xform.rotation.y,
                (float)xform.rotation.z,
                (float)xform.rotation.w
            };
            const Matrix rotMat = QuaternionToMatrix(q);
            // Store rotation indexed by joint index (not push_back!)
            frame.rotations[j.index] = MatrixToEulerZXY(rotMat);

            // =====================================================
            // POSITION: Only the ROOT has position channels in BVH
            // =====================================================
            // The root position is the WORLD position of the character.
            // We get this from node_to_world which ufbx has already converted to meters.
            //
            // DO NOT apply additional scaling here - node_to_world is already in meters!
            // (This was a bug: we were scaling by 0.01 again, making motion 100x too small)
            //
            if (j.index == rootJointIdx) {
                const Vector3 pos = {
                    (float)evalNode->node_to_world.m03,  // Translation X
                    (float)evalNode->node_to_world.m13,  // Translation Y
                    (float)evalNode->node_to_world.m23   // Translation Z
                };
                frame.rootPosition = pos;
            }
        }

        ufbx_free_scene(evalScene);
        clip.frames.push_back(frame);
    }

    animations.push_back(clip);
    return true;
}

// =============================================================================
// BVH Output
// =============================================================================

// Collect joint indices in hierarchy traversal order (matching WriteHierarchy order).
// This ensures motion data channels align with hierarchy channel declarations.
// IMPORTANT: When FBX has multiple skeleton roots (e.g., Hips + props), we only
// write one root's hierarchy, so we must only output that root's joint rotations.
static void CollectHierarchyOrder(
    const std::vector<Joint>& joints,
    int idx,
    std::vector<int>& outOrder)
{
    const Joint& j = joints[idx];
    if (j.isEndSite) return;  // End sites have no channels

    outOrder.push_back(idx);
    for (const int child : j.children) {
        CollectHierarchyOrder(joints, child, outOrder);
    }
}

void FBXtoBVH::SaveBVH() const {
    if (animations.empty()) return;

    std::ofstream file(config.outputFile);
    if (!file) return;

    file << "HIERARCHY\n";

    // Find root joint
    int rootIdx = -1;
    for (const Joint& j : joints) {
        if (j.parentIndex == -1) {
            rootIdx = j.index;
            break;
        }
    }

    WriteHierarchy(file, rootIdx, 0);

    // Collect joint indices in the same order as hierarchy was written.
    // This is critical when FBX has multiple roots - we only output
    // channels for joints that are part of the written hierarchy.
    std::vector<int> hierarchyOrder;
    CollectHierarchyOrder(joints, rootIdx, hierarchyOrder);

    // Write motion data
    const AnimationClip& clip = animations[0];
    file << "MOTION\n";
    file << "Frames: " << clip.frameCount << "\n";
    file << "Frame Time: " << std::fixed << std::setprecision(6) << (1.0 / config.frameRate) << "\n";

    for (const AnimationFrame& f : clip.frames) {
        // Root position (X Y Z)
        file << f.rootPosition.x << " " << f.rootPosition.y << " " << f.rootPosition.z << " ";

        // Joint rotations in hierarchy order (ZXY)
        // Only output rotations for joints in the written hierarchy
        for (const int jointIdx : hierarchyOrder) {
            const Vector3& r = f.rotations[jointIdx];
            file << r.z << " " << r.x << " " << r.y << " ";
        }
        file << "\n";
    }

    std::cout << "Saved: " << config.outputFile << std::endl;
}

// Write hierarchy recursively
void FBXtoBVH::WriteHierarchy(std::ofstream& file, int idx, int depth) const {
    const Joint& j = joints[idx];
    const std::string tab(depth * 2, ' ');

    if (j.isEndSite) {
        file << tab << "End Site\n" << tab << "{\n";
        file << tab << "  OFFSET " << j.offset.x << " " << j.offset.y << " " << j.offset.z << "\n";
        file << tab << "}\n";
        return;
    }

    // ROOT vs JOINT keyword
    file << tab << (j.parentIndex == -1 ? "ROOT " : "JOINT ") << j.name << "\n";
    file << tab << "{\n";
    file << tab << "  OFFSET " << j.offset.x << " " << j.offset.y << " " << j.offset.z << "\n";

    // Root has 6 channels (position + rotation), children have 3 (rotation only)
    if (j.parentIndex == -1)
        file << tab << "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n";
    else
        file << tab << "  CHANNELS 3 Zrotation Xrotation Yrotation\n";

    // Recurse to children
    for (const int child : j.children) {
        WriteHierarchy(file, child, depth + 1);
    }

    file << tab << "}\n";
}
