#pragma once

// Geno character rendering infrastructure
// Ported from GenoView (orangeduck) - C to C++ conversion

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <assert.h>
#include <cstdio>
#include <vector>

//----------------------------------------------------------------------------------
// Shadow Maps
//----------------------------------------------------------------------------------

struct ShadowLight {
    Vector3 target;
    Vector3 position;
    Vector3 up;
    double width;
    double height;
    double near;
    double far;
};

static inline ShadowLight ShadowLightCreate(Vector3 lightDir, Vector3 target, float size, float nearDist, float farDist)
{
    ShadowLight light{};
    light.target = target;
    light.position = Vector3Add(target, Vector3Scale(lightDir, -farDist * 0.5f));
    light.up = Vector3{ 0.0f, 1.0f, 0.0f };
    light.width = size;
    light.height = size;
    light.near = nearDist;
    light.far = farDist;
    return light;
}

static inline RenderTexture2D LoadShadowMap(int width, int height)
{
    RenderTexture2D target{};
    target.id = rlLoadFramebuffer();
    target.texture.width = width;
    target.texture.height = height;
    assert(target.id);

    rlEnableFramebuffer(target.id);

    target.depth.id = rlLoadTextureDepth(width, height, false);
    target.depth.width = width;
    target.depth.height = height;
    target.depth.format = 19;       // DEPTH_COMPONENT_24BIT
    target.depth.mipmaps = 1;
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0);
    assert(rlFramebufferComplete(target.id));

    rlDisableFramebuffer();

    return target;
}

static inline void UnloadShadowMap(RenderTexture2D target)
{
    if (target.id > 0)
    {
        rlUnloadFramebuffer(target.id);
    }
}

static inline void BeginShadowMap(RenderTexture2D target, ShadowLight shadowLight)
{
    BeginTextureMode(target);
    ClearBackground(WHITE);

    rlDrawRenderBatchActive();

    rlMatrixMode(RL_PROJECTION);
    rlPushMatrix();
    rlLoadIdentity();

    rlOrtho(
        -shadowLight.width / 2, shadowLight.width / 2,
        -shadowLight.height / 2, shadowLight.height / 2,
        shadowLight.near, shadowLight.far);

    rlMatrixMode(RL_MODELVIEW);
    rlLoadIdentity();

    Matrix matView = MatrixLookAt(shadowLight.position, shadowLight.target, shadowLight.up);
    rlMultMatrixf(MatrixToFloat(matView));

    rlEnableDepthTest();
}

static inline void EndShadowMap()
{
    rlDrawRenderBatchActive();

    rlMatrixMode(RL_PROJECTION);
    rlPopMatrix();

    rlMatrixMode(RL_MODELVIEW);
    rlLoadIdentity();

    rlDisableDepthTest();

    EndTextureMode();
}

static inline void SetShaderValueShadowMap(Shader shader, int locIndex, RenderTexture2D target)
{
    if (locIndex > -1)
    {
        rlEnableShader(shader.id);
        const int slot = 10; // Can be anything 0 to 15, but 0 will probably be taken up
        rlActiveTextureSlot(slot);
        rlEnableTexture(target.depth.id);
        rlSetUniform(locIndex, &slot, SHADER_UNIFORM_INT, 1);
    }
}

//----------------------------------------------------------------------------------
// GBuffer (Deferred Rendering)
//----------------------------------------------------------------------------------

struct GBuffer {
    unsigned int id;        // OpenGL framebuffer object id
    Texture color;          // Color buffer attachment texture
    Texture normal;         // Normal buffer attachment texture
    Texture depth;          // Depth buffer attachment texture
};

static inline GBuffer LoadGBuffer(int width, int height)
{
    GBuffer target{};
    target.id = rlLoadFramebuffer();
    assert(target.id);

    rlEnableFramebuffer(target.id);

    target.color.id = rlLoadTexture(NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1);
    target.color.width = width;
    target.color.height = height;
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    target.color.mipmaps = 1;
    rlFramebufferAttach(target.id, target.color.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0);

    target.normal.id = rlLoadTexture(NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1);
    target.normal.width = width;
    target.normal.height = height;
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16;
    target.normal.mipmaps = 1;
    rlFramebufferAttach(target.id, target.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1, RL_ATTACHMENT_TEXTURE2D, 0);

    target.depth.id = rlLoadTextureDepth(width, height, false);
    target.depth.width = width;
    target.depth.height = height;
    target.depth.format = 19;       // DEPTH_COMPONENT_24BIT
    target.depth.mipmaps = 1;
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0);

    assert(rlFramebufferComplete(target.id));

    rlDisableFramebuffer();

    return target;
}

static inline void UnloadGBuffer(GBuffer target)
{
    if (target.id > 0)
    {
        rlUnloadFramebuffer(target.id);
    }
}

static inline void BeginGBuffer(GBuffer target, Camera3D camera)
{
    rlDrawRenderBatchActive();

    rlEnableFramebuffer(target.id);
    rlActiveDrawBuffers(2);

    rlViewport(0, 0, target.color.width, target.color.height);
    rlSetFramebufferWidth(target.color.width);
    rlSetFramebufferHeight(target.color.height);

    ClearBackground(BLACK);

    rlMatrixMode(RL_PROJECTION);
    rlPushMatrix();
    rlLoadIdentity();

    const float aspect = (float)target.color.width / (float)target.color.height;

    if (camera.projection == CAMERA_PERSPECTIVE)
    {
        const double top = rlGetCullDistanceNear() * tan(camera.fovy * 0.5 * DEG2RAD);
        const double right = top * aspect;
        rlFrustum(-right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar());
    }
    else if (camera.projection == CAMERA_ORTHOGRAPHIC)
    {
        const double top = camera.fovy / 2.0;
        const double right = top * aspect;
        rlOrtho(-right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar());
    }

    rlMatrixMode(RL_MODELVIEW);
    rlLoadIdentity();

    Matrix matView = MatrixLookAt(camera.position, camera.target, camera.up);
    rlMultMatrixf(MatrixToFloat(matView));

    rlEnableDepthTest();
}

static inline void EndGBuffer(int windowWidth, int windowHeight)
{
    rlDrawRenderBatchActive();

    rlDisableDepthTest();
    rlActiveDrawBuffers(1);
    rlDisableFramebuffer();

    rlMatrixMode(RL_PROJECTION);
    rlPopMatrix();
    rlLoadIdentity();
    rlOrtho(0, windowWidth, windowHeight, 0, 0.0f, 1.0f);

    rlMatrixMode(RL_MODELVIEW);
    rlLoadIdentity();
}

//----------------------------------------------------------------------------------
// Geno Model and Animation Loading
//----------------------------------------------------------------------------------

static inline Model LoadGenoModel(const char* fileName)
{
    Model model{};
    model.transform = MatrixIdentity();

    FILE* f = fopen(fileName, "rb");
    if (f == NULL)
    {
        TraceLog(LOG_ERROR, "GENO: Unable to read model file %s", fileName);
        return model;
    }

    model.materialCount = 1;
    model.materials = (Material*)RL_CALLOC(model.materialCount, sizeof(Material));
    model.materials[0] = LoadMaterialDefault();

    model.meshCount = 1;
    model.meshes = (Mesh*)RL_CALLOC(model.meshCount, sizeof(Mesh));
    model.meshMaterial = (int*)RL_CALLOC(model.meshCount, sizeof(int));
    model.meshMaterial[0] = 0;

    fread(&model.meshes[0].vertexCount, sizeof(int), 1, f);
    fread(&model.meshes[0].triangleCount, sizeof(int), 1, f);
    fread(&model.boneCount, sizeof(int), 1, f);

    model.meshes[0].boneCount = model.boneCount;
    model.meshes[0].vertices = (float*)RL_CALLOC(model.meshes[0].vertexCount * 3, sizeof(float));
    model.meshes[0].texcoords = (float*)RL_CALLOC(model.meshes[0].vertexCount * 2, sizeof(float));
    model.meshes[0].normals = (float*)RL_CALLOC(model.meshes[0].vertexCount * 3, sizeof(float));
    model.meshes[0].boneIds = (unsigned char*)RL_CALLOC(model.meshes[0].vertexCount * 4, sizeof(unsigned char));
    model.meshes[0].boneWeights = (float*)RL_CALLOC(model.meshes[0].vertexCount * 4, sizeof(float));
    model.meshes[0].indices = (unsigned short*)RL_CALLOC(model.meshes[0].triangleCount * 3, sizeof(unsigned short));
    model.meshes[0].animVertices = (float*)RL_CALLOC(model.meshes[0].vertexCount * 3, sizeof(float));
    model.meshes[0].animNormals = (float*)RL_CALLOC(model.meshes[0].vertexCount * 3, sizeof(float));
    model.bones = (BoneInfo*)RL_CALLOC(model.boneCount, sizeof(BoneInfo));
    model.bindPose = (Transform*)RL_CALLOC(model.boneCount, sizeof(Transform));

    fread(model.meshes[0].vertices, sizeof(float), model.meshes[0].vertexCount * 3, f);
    fread(model.meshes[0].texcoords, sizeof(float), model.meshes[0].vertexCount * 2, f);
    fread(model.meshes[0].normals, sizeof(float), model.meshes[0].vertexCount * 3, f);
    fread(model.meshes[0].boneIds, sizeof(unsigned char), model.meshes[0].vertexCount * 4, f);
    fread(model.meshes[0].boneWeights, sizeof(float), model.meshes[0].vertexCount * 4, f);
    fread(model.meshes[0].indices, sizeof(unsigned short), model.meshes[0].triangleCount * 3, f);
    memcpy(model.meshes[0].animVertices, model.meshes[0].vertices, sizeof(float) * model.meshes[0].vertexCount * 3);
    memcpy(model.meshes[0].animNormals, model.meshes[0].normals, sizeof(float) * model.meshes[0].vertexCount * 3);
    fread(model.bones, sizeof(BoneInfo), model.boneCount, f);
    fread(model.bindPose, sizeof(Transform), model.boneCount, f);
    fclose(f);

    model.meshes[0].boneMatrices = (Matrix*)RL_CALLOC(model.boneCount, sizeof(Matrix));
    for (int i = 0; i < model.boneCount; i++)
    {
        model.meshes[0].boneMatrices[i] = MatrixIdentity();
    }

    UploadMesh(&model.meshes[0], true);

    TraceLog(LOG_INFO, "GENO: Loaded model with %d vertices, %d triangles, %d bones",
        model.meshes[0].vertexCount, model.meshes[0].triangleCount, model.boneCount);

    // Debug: print bind pose for first few bones
    TraceLog(LOG_DEBUG, "GENO: Bind pose for first 3 bones:");
    for (int i = 0; i < model.boneCount && i < 3; i++)
    {
        TraceLog(LOG_DEBUG, "  Bone[%d] '%s': pos=(%.3f, %.3f, %.3f) rot=(%.3f, %.3f, %.3f, %.3f)",
            i, model.bones[i].name,
            model.bindPose[i].translation.x, model.bindPose[i].translation.y, model.bindPose[i].translation.z,
            model.bindPose[i].rotation.x, model.bindPose[i].rotation.y, model.bindPose[i].rotation.z, model.bindPose[i].rotation.w);
    }

    return model;
}

static inline ModelAnimation LoadGenoModelAnimation(const char* fileName)
{
    ModelAnimation animation{};

    FILE* f = fopen(fileName, "rb");
    if (f == NULL)
    {
        TraceLog(LOG_ERROR, "GENO: Unable to read animation file %s", fileName);
        return animation;
    }

    fread(&animation.frameCount, sizeof(int), 1, f);
    fread(&animation.boneCount, sizeof(int), 1, f);

    animation.bones = (BoneInfo*)RL_CALLOC(animation.boneCount, sizeof(BoneInfo));
    fread(animation.bones, sizeof(BoneInfo), animation.boneCount, f);

    animation.framePoses = (Transform**)RL_CALLOC(animation.frameCount, sizeof(Transform*));
    for (int i = 0; i < animation.frameCount; i++)
    {
        animation.framePoses[i] = (Transform*)RL_CALLOC(animation.boneCount, sizeof(Transform));
        fread(animation.framePoses[i], sizeof(Transform), animation.boneCount, f);
    }

    fclose(f);

    TraceLog(LOG_INFO, "GENO: Loaded animation with %d frames, %d bones",
        animation.frameCount, animation.boneCount);

    return animation;
}

// Creates an animation with a single frame containing the bind pose
static inline ModelAnimation LoadEmptyModelAnimation(Model model)
{
    ModelAnimation animation{};
    animation.frameCount = 1;
    animation.boneCount = model.boneCount;

    animation.bones = (BoneInfo*)RL_CALLOC(animation.boneCount, sizeof(BoneInfo));
    memcpy(animation.bones, model.bones, animation.boneCount * sizeof(BoneInfo));

    animation.framePoses = (Transform**)RL_CALLOC(animation.frameCount, sizeof(Transform*));
    for (int i = 0; i < animation.frameCount; i++)
    {
        animation.framePoses[i] = (Transform*)RL_CALLOC(animation.boneCount, sizeof(Transform));
        memcpy(animation.framePoses[i], model.bindPose, animation.boneCount * sizeof(Transform));
    }

    return animation;
}

//----------------------------------------------------------------------------------
// BVH to Geno Animation Mapping
//----------------------------------------------------------------------------------

// Maps BVH joint indices to Geno bone indices
// bvhToGeno[bvhJointIndex] = genoBoneIndex (-1 if no match)
struct BVHGenoMapping {
    std::vector<int> bvhToGeno;  // For each BVH joint, the corresponding Geno bone index
    std::vector<int> genoToBvh;  // For each Geno bone, the corresponding BVH joint index
    bool valid;
};

// Create a mapping between BVH joints and Geno bones by matching names
static inline BVHGenoMapping CreateBVHGenoMapping(const BVHData* bvh, const Model* genoModel)
{
    BVHGenoMapping mapping{};
    mapping.bvhToGeno.resize(bvh->jointCount, -1);
    mapping.genoToBvh.resize(genoModel->boneCount, -1);
    mapping.valid = true;

    // Debug: print first few bone names from each
    TraceLog(LOG_DEBUG, "GENO: First 5 BVH joints:");
    for (int i = 0; i < bvh->jointCount && i < 5; i++)
    {
        TraceLog(LOG_DEBUG, "  BVH[%d]: '%s' (endSite=%d)", i, bvh->joints[i].name.c_str(), bvh->joints[i].endSite);
    }
    TraceLog(LOG_DEBUG, "GENO: First 5 Geno bones:");
    for (int i = 0; i < genoModel->boneCount && i < 5; i++)
    {
        TraceLog(LOG_DEBUG, "  Geno[%d]: '%s'", i, genoModel->bones[i].name);
    }

    // Match by name
    int matchCount = 0;
    for (int b = 0; b < bvh->jointCount; b++)
    {
        // Skip end sites in BVH
        if (bvh->joints[b].endSite) continue;

        for (int g = 0; g < genoModel->boneCount; g++)
        {
            // Case-insensitive comparison: bvh name is std::string now
            if (_stricmp(bvh->joints[b].name.c_str(), genoModel->bones[g].name) == 0)
            {
                mapping.bvhToGeno[b] = g;
                mapping.genoToBvh[g] = b;
                matchCount++;
                TraceLog(LOG_DEBUG, "GENO: Matched BVH '%s' -> Geno '%s'",
                    bvh->joints[b].name.c_str(), genoModel->bones[g].name);
                break;
            }
        }
    }

    TraceLog(LOG_INFO, "GENO: Mapped %d BVH joints to Geno bones (BVH: %d joints, Geno: %d bones)",
        matchCount, bvh->jointCount, genoModel->boneCount);

    if (matchCount == 0)
    {
        TraceLog(LOG_WARNING, "GENO: No bone matches found! Animation will not work.");
    }

    return mapping;
}

// Update a ModelAnimation's frame 0 with transforms from BVH TransformData
// Scale converts from BVH units to Geno units (typically 0.01 for cm to m)
static inline void UpdateGenoAnimationFromBVH(
    ModelAnimation* animation,
    const TransformData* xform,
    const BVHGenoMapping* mapping,
    float scale)
{
    if (!mapping->valid || animation->frameCount < 1) return;

    Transform* poses = animation->framePoses[0];

    // Debug: print Hips transform (bone 0) every ~60 frames
    static int debugCounter = 0;
    debugCounter++;
    //const bool shouldLog = (debugCounter % 60 == 1);
    const bool shouldLog = false;

    for (int g = 0; g < animation->boneCount; g++)
    {
        const int b = mapping->genoToBvh[g];
        if (b >= 0 && b < xform->jointCount)
        {
            // Use global transforms from BVH (already computed by TransformDataSampleFrame)
            poses[g].translation = Vector3Scale(xform->globalPositions[b], scale);
            poses[g].rotation = xform->globalRotations[b];
            poses[g].scale = Vector3{ 1.0f, 1.0f, 1.0f };

            if (shouldLog && g == 0)
            {
                TraceLog(LOG_DEBUG, "GENO: Hips pos=(%.3f, %.3f, %.3f) rot=(%.3f, %.3f, %.3f, %.3f)",
                    poses[g].translation.x, poses[g].translation.y, poses[g].translation.z,
                    poses[g].rotation.x, poses[g].rotation.y, poses[g].rotation.z, poses[g].rotation.w);
            }
        }
    }
}

//----------------------------------------------------------------------------------
// Debug Drawing
//----------------------------------------------------------------------------------

static inline void DrawTransformGizmo(Transform t, float scale)
{
    const Matrix rotMatrix = QuaternionToMatrix(t.rotation);

    DrawLine3D(
        t.translation,
        Vector3Add(t.translation, Vector3{ scale * rotMatrix.m0, scale * rotMatrix.m1, scale * rotMatrix.m2 }),
        RED);

    DrawLine3D(
        t.translation,
        Vector3Add(t.translation, Vector3{ scale * rotMatrix.m4, scale * rotMatrix.m5, scale * rotMatrix.m6 }),
        GREEN);

    DrawLine3D(
        t.translation,
        Vector3Add(t.translation, Vector3{ scale * rotMatrix.m8, scale * rotMatrix.m9, scale * rotMatrix.m10 }),
        BLUE);
}

static inline void DrawModelBindPose(Model model, Color color)
{
    for (int i = 0; i < model.boneCount; i++)
    {
        DrawSphereWires(model.bindPose[i].translation, 0.01f, 4, 6, color);
        DrawTransformGizmo(model.bindPose[i], 0.1f);

        if (model.bones[i].parent != -1)
        {
            DrawLine3D(
                model.bindPose[i].translation,
                model.bindPose[model.bones[i].parent].translation,
                color);
        }
    }
}

static inline void DrawModelAnimationFrameSkeleton(ModelAnimation animation, int frame, Color color)
{
    for (int i = 0; i < animation.boneCount; i++)
    {
        DrawSphereWires(animation.framePoses[frame][i].translation, 0.01f, 4, 6, color);
        DrawTransformGizmo(animation.framePoses[frame][i], 0.1f);

        if (animation.bones[i].parent != -1)
        {
            DrawLine3D(
                animation.framePoses[frame][i].translation,
                animation.framePoses[frame][animation.bones[i].parent].translation,
                color);
        }
    }
}
