#pragma once

// =============================================================================
// Capsule Data - Skeletal capsule rendering with AO and shadows
// =============================================================================
//
// Manages capsule representations of skeleton bones for rendering.
// Handles ambient occlusion and shadow calculations between capsules.
//

#include "raylib.h"
#include "raymath.h"
#include "math_utils.h"
#include "transform_data.h"

#include <vector>
#include <algorithm>

// =============================================================================
// Sorting helper
// =============================================================================

struct CapsuleSort {
    int index;
    float value;
};

static int CapsuleSortCompareGreater(const void* lhs, const void* rhs)
{
    const CapsuleSort* a = (const CapsuleSort*)lhs;
    const CapsuleSort* b = (const CapsuleSort*)rhs;
    return a->value > b->value ? 1 : -1;
}

static int CapsuleSortCompareLess(const void* lhs, const void* rhs)
{
    const CapsuleSort* a = (const CapsuleSort*)lhs;
    const CapsuleSort* b = (const CapsuleSort*)rhs;
    return a->value < b->value ? 1 : -1;
}

// =============================================================================
// CapsuleData structure
// =============================================================================

struct CapsuleData {
    // Main capsule data for rendering
    int capsuleCount;
    std::vector<Vector3> capsulePositions;
    std::vector<Quaternion> capsuleRotations;
    std::vector<float> capsuleRadii;
    std::vector<float> capsuleHalfLengths;
    std::vector<Vector3> capsuleColors;
    std::vector<float> capsuleOpacities;
    std::vector<CapsuleSort> capsuleSort;

    // Ambient occlusion capsules
    int aoCapsuleCount;
    std::vector<Vector3> aoCapsuleStarts;
    std::vector<Vector3> aoCapsuleVectors;
    std::vector<float> aoCapsuleRadii;
    std::vector<CapsuleSort> aoCapsuleSort;

    // Shadow casting capsules
    int shadowCapsuleCount;
    std::vector<Vector3> shadowCapsuleStarts;
    std::vector<Vector3> shadowCapsuleVectors;
    std::vector<float> shadowCapsuleRadii;
    std::vector<CapsuleSort> shadowCapsuleSort;

    // AO lookup table texture
    Image aoLookupImage;
    Texture2D aoLookupTable;
    Vector2 aoLookupResolution;

    // Shadow lookup table texture
    Image shadowLookupImage;
    Texture2D shadowLookupTable;
    Vector2 shadowLookupResolution;
};

// =============================================================================
// Lookup table generation
// =============================================================================

static void CapsuleDataUpdateAOLookupTable(CapsuleData* data)
{
    const int width = (int)data->aoLookupResolution.x;
    const int height = (int)data->aoLookupResolution.y;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const float nlAngle = ((float)x / (width - 1)) * PI;
            const float h = 1.0f + (AO_RATIO_MAX - 1.0f) * ((float)y / (height - 1));
            ((unsigned char*)data->aoLookupImage.data)[y * width + x] =
                (unsigned char)Clamp(255.0f * SphereOcclusionLookup(nlAngle, h), 0.0f, 255.0f);
        }
    }
    UpdateTexture(data->aoLookupTable, data->aoLookupImage.data);
}

static void CapsuleDataUpdateShadowLookupTable(CapsuleData* data, float coneAngle)
{
    const int width = (int)data->shadowLookupResolution.x;
    const int height = (int)data->shadowLookupResolution.y;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const float phi = ((float)x / (width - 1)) * PI;
            const float theta = ((float)y / (height - 1)) * (PI / 2.0f);
            ((unsigned char*)data->shadowLookupImage.data)[y * width + x] =
                (unsigned char)Clamp(255.0f * SphereDirectionalOcclusionLookup(phi, theta, coneAngle), 0.0f, 255.0f);
        }
    }
    UpdateTexture(data->shadowLookupTable, data->shadowLookupImage.data);
}

// =============================================================================
// Init / Free / Resize
// =============================================================================

static void CapsuleDataInit(CapsuleData* data)
{
    data->capsuleCount = 0;
    data->capsulePositions.clear();
    data->capsuleRotations.clear();
    data->capsuleRadii.clear();
    data->capsuleHalfLengths.clear();
    data->capsuleColors.clear();
    data->capsuleOpacities.clear();
    data->capsuleSort.clear();

    data->aoCapsuleCount = 0;
    data->aoCapsuleStarts.clear();
    data->aoCapsuleVectors.clear();
    data->aoCapsuleRadii.clear();
    data->aoCapsuleSort.clear();

    data->shadowCapsuleCount = 0;
    data->shadowCapsuleStarts.clear();
    data->shadowCapsuleVectors.clear();
    data->shadowCapsuleRadii.clear();
    data->shadowCapsuleSort.clear();

    // AO lookup table (32x32)
    data->aoLookupImage = Image{
        .data = calloc(32 * 32, 1),
        .width = 32,
        .height = 32,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
    };
    data->aoLookupTable = LoadTextureFromImage(data->aoLookupImage);
    data->aoLookupResolution = Vector2{ 32.0f, 32.0f };
    SetTextureWrap(data->aoLookupTable, TEXTURE_WRAP_CLAMP);
    SetTextureFilter(data->aoLookupTable, TEXTURE_FILTER_BILINEAR);
    CapsuleDataUpdateAOLookupTable(data);

    // Shadow lookup table (256x128)
    data->shadowLookupImage = Image{
        .data = calloc(256 * 128, 1),
        .width = 256,
        .height = 128,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
    };
    data->shadowLookupTable = LoadTextureFromImage(data->shadowLookupImage);
    data->shadowLookupResolution = Vector2{ 256.0f, 128.0f };
    SetTextureWrap(data->shadowLookupTable, TEXTURE_WRAP_CLAMP);
    SetTextureFilter(data->shadowLookupTable, TEXTURE_FILTER_BILINEAR);
    CapsuleDataUpdateShadowLookupTable(data, 0.2f);
}

static void CapsuleDataResize(CapsuleData* data, int maxCount)
{
    data->capsulePositions.resize(maxCount);
    data->capsuleRotations.resize(maxCount);
    data->capsuleRadii.resize(maxCount);
    data->capsuleHalfLengths.resize(maxCount);
    data->capsuleColors.resize(maxCount);
    data->capsuleOpacities.resize(maxCount);
    data->capsuleSort.resize(maxCount);

    data->aoCapsuleStarts.resize(maxCount);
    data->aoCapsuleVectors.resize(maxCount);
    data->aoCapsuleRadii.resize(maxCount);
    data->aoCapsuleSort.resize(maxCount);

    data->shadowCapsuleStarts.resize(maxCount);
    data->shadowCapsuleVectors.resize(maxCount);
    data->shadowCapsuleRadii.resize(maxCount);
    data->shadowCapsuleSort.resize(maxCount);
}

static void CapsuleDataFree(CapsuleData* data)
{
    data->capsulePositions.clear();
    data->capsuleRotations.clear();
    data->capsuleRadii.clear();
    data->capsuleHalfLengths.clear();
    data->capsuleColors.clear();
    data->capsuleOpacities.clear();
    data->capsuleSort.clear();

    data->aoCapsuleStarts.clear();
    data->aoCapsuleVectors.clear();
    data->aoCapsuleRadii.clear();
    data->aoCapsuleSort.clear();

    data->shadowCapsuleStarts.clear();
    data->shadowCapsuleVectors.clear();
    data->shadowCapsuleRadii.clear();
    data->shadowCapsuleSort.clear();

    UnloadImage(data->aoLookupImage);
    UnloadTexture(data->aoLookupTable);
    UnloadImage(data->shadowLookupImage);
    UnloadTexture(data->shadowLookupTable);
}

static void CapsuleDataReset(CapsuleData* data)
{
    data->capsuleCount = 0;
    data->aoCapsuleCount = 0;
    data->shadowCapsuleCount = 0;
}

// =============================================================================
// Capsule generation from skeleton transforms
// =============================================================================

static void CapsuleDataAppendFromTransformData(
    CapsuleData* data,
    TransformData* xforms,
    float maxCapsuleRadius,
    Color color,
    float opacity,
    bool ignoreEndSite)
{
    for (int i = 0; i < xforms->jointCount; i++) {
        const int p = xforms->parents[i];
        if (p == -1) continue;
        if (ignoreEndSite && xforms->endSite[i]) continue;

        const float capsuleHalfLength = Vector3Length(xforms->localPositions[i]) / 2.0f;
        const float capsuleRadius = Min(maxCapsuleRadius, capsuleHalfLength) + (i % 2) * 0.001f;
        if (capsuleRadius < 0.001f) continue;

        const Vector3 capsulePosition = Vector3Scale(
            Vector3Add(xforms->globalPositions[i], xforms->globalPositions[p]), 0.5f);
        const Quaternion capsuleRotation = QuaternionMultiply(
            xforms->globalRotations[p],
            QuaternionBetween(Vector3{ 1.0f, 0.0f, 0.0f }, Vector3Normalize(xforms->localPositions[i])));

        data->capsulePositions[data->capsuleCount] = capsulePosition;
        data->capsuleRotations[data->capsuleCount] = capsuleRotation;
        data->capsuleHalfLengths[data->capsuleCount] = capsuleHalfLength;
        data->capsuleRadii[data->capsuleCount] = capsuleRadius;
        data->capsuleColors[data->capsuleCount] = Vector3{ color.r / 255.0f, color.g / 255.0f, color.b / 255.0f };
        data->capsuleOpacities[data->capsuleCount] = opacity;
        data->capsuleCount++;
    }
}

// =============================================================================
// AO capsule gathering for ground segments
// =============================================================================

static void CapsuleDataUpdateAOCapsulesForGroundSegment(CapsuleData* data, Vector3 groundPos)
{
    data->aoCapsuleCount = 0;

    for (int i = 0; i < data->capsuleCount; i++) {
        const Vector3 pos = data->capsulePositions[i];
        const float halfLen = data->capsuleHalfLengths[i];
        const float radius = data->capsuleRadii[i];

        // Bounding sphere check
        if (Vector3Distance(groundPos, pos) - sqrtf(2.0f) > halfLen + AO_RATIO_MAX * radius) {
            continue;
        }

        const Quaternion rot = data->capsuleRotations[i];
        const Vector3 start = CapsuleStart(pos, rot, halfLen);
        const Vector3 end = CapsuleEnd(pos, rot, halfLen);
        const Vector3 vec = CapsuleVector(pos, rot, halfLen);

        float capsuleTime;
        Vector3 groundPoint;
        NearestPointBetweenLineSegmentAndGroundSegment(
            &capsuleTime, &groundPoint, start, end,
            Vector3{ groundPos.x - 1.0f, 0.0f, groundPos.z - 1.0f },
            Vector3{ groundPos.x + 1.0f, 0.0f, groundPos.z + 1.0f });

        const Vector3 capsulePoint = Vector3Add(start, Vector3Scale(vec, capsuleTime));

        if (Vector3Distance(groundPoint, capsulePoint) > AO_RATIO_MAX * radius) {
            continue;
        }

        const float occlusion = Vector3Distance(groundPoint, capsulePoint) < radius ? 0.0f :
            SphereOcclusion(groundPoint, Vector3{ 0.0f, 1.0f, 0.0f }, capsulePoint, radius);

        if (occlusion < 0.99f) {
            data->aoCapsuleSort[data->aoCapsuleCount] = CapsuleSort{ i, occlusion };
            data->aoCapsuleCount++;
        }
    }

    // Sort by occlusion (darkest first)
    std::sort(data->aoCapsuleSort.begin(), data->aoCapsuleSort.begin() + data->aoCapsuleCount,
        [](const CapsuleSort& a, const CapsuleSort& b) { return a.value > b.value; });

    // Populate sorted capsule data
    for (int i = 0; i < data->aoCapsuleCount; i++) {
        const int j = data->aoCapsuleSort[i].index;
        data->aoCapsuleStarts[i] = CapsuleStart(data->capsulePositions[j], data->capsuleRotations[j], data->capsuleHalfLengths[j]);
        data->aoCapsuleVectors[i] = CapsuleVector(data->capsulePositions[j], data->capsuleRotations[j], data->capsuleHalfLengths[j]);
        data->aoCapsuleRadii[i] = data->capsuleRadii[j];
    }
}

// =============================================================================
// AO capsule gathering for capsule-to-capsule occlusion
// =============================================================================

static void CapsuleDataUpdateAOCapsulesForCapsule(CapsuleData* data, int capsuleIndex)
{
    const Vector3 queryPos = data->capsulePositions[capsuleIndex];
    const float queryHalfLen = data->capsuleHalfLengths[capsuleIndex];
    const float queryRadius = data->capsuleRadii[capsuleIndex];
    const Quaternion queryRot = data->capsuleRotations[capsuleIndex];
    const Vector3 queryStart = CapsuleStart(queryPos, queryRot, queryHalfLen);
    const Vector3 queryEnd = CapsuleEnd(queryPos, queryRot, queryHalfLen);
    const Vector3 queryVec = CapsuleVector(queryPos, queryRot, queryHalfLen);

    data->aoCapsuleCount = 0;

    for (int i = 0; i < data->capsuleCount; i++) {
        if (i == capsuleIndex) continue;

        const Vector3 pos = data->capsulePositions[i];
        const float radius = data->capsuleRadii[i];
        const float halfLen = data->capsuleHalfLengths[i];

        // Bounding sphere check
        if (Vector3Distance(queryPos, pos) - queryHalfLen - queryRadius > halfLen + AO_RATIO_MAX * radius) {
            continue;
        }

        const Quaternion rot = data->capsuleRotations[i];
        const Vector3 start = CapsuleStart(pos, rot, halfLen);
        const Vector3 end = CapsuleEnd(pos, rot, halfLen);
        const Vector3 vec = CapsuleVector(pos, rot, halfLen);

        float capsuleTime, queryTime;
        NearestPointBetweenLineSegments(&capsuleTime, &queryTime, start, end, queryStart, queryEnd);

        const Vector3 capsulePoint = Vector3Add(start, Vector3Scale(vec, capsuleTime));
        const Vector3 queryPoint = Vector3Add(queryStart, Vector3Scale(queryVec, queryTime));

        if (Vector3Distance(queryPoint, capsulePoint) - queryRadius > AO_RATIO_MAX * radius) {
            continue;
        }

        const Vector3 surfaceNormal = Vector3Normalize(Vector3Subtract(capsulePoint, queryPoint));
        const Vector3 surfacePoint = Vector3Add(queryPoint, Vector3Scale(surfaceNormal, queryRadius));
        const float occlusion = Vector3Distance(queryPoint, capsulePoint) <= queryRadius + radius ? 0.0f :
            SphereOcclusion(surfacePoint, surfaceNormal, capsulePoint, radius);

        if (occlusion < 0.99f) {
            data->aoCapsuleSort[data->aoCapsuleCount] = CapsuleSort{ i, occlusion };
            data->aoCapsuleCount++;
        }
    }

    std::sort(data->aoCapsuleSort.begin(), data->aoCapsuleSort.begin() + data->aoCapsuleCount,
        [](const CapsuleSort& a, const CapsuleSort& b) { return a.value > b.value; });

    for (int i = 0; i < data->aoCapsuleCount; i++) {
        const int j = data->aoCapsuleSort[i].index;
        data->aoCapsuleStarts[i] = CapsuleStart(data->capsulePositions[j], data->capsuleRotations[j], data->capsuleHalfLengths[j]);
        data->aoCapsuleVectors[i] = CapsuleVector(data->capsulePositions[j], data->capsuleRotations[j], data->capsuleHalfLengths[j]);
        data->aoCapsuleRadii[i] = data->capsuleRadii[j];
    }
}

// =============================================================================
// Shadow capsule gathering for ground segments
// =============================================================================

static void CapsuleDataUpdateShadowCapsulesForGroundSegment(
    CapsuleData* data,
    Vector3 groundPos,
    Vector3 lightDir,
    float lightConeAngle)
{
    const Vector3 lightRay = Vector3Scale(lightDir, 10.0f);
    constexpr float maxRatio = 4.0f;

    data->shadowCapsuleCount = 0;

    for (int i = 0; i < data->capsuleCount; i++) {
        const Vector3 pos = data->capsulePositions[i];
        const float halfLen = data->capsuleHalfLengths[i];
        const float radius = data->capsuleRadii[i];

        const float midRayTime = NearestPointBetweenLineSegmentAndGroundPlane(pos, lightRay);
        const Vector3 groundMid = Vector3Add(pos, Vector3Scale(lightRay, midRayTime));

        if (Vector3Distance(groundPos, groundMid) - sqrtf(2.0f) > halfLen + maxRatio * radius) {
            continue;
        }

        const Quaternion rot = data->capsuleRotations[i];
        const Vector3 start = CapsuleStart(pos, rot, halfLen);
        const Vector3 end = CapsuleEnd(pos, rot, halfLen);
        const Vector3 vec = CapsuleVector(pos, rot, halfLen);

        const float startRayTime = NearestPointBetweenLineSegmentAndGroundPlane(start, lightRay);
        const float endRayTime = NearestPointBetweenLineSegmentAndGroundPlane(end, lightRay);

        Vector3 groundStart = Vector3Add(start, Vector3Scale(lightRay, startRayTime));
        Vector3 groundEnd = Vector3Add(end, Vector3Scale(lightRay, endRayTime));

        groundStart.x = Clamp(groundStart.x, groundPos.x - 1.0f, groundPos.x + 1.0f);
        groundStart.z = Clamp(groundStart.z, groundPos.z - 1.0f, groundPos.z + 1.0f);
        groundEnd.x = Clamp(groundEnd.x, groundPos.x - 1.0f, groundPos.x + 1.0f);
        groundEnd.z = Clamp(groundEnd.z, groundPos.z - 1.0f, groundPos.z + 1.0f);

        if (Vector3Distance(groundPos, groundStart) - sqrtf(2.0f) > maxRatio * radius &&
            Vector3Distance(groundPos, groundEnd) - sqrtf(2.0f) > maxRatio * radius) {
            continue;
        }

        const float occlusion = Min(
            CapsuleDirectionalOcclusion(groundStart, start, vec, radius, lightDir, lightConeAngle),
            CapsuleDirectionalOcclusion(groundEnd, start, vec, radius, lightDir, lightConeAngle));

        if (occlusion < 0.99f) {
            data->shadowCapsuleSort[data->shadowCapsuleCount] = CapsuleSort{ i, occlusion };
            data->shadowCapsuleCount++;
        }
    }

    std::sort(data->shadowCapsuleSort.begin(), data->shadowCapsuleSort.begin() + data->shadowCapsuleCount,
        [](const CapsuleSort& a, const CapsuleSort& b) { return a.value > b.value; });

    for (int i = 0; i < data->shadowCapsuleCount; i++) {
        const int j = data->shadowCapsuleSort[i].index;
        data->shadowCapsuleStarts[i] = CapsuleStart(data->capsulePositions[j], data->capsuleRotations[j], data->capsuleHalfLengths[j]);
        data->shadowCapsuleVectors[i] = CapsuleVector(data->capsulePositions[j], data->capsuleRotations[j], data->capsuleHalfLengths[j]);
        data->shadowCapsuleRadii[i] = data->capsuleRadii[j];
    }
}

// =============================================================================
// Shadow capsule gathering for capsule-to-capsule shadows
// =============================================================================

static void CapsuleDataUpdateShadowCapsulesForCapsule(
    CapsuleData* data,
    int capsuleIndex,
    Vector3 lightDir,
    float lightConeAngle)
{
    const Vector3 lightRay = Vector3Scale(lightDir, 10.0f);
    constexpr float maxRatio = 4.0f;

    const Vector3 queryPos = data->capsulePositions[capsuleIndex];
    const float queryHalfLen = data->capsuleHalfLengths[capsuleIndex];
    const float queryRadius = data->capsuleRadii[capsuleIndex];
    const Quaternion queryRot = data->capsuleRotations[capsuleIndex];
    const Vector3 queryStart = CapsuleStart(queryPos, queryRot, queryHalfLen);
    const Vector3 queryEnd = CapsuleEnd(queryPos, queryRot, queryHalfLen);
    const Vector3 queryVec = CapsuleVector(queryPos, queryRot, queryHalfLen);

    data->shadowCapsuleCount = 0;

    for (int i = 0; i < data->capsuleCount; i++) {
        if (i == capsuleIndex) continue;

        const Vector3 pos = data->capsulePositions[i];
        const float halfLen = data->capsuleHalfLengths[i];
        const float radius = data->capsuleRadii[i];

        const float midRayTime = NearestPointOnLineSegment(pos, lightRay, queryPos);
        const Vector3 capsuleMid = Vector3Add(pos, Vector3Scale(lightRay, midRayTime));

        if (Vector3Distance(queryPos, capsuleMid) - queryHalfLen - queryRadius > halfLen + maxRatio * radius) {
            continue;
        }

        const Quaternion rot = data->capsuleRotations[i];
        const Vector3 start = CapsuleStart(pos, rot, halfLen);
        const Vector3 end = CapsuleEnd(pos, rot, halfLen);
        const Vector3 vec = CapsuleVector(pos, rot, halfLen);

        float queryTime;
        Vector3 nearestRayPoint;
        NearestPointBetweenLineSegmentAndSweptLine(
            &queryTime, &nearestRayPoint, queryStart, queryEnd, start, end, lightRay);

        const Vector3 queryPoint = Vector3Add(queryStart, Vector3Scale(queryVec, queryTime));

        if (Vector3Distance(queryPoint, nearestRayPoint) - queryRadius > halfLen + maxRatio * radius) {
            continue;
        }

        const Vector3 surfaceNormal = Vector3Normalize(Vector3Subtract(nearestRayPoint, queryPoint));
        const Vector3 surfacePoint = Vector3Add(queryPoint, Vector3Scale(surfaceNormal, queryRadius));

        const float occlusion = Vector3Distance(queryPoint, nearestRayPoint) <= queryRadius + radius ? 0.0f :
            CapsuleDirectionalOcclusion(surfacePoint, start, vec, radius, lightDir, lightConeAngle);

        if (occlusion < 0.99f) {
            data->shadowCapsuleSort[data->shadowCapsuleCount] = CapsuleSort{ i, occlusion };
            data->shadowCapsuleCount++;
        }
    }

    std::sort(data->shadowCapsuleSort.begin(), data->shadowCapsuleSort.begin() + data->shadowCapsuleCount,
        [](const CapsuleSort& a, const CapsuleSort& b) { return a.value > b.value; });

    for (int i = 0; i < data->shadowCapsuleCount; i++) {
        const int j = data->shadowCapsuleSort[i].index;
        data->shadowCapsuleStarts[i] = CapsuleStart(data->capsulePositions[j], data->capsuleRotations[j], data->capsuleHalfLengths[j]);
        data->shadowCapsuleVectors[i] = CapsuleVector(data->capsulePositions[j], data->capsuleRotations[j], data->capsuleHalfLengths[j]);
        data->shadowCapsuleRadii[i] = data->capsuleRadii[j];
    }
}
