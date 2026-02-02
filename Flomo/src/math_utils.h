#pragma once

// need raylib already included
#include "raymath.h"
#include "raylib.h"
#include <cmath>

//----------------------------------------------------------------------------------
// Additional Raylib Functions
//----------------------------------------------------------------------------------

static inline float Max(float x, float y)
{
    return x > y ? x : y;
}

static inline float Min(float x, float y)
{
    return x < y ? x : y;
}

static inline float Saturate(float x)
{
    return Clamp(x, 0.0f, 1.0f);
}

// smoothstep: smooth interpolation with zero derivative at endpoints
// t should be in [0, 1], returns value in [0, 1]
static inline float SmoothStep(float t)
{
    return t * t * (3.0f - 2.0f * t);
}

// Smooth lerp: interpolates from a to b using smoothstep for smoother transitions
static inline float SmoothLerp(float a, float b, float t)
{
    return Lerp(a, b, SmoothStep(t));
}

static inline float Square(float x)
{
    return x * x;
}

static inline int ClampInt(int x, int min, int max)
{
    return x < min ? min : x > max ? max : x;
}

static inline int MaxInt(int x, int y)
{
    return x > y ? x : y;
}

static inline int MinInt(int x, int y)
{
    return x < y ? x : y;
}


// Acos that prevents domain errors (NaNs) if inputs are slightly > 1.0 or < -1.0
static inline float SafeAcos(float x) {
    if (x > 1.0f) return 0.0f;
    if (x < -1.0f) return PI;
    return std::acos(x);
}

// 2D length utilities (XZ plane, ignoring Y)
static inline float Vector3LengthSqr2D(Vector3 v)
{
    return v.x * v.x + v.z * v.z;
}

static inline float Vector3Length2D(Vector3 v)
{
    return sqrtf(v.x * v.x + v.z * v.z);
}   

// Clamped inverse lerp: returns t in [0,1] such that Lerp(a, b, t) = value
// Automatically clamps the result to [0,1] range
static inline float ClampedInvLerp(float a, float b, float value)
{
    if (fabsf(b - a) < 1e-6f) return 0.0f;  // Avoid division by zero
    const float t = (value - a) / (b - a);
    return Clamp(t, 0.0f, 1.0f);
}

// This is a safe version of QuaternionBetween which returns a 180 deg rotation
// at the singularity where vectors are facing exactly in opposite directions
static inline Quaternion QuaternionBetween(Vector3 p, Vector3 q)
{
    const Vector3 c = Vector3CrossProduct(p, q);

    const Quaternion o = {
        c.x,
        c.y,
        c.z,
        sqrtf(Vector3DotProduct(p, p) * Vector3DotProduct(q, q)) + Vector3DotProduct(p, q),
    };

    return QuaternionLength(o) < 1e-8f ?
        QuaternionFromAxisAngle(Vector3{ 1.0f, 0.0f, 0.0f }, PI) :
        QuaternionNormalize(o);
}


// Puts the quaternion in the hemisphere closest to the identity
static inline Quaternion QuaternionAbsolute(Quaternion q)
{
    if (q.w < 0.0f)
    {
        q.x = -q.x;
        q.y = -q.y;
        q.z = -q.z;
        q.w = -q.w;
    }

    return q;
}

// Quaternion exponent, log, and angle axis functions (see: https://theorangeduck.com/page/exponential-map-angle-axis-angular-velocity)

static inline Quaternion QuaternionExp(Vector3 v)
{
    const float halfangle = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

    if (halfangle < 1e-4f)
    {
        return QuaternionNormalize(Quaternion{ v.x, v.y, v.z, 1.0f });
    }
    else
    {
        const float c = cosf(halfangle);
        const float s = sinf(halfangle) / halfangle;
        return Quaternion{ s * v.x, s * v.y, s * v.z, c };
    }
}

static inline Vector3 QuaternionLog(Quaternion q)
{
    const float length = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);

    if (length < 1e-4f)
    {
        return Vector3{ q.x, q.y, q.z };
    }
    else
    {
        const float halfangle = atan2f(length, q.w);
        return Vector3Scale(Vector3{ q.x, q.y, q.z }, halfangle / length);
    }
}


// Extract only the Y-axis rotation component from a quaternion
static inline Quaternion QuaternionYComponent(Quaternion q)
{
    // Project to Y-axis rotation only: keep y and w, zero x and z
    Quaternion yOnly = { 0.0f, q.y, 0.0f, q.w };
    float len = sqrtf(yOnly.y * yOnly.y + yOnly.w * yOnly.w);
    if (len < 1e-8f) return QuaternionIdentity();
    return Quaternion{ 0.0f, yOnly.y / len, 0.0f, yOnly.w / len };
}


static inline Vector3 QuaternionToScaledAngleAxis(Quaternion q)
{
    return Vector3Scale(QuaternionLog(q), 2.0f);
}

static inline Quaternion QuaternionFromScaledAngleAxis(Vector3 v)
{
    return QuaternionExp(Vector3Scale(v, 0.5f));
}

// Cubic Interpolation (see: https://theorangeduck.com/page/cubic-interpolation-quaternions)

static inline Vector3 Vector3Hermite(Vector3 p0, Vector3 p1, Vector3 v0, Vector3 v1, float alpha)
{
    const float x = alpha;
    const float w0 = 2 * x * x * x - 3 * x * x + 1;
    const float w1 = 3 * x * x - 2 * x * x * x;
    const float w2 = x * x * x - 2 * x * x + x;
    const float w3 = x * x * x - x * x;

    return Vector3Add(
        Vector3Add(Vector3Scale(p0, w0), Vector3Scale(p1, w1)),
        Vector3Add(Vector3Scale(v0, w2), Vector3Scale(v1, w3)));
}

static inline Vector3 Vector3InterpolateCubic(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float alpha)
{
    const Vector3 v1 = Vector3Scale(Vector3Add(Vector3Subtract(p1, p0), Vector3Subtract(p2, p1)), 0.5f);
    const Vector3 v2 = Vector3Scale(Vector3Add(Vector3Subtract(p2, p1), Vector3Subtract(p3, p2)), 0.5f);
    return Vector3Hermite(p1, p2, v1, v2, alpha);
}

static inline Quaternion QuaternionHermite(Quaternion r0, Quaternion r1, Vector3 v0, Vector3 v1, float alpha)
{
    const float x = alpha;
    const float w1 = 3 * x * x - 2 * x * x * x;
    const float w2 = x * x * x - 2 * x * x + x;
    const float w3 = x * x * x - x * x;

    const Vector3 r1r0 = QuaternionToScaledAngleAxis(
        QuaternionAbsolute(
            QuaternionMultiply(r1, QuaternionInvert(r0))));

    return QuaternionMultiply(QuaternionFromScaledAngleAxis(
        Vector3Add(
            Vector3Add(
                Vector3Scale(r1r0, w1),
                Vector3Scale(v0, w2)),
            Vector3Scale(v1, w3))),
        r0);
}

static inline Quaternion QuaternionInterpolateCubic(Quaternion r0, Quaternion r1, Quaternion r2, Quaternion r3, float alpha)
{
    const Vector3 r1r0 = QuaternionToScaledAngleAxis(QuaternionAbsolute(QuaternionMultiply(r1, QuaternionInvert(r0))));
    const Vector3 r2r1 = QuaternionToScaledAngleAxis(QuaternionAbsolute(QuaternionMultiply(r2, QuaternionInvert(r1))));
    const Vector3 r3r2 = QuaternionToScaledAngleAxis(QuaternionAbsolute(QuaternionMultiply(r3, QuaternionInvert(r2))));

    const Vector3 v1 = Vector3Scale(Vector3Add(r1r0, r2r1), 0.5f);
    const Vector3 v2 = Vector3Scale(Vector3Add(r2r1, r3r2), 0.5f);
    return QuaternionHermite(r1, r2, v1, v2, alpha);
}

// Frustum culling (based off https://github.com/JeffM2501/raylibExtras)

typedef struct
{
    Vector4 back;
    Vector4 front;
    Vector4 bottom;
    Vector4 top;
    Vector4 right;
    Vector4 left;

} Frustum;

static inline Vector4 FrustumPlaneNormalize(Vector4 plane)
{
    const float magnitude = sqrtf(Square(plane.x) + Square(plane.y) + Square(plane.z));
    plane.x /= magnitude;
    plane.y /= magnitude;
    plane.z /= magnitude;
    plane.w /= magnitude;
    return plane;
}

static inline void FrustumFromCameraMatrices(
    const Matrix& projection,
    const Matrix& modelview,
    Frustum& resultFrustum)
{
    Matrix planes = { 0 };
    planes.m0 = modelview.m0 * projection.m0 + modelview.m1 * projection.m4 + modelview.m2 * projection.m8 + modelview.m3 * projection.m12;
    planes.m1 = modelview.m0 * projection.m1 + modelview.m1 * projection.m5 + modelview.m2 * projection.m9 + modelview.m3 * projection.m13;
    planes.m2 = modelview.m0 * projection.m2 + modelview.m1 * projection.m6 + modelview.m2 * projection.m10 + modelview.m3 * projection.m14;
    planes.m3 = modelview.m0 * projection.m3 + modelview.m1 * projection.m7 + modelview.m2 * projection.m11 + modelview.m3 * projection.m15;
    planes.m4 = modelview.m4 * projection.m0 + modelview.m5 * projection.m4 + modelview.m6 * projection.m8 + modelview.m7 * projection.m12;
    planes.m5 = modelview.m4 * projection.m1 + modelview.m5 * projection.m5 + modelview.m6 * projection.m9 + modelview.m7 * projection.m13;
    planes.m6 = modelview.m4 * projection.m2 + modelview.m5 * projection.m6 + modelview.m6 * projection.m10 + modelview.m7 * projection.m14;
    planes.m7 = modelview.m4 * projection.m3 + modelview.m5 * projection.m7 + modelview.m6 * projection.m11 + modelview.m7 * projection.m15;
    planes.m8 = modelview.m8 * projection.m0 + modelview.m9 * projection.m4 + modelview.m10 * projection.m8 + modelview.m11 * projection.m12;
    planes.m9 = modelview.m8 * projection.m1 + modelview.m9 * projection.m5 + modelview.m10 * projection.m9 + modelview.m11 * projection.m13;
    planes.m10 = modelview.m8 * projection.m2 + modelview.m9 * projection.m6 + modelview.m10 * projection.m10 + modelview.m11 * projection.m14;
    planes.m11 = modelview.m8 * projection.m3 + modelview.m9 * projection.m7 + modelview.m10 * projection.m11 + modelview.m11 * projection.m15;
    planes.m12 = modelview.m12 * projection.m0 + modelview.m13 * projection.m4 + modelview.m14 * projection.m8 + modelview.m15 * projection.m12;
    planes.m13 = modelview.m12 * projection.m1 + modelview.m13 * projection.m5 + modelview.m14 * projection.m9 + modelview.m15 * projection.m13;
    planes.m14 = modelview.m12 * projection.m2 + modelview.m13 * projection.m6 + modelview.m14 * projection.m10 + modelview.m15 * projection.m14;
    planes.m15 = modelview.m12 * projection.m3 + modelview.m13 * projection.m7 + modelview.m14 * projection.m11 + modelview.m15 * projection.m15;

    resultFrustum.back = FrustumPlaneNormalize(Vector4{ planes.m3 - planes.m2, planes.m7 - planes.m6, planes.m11 - planes.m10, planes.m15 - planes.m14 });
    resultFrustum.front = FrustumPlaneNormalize(Vector4{ planes.m3 + planes.m2, planes.m7 + planes.m6, planes.m11 + planes.m10, planes.m15 + planes.m14 });
    resultFrustum.bottom = FrustumPlaneNormalize(Vector4{ planes.m3 + planes.m1, planes.m7 + planes.m5, planes.m11 + planes.m9, planes.m15 + planes.m13 });
    resultFrustum.top = FrustumPlaneNormalize(Vector4{ planes.m3 - planes.m1, planes.m7 - planes.m5, planes.m11 - planes.m9, planes.m15 - planes.m13 });
    resultFrustum.left = FrustumPlaneNormalize(Vector4{ planes.m3 + planes.m0, planes.m7 + planes.m4, planes.m11 + planes.m8, planes.m15 + planes.m12 });
    resultFrustum.right = FrustumPlaneNormalize(Vector4{ planes.m3 - planes.m0, planes.m7 - planes.m4, planes.m11 - planes.m8, planes.m15 - planes.m12 });
}

static inline float FrustumPlaneDistanceToPoint(Vector4 plane, Vector3 position)
{
    return (plane.x * position.x + plane.y * position.y + plane.z * position.z + plane.w);
}

static inline bool FrustumContainsSphere(Frustum frustum, Vector3 position, float radius)
{
    if (FrustumPlaneDistanceToPoint(frustum.back, position) < -radius) { return false; }
    if (FrustumPlaneDistanceToPoint(frustum.front, position) < -radius) { return false; }
    if (FrustumPlaneDistanceToPoint(frustum.bottom, position) < -radius) { return false; }
    if (FrustumPlaneDistanceToPoint(frustum.top, position) < -radius) { return false; }
    if (FrustumPlaneDistanceToPoint(frustum.left, position) < -radius) { return false; }
    if (FrustumPlaneDistanceToPoint(frustum.right, position) < -radius) { return false; }
    return true;
}

static inline float FastNegExp(float x)
{
    return 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
}

static inline float HalflifeToDamping(float halflife, float eps = 1e-5f)
{
    return (4.0f * 0.69314718056f) / (halflife + eps);
}

static inline void SimpleSpringDamperUsingDampingEydt(
    float& x,
    float& v,
    float x_goal,
    float damping,
    float eydt,
    float dt)
{
    const float y = damping;
    const float j0 = x - x_goal;
    const float j1 = v + j0 * y;

    x = eydt * (j0 + j1 * dt) + x_goal;
    v = eydt * (v - j1 * y * dt);
}


static inline void SimpleSpringDamper(
    float& x,
    float& v,
    float x_goal,
    float blendtime,
    float dt)
{
    const float y = HalflifeToDamping(blendtime / 2.0f) / 2.0f;
    const float eydt = FastNegExp(y * dt);

    SimpleSpringDamperUsingDampingEydt(x, v, x_goal, y, eydt, dt);
}

struct DoubleSpringDamperState
{
    float x;
    float v;
    float xi;
    float vi;
};

static inline void DoubleSpringDamper(
    DoubleSpringDamperState& state,
    float x_goal,
    float blendtime,
    float dt)
{
    // half of the blendtime for each damper
    const float y = HalflifeToDamping(blendtime / 4.0f) / 2.0f;
    const float eydt = FastNegExp(y * dt);

    SimpleSpringDamperUsingDampingEydt(state.xi, state.vi, x_goal, y, eydt, dt);
    SimpleSpringDamperUsingDampingEydt(state.x, state.v, state.xi, y, eydt, dt);
}





struct Rot6d
{
    float ax, ay, az;
    float bx, by, bz;
};

inline float FastInvSqrt(const float x)
{
    const float xhalf = 0.5f * x;
    int32_t i = *(int32_t*)&x;
    i = 0x5f3759df - (i >> 1);
    float y = *(float*)&i;
    y = y * (1.5f - xhalf * y * y);
    return y;
}

void Rot6dRotate(Rot6d& rot, const Vector3& omega, float dt)
{
    const float wx = omega.x * dt;
    const float wy = omega.y * dt;
    const float wz = omega.z * dt;

    const float thetaSq = wx * wx + wy * wy + wz * wz;
    if (thetaSq < 1e-12f)
    {
        return;
    }

    const float nax = rot.ax + (wy * rot.az - wz * rot.ay);
    const float nay = rot.ay + (wz * rot.ax - wx * rot.az);
    const float naz = rot.az + (wx * rot.ay - wy * rot.ax);

    const float nbx = rot.bx + (wy * rot.bz - wz * rot.by);
    const float nby = rot.by + (wz * rot.bx - wx * rot.bz);
    const float nbz = rot.bz + (wx * rot.by - wy * rot.bx);

    const float invLenA = FastInvSqrt(nax * nax + nay * nay + naz * naz);
    rot.ax = nax * invLenA;
    rot.ay = nay * invLenA;
    rot.az = naz * invLenA;

    const float dot = rot.ax * nbx + rot.ay * nby + rot.az * nbz;
    const float rbx = nbx - dot * rot.ax;
    const float rby = nby - dot * rot.ay;
    const float rbz = nbz - dot * rot.az;

    const float invLenB = FastInvSqrt(rbx * rbx + rby * rby + rbz * rbz);
    rot.bx = rbx * invLenB;
    rot.by = rby * invLenB;
    rot.bz = rbz * invLenB;
}

void Rot6dMultiply(const Rot6d& lhs, const Rot6d& rhs, Rot6d& out)
{
    const float lcx = lhs.ay * lhs.bz - lhs.az * lhs.by;
    const float lcy = lhs.az * lhs.bx - lhs.ax * lhs.bz;
    const float lcz = lhs.ax * lhs.by - lhs.ay * lhs.bx;

    out.ax = lhs.ax * rhs.ax + lhs.bx * rhs.ay + lcx * rhs.az;
    out.ay = lhs.ay * rhs.ax + lhs.by * rhs.ay + lcy * rhs.az;
    out.az = lhs.az * rhs.ax + lhs.bz * rhs.ay + lcz * rhs.az;

    out.bx = lhs.ax * rhs.bx + lhs.bx * rhs.by + lcx * rhs.bz;
    out.by = lhs.ay * rhs.bx + lhs.by * rhs.by + lcy * rhs.bz;
    out.bz = lhs.az * rhs.bx + lhs.bz * rhs.by + lcz * rhs.bz;
}

void Rot6dInverse(const Rot6d& rot, Rot6d& out)
{
    const float cx = rot.ay * rot.bz - rot.az * rot.by;
    const float cy = rot.az * rot.bx - rot.ax * rot.bz;
    
    out.ax = rot.ax; out.ay = rot.bx; out.az = cx;
    out.bx = rot.ay; out.by = rot.by; out.bz = cy;
}

void Rot6dTransformVector(const Rot6d& rot, const Vector3& v, Vector3& out)
{
    const float cx = rot.ay * rot.bz - rot.az * rot.by;
    const float cy = rot.az * rot.bx - rot.ax * rot.bz;
    const float cz = rot.ax * rot.by - rot.ay * rot.bx;

    out.x = v.x * rot.ax + v.y * rot.bx + v.z * cx;
    out.y = v.x * rot.ay + v.y * rot.by + v.z * cy;
    out.z = v.x * rot.az + v.y * rot.bz + v.z * cz;
}


void Rot6dToQuaternion(const Rot6d& rot, Quaternion& outQ)
{
    // third column via cross product: c = a × b
    const float cx = rot.ay * rot.bz - rot.az * rot.by;
    const float cy = rot.az * rot.bx - rot.ax * rot.bz;
    const float cz = rot.ax * rot.by - rot.ay * rot.bx;

    // matrix layout:
    // m00=rot.ax  m01=rot.bx  m02=cx
    // m10=rot.ay  m11=rot.by  m12=cy
    // m20=rot.az  m21=rot.bz  m22=cz
    //
    // standard matrix-to-quat formulas:
    // x = (m21 - m12) / s = (rot.bz - cy) / s
    // y = (m02 - m20) / s = (cx - rot.az) / s
    // z = (m10 - m01) / s = (rot.ay - rot.bx) / s
    // w = (varies by branch)

    const float tr = rot.ax + rot.by + cz;
    if (tr > 0.0f)
    {
        const float s = std::sqrt(tr + 1.0f) * 2.0f;
        outQ.w = 0.25f * s;
        outQ.x = (rot.bz - cy) / s;
        outQ.y = (cx - rot.az) / s;
        outQ.z = (rot.ay - rot.bx) / s;
    }
    else if ((rot.ax > rot.by) && (rot.ax > cz))
    {
        const float s = std::sqrt(1.0f + rot.ax - rot.by - cz) * 2.0f;
        outQ.w = (rot.bz - cy) / s;
        outQ.x = 0.25f * s;
        outQ.y = (rot.ay + rot.bx) / s;
        outQ.z = (rot.az + cx) / s;
    }
    else if (rot.by > cz)
    {
        const float s = std::sqrt(1.0f + rot.by - rot.ax - cz) * 2.0f;
        outQ.w = (cx - rot.az) / s;
        outQ.x = (rot.ay + rot.bx) / s;
        outQ.y = 0.25f * s;
        outQ.z = (rot.bz + cy) / s;
    }
    else
    {
        const float s = std::sqrt(1.0f + cz - rot.ax - rot.by) * 2.0f;
        outQ.w = (rot.ay - rot.bx) / s;
        outQ.x = (rot.az + cx) / s;
        outQ.y = (rot.bz + cy) / s;
        outQ.z = 0.25f * s;
    }
}

void Rot6dFromQuaternion(const Quaternion& q, Rot6d& outRot)
{
    const float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
    const float xx = q.x * x2, xy = q.x * y2, xz = q.x * z2;
    const float yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
    const float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;

    outRot.ax = 1.0f - (yy + zz);
    outRot.ay = xy + wz;
    outRot.az = xz - wy;

    outRot.bx = xy - wz;
    outRot.by = 1.0f - (xx + zz);
    outRot.bz = yz + wx;
}

void Rot6dToMatrix(const Rot6d& rot, Matrix& outMat)
{
    const float cx = rot.ay * rot.bz - rot.az * rot.by;
    const float cy = rot.az * rot.bx - rot.ax * rot.bz;
    const float cz = rot.ax * rot.by - rot.ay * rot.bx;

    outMat.m0 = rot.ax; outMat.m4 = rot.bx; outMat.m8 = cx;   outMat.m12 = 0.0f;
    outMat.m1 = rot.ay; outMat.m5 = rot.by; outMat.m9 = cy;   outMat.m13 = 0.0f;
    outMat.m2 = rot.az; outMat.m6 = rot.bz; outMat.m10 = cz;   outMat.m14 = 0.0f;
    outMat.m3 = 0.0f;   outMat.m7 = 0.0f;   outMat.m11 = 0.0f; outMat.m15 = 1.0f;
}

void Rot6dFromMatrix(const Matrix& mat, Rot6d& outRot)
{
    outRot.ax = mat.m0; outRot.ay = mat.m1; outRot.az = mat.m2;
    outRot.bx = mat.m4; outRot.by = mat.m5; outRot.bz = mat.m6;
}

void Rot6dLerp(const Rot6d& start, const Rot6d& end, float t, Rot6d& out)
{
    const float invT = 1.0f - t;

    const float nax = start.ax * invT + end.ax * t;
    const float nay = start.ay * invT + end.ay * t;
    const float naz = start.az * invT + end.az * t;

    const float nbx = start.bx * invT + end.bx * t;
    const float nby = start.by * invT + end.by * t;
    const float nbz = start.bz * invT + end.bz * t;

    // Normalize a-column
    const float invLenA = FastInvSqrt(nax * nax + nay * nay + naz * naz);
    const float ax = nax * invLenA;
    const float ay = nay * invLenA;
    const float az = naz * invLenA;

    // Gram-Schmidt: remove a-component from b
    const float dot = ax * nbx + ay * nby + az * nbz;
    const float rbx = nbx - dot * ax;
    const float rby = nby - dot * ay;
    const float rbz = nbz - dot * az;

    // Normalize b-column
    const float invLenB = FastInvSqrt(rbx * rbx + rby * rby + rbz * rbz);

    // Write all results at once (safe for aliasing)
    out.ax = ax;
    out.ay = ay;
    out.az = az;
    out.bx = rbx * invLenB;
    out.by = rby * invLenB;
    out.bz = rbz * invLenB;
}

void Rot6dSlerp(const Rot6d& start, const Rot6d& end, float t, Rot6d& out)
{
    Quaternion qStart, qEnd;
    Rot6dToQuaternion(start, qStart);
    Rot6dToQuaternion(end, qEnd);
    Rot6dFromQuaternion(QuaternionSlerp(qStart, qEnd, t), out);
}

// orthonormalize a Rot6d (useful after weighted accumulation)
void Rot6dNormalize(Rot6d& rot)
{
    // normalize a-column
    const float invLenA = FastInvSqrt(rot.ax * rot.ax + rot.ay * rot.ay + rot.az * rot.az);
    rot.ax *= invLenA;
    rot.ay *= invLenA;
    rot.az *= invLenA;

    // remove a-component from b, then normalize b
    const float dot = rot.ax * rot.bx + rot.ay * rot.by + rot.az * rot.bz;
    rot.bx -= dot * rot.ax;
    rot.by -= dot * rot.ay;
    rot.bz -= dot * rot.az;

    const float invLenB = FastInvSqrt(rot.bx * rot.bx + rot.by * rot.by + rot.bz * rot.bz);
    rot.bx *= invLenB;
    rot.by *= invLenB;
    rot.bz *= invLenB;
}

// identity Rot6d
static inline Rot6d Rot6dIdentity()
{
    return Rot6d{ 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
}

// extract Y-rotation (yaw) angle from Rot6d
// the first column (a) of a Y-rotation matrix is [cos(y), 0, -sin(y)]
// so yaw = atan2(-az, ax)
static inline float Rot6dGetYaw(const Rot6d& rot)
{
    return atan2f(-rot.az, rot.ax);
}

// create a Y-only rotation Rot6d from an angle
static inline Rot6d Rot6dFromYaw(float yaw)
{
    const float c = cosf(yaw);
    const float s = sinf(yaw);
    // Y-rotation matrix:
    // | cos   0  sin |
    // |  0    1   0  |
    // |-sin   0  cos |
    // first column (a) = [cos, 0, -sin], second column (b) = [0, 1, 0]
    return Rot6d{ c, 0.0f, -s, 0.0f, 1.0f, 0.0f };
}

// remove Y-rotation component from a Rot6d, returning just the tilt/roll
// result = inverse(yaw) * rot
// safe to call with rot == out (handles aliasing)
static inline void Rot6dRemoveYComponent(const Rot6d& rot, Rot6d& out)
{
    const float yaw = Rot6dGetYaw(rot);
    const Rot6d invYaw = Rot6dFromYaw(-yaw);
    Rot6d tmp;
    Rot6dMultiply(invYaw, rot, tmp);
    Rot6dNormalize(tmp);  // ensure orthonormal after multiplication
    out = tmp;
}

void Rot6dGetVelocity(const Rot6d& current, const Rot6d& target, float dt, Vector3& outOmega)
{
    Quaternion qCurrent, qTarget;
    Rot6dToQuaternion(current, qCurrent);
    Rot6dToQuaternion(target, qTarget);

    const float qx = qTarget.w * -qCurrent.x + qTarget.x * qCurrent.w + qTarget.y * -qCurrent.z - qTarget.z * -qCurrent.y;
    const float qy = qTarget.w * -qCurrent.y - qTarget.x * -qCurrent.z + qTarget.y * qCurrent.w + qTarget.z * -qCurrent.x;
    const float qz = qTarget.w * -qCurrent.z + qTarget.x * -qCurrent.y - qTarget.y * -qCurrent.x + qTarget.z * qCurrent.w;
    const float qw = qTarget.w * qCurrent.w - qTarget.x * -qCurrent.x - qTarget.y * -qCurrent.y - qTarget.z * -qCurrent.z;

    float absQw = (qw < 0.0f) ? -qw : qw;
    if (absQw > 1.0f) absQw = 1.0f;

    const float angle = 2.0f * std::acos(absQw);

    if (angle < 1e-6f)
    {
        outOmega.x = 0.0f; outOmega.y = 0.0f; outOmega.z = 0.0f;
        return;
    }

    const float sinHalf = std::sqrt(1.0f - absQw * absQw);
    float factor = (angle / dt) / (sinHalf < 1e-6f ? 1.0f : sinHalf);
    if (qw < 0.0f) factor = -factor;

    outOmega.x = qx * factor;
    outOmega.y = qy * factor;
    outOmega.z = qz * factor;
}