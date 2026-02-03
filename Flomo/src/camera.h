#pragma once

#include "raylib.h"
#include "raymath.h"
#include "utils.h"
#include "math_utils.h"

//----------------------------------------------------------------------------------
// Camera System
//----------------------------------------------------------------------------------

// Camera mode selection (named FlomoCameraMode to avoid raylib's CameraMode)
enum class FlomoCameraMode {
    Orbit,            // Orbit around a target (character bone)
    UnrealEditor,     // Unreal Engine style free camera
    LazyTurretFollower // Follows target, maintains distance, allows character control
};

// Orbit Camera - rotates around a target point
struct OrbitCamera {
    float azimuth;
    float altitude;
    float distance;
    Vector3 offset;
    bool track;
    int trackBone;
    bool trackControlledCharacter;  // If true, track controlled character instead of active character
};

// Unreal-style free camera
// Controls: LMB + WASD to move, Q/E for down/up, scroll wheel for speed
struct UnrealCamera {
    Vector3 position;
    float yaw;          // Horizontal rotation (radians)
    float pitch;        // Vertical rotation (radians)
    float moveSpeed;    // Current movement speed
    float minSpeed;     // Minimum speed (scroll wheel limit)
    float maxSpeed;     // Maximum speed (scroll wheel limit)
    Vector3 velocity;   // Current velocity for acceleration limiting
};

// Lazy turret follower - follows target but allows character control
// Hybrid: can manually rotate/zoom like orbit, rotates to track target when passive
struct LazyTurretCamera {
    float azimuth;      // Horizontal angle around target (user-controlled or computed)
    float altitude;     // Vertical angle (elevation)
    float distance;     // Current distance from target
    float minDistance;  // Minimum distance from target (avoid clipping)
    float maxDistance;  // Maximum distance from target
    float smoothTime;   // Smoothing time for position updates (spring-like follow)
    Vector3 velocity;   // Current velocity for smooth damping
    Vector3 position;   // Actual camera position (smoothed when passive)
};

// Combined camera system
struct CameraSystem {
    Camera3D cam3d;             // The actual raylib camera used for rendering
    FlomoCameraMode mode;       // Current camera mode
    OrbitCamera orbit;          // Orbit camera state
    UnrealCamera unreal;        // Unreal camera state
    LazyTurretCamera turret;    // Lazy turret follower state

    // Smooth target following (used by Orbit and LazyTurretFollower)
    DoubleSpringDamperStateVector3 smoothedTarget;  // Damped target position
};

static inline void CameraSystemInit(CameraSystem* cam, int argc, char** argv)
{
    // Initialize raylib Camera3D
    memset(&cam->cam3d, 0, sizeof(Camera3D));
    cam->cam3d.position = Vector3{ 2.0f, 3.0f, 5.0f };
    cam->cam3d.target = Vector3{ -0.5f, 1.0f, 0.0f };
    cam->cam3d.up = Vector3{ 0.0f, 1.0f, 0.0f };
    cam->cam3d.fovy = ArgFloat(argc, argv, "cameraFOV", 45.0f);
    cam->cam3d.projection = CAMERA_PERSPECTIVE;

    // Default mode
    cam->mode = FlomoCameraMode::UnrealEditor;

    // Initialize Orbit camera
    cam->orbit.azimuth = ArgFloat(argc, argv, "cameraAzimuth", 0.0f);
    cam->orbit.altitude = ArgFloat(argc, argv, "cameraAltitude", 0.4f);
    cam->orbit.distance = ArgFloat(argc, argv, "cameraDistance", 4.0f);
    cam->orbit.offset = ArgVector3(argc, argv, "cameraOffset", Vector3Zero());
    cam->orbit.track = ArgBool(argc, argv, "cameraTrack", true);
    cam->orbit.trackBone = ArgInt(argc, argv, "cameraTrackBone", 0);
    cam->orbit.trackControlledCharacter = true;

    // Initialize Unreal camera
    cam->unreal.position = Vector3{ 2.0f, 1.5f, 5.0f };
    cam->unreal.yaw = PI;       // Face towards -Z (towards origin)
    cam->unreal.pitch = 0.0f;
    cam->unreal.moveSpeed = 5.0f;
    cam->unreal.minSpeed = 0.5f;
    cam->unreal.maxSpeed = 50.0f;
    cam->unreal.velocity = Vector3Zero();

    // Initialize Lazy Turret camera
    cam->turret.azimuth = 0.0f;
    cam->turret.altitude = 0.4f;
    cam->turret.distance = 5.0f;
    cam->turret.minDistance = 3.0f;
    cam->turret.maxDistance = 6.0f;
    cam->turret.smoothTime = 0.3f;  // Smooth following with slight lag
    cam->turret.velocity = Vector3Zero();
    cam->turret.position = Vector3{ 2.0f, 1.5f, 5.0f };
}

// Legacy init for compatibility
static inline void OrbitCameraInit(OrbitCamera* camera, int argc, char** argv)
{
    camera->azimuth = ArgFloat(argc, argv, "cameraAzimuth", 0.0f);
    camera->altitude = ArgFloat(argc, argv, "cameraAltitude", 0.4f);
    camera->distance = ArgFloat(argc, argv, "cameraDistance", 4.0f);
    camera->offset = ArgVector3(argc, argv, "cameraOffset", Vector3Zero());
    camera->track = ArgBool(argc, argv, "cameraTrack", true);
    camera->trackBone = ArgInt(argc, argv, "cameraTrackBone", 0);
}

// Update orbit camera and write results to cam3d
static inline void OrbitCameraUpdate(
    CameraSystem* cam,
    Vector3 target,
    float targetBlendtime,
    float azimuthDelta,
    float altitudeDelta,
    float offsetDeltaX,
    float offsetDeltaY,
    float mouseWheel,
    float dt)
{
    OrbitCamera* orbit = &cam->orbit;

    // Update smoothed target using double spring damper
    DoubleSpringDamperVector3(cam->smoothedTarget, target, targetBlendtime, dt);

    orbit->azimuth = orbit->azimuth + 1.0f * dt * -azimuthDelta;
    orbit->altitude = Clamp(orbit->altitude + 1.0f * dt * altitudeDelta, 0.0f, 0.4f * PI);
    orbit->distance = Clamp(orbit->distance + 20.0f * dt * -mouseWheel, 0.1f, 100.0f);

    const Quaternion rotationAzimuth = QuaternionFromAxisAngle(Vector3{0, 1, 0}, orbit->azimuth);
    const Vector3 position = Vector3RotateByQuaternion(Vector3{0, 0, orbit->distance}, rotationAzimuth);
    const Vector3 axis = Vector3Normalize(Vector3CrossProduct(position, Vector3{0, 1, 0}));

    const Quaternion rotationAltitude = QuaternionFromAxisAngle(axis, orbit->altitude);

    Vector3 localOffset = Vector3{ dt * -offsetDeltaX, dt * offsetDeltaY, 0.0f };
    localOffset = Vector3RotateByQuaternion(localOffset, rotationAzimuth);

    orbit->offset = Vector3Add(orbit->offset, Vector3RotateByQuaternion(localOffset, rotationAltitude));

    // Use smoothed target instead of raw target
    const Vector3 cameraTarget = Vector3Add(orbit->offset, cam->smoothedTarget.x);
    const Vector3 eye = Vector3Add(cameraTarget, Vector3RotateByQuaternion(position, rotationAltitude));

    cam->cam3d.target = cameraTarget;
    cam->cam3d.position = eye;
}

// Update Lazy Turret Follower camera - follows target, allows character control
// Hybrid: manual control with mouse (orbit-like), auto-tracks target when passive (turret-like)
static inline void LazyTurretCameraUpdate(
    CameraSystem* cam,
    Vector3 target,
    float targetBlendtime,
    float azimuthDelta,
    float altitudeDelta,
    float mouseWheel,
    float dt)
{
    LazyTurretCamera* turret = &cam->turret;

    // Update smoothed target (like Orbit mode)
    DoubleSpringDamperVector3(cam->smoothedTarget, target, targetBlendtime, dt);

    // Detect if user is actively controlling (has rotation or zoom input)
    const bool hasRotationInput = (fabsf(azimuthDelta) > 0.001f || fabsf(altitudeDelta) > 0.001f);
    const bool hasZoomInput = (fabsf(mouseWheel) > 0.001f);
    const bool isUserControlling = hasRotationInput || hasZoomInput;

    // Update distance from mouse wheel
    if (hasZoomInput)
    {
        turret->distance = Clamp(turret->distance + 20.0f * dt * -mouseWheel, turret->minDistance, turret->maxDistance);
    }

    if (isUserControlling)
    {
        // Active mode: user controls azimuth/altitude (orbit-like behavior)
        turret->azimuth = turret->azimuth + 1.0f * dt * -azimuthDelta;
        turret->altitude = Clamp(turret->altitude + 1.0f * dt * altitudeDelta, 0.0f, 0.4f * PI);

        // Compute position from user-controlled angles (no position smoothing - direct control)
        const Quaternion rotationAzimuth = QuaternionFromAxisAngle(Vector3{0, 1, 0}, turret->azimuth);
        const Vector3 offset = Vector3RotateByQuaternion(Vector3{0, 0, turret->distance}, rotationAzimuth);
        const Vector3 axis = Vector3Normalize(Vector3CrossProduct(offset, Vector3{0, 1, 0}));
        const Quaternion rotationAltitude = QuaternionFromAxisAngle(axis, turret->altitude);

        turret->position = Vector3Add(cam->smoothedTarget.x, Vector3RotateByQuaternion(offset, rotationAltitude));
        turret->velocity = Vector3Zero();  // Reset velocity when user takes control
    }
    else
    {
        // Passive mode: turret tracks target, smooths position for distance constraints

        // Compute desired position: maintain current direction from smoothed target, clamp distance
        const Vector3 toCamera = Vector3Subtract(turret->position, cam->smoothedTarget.x);
        const float currentDist = Vector3Length(toCamera);

        Vector3 desiredOffset;
        if (currentDist < 0.1f)
        {
            // Fallback: place camera at current azimuth/altitude
            const Quaternion rotationAzimuth = QuaternionFromAxisAngle(Vector3{0, 1, 0}, turret->azimuth);
            const Vector3 offset = Vector3RotateByQuaternion(Vector3{0, 0, turret->distance}, rotationAzimuth);
            const Vector3 axis = Vector3Normalize(Vector3CrossProduct(offset, Vector3{0, 1, 0}));
            const Quaternion rotationAltitude = QuaternionFromAxisAngle(axis, turret->altitude);
            desiredOffset = Vector3RotateByQuaternion(offset, rotationAltitude);
        }
        else
        {
            // Maintain current direction but clamp distance
            const float clampedDist = Clamp(currentDist, turret->minDistance, turret->maxDistance);
            desiredOffset = Vector3Scale(toCamera, clampedDist / currentDist);
        }

        const Vector3 desiredPosition = Vector3Add(cam->smoothedTarget.x, desiredOffset);

        // Smooth position for distance constraints
        SimpleSpringDamperVector3(turret->position, turret->velocity, desiredPosition, turret->smoothTime, dt);

        // Update azimuth/altitude from actual camera position (so user control starts smoothly)
        const Vector3 actualOffset = Vector3Subtract(turret->position, cam->smoothedTarget.x);
        const Vector3 horizontal = Vector3{ actualOffset.x, 0.0f, actualOffset.z };
        const float horizontalDist = Vector3Length(horizontal);

        turret->azimuth = atan2f(actualOffset.x, actualOffset.z);
        turret->altitude = (horizontalDist > 0.001f) ? atanf(actualOffset.y / horizontalDist) : 0.0f;
        turret->altitude = Clamp(turret->altitude, 0.0f, 0.4f * PI);
    }

    // Update cam3d - always look at smoothed target
    cam->cam3d.position = turret->position;
    cam->cam3d.target = cam->smoothedTarget.x;
}

// Calculate forward vector from yaw/pitch
static inline Vector3 UnrealCameraForward(float yaw, float pitch)
{
    return Vector3{
        cosf(pitch) * sinf(yaw),
        sinf(pitch),
        cosf(pitch) * cosf(yaw)
    };
}

// Update Unreal-style camera
// Controls: RMB held + WASD move, Q down, E up
// RMB held + scroll = adjust speed, scroll alone = dolly forward/back
// MMB = pan (strafe left/right, up/down)
static inline void UnrealCameraUpdate(
    CameraSystem* cam,
    float mouseDeltaX,
    float mouseDeltaY,
    float mouseWheel,
    bool moveForward,
    bool moveBack,
    bool moveLeft,
    bool moveRight,
    bool moveUp,
    bool moveDown,
    bool isActive,      // RMB held - enables movement and look
    bool isPanning,     // MMB held - enables panning
    float dt)
{
    UnrealCamera* unreal = &cam->unreal;

    // Scroll wheel behavior depends on whether RMB is held
    if (mouseWheel != 0.0f)
    {
        if (isActive)
        {
            // RMB held: adjust speed (10% per tick, logarithmic)
            unreal->moveSpeed *= (mouseWheel > 0) ? 1.1f : (1.0f / 1.1f);
            unreal->moveSpeed = Clamp(unreal->moveSpeed, unreal->minSpeed, unreal->maxSpeed);
        }
        else
        {
            // RMB not held: dolly forward/backward at current speed
            const Vector3 fwd = UnrealCameraForward(unreal->yaw, unreal->pitch);
            const Vector3 dolly = Vector3Scale(fwd, mouseWheel * unreal->moveSpeed * 0.2f);
            unreal->position = Vector3Add(unreal->position, dolly);
        }
    }

    // MMB panning - strafe in screen space
    if (isPanning)
    {
        const Vector3 forward = UnrealCameraForward(unreal->yaw, unreal->pitch);
        const Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, Vector3{0, 1, 0}));
        const Vector3 up = Vector3{ 0, 1, 0 };

        // Pan speed scales with move speed
        const float panScale = unreal->moveSpeed * 0.002f;
        const Vector3 panOffset = Vector3Add(
            Vector3Scale(right, -mouseDeltaX * panScale),
            Vector3Scale(up, mouseDeltaY * panScale)
        );
        unreal->position = Vector3Add(unreal->position, panOffset);
    }

    // Only update look and movement when active (RMB held)
    if (isActive)
    {
        // Mouse look
        const float sensitivity = 0.003f;
        unreal->yaw -= mouseDeltaX * sensitivity;
        unreal->pitch -= mouseDeltaY * sensitivity;
        unreal->pitch = Clamp(unreal->pitch, -PI * 0.49f, PI * 0.49f);

        // Calculate forward and right vectors (after updating yaw/pitch)
        const Vector3 forward = UnrealCameraForward(unreal->yaw, unreal->pitch);
        const Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, Vector3{0, 1, 0}));
        const Vector3 up = Vector3{ 0, 1, 0 };

        // Compute desired velocity from input
        Vector3 desiredDirection = Vector3Zero();
        if (moveForward) desiredDirection = Vector3Add(desiredDirection, forward);
        if (moveBack)    desiredDirection = Vector3Subtract(desiredDirection, forward);
        if (moveRight)   desiredDirection = Vector3Add(desiredDirection, right);
        if (moveLeft)    desiredDirection = Vector3Subtract(desiredDirection, right);
        if (moveUp)      desiredDirection = Vector3Add(desiredDirection, up);
        if (moveDown)    desiredDirection = Vector3Subtract(desiredDirection, up);

        // Normalize and compute desired velocity
        Vector3 desiredVelocity = Vector3Zero();
        if (Vector3Length(desiredDirection) > 0.001f)
        {
            desiredVelocity = Vector3Normalize(desiredDirection);
            desiredVelocity = Vector3Scale(desiredVelocity, unreal->moveSpeed);
        }

        // Apply acceleration limiting
        const Vector3 velocityDiff = Vector3Subtract(desiredVelocity, unreal->velocity);
        const float velocityDiffLen = Vector3Length(velocityDiff);

        if (velocityDiffLen > 0.001f)
        {
            // Max acceleration depends on whether we're stopping or accelerating
            const bool isStopping = (Vector3Length(desiredVelocity) < 0.001f);
            const float maxAccel = isStopping ? (9.0f * unreal->moveSpeed) : (3.0f * unreal->moveSpeed);
            const float maxDeltaV = maxAccel * dt;

            if (velocityDiffLen <= maxDeltaV)
            {
                // Can reach desired velocity this frame
                unreal->velocity = desiredVelocity;
            }
            else
            {
                // Clamp to max acceleration
                const Vector3 velocityDiffNorm = Vector3Scale(velocityDiff, 1.0f / velocityDiffLen);
                unreal->velocity = Vector3Add(unreal->velocity, Vector3Scale(velocityDiffNorm, maxDeltaV));
            }
        }

        // Apply velocity to position
        unreal->position = Vector3Add(unreal->position, Vector3Scale(unreal->velocity, dt));
    }
    else
    {
        // Not active - apply deceleration
        const float velocityLen = Vector3Length(unreal->velocity);
        if (velocityLen > 0.001f)
        {
            const float maxAccel = 9.0f * unreal->moveSpeed;
            const float maxDeltaV = maxAccel * dt;

            if (velocityLen <= maxDeltaV)
            {
                // Come to complete stop
                unreal->velocity = Vector3Zero();
            }
            else
            {
                // Decelerate
                const Vector3 velocityNorm = Vector3Scale(unreal->velocity, 1.0f / velocityLen);
                unreal->velocity = Vector3Subtract(unreal->velocity, Vector3Scale(velocityNorm, maxDeltaV));
            }

            // Apply velocity to position
            unreal->position = Vector3Add(unreal->position, Vector3Scale(unreal->velocity, dt));
        }
    }

    // Update cam3d with final forward direction
    const Vector3 finalForward = UnrealCameraForward(unreal->yaw, unreal->pitch);
    cam->cam3d.position = unreal->position;
    cam->cam3d.target = Vector3Add(unreal->position, finalForward);
}

// Switch to Unreal mode: keep current camera position and orientation
static inline void CameraSwitchToUnreal(CameraSystem* cam)
{
    // Copy current camera state to unreal camera
    cam->unreal.position = cam->cam3d.position;

    // Calculate yaw and pitch from current viewing direction
    const Vector3 dir = Vector3Normalize(Vector3Subtract(cam->cam3d.target, cam->cam3d.position));
    cam->unreal.yaw = atan2f(dir.x, dir.z);
    cam->unreal.pitch = asinf(Clamp(dir.y, -1.0f, 1.0f));

    cam->mode = FlomoCameraMode::UnrealEditor;
}

// Switch to Orbit mode: aim at target bone, keep current viewing angle
static inline void CameraSwitchToOrbit(CameraSystem* cam, Vector3 boneTarget)
{
    // Calculate current viewing direction
    const Vector3 camPos = cam->cam3d.position;
    const Vector3 toCamera = Vector3Subtract(camPos, boneTarget);
    const float distance = Vector3Length(toCamera);

    // Calculate azimuth (horizontal angle) and altitude (vertical angle)
    const Vector3 horizontal = Vector3{ toCamera.x, 0.0f, toCamera.z };
    const float horizontalDist = Vector3Length(horizontal);

    cam->orbit.azimuth = atan2f(toCamera.x, toCamera.z);
    cam->orbit.altitude = (horizontalDist > 0.001f) ? atanf(toCamera.y / horizontalDist) : 0.0f;
    cam->orbit.altitude = Clamp(cam->orbit.altitude, 0.0f, 0.4f * PI);
    cam->orbit.distance = Clamp(distance, 0.1f, 100.0f);
    cam->orbit.offset = Vector3Zero();

    cam->mode = FlomoCameraMode::Orbit;
}

// Sync all camera modes from current cam3d state
// Call this before switching modes to ensure smooth transitions
static inline void CameraSyncAllModesFromCurrent(CameraSystem* cam, Vector3 targetPosition)
{
    const Vector3 camPos = cam->cam3d.position;
    const Vector3 viewDir = Vector3Normalize(Vector3Subtract(cam->cam3d.target, camPos));

    // Sync Unreal mode: extract yaw/pitch from viewing direction
    cam->unreal.position = camPos;
    cam->unreal.yaw = atan2f(viewDir.x, viewDir.z);
    cam->unreal.pitch = asinf(Clamp(viewDir.y, -1.0f, 1.0f));

    // Sync Orbit mode: compute azimuth/altitude/distance relative to target
    const Vector3 toCamera = Vector3Subtract(camPos, targetPosition);
    const float distance = Vector3Length(toCamera);
    const Vector3 horizontal = Vector3{ toCamera.x, 0.0f, toCamera.z };
    const float horizontalDist = Vector3Length(horizontal);

    cam->orbit.azimuth = atan2f(toCamera.x, toCamera.z);
    cam->orbit.altitude = (horizontalDist > 0.001f) ? atanf(toCamera.y / horizontalDist) : 0.0f;
    cam->orbit.altitude = Clamp(cam->orbit.altitude, 0.0f, 0.4f * PI);
    cam->orbit.distance = Clamp(distance, 0.1f, 100.0f);
    cam->orbit.offset = Vector3Zero();

    // Sync Turret mode: compute azimuth/altitude/distance (same as orbit), set position
    cam->turret.azimuth = cam->orbit.azimuth;
    cam->turret.altitude = cam->orbit.altitude;
    cam->turret.distance = cam->orbit.distance;
    cam->turret.position = camPos;
    cam->turret.velocity = Vector3Zero();  // Reset velocity for smooth start
}
