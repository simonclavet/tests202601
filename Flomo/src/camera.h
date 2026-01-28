#pragma once

#include "raylib.h"
#include "raymath.h"
#include "utils.h"

//----------------------------------------------------------------------------------
// Camera System
//----------------------------------------------------------------------------------

// Camera mode selection (named FlomoCameraMode to avoid raylib's CameraMode)
enum class FlomoCameraMode {
    Orbit,      // Orbit around a target (character bone)
    Unreal      // Unreal Engine style free camera
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
};

// Combined camera system
struct CameraSystem {
    Camera3D cam3d;         // The actual raylib camera used for rendering
    FlomoCameraMode mode;   // Current camera mode
    OrbitCamera orbit;      // Orbit camera state
    UnrealCamera unreal;    // Unreal camera state
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
    cam->mode = FlomoCameraMode::Unreal;

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
    float azimuthDelta,
    float altitudeDelta,
    float offsetDeltaX,
    float offsetDeltaY,
    float mouseWheel,
    float dt)
{
    OrbitCamera* orbit = &cam->orbit;

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

    const Vector3 cameraTarget = Vector3Add(orbit->offset, target);
    const Vector3 eye = Vector3Add(cameraTarget, Vector3RotateByQuaternion(position, rotationAltitude));

    cam->cam3d.target = cameraTarget;
    cam->cam3d.position = eye;
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

        // Movement
        Vector3 velocity = Vector3Zero();
        if (moveForward) velocity = Vector3Add(velocity, forward);
        if (moveBack)    velocity = Vector3Subtract(velocity, forward);
        if (moveRight)   velocity = Vector3Add(velocity, right);
        if (moveLeft)    velocity = Vector3Subtract(velocity, right);
        if (moveUp)      velocity = Vector3Add(velocity, up);
        if (moveDown)    velocity = Vector3Subtract(velocity, up);

        // Normalize and apply speed
        if (Vector3Length(velocity) > 0.001f)
        {
            velocity = Vector3Normalize(velocity);
            velocity = Vector3Scale(velocity, unreal->moveSpeed * dt);
            unreal->position = Vector3Add(unreal->position, velocity);
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

    cam->mode = FlomoCameraMode::Unreal;
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
