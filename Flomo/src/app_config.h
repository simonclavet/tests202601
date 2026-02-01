#pragma once

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "utils.h"
#include "raylib.h"
#include <fstream>

// Simple app config that persists between runs
struct AppConfig {
    // Window
    int windowX = 100;
    int windowY = 100;
    int windowWidth = 2200;
    int windowHeight = 1500;

    // Camera (Unreal mode state - most general representation)
    float cameraPosX = 2.0f;
    float cameraPosY = 1.5f;
    float cameraPosZ = 5.0f;
    float cameraYaw = 3.14159f;  // PI - facing towards -Z
    float cameraPitch = 0.0f;
    float cameraMoveSpeed = 5.0f;
    int cameraMode = 1;  // 0 = Orbit, 1 = Unreal

    // Render settings (persisted)
    // Colors stored as separate ints so parsing/writing JSON remains simple here.
    Color backgroundColor = { 255, 255, 255, 255 };
    float sunLightConeAngle = 0.2f;
    float sunLightStrength = 0.25f;
    float sunAzimuth = 3.14159f / 4.0f;
    float sunAltitude = 0.8f;
    Color sunColor = { 253, 255, 232, 255 };

    float skyLightStrength = 0.15f;
    Color skyColor = { 174, 183, 190, 255 };

    float groundLightStrength = 0.1f;
    float ambientLightStrength = 1.0f;

    float exposure = 0.9f;

    // Toggles
    bool drawOrigin = true;
    bool drawGrid = false;
    bool drawChecker = true;
    bool drawCapsules = true;
    bool drawWireframes = false;
    bool drawSkeleton = true;
    bool drawTransforms = false;
    bool drawAO = true;
    bool drawShadows = true;
    bool drawEndSites = true;
    bool drawFPS = false;
    bool drawUI = true;

    bool drawFeatures = false;
    bool drawBlendCursors = true;  // Debug: show individual blend cursor skeletons
    bool drawVelocities = false;   // Draw joint velocity vectors
    bool drawAccelerations = false; // Draw joint acceleration vectors
    bool drawRootVelocities = false; // Draw root motion velocity from each cursor
    bool drawToeVelocities = false;  // Draw toe velocity vectors (actual vs blended)
    bool drawFootIK = false;         // Draw foot IK debug (virtual toe positions, etc.)

    // Animation settings
    float defaultBlendTime = 0.1f;  // time for blend cursor spring to reach 95% of target
    float switchInterval = 3.0f;    // time between random animation switches

    // Velocity-based blending
    bool useVelBlending = false;    // enable velocity-based rotation blending
    float blendPosReturnTime = 0.1f; // time to bring the velocity advanced pose to the blended position

    // Foot IK
    bool enableFootIK = true;  // enable/disable foot IK towards virtual toe positions
    bool enableTimedUnlocking = true;  // enable/disable timed unlock mechanism for virtual toes
    float unlockDistance = 0.2f;  // distance threshold to unlock virtual toe (meters)
    float unlockDuration = 0.3f;  // time to gradually re-lock virtual toe (seconds)
    
    
    bool drawPlayerInput = false;

    // Validity
    bool valid = false;
};


static const char* CONFIG_FILENAME = "flomo_config.json";

// Get config file path (next to executable or in working directory)
static inline const char* GetConfigPath()
{
    static char path[512];
    snprintf(path, sizeof(path), "%s", CONFIG_FILENAME);
    return path;
}


// Load config from JSON file
static inline AppConfig LoadAppConfig(int argc, char** argv)
{
    AppConfig config = {};

    std::vector<char> buffer_vec;
    std::ifstream file(GetConfigPath(), std::ios::binary | std::ios::ate);

    if (file) {
        buffer_vec.resize(static_cast<size_t>(file.tellg()) + 1);
        file.seekg(0);
        file.read(buffer_vec.data(), buffer_vec.size() - 1);
        buffer_vec.back() = '\0';  // Ensure null termination
    }
    const char* buffer = buffer_vec.data();

    //// Try to open file; if absent we still apply command-line overrides below.
    //FILE* file = fopen(GetConfigPath(), "r");
    //char* buffer = nullptr;
    //if (file) {
    //    fseek(file, 0, SEEK_END);
    //    const long size = ftell(file);
    //    fseek(file, 0, SEEK_SET);

    //    buffer = (char*)malloc(size + 1);
    //    if (buffer) {
    //        fread(buffer, 1, size, file);
    //        buffer[size] = '\0';
    //    }
    //    fclose(file);
    //}

    // Apply JSON (if any) and then command-line overrides (via Resolve helpers).
    config.windowX = ResolveIntConfig(buffer, "windowX", config.windowX, argc, argv);
    config.windowY = ResolveIntConfig(buffer, "windowY", config.windowY, argc, argv);
    config.windowWidth = ResolveIntConfig(buffer, "windowWidth", config.windowWidth, argc, argv);
    config.windowHeight = ResolveIntConfig(buffer, "windowHeight", config.windowHeight, argc, argv);

    config.cameraPosX = ResolveFloatConfig(buffer, "cameraPosX", config.cameraPosX, argc, argv);
    config.cameraPosY = ResolveFloatConfig(buffer, "cameraPosY", config.cameraPosY, argc, argv);
    config.cameraPosZ = ResolveFloatConfig(buffer, "cameraPosZ", config.cameraPosZ, argc, argv);
    config.cameraYaw = ResolveFloatConfig(buffer, "cameraYaw", config.cameraYaw, argc, argv);
    config.cameraPitch = ResolveFloatConfig(buffer, "cameraPitch", config.cameraPitch, argc, argv);
    config.cameraMoveSpeed = ResolveFloatConfig(buffer, "cameraMoveSpeed", config.cameraMoveSpeed, argc, argv);
    config.cameraMode = ResolveIntConfig(buffer, "cameraMode", config.cameraMode, argc, argv);


    // Render fields (colors as arrays)
    config.backgroundColor = ResolveColorConfig(buffer, "backgroundColor", config.backgroundColor, argc, argv);

    config.sunLightConeAngle = ResolveFloatConfig(buffer, "sunLightConeAngle", config.sunLightConeAngle, argc, argv);
    config.sunLightStrength = ResolveFloatConfig(buffer, "sunLightStrength", config.sunLightStrength, argc, argv);
    config.sunAzimuth = ResolveFloatConfig(buffer, "sunAzimuth", config.sunAzimuth, argc, argv);
    config.sunAltitude = ResolveFloatConfig(buffer, "sunAltitude", config.sunAltitude, argc, argv);
    config.sunColor = ResolveColorConfig(buffer, "sunColor", config.sunColor, argc, argv);

    config.skyLightStrength = ResolveFloatConfig(buffer, "skyLightStrength", config.skyLightStrength, argc, argv);
    config.skyColor = ResolveColorConfig(buffer, "skyColor", config.skyColor, argc, argv);


    config.groundLightStrength = ResolveFloatConfig(buffer, "groundLightStrength", config.groundLightStrength, argc, argv);
    config.ambientLightStrength = ResolveFloatConfig(buffer, "ambientLightStrength", config.ambientLightStrength, argc, argv);

    config.exposure = ResolveFloatConfig(buffer, "exposure", config.exposure, argc, argv);

    config.drawOrigin = ResolveBoolConfig(buffer, "drawOrigin", config.drawOrigin, argc, argv);
    config.drawGrid = ResolveBoolConfig(buffer, "drawGrid", config.drawGrid, argc, argv);
    config.drawChecker = ResolveBoolConfig(buffer, "drawChecker", config.drawChecker, argc, argv);
    config.drawCapsules = ResolveBoolConfig(buffer, "drawCapsules", config.drawCapsules, argc, argv);
    config.drawWireframes = ResolveBoolConfig(buffer, "drawWireframes", config.drawWireframes, argc, argv);
    config.drawSkeleton = ResolveBoolConfig(buffer, "drawSkeleton", config.drawSkeleton, argc, argv);
    config.drawTransforms = ResolveBoolConfig(buffer, "drawTransforms", config.drawTransforms, argc, argv);
    config.drawAO = ResolveBoolConfig(buffer, "drawAO", config.drawAO, argc, argv);
    config.drawShadows = ResolveBoolConfig(buffer, "drawShadows", config.drawShadows, argc, argv);
    config.drawEndSites = ResolveBoolConfig(buffer, "drawEndSites", config.drawEndSites, argc, argv);
    config.drawFPS = ResolveBoolConfig(buffer, "drawFPS", config.drawFPS, argc, argv);
    config.drawUI = ResolveBoolConfig(buffer, "drawUI", config.drawUI, argc, argv);

    config.drawFeatures = ResolveBoolConfig(buffer, "drawFeatures", config.drawFeatures, argc, argv);
    config.drawBlendCursors = ResolveBoolConfig(buffer, "drawBlendCursors", config.drawBlendCursors, argc, argv);
    config.drawVelocities = ResolveBoolConfig(buffer, "drawVelocities", config.drawVelocities, argc, argv);
    config.drawAccelerations = ResolveBoolConfig(buffer, "drawAccelerations", config.drawAccelerations, argc, argv);
    config.drawRootVelocities = ResolveBoolConfig(buffer, "drawRootVelocities", config.drawRootVelocities, argc, argv);
    config.drawToeVelocities = ResolveBoolConfig(buffer, "drawToeVelocities", config.drawToeVelocities, argc, argv);
    config.drawFootIK = ResolveBoolConfig(buffer, "drawFootIK", config.drawFootIK, argc, argv);

    config.defaultBlendTime = ResolveFloatConfig(buffer, "defaultBlendTime", config.defaultBlendTime, argc, argv);
    config.switchInterval = ResolveFloatConfig(buffer, "switchInterval", config.switchInterval, argc, argv);

    config.useVelBlending = ResolveBoolConfig(buffer, "useVelBlending", config.useVelBlending, argc, argv);
    config.blendPosReturnTime = ResolveFloatConfig(buffer, "blendPosReturnTime", config.blendPosReturnTime, argc, argv);
    
    config.enableFootIK = ResolveBoolConfig(buffer, "enableFootIK", config.enableFootIK, argc, argv);
    
    config.enableTimedUnlocking = ResolveBoolConfig(buffer, "enableTimedUnlocking", config.enableTimedUnlocking, argc, argv);
    config.unlockDistance = ResolveFloatConfig(buffer, "unlockDistance", config.unlockDistance, argc, argv);
    config.unlockDuration = ResolveFloatConfig(buffer, "unlockDuration", config.unlockDuration, argc, argv);
    config.drawPlayerInput = ResolveBoolConfig(buffer, "drawPlayerInput", false, argc, argv);

    // Validate window values
    if (config.windowX >= 0 && config.windowY >= 0 &&
        config.windowWidth > 100 && config.windowHeight > 100) {
        config.valid = true;
    }

    return config;
}

static inline void SaveAppConfig(const AppConfig& cfg)
{
    FILE* file = fopen(GetConfigPath(), "w");
    if (!file) {
        return;
    }

    fprintf(file, "{\n");
    fprintf(file, "    \"windowX\": %d,\n", cfg.windowX);
    fprintf(file, "    \"windowY\": %d,\n", cfg.windowY);
    fprintf(file, "    \"windowWidth\": %d,\n", cfg.windowWidth);
    fprintf(file, "    \"windowHeight\": %d,\n", cfg.windowHeight);
    fprintf(file, "    \"cameraPosX\": %.4f,\n", cfg.cameraPosX);
    fprintf(file, "    \"cameraPosY\": %.4f,\n", cfg.cameraPosY);
    fprintf(file, "    \"cameraPosZ\": %.4f,\n", cfg.cameraPosZ);
    fprintf(file, "    \"cameraYaw\": %.4f,\n", cfg.cameraYaw);
    fprintf(file, "    \"cameraPitch\": %.4f,\n", cfg.cameraPitch);
    fprintf(file, "    \"cameraMoveSpeed\": %.4f,\n", cfg.cameraMoveSpeed);
    fprintf(file, "    \"cameraMode\": %d,\n", cfg.cameraMode);

    // Render settings (colors as arrays)
    fprintf(file, "    \"backgroundColor\": [ %d, %d, %d ],\n", cfg.backgroundColor.r, cfg.backgroundColor.g, cfg.backgroundColor.b);

    fprintf(file, "    \"sunLightConeAngle\": %.6f,\n", cfg.sunLightConeAngle);
    fprintf(file, "    \"sunLightStrength\": %.6f,\n", cfg.sunLightStrength);
    fprintf(file, "    \"sunAzimuth\": %.6f,\n", cfg.sunAzimuth);
    fprintf(file, "    \"sunAltitude\": %.6f,\n", cfg.sunAltitude);
    fprintf(file, "    \"sunColor\": [ %d, %d, %d ],\n", cfg.sunColor.r, cfg.sunColor.g, cfg.sunColor.b);

    fprintf(file, "    \"skyLightStrength\": %.6f,\n", cfg.skyLightStrength);
    fprintf(file, "    \"skyColor\": [ %d, %d, %d ],\n", cfg.skyColor.r, cfg.skyColor.g, cfg.skyColor.b);

    fprintf(file, "    \"groundLightStrength\": %.6f,\n", cfg.groundLightStrength);
    fprintf(file, "    \"ambientLightStrength\": %.6f,\n", cfg.ambientLightStrength);

    fprintf(file, "    \"exposure\": %.6f,\n", cfg.exposure);

    fprintf(file, "    \"drawOrigin\": %s,\n", cfg.drawOrigin ? "true" : "false");
    fprintf(file, "    \"drawGrid\": %s,\n", cfg.drawGrid ? "true" : "false");
    fprintf(file, "    \"drawChecker\": %s,\n", cfg.drawChecker ? "true" : "false");
    fprintf(file, "    \"drawCapsules\": %s,\n", cfg.drawCapsules ? "true" : "false");
    fprintf(file, "    \"drawWireframes\": %s,\n", cfg.drawWireframes ? "true" : "false");
    fprintf(file, "    \"drawSkeleton\": %s,\n", cfg.drawSkeleton ? "true" : "false");
    fprintf(file, "    \"drawTransforms\": %s,\n", cfg.drawTransforms ? "true" : "false");
    fprintf(file, "    \"drawAO\": %s,\n", cfg.drawAO ? "true" : "false");
    fprintf(file, "    \"drawShadows\": %s,\n", cfg.drawShadows ? "true" : "false");
    fprintf(file, "    \"drawEndSites\": %s,\n", cfg.drawEndSites ? "true" : "false");
    fprintf(file, "    \"drawFPS\": %s,\n", cfg.drawFPS ? "true" : "false");
    fprintf(file, "    \"drawUI\": %s,\n", cfg.drawUI ? "true" : "false");

    fprintf(file, "    \"drawFeatures\": %s,\n", cfg.drawFeatures ? "true" : "false");
    fprintf(file, "    \"drawBlendCursors\": %s,\n", cfg.drawBlendCursors ? "true" : "false");
    fprintf(file, "    \"drawVelocities\": %s,\n", cfg.drawVelocities ? "true" : "false");
    fprintf(file, "    \"drawAccelerations\": %s,\n", cfg.drawAccelerations ? "true" : "false");
    fprintf(file, "    \"drawRootVelocities\": %s,\n", cfg.drawRootVelocities ? "true" : "false");
    fprintf(file, "    \"drawToeVelocities\": %s,\n", cfg.drawToeVelocities ? "true" : "false");
    fprintf(file, "    \"drawFootIK\": %s,\n", cfg.drawFootIK ? "true" : "false");

    fprintf(file, "    \"defaultBlendTime\": %.4f,\n", cfg.defaultBlendTime);
    fprintf(file, "    \"switchInterval\": %.4f,\n", cfg.switchInterval);

    fprintf(file, "    \"useVelBlending\": %s,\n", cfg.useVelBlending ? "true" : "false");
    fprintf(file, "    \"blendPosReturnTime\": %.4f,\n", cfg.blendPosReturnTime);
    fprintf(file, "    \"enableFootIK\": %s,\n", cfg.enableFootIK ? "true" : "false");
    
    fprintf(file, "    \"enableTimedUnlocking\": %s,\n", cfg.enableTimedUnlocking ? "true" : "false");
    fprintf(file, "    \"unlockDistance\": %.4f,\n", cfg.unlockDistance);
    fprintf(file, "    \"unlockDuration\": %.4f,\n", cfg.unlockDuration);
    fprintf(file, "    \"drawPlayerInput\": %s\n", cfg.drawPlayerInput ? "true" : "false");

    fprintf(file, "}\n");

    fclose(file);
}