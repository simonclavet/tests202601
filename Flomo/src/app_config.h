#pragma once

#include "utils.h"
#include "raylib.h"
#include "definitions.h"



static const char* CONFIG_FILENAME = "flomo_config.json";

// Get config file path (next to executable or in working directory)
static inline const char* GetConfigPath()
{
    static char path[512];
    snprintf(path, sizeof(path), "%s", CONFIG_FILENAME);
    return path;
}


// Load motion matching config from JSON buffer
static inline void MotionMatchingConfigFromJson(const char* jsonBuffer, MotionMatchingFeaturesConfig& config)
{
    if (!jsonBuffer) return;

    // Parse feature weights
    for (int i = 0; i < static_cast<int>(FeatureType::COUNT); ++i)
    {
        const FeatureType type = static_cast<FeatureType>(i);
        const char* name = FeatureTypeName(type);
        config.features[i].weight = ParseFloatValue(jsonBuffer, name, config.features[i].weight);
    }

    // Parse pastTimeOffset
    config.pastTimeOffset = ParseFloatValue(jsonBuffer, "pastTimeOffset", config.pastTimeOffset);

    // Parse future trajectory times array
    // Look for "futureTrajPointTimes": [0.2, 0.4, 0.8]
    const char* arrayKey = "futureTrajPointTimes";
    const char* found = strstr(jsonBuffer, arrayKey);
    if (found)
    {
        const char* bracket = strchr(found, '[');
        if (bracket)
        {
            config.futureTrajPointTimes.clear();
            const char* p = bracket + 1;

            // Parse comma-separated floats until we hit ']'
            while (*p && *p != ']')
            {
                // Skip whitespace
                while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
                if (*p == ']') break;

                // Parse float
                char* end = nullptr;
                float val = strtof(p, &end);
                if (end != p)  // Successfully parsed
                {
                    config.futureTrajPointTimes.push_back(val);
                    p = end;
                }

                // Skip comma and whitespace
                while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
            }
        }
    }
}


// Save motion matching config to JSON string
static inline std::string MotionMatchingConfigToJson(const MotionMatchingFeaturesConfig& config)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"featureWeights\": {\n";

    for (int i = 0; i < static_cast<int>(FeatureType::COUNT); ++i)
    {
        const FeatureType type = static_cast<FeatureType>(i);
        oss << "    \"" << FeatureTypeName(type) << "\": " << config.features[i].weight;
        if (i < static_cast<int>(FeatureType::COUNT) - 1) oss << ",";
        oss << "\n";
    }

    oss << "  },\n";

    oss << "  \"pastTimeOffset\": " << config.pastTimeOffset << ",\n";

    oss << "  \"futureTrajPointTimes\": [";
    for (size_t i = 0; i < config.futureTrajPointTimes.size(); ++i)
    {
        oss << config.futureTrajPointTimes[i];
        if (i < config.futureTrajPointTimes.size() - 1) oss << ", ";
    }
    oss << "]\n";
    oss << "}";

    return oss.str();
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
    config.trackHipsProjectedOnGround = ResolveBoolConfig(buffer, "trackHipsProjectedOnGround", config.trackHipsProjectedOnGround, argc, argv);


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
    config.drawPastHistory = ResolveBoolConfig(buffer, "drawPastHistory", config.drawPastHistory, argc, argv);

    config.animationMode = static_cast<AnimationMode>(ResolveIntConfig(buffer, "animationMode", static_cast<int>(config.animationMode), argc, argv));
    config.defaultBlendTime = ResolveFloatConfig(buffer, "defaultBlendTime", config.defaultBlendTime, argc, argv);
    config.switchInterval = ResolveFloatConfig(buffer, "switchInterval", config.switchInterval, argc, argv);
    config.mmSearchPeriod = ResolveFloatConfig(buffer, "mmSearchPeriod", config.mmSearchPeriod, argc, argv);
    config.virtualControlMaxAcceleration = ResolveFloatConfig(buffer, "virtualControlMaxAcceleration", config.virtualControlMaxAcceleration, argc, argv);
    config.poseDragLookaheadTimeEditor = ResolveFloatConfig(buffer, "poseDragLookaheadTime", config.poseDragLookaheadTimeEditor, argc, argv);

    config.cursorBlendMode = static_cast<CursorBlendMode>(ResolveIntConfig(buffer, "cursorBlendMode", static_cast<int>(config.cursorBlendMode), argc, argv));
    config.blendPosReturnTime = ResolveFloatConfig(buffer, "blendPosReturnTime", config.blendPosReturnTime, argc, argv);

    config.enableFootIK = ResolveBoolConfig(buffer, "enableFootIK", config.enableFootIK, argc, argv);
    
    config.enableTimedUnlocking = ResolveBoolConfig(buffer, "enableTimedUnlocking", config.enableTimedUnlocking, argc, argv);
    config.unlockDistance = ResolveFloatConfig(buffer, "unlockDistance", config.unlockDistance, argc, argv);
    config.unlockDuration = ResolveFloatConfig(buffer, "unlockDuration", config.unlockDuration, argc, argv);
    config.drawPlayerInput = ResolveBoolConfig(buffer, "drawPlayerInput", false, argc, argv);

    // Initialize motion matching config with defaults first
    MotionMatchingConfigInit(config.mmConfigEditor);

    // Then load from JSON (if available)
    if (buffer)
    {
        MotionMatchingConfigFromJson(buffer, config.mmConfigEditor);
    }


    // Validate window values
    if (config.windowX >= 0 && config.windowY >= 0 &&
        config.windowWidth > 100 && config.windowHeight > 100) {
        config.valid = true;
    }

    return config;
}



static inline void SaveAppConfig(const AppConfig& config)
{
    FILE* file = fopen(GetConfigPath(), "w");
    if (!file) {
        return;
    }

    fprintf(file, "{\n");
    fprintf(file, "    \"windowX\": %d,\n", config.windowX);
    fprintf(file, "    \"windowY\": %d,\n", config.windowY);
    fprintf(file, "    \"windowWidth\": %d,\n", config.windowWidth);
    fprintf(file, "    \"windowHeight\": %d,\n", config.windowHeight);
    fprintf(file, "    \"cameraPosX\": %.4f,\n", config.cameraPosX);
    fprintf(file, "    \"cameraPosY\": %.4f,\n", config.cameraPosY);
    fprintf(file, "    \"cameraPosZ\": %.4f,\n", config.cameraPosZ);
    fprintf(file, "    \"cameraYaw\": %.4f,\n", config.cameraYaw);
    fprintf(file, "    \"cameraPitch\": %.4f,\n", config.cameraPitch);
    fprintf(file, "    \"cameraMoveSpeed\": %.4f,\n", config.cameraMoveSpeed);
    fprintf(file, "    \"cameraMode\": %d,\n", config.cameraMode);
    fprintf(file, "    \"trackHipsProjectedOnGround\": %s,\n", config.trackHipsProjectedOnGround ? "true" : "false");

    // Render settings (colors as arrays)
    fprintf(file, "    \"backgroundColor\": [ %d, %d, %d ],\n", config.backgroundColor.r, config.backgroundColor.g, config.backgroundColor.b);

    fprintf(file, "    \"sunLightConeAngle\": %.6f,\n", config.sunLightConeAngle);
    fprintf(file, "    \"sunLightStrength\": %.6f,\n", config.sunLightStrength);
    fprintf(file, "    \"sunAzimuth\": %.6f,\n", config.sunAzimuth);
    fprintf(file, "    \"sunAltitude\": %.6f,\n", config.sunAltitude);
    fprintf(file, "    \"sunColor\": [ %d, %d, %d ],\n", config.sunColor.r, config.sunColor.g, config.sunColor.b);

    fprintf(file, "    \"skyLightStrength\": %.6f,\n", config.skyLightStrength);
    fprintf(file, "    \"skyColor\": [ %d, %d, %d ],\n", config.skyColor.r, config.skyColor.g, config.skyColor.b);

    fprintf(file, "    \"groundLightStrength\": %.6f,\n", config.groundLightStrength);
    fprintf(file, "    \"ambientLightStrength\": %.6f,\n", config.ambientLightStrength);

    fprintf(file, "    \"exposure\": %.6f,\n", config.exposure);

    fprintf(file, "    \"drawOrigin\": %s,\n", config.drawOrigin ? "true" : "false");
    fprintf(file, "    \"drawGrid\": %s,\n", config.drawGrid ? "true" : "false");
    fprintf(file, "    \"drawChecker\": %s,\n", config.drawChecker ? "true" : "false");
    fprintf(file, "    \"drawCapsules\": %s,\n", config.drawCapsules ? "true" : "false");
    fprintf(file, "    \"drawWireframes\": %s,\n", config.drawWireframes ? "true" : "false");
    fprintf(file, "    \"drawSkeleton\": %s,\n", config.drawSkeleton ? "true" : "false");
    fprintf(file, "    \"drawTransforms\": %s,\n", config.drawTransforms ? "true" : "false");
    fprintf(file, "    \"drawAO\": %s,\n", config.drawAO ? "true" : "false");
    fprintf(file, "    \"drawShadows\": %s,\n", config.drawShadows ? "true" : "false");
    fprintf(file, "    \"drawEndSites\": %s,\n", config.drawEndSites ? "true" : "false");
    fprintf(file, "    \"drawFPS\": %s,\n", config.drawFPS ? "true" : "false");
    fprintf(file, "    \"drawUI\": %s,\n", config.drawUI ? "true" : "false");

    fprintf(file, "    \"drawFeatures\": %s,\n", config.drawFeatures ? "true" : "false");
    fprintf(file, "    \"drawBlendCursors\": %s,\n", config.drawBlendCursors ? "true" : "false");
    fprintf(file, "    \"drawVelocities\": %s,\n", config.drawVelocities ? "true" : "false");
    fprintf(file, "    \"drawAccelerations\": %s,\n", config.drawAccelerations ? "true" : "false");
    fprintf(file, "    \"drawRootVelocities\": %s,\n", config.drawRootVelocities ? "true" : "false");
    fprintf(file, "    \"drawToeVelocities\": %s,\n", config.drawToeVelocities ? "true" : "false");
    fprintf(file, "    \"drawFootIK\": %s,\n", config.drawFootIK ? "true" : "false");
    fprintf(file, "  \"drawPastHistory\": %s,\n", config.drawPastHistory ? "true" : "false");

    fprintf(file, "    \"animationMode\": %d,\n", static_cast<int>(config.animationMode));
    fprintf(file, "    \"defaultBlendTime\": %.4f,\n", config.defaultBlendTime);
    fprintf(file, "    \"switchInterval\": %.4f,\n", config.switchInterval);
    fprintf(file, "    \"mmSearchPeriod\": %.4f,\n", config.mmSearchPeriod);
    fprintf(file, "    \"virtualControlMaxAcceleration\": %.4f,\n", config.virtualControlMaxAcceleration);

    fprintf(file, "    \"cursorBlendMode\": %d,\n", static_cast<int>(config.cursorBlendMode));
    fprintf(file, "    \"blendPosReturnTime\": %.4f,\n", config.blendPosReturnTime);
    fprintf(file, "    \"enableFootIK\": %s,\n", config.enableFootIK ? "true" : "false");
    
    fprintf(file, "    \"enableTimedUnlocking\": %s,\n", config.enableTimedUnlocking ? "true" : "false");
    fprintf(file, "    \"unlockDistance\": %.4f,\n", config.unlockDistance);
    fprintf(file, "    \"unlockDuration\": %.4f,\n", config.unlockDuration);
    fprintf(file, "    \"drawPlayerInput\": %s\n", config.drawPlayerInput ? "true" : "false");


    // Save motion matching config
    std::string mmConfigJson = MotionMatchingConfigToJson(config.mmConfigEditor);
    fprintf(file, "  \"motionMatchingConfig\": %s,\n", mmConfigJson.c_str());
    fprintf(file, "    \"poseDragLookaheadTime\": %.4f,\n", config.poseDragLookaheadTimeEditor);

    fprintf(file, "}\n");

    fclose(file);
}


