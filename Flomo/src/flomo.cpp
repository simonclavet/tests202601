// These must stay here (implementation headers with #define macros)
//#define RAYGUI_WINDOWBOX_STATUSBAR_HEIGHT 24
//#define GUI_WINDOW_FILE_DIALOG_IMPLEMENTATION
//#include "gui_window_file_dialog.h"
//#define RAYGUI_IMPLEMENTATION
//#include "raygui.h"
// 
// Dear ImGui with raylib backend
#include "imgui.h"
#include "rlImGui.h"

#include "math_utils.h"
#include "utils.h"

// Un-comment to enable profiling
//#define ENABLE_PROFILE
#include "profiler.h"


#include "definitions.h"
#include "camera.h"
#include "bvh_parser.h"
#include "fbx_loader.h"
#include "transform_data.h"
#include "geometry_utils.h"
#include "capsule_data.h"
#include "mesh_utils.h"
#include "geno_renderer.h"
#include "app_config.h"
#include "balltree.h"
#include "character_data.h"
#include "anim_database.h"
#include "leg_ik.h"
#include "controlled_character.h"


using namespace std;

// Declare the CUDA functions
extern "C" void run_cuda_addition(float* a, float* b, float* c, int n);
extern "C" void cuda_check_error(const char* msg);
extern "C" void test_tiny_cuda_nn();

static void TestCudaAndLibtorchAndTCN()
{
    const int N = 1000000;  // 1 million elements
    vector<float> a(N, 1.0f);
    vector<float> b(N, 2.0f);
    vector<float> c(N, 0.0f);

    cout << "Running CUDA addition..." << endl;

    auto start = chrono::high_resolution_clock::now();

    run_cuda_addition(a.data(), b.data(), c.data(), N);
    cuda_check_error("main execution");

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (c[i] != 3.0f) {
            correct = false;
            break;
        }
    }

    cout << "CUDA addition " << (correct ? "PASSED" : "FAILED") << endl;
    cout << "Time: " << duration.count() << " microseconds" << endl;
    cout << "First 5 results: ";
    for (int i = 0; i < 5; i++) {
        cout << c[i] << " ";
    }
    cout << endl;

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        cout << "LibTorch: Using CUDA device" << endl;
    }
    else {
        cout << "LibTorch: Using CPU device" << endl;
    }

    torch::Tensor tensor = torch::rand({ 3, 3 }).to(device);
    auto result = tensor * 2;
    cout << "Random tensor:\n" << tensor << endl;
    cout << "Tensor * 2:\n" << result << endl;

    // Test tiny-cuda-nn (implemented in cuda_kernels.cu)
    test_tiny_cuda_nn();
}



//----------------------------------------------------------------------------------
// Shaders
//----------------------------------------------------------------------------------


#define AO_CAPSULES_MAX 32
#define SHADOW_CAPSULES_MAX 64
//#define SHADOW_CAPSULES_MAX 16



// Shader uniform location indices (cached for performance)
struct ShaderUniforms {
    // Capsule geometry
    int isCapsule;
    int capsulePosition;
    int capsuleRotation;
    int capsuleHalfLength;
    int capsuleRadius;
    int capsuleStart;
    int capsuleVector;

    // Shadow casting capsules
    int shadowCapsuleCount;
    int shadowCapsuleStarts;
    int shadowCapsuleVectors;
    int shadowCapsuleRadii;
    int shadowLookupTable;
    int shadowLookupResolution;

    // Ambient occlusion capsules
    int aoCapsuleCount;
    int aoCapsuleStarts;
    int aoCapsuleVectors;
    int aoCapsuleRadii;
    int aoLookupTable;
    int aoLookupResolution;

    // Camera
    int cameraPosition;

    // Material properties
    int objectColor;
    int objectSpecularity;
    int objectGlossiness;
    int objectOpacity;

    // Lighting
    int sunStrength;
    int sunDir;
    int sunColor;
    int skyStrength;
    int skyColor;
    int ambientStrength;
    int groundStrength;

    // Tonemapping
    int exposure;
};

// Lookup all shader uniform indices
static void ShaderUniformsInit(ShaderUniforms* uniforms, Shader shader)
{
    uniforms->isCapsule = GetShaderLocation(shader, "isCapsule");
    uniforms->capsulePosition = GetShaderLocation(shader, "capsulePosition");
    uniforms->capsuleRotation = GetShaderLocation(shader, "capsuleRotation");
    uniforms->capsuleHalfLength = GetShaderLocation(shader, "capsuleHalfLength");
    uniforms->capsuleRadius = GetShaderLocation(shader, "capsuleRadius");
    uniforms->capsuleStart = GetShaderLocation(shader, "capsuleStart");
    uniforms->capsuleVector = GetShaderLocation(shader, "capsuleVector");

    uniforms->shadowCapsuleCount = GetShaderLocation(shader, "shadowCapsuleCount");
    uniforms->shadowCapsuleStarts = GetShaderLocation(shader, "shadowCapsuleStarts");
    uniforms->shadowCapsuleVectors = GetShaderLocation(shader, "shadowCapsuleVectors");
    uniforms->shadowCapsuleRadii = GetShaderLocation(shader, "shadowCapsuleRadii");
    uniforms->shadowLookupTable = GetShaderLocation(shader, "shadowLookupTable");
    uniforms->shadowLookupResolution = GetShaderLocation(shader, "shadowLookupResolution");

    uniforms->aoCapsuleCount = GetShaderLocation(shader, "aoCapsuleCount");
    uniforms->aoCapsuleStarts = GetShaderLocation(shader, "aoCapsuleStarts");
    uniforms->aoCapsuleVectors = GetShaderLocation(shader, "aoCapsuleVectors");
    uniforms->aoCapsuleRadii = GetShaderLocation(shader, "aoCapsuleRadii");
    uniforms->aoLookupTable = GetShaderLocation(shader, "aoLookupTable");
    uniforms->aoLookupResolution = GetShaderLocation(shader, "aoLookupResolution");

    uniforms->cameraPosition = GetShaderLocation(shader, "cameraPosition");

    uniforms->objectColor = GetShaderLocation(shader, "objectColor");
    uniforms->objectSpecularity = GetShaderLocation(shader, "objectSpecularity");
    uniforms->objectGlossiness = GetShaderLocation(shader, "objectGlossiness");
    uniforms->objectOpacity = GetShaderLocation(shader, "objectOpacity");

    uniforms->sunStrength = GetShaderLocation(shader, "sunStrength");
    uniforms->sunDir = GetShaderLocation(shader, "sunDir");
    uniforms->sunColor = GetShaderLocation(shader, "sunColor");
    uniforms->skyStrength = GetShaderLocation(shader, "skyStrength");
    uniforms->skyColor = GetShaderLocation(shader, "skyColor");
    uniforms->ambientStrength = GetShaderLocation(shader, "ambientStrength");
    uniforms->groundStrength = GetShaderLocation(shader, "groundStrength");

    uniforms->exposure = GetShaderLocation(shader, "exposure");

    TraceLog(LOG_INFO, "Shader uniform locations:");
    TraceLog(LOG_INFO, "isCapsule: %d", uniforms->isCapsule);
    TraceLog(LOG_INFO, "capsulePosition: %d", uniforms->capsulePosition);
}


//--------------------------------------
// Scrubber
//--------------------------------------

// Animation playback state and settings
struct ScrubberSettings {
    // Playback controls
    bool playing;
    bool looping;
    bool inplace;       // Lock root position during playback
    float playTime;
    bool frameSnap;     // Snap to frame boundaries
    int sampleMode;     // 0=nearest, 1=linear, 2=cubic

    // Frame range limits
    float timeLimit;
    int frameLimit;
    int frameMin;
    int frameMax;
    int frameMinSelect;
    int frameMaxSelect;
    bool frameMinEdit;
    bool frameMaxEdit;
    float timeMin;
    float timeMax;
};

static inline void ScrubberSettingsInit(ScrubberSettings* settings, int argc, char** argv)
{
    settings->playing = ArgBool(argc, argv, "playing", true);
    settings->looping = ArgBool(argc, argv, "looping", false);
    settings->inplace = ArgBool(argc, argv, "inplace", false);
    settings->playTime = ArgFloat(argc, argv, "playTime", 0.0f);
    settings->frameSnap = ArgBool(argc, argv, "frameSnap", true);
    static const char* sampleModeOptions[] = { "nearest", "linear", "cubic" };
    settings->sampleMode = ArgEnum(argc, argv, "sampleMode", 3, sampleModeOptions, 1);

    settings->timeLimit = 0.0f;
    settings->frameLimit = 0;
    settings->frameMin = 0;
    settings->frameMax = 0;
    settings->frameMinSelect = 0;
    settings->frameMaxSelect = 0;
    settings->frameMinEdit = false;
    settings->frameMaxEdit = false;
    settings->timeMin = 0.0f;
    settings->timeMax = 0.0f;
}

static inline void ScrubberSettingsRecomputeLimits(ScrubberSettings* settings, CharacterData* characterData)
{
    settings->frameLimit = 0;
    settings->timeLimit = 0.0f;
    for (int i = 0; i < characterData->count; i++)
    {
        settings->frameLimit = MaxInt(settings->frameLimit, characterData->bvhData[i].frameCount - 1);
        settings->timeLimit = Max(settings->timeLimit, (characterData->bvhData[i].frameCount - 1) * characterData->bvhData[i].frameTime);
    }
}

static inline void ScrubberSettingsInitMaxs(ScrubberSettings* settings, CharacterData* characterData)
{
    if (characterData->count == 0) { return; }

    settings->frameMax = characterData->bvhData[characterData->active].frameCount - 1;
    settings->frameMaxSelect = settings->frameMax;
    settings->timeMax = settings->frameMax * characterData->bvhData[characterData->active].frameTime;

    settings->frameMin = 0;
    settings->frameMinSelect = settings->frameMin;
    settings->timeMin = 0.0f;
}

static inline void ScrubberSettingsClamp(ScrubberSettings* settings, CharacterData* characterData)
{
    if (characterData->count == 0) { return; }

    settings->frameMax = ClampInt(settings->frameMax, 0, settings->frameLimit);
    settings->frameMaxSelect = settings->frameMax;
    settings->timeMax = settings->frameMax * characterData->bvhData[characterData->active].frameTime;

    settings->frameMin = ClampInt(settings->frameMin, 0, settings->frameMax);
    settings->frameMinSelect = settings->frameMin;
    settings->timeMin = settings->frameMin * characterData->bvhData[characterData->active].frameTime;

    settings->playTime = Clamp(settings->playTime, settings->timeMin, settings->timeMax);
}


// Main application state - passed to update/render functions
struct ApplicationState {
    int argc;
    char** argv;

    // Window
    int screenWidth;
    int screenHeight;

    // Camera
    CameraSystem camera;

    // Rendering resources
    Shader shader;
    ShaderUniforms uniforms;
    Mesh groundPlaneMesh;
    Model groundPlaneModel;
    Model capsuleModel;

    // Animation data
    CharacterData characterData;
    CapsuleData capsuleData;
    AnimDatabase animDatabase;
    ControlledCharacter controlledCharacter;

    // UI state
    ScrubberSettings scrubberSettings;
    AppConfig config;
    //GuiWindowFileDialogState fileDialogState;

    char errMsg[512];

    // Geno character rendering (experimental skinned mesh)
    bool genoRenderMode;
    Model genoModel;
    ModelAnimation genoAnimation;
    Shader genoBasicShader;
    bool genoModelLoaded;
    vector<BVHGenoMapping> genoMappings;

    // Debug timescale system
    // numpad-: halve debugTimescale
    // numpad+: double debugTimescale (max 1.0), also unpause
    // numpad*: toggle pause, hold while paused to advance at half speed
    float debugTimescale = 1.0f;
    bool debugPaused = false;
    double worldTime = 0.0;  // accumulated time with timescale applied

    // Flag for pending database rebuild (set when config changes, cleared when rebuilt)
    bool animDatabaseRebuildPending = false;
};


//----------------------------------------------------------------------------------
// Drawing
//----------------------------------------------------------------------------------

static inline void DrawTransform(const Vector3 position, const Quaternion rotation, const float size)
{
    DrawLine3D(position, Vector3Add(position, Vector3RotateByQuaternion(Vector3{ size, 0.0, 0.0 }, rotation)), RED);
    DrawLine3D(position, Vector3Add(position, Vector3RotateByQuaternion(Vector3{ 0.0, size, 0.0 }, rotation)), GREEN);
    DrawLine3D(position, Vector3Add(position, Vector3RotateByQuaternion(Vector3{ 0.0, 0.0, size }, rotation)), BLUE);
}

static inline void DrawSkeleton(TransformData* xformData, bool drawEndSites, Color color, Color endSiteColor)
{
    for (int i = 0; i < xformData->jointCount; i++)
    {
        if (!xformData->endSite[i])
        {
            DrawSphereWires(
                xformData->globalPositions[i],
                0.01f,
                4,
                6,
                color);
        }
        else if (drawEndSites)
        {
            DrawCubeWiresV(
                xformData->globalPositions[i],
                Vector3{ 0.02f, 0.02f, 0.02f },
                endSiteColor);
        }

        if (xformData->parents[i] != -1)
        {
            if (!xformData->endSite[i])
            {
                DrawLine3D(
                    xformData->globalPositions[i],
                    xformData->globalPositions[xformData->parents[i]],
                    color);
            }
            else if (drawEndSites)
            {
                DrawLine3D(
                    xformData->globalPositions[i],
                    xformData->globalPositions[xformData->parents[i]],
                    endSiteColor);
            }
        }
    }
}

static inline void DrawTransforms(TransformData* xformData)
{
    for (int i = 0; i < xformData->jointCount; i++)
    {
        if (!xformData->endSite[i])
        {
            DrawTransform(
                xformData->globalPositions[i],
                xformData->globalRotations[i],
                0.1f);
        }
    }
}

static inline void DrawWireFrames(CapsuleData* capsuleData, Color color)
{
    for (int i = 0; i < capsuleData->capsuleCount; i++)
    {
        const Vector3 capsuleStart = CapsuleStart(capsuleData->capsulePositions[i], capsuleData->capsuleRotations[i], capsuleData->capsuleHalfLengths[i]);
        const Vector3 capsuleEnd = CapsuleEnd(capsuleData->capsulePositions[i], capsuleData->capsuleRotations[i], capsuleData->capsuleHalfLengths[i]);
        const float capsuleRadius = capsuleData->capsuleRadii[i];

        DrawSphereWires(capsuleStart, capsuleRadius, 4, 6, color);
        DrawSphereWires(capsuleEnd, capsuleRadius, 4, 6, color);
        DrawCylinderWiresEx(capsuleStart, capsuleEnd, capsuleRadius, capsuleRadius, 6, color);
    }
}

//----------------------------------------------------------------------------------
// GUI
//----------------------------------------------------------------------------------
static inline void ImGuiCamera(CameraSystem* camera, CharacterData* characterData,
    const ControlledCharacter* controlledCharacter, AppConfig* config, int argc, char** argv)
{
    ImGui::SetNextWindowPos(ImVec2(20, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(220, 320), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Camera")) {
        // Camera mode selector
        const char* modeNames[] = { "Orbit", "Unreal", "Turret Follower" };
        int currentMode = static_cast<int>(camera->mode);
        if (ImGui::Combo("Mode", &currentMode, modeNames, 3)) {
            const FlomoCameraMode newMode = static_cast<FlomoCameraMode>(currentMode);

            // Compute target position for syncing (used by Orbit and Turret modes)
            Vector3 targetPosition = Vector3{ 0.0f, 1.0f, 0.0f };
            if (camera->track)
            {
                if (camera->trackControlledCharacter && controlledCharacter->active)
                {
                    const int trackBone = MinInt(camera->trackBone,
                        controlledCharacter->xformData.jointCount - 1);
                    targetPosition = controlledCharacter->xformData.globalPositions[trackBone];
                }
                else if (characterData->count > 0 &&
                    camera->trackBone < characterData->xformData[characterData->active].jointCount)
                {
                    targetPosition = characterData->xformData[characterData->active].globalPositions[camera->trackBone];
                }
            }
            else if (camera->trackHipsProjectedOnGround)
            {
                // Track bone 0 (hips) at Y=1m
                if (camera->trackControlledCharacter && controlledCharacter->active)
                {
                    targetPosition = controlledCharacter->xformData.globalPositions[0];
                }
                else if (characterData->count > 0 && characterData->xformData[characterData->active].jointCount > 0)
                {
                    targetPosition = characterData->xformData[characterData->active].globalPositions[0];
                }
                targetPosition.y = 1.0f;
            }

            // Sync all modes from current state, then switch
            CameraSyncAllModesFromCurrent(camera, targetPosition);
            camera->mode = newMode;
        }

        ImGui::Separator();

        // Target blending (used by Orbit and LazyTurretFollower)
        if (camera->mode == FlomoCameraMode::Orbit || camera->mode == FlomoCameraMode::LazyTurretFollower)
        {
            ImGui::SliderFloat("Target Blend", &config->cameraTargetBlendtime, 0.0f, 1.0f, "%.2f");
            ImGui::Separator();
        }

        if (camera->mode == FlomoCameraMode::Orbit)
        {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "RMB - Rotate");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "MMB - Pan");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Scroll - Zoom");
            ImGui::Separator();

            ImGui::Text("Target: [%.2f, %.2f, %.2f]",
                camera->cam3d.target.x, camera->cam3d.target.y, camera->cam3d.target.z);
            ImGui::Text("Azimuth: %.2f", camera->orbit.azimuth);
            ImGui::Text("Altitude: %.2f", camera->orbit.altitude);
            ImGui::Text("Distance: %.2f", camera->orbit.distance);

            if (ImGui::Button("Reset Orbit")) {
                camera->orbit.azimuth = ArgFloat(argc, argv, "cameraAzimuth", 0.0f);
                camera->orbit.altitude = ArgFloat(argc, argv, "cameraAltitude", 0.4f);
                camera->orbit.distance = ArgFloat(argc, argv, "cameraDistance", 4.0f);
                camera->orbit.offset = ArgVector3(argc, argv, "cameraOffset", Vector3Zero());
            }

            ImGui::Separator();

            if (characterData->count > 0 || controlledCharacter->active) {
                // Track Bone checkbox - mutually exclusive with trackHipsProjectedOnGround
                if (ImGui::Checkbox("Track Bone", &camera->track))
                {
                    if (camera->track) camera->trackHipsProjectedOnGround = false;
                }

                // Build joint name list for combo (use controlled character joints if tracking it)
                vector<string> joints;
                string comboStr;
                if (camera->trackControlledCharacter && controlledCharacter->active)
                {
                    comboStr = controlledCharacter->jointNamesCombo;
                }
                else if (characterData->count > 0)
                {
                    comboStr = characterData->jointNamesCombo[characterData->active];
                }
                stringstream ss(comboStr);
                string token;
                while (getline(ss, token, ';')) {
                    joints.push_back(token);
                }
                vector<const char*> items;
                for (const string& s : joints) items.push_back(s.c_str());

                if (!items.empty() && camera->track)
                {
                    ImGui::Combo("##trackbone", &camera->trackBone, items.data(), (int)items.size());
                }

                // Track Hips Projected on Ground - mutually exclusive with track
                if (ImGui::Checkbox("Track Hips on Ground", &camera->trackHipsProjectedOnGround))
                {
                    if (camera->trackHipsProjectedOnGround) camera->track = false;
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Track bone 0 (hips) projected at Y=1m for stable camera");
                }
            }
        }
        else if (camera->mode == FlomoCameraMode::UnrealEditor)
        {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "RMB + WASD - Move");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "RMB + Q/E - Down/Up");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "MMB - Pan");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Scroll - Dolly");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "RMB + Scroll - Speed");
            ImGui::Separator();

            ImGui::Text("Position: [%.2f, %.2f, %.2f]",
                camera->unreal.position.x, camera->unreal.position.y, camera->unreal.position.z);
            ImGui::Text("Yaw: %.2f  Pitch: %.2f", camera->unreal.yaw, camera->unreal.pitch);
            ImGui::SliderFloat("Move Speed", &camera->unreal.moveSpeed,
                camera->unreal.minSpeed, camera->unreal.maxSpeed, "%.1f");

            if (ImGui::Button("Reset Unreal")) {
                camera->unreal.position = Vector3{ 2.0f, 1.5f, 5.0f };
                camera->unreal.yaw = PI;
                camera->unreal.pitch = 0.0f;
                camera->unreal.moveSpeed = 5.0f;
            }
        }
        else if (camera->mode == FlomoCameraMode::LazyTurretFollower)
        {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "WASD - Move Character");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Camera follows passively");
            ImGui::Separator();

            ImGui::Text("Position: [%.2f, %.2f, %.2f]",
                camera->turret.position.x, camera->turret.position.y, camera->turret.position.z);
            ImGui::Text("Azimuth: %.2f  Altitude: %.2f", camera->turret.azimuth, camera->turret.altitude);
            ImGui::Text("Distance: %.2f", camera->turret.distance);

            ImGui::SliderFloat("Min Distance", &camera->turret.minDistance, 1.0f, 10.0f, "%.1f");
            ImGui::SliderFloat("Max Distance", &camera->turret.maxDistance,
                camera->turret.minDistance, 20.0f, "%.1f");
            ImGui::SliderFloat("Smooth Time", &camera->turret.smoothTime, 0.05f, 1.0f, "%.2f");

            ImGui::Separator();

            if (characterData->count > 0 || controlledCharacter->active) {
                // Track Bone checkbox - mutually exclusive with trackHipsProjectedOnGround
                if (ImGui::Checkbox("Track Bone##turret", &camera->track))
                {
                    if (camera->track) camera->trackHipsProjectedOnGround = false;
                }

                // Build joint name list for combo (use controlled character joints if tracking it)
                vector<string> joints;
                string comboStr;
                if (camera->trackControlledCharacter && controlledCharacter->active)
                {
                    comboStr = controlledCharacter->jointNamesCombo;
                }
                else if (characterData->count > 0)
                {
                    comboStr = characterData->jointNamesCombo[characterData->active];
                }
                stringstream ss(comboStr);
                string token;
                while (getline(ss, token, ';')) {
                    joints.push_back(token);
                }
                vector<const char*> items;
                for (const string& s : joints) items.push_back(s.c_str());

                if (!items.empty() && camera->track)
                {
                    ImGui::Combo("##trackboneturret", &camera->trackBone, items.data(), (int)items.size());
                }

                // Track Hips Projected on Ground - mutually exclusive with track
                if (ImGui::Checkbox("Track Hips on Ground##turret", &camera->trackHipsProjectedOnGround))
                {
                    if (camera->trackHipsProjectedOnGround) camera->track = false;
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Track bone 0 (hips) projected at Y=1m for stable camera");
                }
            }
        }
    }
    ImGui::End();
}

static inline void ImGuiRenderSettings(AppConfig* config,
    CapsuleData* capsuleData, int screenWidth, int screenHeight,
    bool* genoRenderMode, bool genoModelLoaded)
{
    const float sw = (float)screenWidth;
    ImGui::SetNextWindowPos(ImVec2(sw - 260, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(240, 430), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Rendering")) {
        ImGui::SliderFloat("Exposure", &config->exposure, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("Sun Light", &config->sunLightStrength, 0.0f, 1.0f, "%.2f");
        if (ImGui::SliderFloat("Sun Softness", &config->sunLightConeAngle, 0.02f, PI / 4.0f, "%.2f")) {
            CapsuleDataUpdateShadowLookupTable(capsuleData, config->sunLightConeAngle);
        }
        ImGui::SliderFloat("Sky Light", &config->skyLightStrength, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Ambient Light", &config->ambientLightStrength, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Ground Light", &config->groundLightStrength, 0.0f, 0.5f, "%.2f");
        ImGui::SliderFloat("Sun Azimuth", &config->sunAzimuth, -PI, PI, "%.2f");
        ImGui::SliderFloat("Sun Altitude", &config->sunAltitude, 0.0f, 0.49f * PI, "%.2f");

        // Single column layout for all checkboxes
        ImGui::Columns(2, "render_checkboxes", false);

        ImGui::Checkbox("Draw Origin", &config->drawOrigin);
        ImGui::Checkbox("Draw Checker", &config->drawChecker);
        ImGui::Checkbox("Draw Wireframes", &config->drawWireframes);
        ImGui::Checkbox("Draw Transforms", &config->drawTransforms);
        ImGui::Checkbox("Draw Shadows", &config->drawShadows);
        ImGui::Checkbox("Draw FPS", &config->drawFPS);

        ImGui::NextColumn();

        ImGui::Checkbox("Draw Grid", &config->drawGrid);
        ImGui::Checkbox("Draw Capsules", &config->drawCapsules);
        ImGui::Checkbox("Draw Skeleton", &config->drawSkeleton);
        ImGui::Checkbox("Draw AO", &config->drawAO);
        ImGui::Checkbox("Draw End Sites", &config->drawEndSites);
        if (genoModelLoaded) {
            ImGui::Checkbox("Mesh Character", genoRenderMode);
        }

        ImGui::Columns(1);

        ImGui::Separator();
        ImGui::Checkbox("Draw Features", &config->drawFeatures);
        ImGui::Checkbox("Draw Blend Cursors", &config->drawBlendCursors);
        ImGui::Checkbox("Draw Velocities", &config->drawVelocities);
        ImGui::Checkbox("Draw Accelerations", &config->drawAccelerations);
        ImGui::Checkbox("Draw Root Velocities", &config->drawRootVelocities);
        ImGui::Checkbox("Draw Toe Velocities", &config->drawToeVelocities);
        ImGui::Checkbox("Draw Foot IK", &config->drawFootIK);
        ImGui::Checkbox("Draw Basic Blend", &config->drawBasicBlend);
        ImGui::Checkbox("Draw Magic Anchor", &config->drawMagicAnchor);
        ImGui::Checkbox("Draw Player Input", &config->drawPlayerInput);
        ImGui::Checkbox("Draw Past History", &config->drawPastHistory);

    }
    ImGui::End();
}

static inline void ImGuiCharacterData(
    CharacterData* characterData,
    CameraSystem* camera,
    ControlledCharacter* controlledCharacter,
    char* errMsg,
    int argc,
    char** argv)
{
    float offsetHeight = 280.0f;
    ImGui::SetNextWindowPos(ImVec2(20, offsetHeight), ImGuiCond_FirstUseEver);

    // Maximum characters to show in the GUI layout (cosmetic limit only)
    constexpr int CHARACTERS_GUI_SLOTS = 15;

    ImGui::SetNextWindowSize(ImVec2(190, (CHARACTERS_GUI_SLOTS - 1) * 30 + 150), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Characters")) {
        //#if !defined(PLATFORM_WEB)
        //        if (ImGui::Button("Open")) {
        //            fileDialogState->windowActive = true;
        //        }
        //        ImGui::SameLine();
        //#endif
        if (ImGui::Button("Clear")) {
            characterData->clearRequested = true;  // Handled in main update with full state access
            errMsg[0] = '\0';
        }

        // Show controlled character first if active (as camera target option)
        if (controlledCharacter->active)
        {
            const bool isSelected = camera->trackControlledCharacter;
            if (ImGui::RadioButton("Controlled", isSelected))
            {
                camera->trackControlledCharacter = true;
            }
            // Color swatch for controlled character
            ImGui::SameLine();
            const Color ctrlColor = Color{ 100, 200, 255, 255 };  // light blue
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(ctrlColor.r / 255.f, ctrlColor.g / 255.f, ctrlColor.b / 255.f, ctrlColor.a / 255.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(ctrlColor.r / 255.f, ctrlColor.g / 255.f, ctrlColor.b / 255.f, ctrlColor.a / 255.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(ctrlColor.r / 255.f, ctrlColor.g / 255.f, ctrlColor.b / 255.f, ctrlColor.a / 255.f));
            ImGui::Button("##colorCtrl", ImVec2(20, 20));
            ImGui::PopStyleColor(3);
            ImVec2 min = ImGui::GetItemRectMin();
            ImVec2 max = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddRect(min, max, IM_COL32(128, 128, 128, 255));
        }

        for (int i = 0; i < characterData->count; i++) {
            string bvhNameShort;
            if (characterData->names[i].length() < 100) {
                bvhNameShort = characterData->names[i];
            }
            else {
                bvhNameShort = characterData->names[i].substr(0, 96) + "...";
            }

            // When controlled character exists, use visual selection based on trackControlledCharacter
            const bool isSelected = controlledCharacter->active
                ? (!camera->trackControlledCharacter && i == characterData->active)
                : (i == characterData->active);

            if (ImGui::RadioButton(bvhNameShort.c_str(), isSelected))
            {
                characterData->active = i;
                camera->trackControlledCharacter = false;  // Switch to tracking this character
            }

            ImGui::SameLine();
            Color color = characterData->colors[i];
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(color.r / 255.f, color.g / 255.f, color.b / 255.f, color.a / 255.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(color.r / 255.f, color.g / 255.f, color.b / 255.f, color.a / 255.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(color.r / 255.f, color.g / 255.f, color.b / 255.f, color.a / 255.f));
            if (ImGui::Button((string("##color") + to_string(i)).c_str(), ImVec2(20, 20))) {
                characterData->colorPickerActive = !characterData->colorPickerActive;
            }
            ImGui::PopStyleColor(3);
            // Add border
            ImVec2 min = ImGui::GetItemRectMin();
            ImVec2 max = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddRect(min, max, IM_COL32(128, 128, 128, 255));
        }

        if (characterData->count > 0) {
            bool scaleM = characterData->scales[characterData->active] == 1.0f;
            ImGui::Checkbox("m", &scaleM);
            if (scaleM) characterData->scales[characterData->active] = 1.0f;
            ImGui::SameLine();

            bool scaleCM = characterData->scales[characterData->active] == 0.01f;
            ImGui::Checkbox("cm", &scaleCM);
            if (scaleCM) characterData->scales[characterData->active] = 0.01f;
            ImGui::SameLine();

            bool scaleInches = characterData->scales[characterData->active] == 0.0254f;
            ImGui::Checkbox("inch", &scaleInches);
            if (scaleInches) characterData->scales[characterData->active] = 0.0254f;
            ImGui::SameLine();

            bool scaleFeet = characterData->scales[characterData->active] == 0.3048f;
            ImGui::Checkbox("feet", &scaleFeet);
            if (scaleFeet) characterData->scales[characterData->active] = 0.3048f;
            ImGui::SameLine();

            bool scaleAuto = characterData->scales[characterData->active] == characterData->autoScales[characterData->active];
            ImGui::Checkbox("auto", &scaleAuto);
            if (scaleAuto) characterData->scales[characterData->active] = characterData->autoScales[characterData->active];

            // Enforce mutual exclusivity manually
            if (scaleM) { scaleCM = scaleInches = scaleFeet = scaleAuto = false; }
            if (scaleCM) { scaleM = scaleInches = scaleFeet = scaleAuto = false; }
            if (scaleInches) { scaleM = scaleCM = scaleFeet = scaleAuto = false; }
            if (scaleFeet) { scaleM = scaleCM = scaleInches = scaleAuto = false; }
            if (scaleAuto) { scaleM = scaleCM = scaleInches = scaleFeet = false; }

            ImGui::SliderFloat("Radius", &characterData->radii[characterData->active], 0.01f, 0.1f, "%.2f");
            ImGui::SliderFloat("Opacity", &characterData->opacities[characterData->active], 0.0f, 1.0f, "%.2f");
        }
    }
    ImGui::End();
}

static inline void ImGuiScrubberSettings(
    ScrubberSettings* settings,
    CharacterData* characterData,
    int screenWidth,
    int screenHeight)
{
    if (characterData->count == 0) { return; }
    const float sw = (float)screenWidth;
    const float sh = (float)screenHeight;
    const float frameTime = characterData->bvhData[characterData->active].frameTime;

    const float padding = 20.0f;
    ImGui::SetNextWindowPos(ImVec2(padding, sh - 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(sw - padding * 2, 90), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Scrubber")) {
        ImGui::Text("Frame Time: %f", frameTime);
        ImGui::SameLine();
        ImGui::Checkbox("Snap to Frame", &settings->frameSnap);
        ImGui::SameLine();

        const char* sampleModes[] = { "Nearest", "Linear", "Cubic" };
        ImGui::Combo("##samplemode", &settings->sampleMode, sampleModes, IM_ARRAYSIZE(sampleModes));
        ImGui::SameLine();

        ImGui::Checkbox("Inplace", &settings->inplace);
        ImGui::SameLine();
        ImGui::Checkbox("Loop", &settings->looping);
        ImGui::SameLine();
        ImGui::Checkbox("Play", &settings->playing);
        if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Numpad+: unpause\nNumpad*: pause (hold to advance slow)\nNumpad-/+: adjust speed"); }

        int frame = ClampInt((int)(settings->playTime / frameTime + 0.5f), settings->frameMin, settings->frameMax);

        ImGui::InputInt("Min", &settings->frameMinSelect);
        settings->frameMinSelect = ClampInt(settings->frameMinSelect, 0, settings->frameLimit);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            settings->frameMin = settings->frameMinSelect;
            ScrubberSettingsClamp(settings, characterData);
        }
        ImGui::SameLine();

        ImGui::InputInt("Max", &settings->frameMaxSelect);
        settings->frameMaxSelect = ClampInt(settings->frameMaxSelect, 0, settings->frameLimit);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            settings->frameMax = settings->frameMaxSelect;
            ScrubberSettingsClamp(settings, characterData);
        }
        ImGui::SameLine();
        ImGui::Text("of %i", settings->frameLimit);

        float frameFloatPrev = settings->frameSnap ? (float)frame : settings->playTime / frameTime;
        float frameFloat = frameFloatPrev;
        ImGui::Text("%5.2f", settings->playTime);
        ImGui::SameLine();
        ImGui::SliderFloat("##framefloat", &frameFloat, (float)settings->frameMin, (float)settings->frameMax, "");
        ImGui::SameLine();
        ImGui::Text("%i", frame);

        if (frameFloat != frameFloatPrev) {
            if (settings->frameSnap) {
                frame = ClampInt((int)(frameFloat + 0.5f), settings->frameMin, settings->frameMax);
                settings->playTime = Clamp(frame * frameTime, settings->timeMin, settings->timeMax);
            }
            else {
                settings->playTime = Clamp(frameFloat * frameTime, settings->timeMin, settings->timeMax);
            }
        }
    }
    ImGui::End();
}

static inline void ImGuiAnimSettings(ApplicationState* app)
{
    AppConfig* config = &app->config;
    ImGui::SetNextWindowPos(ImVec2(250, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(280, 300), ImGuiCond_FirstUseEver);  // increased width and height
    if (ImGui::Begin("Anim Settings")) {
        ImGui::SliderFloat("Blend Time", &config->defaultBlendTime, 0.0f, 2.0f, "%.2f s");
        ImGui::SliderFloat("Switch Interval", &config->switchInterval, 0.1f, 5.0f, "%.2f s");
        ImGui::SliderFloat("MM Search Period", &config->mmSearchPeriod, 0.01f, 1.0f, "%.2f s");
        ImGui::Separator();
        // Cursor blend mode dropdown
        if (ImGui::BeginCombo("Blend Mode", CursorBlendModeName(config->cursorBlendMode)))
        {
            for (int i = 0; i < static_cast<int>(CursorBlendMode::COUNT); ++i)
            {
                const bool isSelected = (static_cast<int>(config->cursorBlendMode) == i);
                if (ImGui::Selectable(CursorBlendModeName(static_cast<CursorBlendMode>(i)), isSelected))
                {
                    config->cursorBlendMode = static_cast<CursorBlendMode>(i);
                }
                if (isSelected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (config->cursorBlendMode == CursorBlendMode::LookaheadDragging)
        {
            ImGui::SetNextItemWidth(80);
            if (ImGui::InputFloat("Lookahead Time", &config->poseDragLookaheadTimeEditor, 0.0f, 0.0f, "%.3f"))
            {
                config->poseDragLookaheadTimeEditor = Clamp(config->poseDragLookaheadTimeEditor, 0.01f, 0.2f);
                app->animDatabaseRebuildPending = true;
            }
        }

        ImGui::Separator();
        ImGui::Checkbox("Enable Foot IK", &config->enableFootIK);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Enable IK to pull feet towards virtual toe positions");
        }

        if (config->enableFootIK) {
            ImGui::Checkbox("Timed Unlocking", &config->enableTimedUnlocking);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Allow virtual toes to break free when drifting too far");
            }

            if (config->enableTimedUnlocking) {
                ImGui::SliderFloat("Unlock Distance", &config->unlockDistance, 0.01f, 1.0f, "%.2f m");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Distance threshold to trigger unlock");
                }

                ImGui::SliderFloat("Unlock Duration", &config->unlockDuration, 0.001f, 2.0f, "%.2f s");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Time to gradually re-lock after unlock");
                }
            }
        }

        // Motion Matching Features
        AnimDatabase& db = app->animDatabase;
        MotionMatchingFeaturesConfig& editedMMConfig = config->mmConfigEditor;
        bool configChanged = false;

        ImGui::Text("Feature Weights:");
        ImGui::Separator();

        // Feature weights as input fields
        for (int i = 0; i < static_cast<int>(FeatureType::COUNT); ++i)
        {
            const FeatureType type = static_cast<FeatureType>(i);
            const char* name = FeatureTypeName(type);
            float weight = editedMMConfig.featureTypeWeights[i];

            ImGui::PushID(i);
            ImGui::Text("%s:", name);
            ImGui::SameLine(180);  // Align input fields
            ImGui::SetNextItemWidth(80);
            if (ImGui::InputFloat("##weight", &weight, 0.0f, 0.0f, "%.3f"))
            {
                // Clamp to >= 0
                editedMMConfig.featureTypeWeights[i] = (weight < 0.0f) ? 0.0f : weight;
                configChanged = true;
            }
            ImGui::PopID();
        }

        ImGui::Separator();

        ImGui::Separator();

        // Future trajectory times configuration
        ImGui::Text("Future Trajectory Times (s):");
        const size_t timeCount = editedMMConfig.futureTrajPointTimes.size();

        for (size_t i = 0; i < timeCount; ++i)
        {
            ImGui::PushID((int)i + 1000);

            ImGui::Text("Time %d:", (int)i + 1);
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(80);

            float timeVal = editedMMConfig.futureTrajPointTimes[i];
            if (ImGui::InputFloat("##time", &timeVal, 0.0f, 0.0f, "%.3f"))
            {
                // Clamp to >= 0.001
                timeVal = (timeVal < 0.001f) ? 0.001f : timeVal;

                // Enforce increasing order: clamp to be >= previous time
                if (i > 0)
                {
                    const float prevTime = editedMMConfig.futureTrajPointTimes[i - 1];
                    timeVal = (timeVal < prevTime) ? prevTime : timeVal;
                }

                // Enforce increasing order: clamp next times to be >= this time
                editedMMConfig.futureTrajPointTimes[i] = timeVal;
                for (size_t j = i + 1; j < timeCount; ++j)
                {
                    if (editedMMConfig.futureTrajPointTimes[j] < timeVal)
                    {
                        editedMMConfig.futureTrajPointTimes[j] = timeVal;
                    }
                }

                configChanged = true;
            }

            ImGui::PopID();
        }

        ImGui::Separator();

        // Past time offset configuration
        ImGui::Text("Past Time Offset (s):");
        ImGui::SameLine(180);
        ImGui::SetNextItemWidth(80);
        float pastTime = editedMMConfig.pastTimeOffset;
        if (ImGui::InputFloat("##pasttime", &pastTime, 0.0f, 0.0f, "%.3f"))
        {
            // Clamp to >= 0.001
            editedMMConfig.pastTimeOffset = (pastTime < 0.001f) ? 0.001f : pastTime;
            configChanged = true;
        }

        ImGui::Separator();

        // Blend Root Mode Position
        ImGui::Text("Root Position Mode:");
        ImGui::SameLine(180);
        if (ImGui::BeginCombo("##rootPosMode", BlendRootModePositionName(config->blendRootModePositionEditor)))
        {
            for (int i = 0; i < static_cast<int>(BlendRootModePosition::COUNT); ++i)
            {
                const BlendRootModePosition mode = static_cast<BlendRootModePosition>(i);
                const bool isSelected = (config->blendRootModePositionEditor == mode);
                if (ImGui::Selectable(BlendRootModePositionName(mode), isSelected))
                {
                    config->blendRootModePositionEditor = mode;
                    configChanged = true;
                }
                if (isSelected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        // Blend Root Mode Rotation
        ImGui::Text("Root Rotation Mode:");
        ImGui::SameLine(180);
        if (ImGui::BeginCombo("##rootRotMode", BlendRootModeRotationName(config->blendRootModeRotationEditor)))
        {
            for (int i = 0; i < static_cast<int>(BlendRootModeRotation::COUNT); ++i)
            {
                const BlendRootModeRotation mode = static_cast<BlendRootModeRotation>(i);
                const bool isSelected = (config->blendRootModeRotationEditor == mode);
                if (ImGui::Selectable(BlendRootModeRotationName(mode), isSelected))
                {
                    config->blendRootModeRotationEditor = mode;
                    configChanged = true;
                }
                if (isSelected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (configChanged)
        {
            // Mark for rebuild (don't rebuild on every keypress)
            app->animDatabaseRebuildPending = true;
        }

        // Show rebuild button when changes are pending
        if (app->animDatabaseRebuildPending)
        {
            ImGui::Separator();
            if (ImGui::Button("Rebuild Database"))
            {
                db.featuresConfig = editedMMConfig;
                db.poseDragLookaheadTime = config->poseDragLookaheadTimeEditor;
                db.blendRootModePosition = config->blendRootModePositionEditor;
                db.blendRootModeRotation = config->blendRootModeRotationEditor;
                AnimDatabaseRebuild(&db, &app->characterData);
                app->animDatabaseRebuildPending = false;
                TraceLog(LOG_INFO, "Motion matching config updated");
            }
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Changes pending");
        }
    }
    ImGui::End();
}

static inline void ImGuiPlayerControl(ControlledCharacter* controlledCharacter, AppConfig* config)
{
    if (!controlledCharacter->active) return;

    ImGui::SetNextWindowPos(ImVec2(250, 200), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(220, 130), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Player Control"))
    {
        // Animation mode dropdown (stored in config for persistence)
        int currentMode = static_cast<int>(config->animationMode);
        if (ImGui::BeginCombo("Mode", AnimationModeName(config->animationMode)))
        {
            for (int i = 0; i < static_cast<int>(AnimationMode::COUNT); ++i)
            {
                const bool isSelected = (currentMode == i);
                if (ImGui::Selectable(AnimationModeName(static_cast<AnimationMode>(i)), isSelected))
                {
                    config->animationMode = static_cast<AnimationMode>(i);
                }
                if (isSelected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        PlayerControlInput& input = controlledCharacter->playerInput;
        ImGui::SliderFloat("Max Speed", &input.maxSpeed, 0.1f, 10.0f, "%.2f m/s");

        const float velMag = Vector3Length(input.desiredVelocity);
        ImGui::Text("Input: %.2f m/s", velMag);

        // Show MM info when in motion matching mode
        if (config->animationMode == AnimationMode::MotionMatching)
        {
            ImGui::Separator();
            ImGui::Text("Best Frame: %d", controlledCharacter->mmBestFrame);
            ImGui::Text("Best Cost: %.4f", controlledCharacter->mmBestCost);
        }
    }
    ImGui::End();
}

//----------------------------------------------------------------------------------
// Application
//----------------------------------------------------------------------------------

// Update function - what is called to "tick" the application.
static void ApplicationUpdate(void* voidApplicationState)
{
    ApplicationState* app = (ApplicationState*)voidApplicationState;

    // Update window dimensions if resized
    if (IsWindowResized()) {
        app->screenWidth = GetScreenWidth();
        app->screenHeight = GetScreenHeight();
    }

    BeginDrawing();

    // Process Dragged and Dropped Files

    if (IsFileDropped())
    {
        FilePathList droppedFiles = LoadDroppedFiles();

        int prevBvhCount = app->characterData.count;

        for (int i = 0; i < droppedFiles.count; i++)
        {
            if (CharacterDataLoadFromFile(&app->characterData, droppedFiles.paths[i], app->errMsg, 512))
            {
                app->characterData.active = app->characterData.count - 1;
            }
        }

        UnloadDroppedFiles(droppedFiles);

        if (app->characterData.count > prevBvhCount)
        {
            // Ensure active character is valid
            if (app->characterData.active < 0 || app->characterData.active >= app->characterData.count)
            {
                app->characterData.active = app->characterData.count - 1;
            }

            // Reset scrubber to known state before updating
            ScrubberSettingsInit(&app->scrubberSettings, app->argc, app->argv);

            ScrubberSettingsRecomputeLimits(&app->scrubberSettings, &app->characterData);
            ScrubberSettingsInitMaxs(&app->scrubberSettings, &app->characterData);

            app->animDatabase.featuresConfig = app->config.mmConfigEditor;
            app->animDatabase.poseDragLookaheadTime = app->config.poseDragLookaheadTimeEditor;
            app->animDatabase.blendRootModePosition = app->config.blendRootModePositionEditor;
            app->animDatabase.blendRootModeRotation = app->config.blendRootModeRotationEditor;

            // Rebuild animation database
            AnimDatabaseRebuild(&app->animDatabase, &app->characterData);
            if (!app->animDatabase.valid) {
                TraceLog(LOG_WARNING, "AnimDatabase invalid after rebuild - disabling controlled character.");
                app->controlledCharacter.active = false;
            }
            else {
                // initialize controlled character
                if (!app->controlledCharacter.active) {
                    ControlledCharacterInit(
                        &app->controlledCharacter,
                        &app->characterData.bvhData[0],
                        app->characterData.scales[0],
                        app->config.switchInterval);
                }
            }

            // Resize capsule buffer for all characters + controlled character
            // Account for up to 3x joints: normal + footIK debug + basicBlend debug
            CapsuleDataUpdateForCharacters(&app->capsuleData, &app->characterData);
            if (app->controlledCharacter.active)
            {
                const int totalJoints = (int)app->capsuleData.capsulePositions.size() +
                    app->controlledCharacter.xformData.jointCount * 3;
                CapsuleDataResize(&app->capsuleData, totalJoints);
            }

            string windowTitle = app->characterData.filePaths[app->characterData.active] + " - BVHView";
            SetWindowTitle(windowTitle.c_str());
        }
    }

    // Handle clear request (with full state access)
    if (app->characterData.clearRequested)
    {
        app->characterData = {};

        // Free and reset character data
        CharacterDataInit(&app->characterData, app->argc, app->argv);

        // Reset AnimDatabase
        AnimDatabaseFree(&app->animDatabase);

        // Disable and free controlled character
        if (app->controlledCharacter.active)
        {
            //ControlledCharacterFree(&app->controlledCharacter);
            app->controlledCharacter.active = false;
        }

        // Reset scrubber
        ScrubberSettingsInit(&app->scrubberSettings, app->argc, app->argv);

        // Reset capsule data
        CapsuleDataReset(&app->capsuleData);

        SetWindowTitle("Flomo");
        TraceLog(LOG_INFO, "Cleared all animations");
    }

    // Process Key Presses

    if (IsKeyPressed(KEY_H))// && !app->fileDialogState.windowActive)
    {
        app->config.drawUI = !app->config.drawUI;
    }

    PROFILE_BEGIN(Update);

    const bool imguiWantsMouse = ImGui::GetIO().WantCaptureMouse;
    const bool imguiWantsKeyboard = ImGui::GetIO().WantCaptureKeyboard;


    // Compute effective dt based on debug timescale
    const float rawDt = GetFrameTime();
    const float clampedRawDt = Clamp(rawDt, 0.0f, 0.1f); // Prevent extreme spikes when debugging
    float effectiveDt = 0.0f;
    if (!imguiWantsKeyboard)
    {
        // Check numpad input for debug timescale (need to check here before ImGui captures input)
        const bool numpadMinusPressed = IsKeyPressed(KEY_KP_SUBTRACT);
        const bool numpadPlusPressed = IsKeyPressed(KEY_KP_ADD);
        const bool numpadMultiplyPressed = IsKeyPressed(KEY_KP_MULTIPLY);
        const bool numpadMultiplyHeld = IsKeyDown(KEY_KP_MULTIPLY);
        const bool shiftHeld = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);

        if (numpadMinusPressed)
        {
            app->debugTimescale *= 0.7f;
            TraceLog(LOG_INFO, "Debug timescale: %.4f", app->debugTimescale);
        }
        if (numpadPlusPressed)
        {
            if (app->debugPaused)
            {
                // Just unpause, don't change timescale
                app->debugPaused = false;
                TraceLog(LOG_INFO, "Unpaused at timescale: %.4f", app->debugTimescale);
            }
            else if (shiftHeld)
            {
                // Shift+numpad+: double timescale without clamping (allows >1x speed)
                app->debugTimescale *= 1.0f / 0.7f;
                TraceLog(LOG_INFO, "Debug timescale: %.4f (fast forward)", app->debugTimescale);
            }
            else
            {
                // Double timescale up to max 1.0
                app->debugTimescale = Clamp(app->debugTimescale * 1.0f / 0.7f, 0.0f, 1.0f);
                TraceLog(LOG_INFO, "Debug timescale: %.4f", app->debugTimescale);
            }
        }
        if (numpadMultiplyPressed)
        {
            app->debugPaused = true;
            TraceLog(LOG_INFO, "Paused (hold * to advance at half speed)");
        }

        // Compute effective dt
        if (app->debugPaused)
        {
            if (numpadMultiplyHeld)
            {
                // Holding * while paused: advance at half the debug timescale
                effectiveDt = clampedRawDt * app->debugTimescale * 0.5f;
            }
            else
            {
                effectiveDt = 0.0f;
            }
        }
        else
        {
            effectiveDt = clampedRawDt * app->debugTimescale;
        }
    }

    // Accumulate world time (with timescale applied)
    app->worldTime += effectiveDt;

    // Tick time forward

    if (app->scrubberSettings.playing)
    {
        app->scrubberSettings.playTime += effectiveDt;

        if (app->scrubberSettings.playTime >= app->scrubberSettings.timeMax)
        {
            app->scrubberSettings.playTime = (app->scrubberSettings.looping && app->scrubberSettings.timeMax >= 1e-8f) ?
                fmodf(app->scrubberSettings.playTime, app->scrubberSettings.timeMax) + app->scrubberSettings.timeMin :
                app->scrubberSettings.timeMax;
        }
    }

    // Sample Animation Data

    for (int i = 0; i < app->characterData.count; i++)
    {
        if (app->scrubberSettings.sampleMode == 0)
        {
            TransformDataSampleFrameNearest(
                &app->characterData.xformData[i],
                &app->characterData.bvhData[i],
                app->scrubberSettings.playTime,
                app->characterData.scales[i]);
        }
        else if (app->scrubberSettings.sampleMode == 1)
        {
            TransformDataSampleFrameLinear(
                &app->characterData.xformData[i],
                &app->characterData.xformTmp0[i],
                &app->characterData.xformTmp1[i],
                &app->characterData.bvhData[i],
                app->scrubberSettings.playTime,
                app->characterData.scales[i]);
        }
        else
        {
            TransformDataSampleFrameCubic(
                &app->characterData.xformData[i],
                &app->characterData.xformTmp0[i],
                &app->characterData.xformTmp1[i],
                &app->characterData.xformTmp2[i],
                &app->characterData.xformTmp3[i],
                &app->characterData.bvhData[i],
                app->scrubberSettings.playTime,
                app->characterData.scales[i]);
        }

        if (app->scrubberSettings.inplace)
        {
            // Remove Translation on ground Plane

            app->characterData.xformData[i].localPositions[0].x = 0.0f;
            app->characterData.xformData[i].localPositions[0].z = 0.0f;

            // Attempt to extract rotation around vertical axis (this does not work 
            // for all animations but is pretty effective for almost all of them)

            Quaternion verticalRotation = QuaternionInvert(QuaternionNormalize(Quaternion{
                0.0f,
                app->characterData.xformData[i].localRotations[0].y,
                0.0f,
                app->characterData.xformData[i].localRotations[0].w,
                }));

            // Remove rotation around vertical axis

            app->characterData.xformData[i].localRotations[0] = QuaternionMultiply(
                verticalRotation,
                app->characterData.xformData[i].localRotations[0]);
        }

        TransformDataForwardKinematics(&app->characterData.xformData[i]);
    }

    // Update Controlled Character (root motion playback)

    if (app->controlledCharacter.active && effectiveDt > 0.0f)
    {
        // sync settings from config
        app->controlledCharacter.animMode = app->config.animationMode;
        app->controlledCharacter.cursorBlendMode = app->config.cursorBlendMode;
        app->controlledCharacter.blendPosReturnTime = app->config.blendPosReturnTime;

        ControlledCharacterUpdate(
            &app->controlledCharacter,
            &app->characterData,
            &app->animDatabase,
            effectiveDt,
            app->worldTime,
            app->config);

    }

    // Update Camera
    const Vector2 mouseDelta = GetMouseDelta();
    const float mouseWheel = GetMouseWheelMove();

    // Get bone target for orbit and turret camera modes
    Vector3 boneTarget = Vector3{ 0.0f, 1.0f, 0.0f };
    if (app->camera.track)
    {
        if (app->camera.trackControlledCharacter && app->controlledCharacter.active)
        {
            // Track controlled character's root (joint 0) or hips
            const int trackBone = MinInt(app->camera.trackBone,
                app->controlledCharacter.xformData.jointCount - 1);
            boneTarget = app->controlledCharacter.xformData.globalPositions[trackBone];
        }
        else if (app->characterData.count > 0 &&
            app->camera.trackBone < app->characterData.xformData[app->characterData.active].jointCount)
        {
            boneTarget = app->characterData.xformData[app->characterData.active].globalPositions[app->camera.trackBone];
        }
    }
    else if (app->camera.trackHipsProjectedOnGround)
    {
        // Track bone 0 (hips) but projected on ground at Y=1m for stable camera
        if (app->camera.trackControlledCharacter && app->controlledCharacter.active)
        {
            boneTarget = app->controlledCharacter.xformData.globalPositions[0];
        }
        else if (app->characterData.count > 0 && app->characterData.xformData[app->characterData.active].jointCount > 0)
        {
            boneTarget = app->characterData.xformData[app->characterData.active].globalPositions[0];
        }
        boneTarget.y = 1.0f;
    }

    // 'F' key toggles camera mode
    if (!imguiWantsKeyboard && IsKeyPressed(KEY_F))
    {
        if (app->camera.mode == FlomoCameraMode::Orbit)
        {
            CameraSwitchToUnreal(&app->camera);
        }
        else
        {
            CameraSwitchToOrbit(&app->camera, boneTarget);
        }
    }

    // Debug timescale controls (numpad)
    // numpad-: halve debugTimescale
    // numpad+: double debugTimescale (max 1.0), also unpause if paused
    // numpad*: pause, hold while paused to advance at half speed
    //if (!imguiWantsKeyboard)
    //{
    //    if (IsKeyPressed(KEY_KP_SUBTRACT))
    //    {
    //        app->debugTimescale *= 0.5f;
    //        TraceLog(LOG_INFO, "Debug timescale: %.4f", app->debugTimescale);
    //    }
    //    if (IsKeyPressed(KEY_KP_ADD))
    //    {
    //        if (app->debugPaused)
    //        {
    //            // Just unpause, don't change timescale
    //            app->debugPaused = false;
    //            TraceLog(LOG_INFO, "Unpaused at timescale: %.4f", app->debugTimescale);
    //        }
    //        else
    //        {
    //            // Double timescale up to max 1.0
    //            app->debugTimescale = Clamp(app->debugTimescale * 2.0f, 0.0f, 1.0f);
    //            TraceLog(LOG_INFO, "Debug timescale: %.4f", app->debugTimescale);
    //        }
    //    }
    //    if (IsKeyPressed(KEY_KP_MULTIPLY))
    //    {
    //        // Toggle pause
    //        app->debugPaused = true;
    //        TraceLog(LOG_INFO, "Paused (hold * to advance at half speed)");
    //    }
    //}

    if (app->camera.mode == FlomoCameraMode::Orbit)
    {
        // Orbit camera: RMB rotates, MMB pans, scroll zooms
        // Always update to follow target, but zero input when ImGui has mouse
        const bool acceptInput = !imguiWantsMouse;
        const float scrollInput = (acceptInput && !IsMouseButtonDown(2)) ? mouseWheel : 0.0f;

        OrbitCameraUpdate(
            &app->camera,
            boneTarget,
            app->config.cameraTargetBlendtime,
            (acceptInput && IsMouseButtonDown(1)) ? mouseDelta.x : 0.0f,  // RMB rotates
            (acceptInput && IsMouseButtonDown(1)) ? mouseDelta.y : 0.0f,
            (acceptInput && IsMouseButtonDown(2)) ? mouseDelta.x : 0.0f,  // MMB pans
            (acceptInput && IsMouseButtonDown(2)) ? mouseDelta.y : 0.0f,
            scrollInput,  // Scroll zooms
            rawDt);
    }
    else if (app->camera.mode == FlomoCameraMode::UnrealEditor)
    {
        // Unreal camera: RMB + WASD/QE moves, scroll adjusts speed
        const bool isActive = IsMouseButtonDown(1) && !imguiWantsMouse;
        // Ignore scroll wheel when MMB is held
        const float scrollInput = (imguiWantsMouse || IsMouseButtonDown(2)) ? 0.0f : mouseWheel;

        const bool isPanning = IsMouseButtonDown(2) && !imguiWantsMouse;

        UnrealCameraUpdate(
            &app->camera,
            mouseDelta.x,
            mouseDelta.y,
            scrollInput,
            !imguiWantsKeyboard && IsKeyDown(KEY_W),
            !imguiWantsKeyboard && IsKeyDown(KEY_S),
            !imguiWantsKeyboard && IsKeyDown(KEY_A),
            !imguiWantsKeyboard && IsKeyDown(KEY_D),
            !imguiWantsKeyboard && IsKeyDown(KEY_E),
            !imguiWantsKeyboard && IsKeyDown(KEY_Q),
            isActive,
            isPanning,
            rawDt);
    }
    else if (app->camera.mode == FlomoCameraMode::LazyTurretFollower)
    {
        // Lazy turret follower: hybrid mode - can rotate/zoom like orbit, smoothly follows otherwise
        const bool acceptInput = !imguiWantsMouse;
        const float scrollInput = (acceptInput && !IsMouseButtonDown(2)) ? mouseWheel : 0.0f;

        LazyTurretCameraUpdate(
            &app->camera,
            boneTarget,
            app->config.cameraTargetBlendtime,
            (acceptInput && IsMouseButtonDown(1)) ? mouseDelta.x : 0.0f,  // RMB rotates
            (acceptInput && IsMouseButtonDown(1)) ? mouseDelta.y : 0.0f,
            scrollInput,  // Scroll zooms
            rawDt);
    }

    // Update Player Control Input (WASD relative to camera)
    if (app->controlledCharacter.active && 
        !imguiWantsKeyboard &&
        app->camera.mode != FlomoCameraMode::UnrealEditor)
    {
        PlayerControlInput& input = app->controlledCharacter.playerInput;

        // Get input direction (camera-relative)
        Vector2 inputDir = Vector2Zero();

        // Check for gamepad input (left stick)
        if (IsGamepadAvailable(0))
        {
            const float leftX = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X);
            const float leftY = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_Y);

            // Apply deadzone (small values near zero are ignored)
            const float deadzone = 0.15f;
            if (fabsf(leftX) > deadzone || fabsf(leftY) > deadzone)
            {
                inputDir.x = leftX;
                inputDir.y = -leftY;  // invert Y for typical forward/back convention
            }
        }

        if (IsKeyDown(KEY_W)) inputDir.y += 1.0f;
        if (IsKeyDown(KEY_S)) inputDir.y -= 1.0f;
        if (IsKeyDown(KEY_D)) inputDir.x += 1.0f;
        if (IsKeyDown(KEY_A)) inputDir.x -= 1.0f;

        // Clamp magnitude to 1 (prevent diagonal speed boost)
        const float mag = Vector2Length(inputDir);
        if (mag > 1.0f) {
            inputDir = Vector2Scale(inputDir, 1.0f / mag);
        }

        // Get camera's actual forward and right vectors from cam3d
        const Vector3 camForward = Vector3Normalize(Vector3Subtract(app->camera.cam3d.target, app->camera.cam3d.position));
        const Vector3 camRight = Vector3Normalize(Vector3CrossProduct(camForward, Vector3{ 0.0f, 1.0f, 0.0f }));

        // Project camera vectors onto XZ plane (ignore Y component for ground movement)
        const Vector3 forward = Vector3Normalize(Vector3{ camForward.x, 0.0f, camForward.z });
        const Vector3 right = Vector3Normalize(Vector3{ camRight.x, 0.0f, camRight.z });

        // Combine input with camera orientation
        const Vector3 worldDir = Vector3Add(
            Vector3Scale(right, inputDir.x),
            Vector3Scale(forward, inputDir.y));

        input.desiredVelocity = Vector3Scale(worldDir, input.maxSpeed);

        // Aim direction from right stick (or camera direction when zero)
        Vector2 aimInput = Vector2Zero();
        if (IsGamepadAvailable(0))
        {
            const float rightX = GetGamepadAxisMovement(0, GAMEPAD_AXIS_RIGHT_X);
            const float rightY = GetGamepadAxisMovement(0, GAMEPAD_AXIS_RIGHT_Y);

            const float deadzone = 0.15f;
            if (fabsf(rightX) > deadzone || fabsf(rightY) > deadzone)
            {
                aimInput.x = rightX;
                aimInput.y = -rightY;  // invert Y
            }
        }

        if (Vector2Length(aimInput) > 0.01f)
        {
            // Right stick has input: use it for aim direction (camera-relative)
            const Vector3 aimWorldDir = Vector3Add(
                Vector3Scale(right, aimInput.x),
                Vector3Scale(forward, aimInput.y));
            const float aimLen = Vector3Length(aimWorldDir);
            if (aimLen > 1e-6f)
            {
                input.desiredAimDirection = Vector3Scale(aimWorldDir, 1.0f / aimLen);
            }
        }
        else
        {
            // No right stick input: use camera forward direction (horizontal)
            input.desiredAimDirection = forward;
        }
    }


    // Create Capsules

    CapsuleDataReset(&app->capsuleData);
    for (int i = 0; i < app->characterData.count; i++)
    {
        CapsuleDataAppendFromTransformData(
            &app->capsuleData,
            &app->characterData.xformData[i],
            app->characterData.radii[i],
            app->characterData.colors[i],
            app->characterData.opacities[i],
            !app->config.drawEndSites);
    }

    // Add controlled character's capsules
    if (app->controlledCharacter.active)
    {
        CapsuleDataAppendFromTransformData(
            &app->capsuleData,
            &app->controlledCharacter.xformData,
            app->controlledCharacter.radius,
            app->controlledCharacter.color,
            app->controlledCharacter.opacity,
            !app->config.drawEndSites);

        // Add pre-IK capsules (semi-transparent yellow/orange) when foot IK visualization is enabled
        if (app->config.drawFootIK && app->controlledCharacter.debugSaveBeforeIK)
        {
            CapsuleDataAppendFromTransformData(
                &app->capsuleData,
                &app->controlledCharacter.xformBeforeIK,
                app->controlledCharacter.radius,
                Color{ 255, 180, 0, 255 },  // orange/yellow to match skeleton
                0.4f,  // semi-transparent
                !app->config.drawEndSites);
        }

        // Add basic blend capsules (semi-transparent green) for comparison with lookahead
        if (app->config.drawBasicBlend)
        {
            CapsuleDataAppendFromTransformData(
                &app->capsuleData,
                &app->controlledCharacter.xformBasicBlend,
                app->controlledCharacter.radius,
                Color{ 100, 255, 100, 255 },  // green
                0.4f,  // semi-transparent
                !app->config.drawEndSites);
        }
    }

    PROFILE_END(Update);

    // Rendering

    Frustum frustum;
    FrustumFromCameraMatrices(
        //GetCameraProjectionMatrix(&app->camera.cam3d, (float)app->screenHeight / (float)app->screenWidth),
        GetCameraProjectionMatrix(&app->camera.cam3d, (float)app->screenWidth / (float)app->screenHeight),
        GetCameraViewMatrix(&app->camera.cam3d),
        frustum);


    PROFILE_BEGIN(Rendering);

    ClearBackground(app->config.backgroundColor);

    BeginMode3D(app->camera.cam3d);

    //DrawSphere(Vector3{ 0, 2, 0 }, 0.5f, RED);  // Simple red sphere
    //DrawCube(Vector3{ 2, 1, 0 }, 1, 1, 1, BLUE); // Simple blue cube


    // Set shader uniforms that don't change based on the object being drawn

    const Vector3 sunColorValue = { app->config.sunColor.r / 255.0f, app->config.sunColor.g / 255.0f, app->config.sunColor.b / 255.0f };
    const Vector3 skyColorValue = { app->config.skyColor.r / 255.0f, app->config.skyColor.g / 255.0f, app->config.skyColor.b / 255.0f };
    const float objectSpecularity = 0.5f;
    const float objectGlossiness = 10.0f;
    const float objectOpacity = 1.0f;

    const Vector3 sunLightPosition = Vector3RotateByQuaternion(Vector3{ 0.0f, 0.0f, 1.0f }, QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, app->config.sunAzimuth));
    const Vector3 sunLightAxis = Vector3Normalize(Vector3CrossProduct(sunLightPosition, Vector3{ 0.0f, 1.0f, 0.0f }));
    const Vector3 sunLightDir = Vector3Negate(Vector3RotateByQuaternion(sunLightPosition, QuaternionFromAxisAngle(sunLightAxis, app->config.sunAltitude)));

    SetShaderValue(app->shader, app->uniforms.cameraPosition, &app->camera.cam3d.position, SHADER_UNIFORM_VEC3);
    SetShaderValue(app->shader, app->uniforms.exposure, &app->config.exposure, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.sunDir, &sunLightDir, SHADER_UNIFORM_VEC3);
    SetShaderValue(app->shader, app->uniforms.sunStrength, &app->config.sunLightStrength, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.sunColor, &sunColorValue, SHADER_UNIFORM_VEC3);
    SetShaderValue(app->shader, app->uniforms.skyStrength, &app->config.skyLightStrength, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.skyColor, &skyColorValue, SHADER_UNIFORM_VEC3);
    SetShaderValue(app->shader, app->uniforms.ambientStrength, &app->config.ambientLightStrength, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.groundStrength, &app->config.groundLightStrength, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.objectSpecularity, &objectSpecularity, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.objectGlossiness, &objectGlossiness, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.objectOpacity, &objectOpacity, SHADER_UNIFORM_FLOAT);
    SetShaderValue(app->shader, app->uniforms.aoLookupResolution, &app->capsuleData.aoLookupResolution, SHADER_UNIFORM_VEC2);
    SetShaderValue(app->shader, app->uniforms.shadowLookupResolution, &app->capsuleData.shadowLookupResolution, SHADER_UNIFORM_VEC2);
    SetShaderValueTexture(app->shader, app->uniforms.aoLookupTable, app->capsuleData.aoLookupTable);
    SetShaderValueTexture(app->shader, app->uniforms.shadowLookupTable, app->capsuleData.shadowLookupTable);

    // Draw Ground

    PROFILE_BEGIN(RenderingGround);

    if (app->config.drawChecker)
    {
        const int groundIsCapsule = 0;
        const Vector3 groundColor = { 0.75f, 0.75f, 0.75f };

        SetShaderValue(app->shader, app->uniforms.isCapsule, &groundIsCapsule, SHADER_UNIFORM_INT);
        SetShaderValue(app->shader, app->uniforms.objectColor, &groundColor, SHADER_UNIFORM_VEC3);

        // Ground tile parameters
        const float tileSize = 2.0f;       // Size of each tile (matches groundPlaneMesh)
        const int tilesPerSide = 25;       // How many tiles to draw in each direction from center
        const float segmentRadius = tileSize * 0.707f;  // For frustum culling

        // Center the grid around the camera target (snapped to tile boundaries)
        const Vector3 camTarget = app->camera.cam3d.target;
        const float centerX = floorf(camTarget.x / tileSize) * tileSize;
        const float centerZ = floorf(camTarget.z / tileSize) * tileSize;

        for (int i = -tilesPerSide; i <= tilesPerSide; i++)
        {
            for (int j = -tilesPerSide; j <= tilesPerSide; j++)
            {
                // Tile position at fixed world coordinates
                const Vector3 groundSegmentPosition =
                {
                    centerX + i * tileSize,
                    0.0f,
                    centerZ + j * tileSize,
                };

                // Frustum culling
                if (!FrustumContainsSphere(frustum, groundSegmentPosition, segmentRadius * 1.1f))
                {
                    continue;
                }

                PROFILE_BEGIN(RenderingGroundSegment);

                // Gather all capsules casting AO on this ground segment

                PROFILE_BEGIN(RenderingGroundSegmentAO);

                app->capsuleData.aoCapsuleCount = 0;
                if (app->config.drawCapsules && app->config.drawAO)
                {
                    CapsuleDataUpdateAOCapsulesForGroundSegment(&app->capsuleData, groundSegmentPosition);
                }
                int aoCapsuleCount = MinInt(app->capsuleData.aoCapsuleCount, AO_CAPSULES_MAX);

                PROFILE_END(RenderingGroundSegmentAO);

                SetShaderValue(app->shader, app->uniforms.aoCapsuleCount, &aoCapsuleCount, SHADER_UNIFORM_INT);
                SetShaderValueV(app->shader, app->uniforms.aoCapsuleStarts, app->capsuleData.aoCapsuleStarts.data(), SHADER_UNIFORM_VEC3, aoCapsuleCount);
                SetShaderValueV(app->shader, app->uniforms.aoCapsuleVectors, app->capsuleData.aoCapsuleVectors.data(), SHADER_UNIFORM_VEC3, aoCapsuleCount);
                SetShaderValueV(app->shader, app->uniforms.aoCapsuleRadii, app->capsuleData.aoCapsuleRadii.data(), SHADER_UNIFORM_FLOAT, aoCapsuleCount);

                // Gather all capsules casting shadows on this ground segment

                PROFILE_BEGIN(RenderingGroundSegmentShadow);
                app->capsuleData.shadowCapsuleCount = 0;
                const float groundShadowDistance = 20.0f;
                if (Vector3DistanceSqr(groundSegmentPosition, camTarget) < Square(groundShadowDistance))
                {
                    // Only compute shadows for ground segments near the camera
                    if (app->config.drawCapsules && app->config.drawShadows)
                    {
                        CapsuleDataUpdateShadowCapsulesForGroundSegment(&app->capsuleData, groundSegmentPosition, sunLightDir, app->config.sunLightConeAngle);
                    }
                }

                const int shadowCapsuleCount = MinInt(app->capsuleData.shadowCapsuleCount, SHADOW_CAPSULES_MAX);

                PROFILE_END(RenderingGroundSegmentShadow);

                SetShaderValue(app->shader, app->uniforms.shadowCapsuleCount, &shadowCapsuleCount, SHADER_UNIFORM_INT);
                SetShaderValueV(app->shader, app->uniforms.shadowCapsuleStarts, app->capsuleData.shadowCapsuleStarts.data(), SHADER_UNIFORM_VEC3, shadowCapsuleCount);
                SetShaderValueV(app->shader, app->uniforms.shadowCapsuleVectors, app->capsuleData.shadowCapsuleVectors.data(), SHADER_UNIFORM_VEC3, shadowCapsuleCount);
                SetShaderValueV(app->shader, app->uniforms.shadowCapsuleRadii, app->capsuleData.shadowCapsuleRadii.data(), SHADER_UNIFORM_FLOAT, shadowCapsuleCount);

                // Draw

                DrawModel(app->groundPlaneModel, groundSegmentPosition, 1.0f, WHITE);

                PROFILE_END(RenderingGroundSegment);
            }
        }
    }

    PROFILE_END(RenderingGround);

    // Draw Capsules

    PROFILE_BEGIN(RenderingCapsules);

    if (app->config.drawCapsules && !app->genoRenderMode)
    {
        // Depth sort back to front for transparency

        for (int i = 0; i < app->capsuleData.capsuleCount; i++)
        {
            app->capsuleData.capsuleSort[i].index = i;
            app->capsuleData.capsuleSort[i].value = Vector3Distance(app->camera.cam3d.position, app->capsuleData.capsulePositions[i]);
        }

        //qsort(app->capsuleData.capsuleSort, app->capsuleData.capsuleCount, sizeof(CapsuleSort), CapsuleSortCompareLess);
        sort(app->capsuleData.capsuleSort.begin(), app->capsuleData.capsuleSort.begin() + app->capsuleData.capsuleCount,
            [](const CapsuleSort& a, const CapsuleSort& b) { return a.value < b.value; });

        // Render

        const int capsuleIsCapsule = 1;
        SetShaderValue(app->shader, app->uniforms.isCapsule, &capsuleIsCapsule, SHADER_UNIFORM_INT);

        for (int i = 0; i < app->capsuleData.capsuleCount; i++)
        {
            const int j = app->capsuleData.capsuleSort[i].index;

            // Check if we can cull capsule

            const Vector3 capsulePosition = app->capsuleData.capsulePositions[j];
            const float capsuleHalfLength = app->capsuleData.capsuleHalfLengths[j];
            const float capsuleRadius = app->capsuleData.capsuleRadii[j];

            if (!FrustumContainsSphere(frustum, capsulePosition, capsuleHalfLength + capsuleRadius))
            {
                continue;
            }

            PROFILE_BEGIN(RenderingCapsulesCapsule);

            // If capsule is semi-transparent disable depth mask

            if (app->capsuleData.capsuleOpacities[j] < 1.0f)
            {
                rlDrawRenderBatchActive();
                rlDisableDepthMask();
            }

            // Set shader properties

            const Quaternion capsuleRotation = app->capsuleData.capsuleRotations[j];
            const Vector3 capsuleStart = CapsuleStart(capsulePosition, capsuleRotation, capsuleHalfLength);
            const Vector3 capsuleVector = CapsuleVector(capsulePosition, capsuleRotation, capsuleHalfLength);

            SetShaderValue(app->shader, app->uniforms.objectColor, &app->capsuleData.capsuleColors[j], SHADER_UNIFORM_VEC3);
            SetShaderValue(app->shader, app->uniforms.objectOpacity, &app->capsuleData.capsuleOpacities[j], SHADER_UNIFORM_FLOAT);
            SetShaderValue(app->shader, app->uniforms.capsulePosition, &app->capsuleData.capsulePositions[j], SHADER_UNIFORM_VEC3);
            SetShaderValue(app->shader, app->uniforms.capsuleRotation, &app->capsuleData.capsuleRotations[j], SHADER_UNIFORM_VEC4);
            SetShaderValue(app->shader, app->uniforms.capsuleHalfLength, &app->capsuleData.capsuleHalfLengths[j], SHADER_UNIFORM_FLOAT);
            SetShaderValue(app->shader, app->uniforms.capsuleRadius, &app->capsuleData.capsuleRadii[j], SHADER_UNIFORM_FLOAT);
            SetShaderValue(app->shader, app->uniforms.capsuleStart, &capsuleStart, SHADER_UNIFORM_VEC3);
            SetShaderValue(app->shader, app->uniforms.capsuleVector, &capsuleVector, SHADER_UNIFORM_VEC3);

            // Find all capsules casting AO on this capsule

            PROFILE_BEGIN(RenderingCapsulesCapsuleAO);

            app->capsuleData.aoCapsuleCount = 0;
            if (app->config.drawAO)
            {
                CapsuleDataUpdateAOCapsulesForCapsule(&app->capsuleData, j);
            }
            const int aoCapsuleCount = MinInt(app->capsuleData.aoCapsuleCount, AO_CAPSULES_MAX);

            PROFILE_END(RenderingCapsulesCapsuleAO);

            SetShaderValue(app->shader, app->uniforms.aoCapsuleCount, &aoCapsuleCount, SHADER_UNIFORM_INT);
            SetShaderValueV(app->shader, app->uniforms.aoCapsuleStarts, app->capsuleData.aoCapsuleStarts.data(), SHADER_UNIFORM_VEC3, aoCapsuleCount);
            SetShaderValueV(app->shader, app->uniforms.aoCapsuleVectors, app->capsuleData.aoCapsuleVectors.data(), SHADER_UNIFORM_VEC3, aoCapsuleCount);
            SetShaderValueV(app->shader, app->uniforms.aoCapsuleRadii, app->capsuleData.aoCapsuleRadii.data(), SHADER_UNIFORM_FLOAT, aoCapsuleCount);

            // Find all capsules casting shadows on this capsule

            PROFILE_BEGIN(RenderingCapsulesCapsuleShadow);

            app->capsuleData.shadowCapsuleCount = 0;
            if (app->config.drawShadows)
            {
                const bool isCloseToCamera = i < 50; // Approximate check based on sort order
                if (isCloseToCamera)
                {
                    CapsuleDataUpdateShadowCapsulesForCapsule(&app->capsuleData, j, sunLightDir, app->config.sunLightConeAngle);
                }
            }
            const int shadowCapsuleCount = MinInt(app->capsuleData.shadowCapsuleCount, SHADOW_CAPSULES_MAX);

            PROFILE_END(RenderingCapsulesCapsuleShadow);

            SetShaderValue(app->shader, app->uniforms.shadowCapsuleCount, &shadowCapsuleCount, SHADER_UNIFORM_INT);
            SetShaderValueV(app->shader, app->uniforms.shadowCapsuleStarts, app->capsuleData.shadowCapsuleStarts.data(), SHADER_UNIFORM_VEC3, shadowCapsuleCount);
            SetShaderValueV(app->shader, app->uniforms.shadowCapsuleVectors, app->capsuleData.shadowCapsuleVectors.data(), SHADER_UNIFORM_VEC3, shadowCapsuleCount);
            SetShaderValueV(app->shader, app->uniforms.shadowCapsuleRadii, app->capsuleData.shadowCapsuleRadii.data(), SHADER_UNIFORM_FLOAT, shadowCapsuleCount);

            // Draw

            DrawModel(app->capsuleModel, Vector3Zero(), 1.0f, WHITE);

            // Reset depth mask if rendered semi-transparent

            if (app->capsuleData.capsuleOpacities[j] < 1.0f)
            {
                rlDrawRenderBatchActive();
                rlEnableDepthMask();
            }

            PROFILE_END(RenderingCapsulesCapsule);
        }
    }

    PROFILE_END(RenderingCapsules);

    // Geno Character Rendering

    if (app->genoRenderMode && app->genoModelLoaded && app->characterData.count > 0)
    {
        // Ensure we have mappings for all characters
        while ((int)app->genoMappings.size() < app->characterData.count)
        {
            const int idx = (int)app->genoMappings.size();
            BVHGenoMapping mapping = CreateBVHGenoMapping(
                &app->characterData.bvhData[idx],
                &app->genoModel);
            app->genoMappings.push_back(mapping);
        }

        // Draw all characters
        for (int c = 0; c < app->characterData.count; c++)
        {
            // Update Geno animation from current BVH pose
            UpdateGenoAnimationFromBVH(
                &app->genoAnimation,
                &app->characterData.xformData[c],
                &app->genoMappings[c],
                1.0f);  // Scale already applied by TransformDataSampleFrame

            // Update animation bones (raylib handles GPU skinning internally)
            UpdateModelAnimationBones(app->genoModel, app->genoAnimation, 0);

            // Draw the Geno model with character's color
            const Color charColor = app->characterData.colors[c];
            DrawModel(app->genoModel, Vector3Zero(), 1.0f, charColor);
        }
    }

    // Grid

    if (app->config.drawGrid)
    {
        DrawGrid(20, 1.0f);
    }

    // Origin

    if (app->config.drawOrigin)
    {
        DrawTransform(
            Vector3{ 0.0f, 0.01f, 0.0f },
            QuaternionIdentity(),
            1.0f);
    }

    // Disable Depth Test

    rlDrawRenderBatchActive();
    rlDisableDepthTest();

    // Draw Capsule Wireframes

    if (app->config.drawWireframes)
    {
        DrawWireFrames(&app->capsuleData, DARKGRAY);
    }

    // Draw Bones

    if (app->config.drawSkeleton)
    {
        for (int i = 0; i < app->characterData.count; i++)
        {
            DrawSkeleton(
                &app->characterData.xformData[i],
                app->config.drawEndSites,
                DARKGRAY,
                GRAY);
        }

        // Draw controlled character skeleton (uses character's color)
        if (app->controlledCharacter.active && app->controlledCharacter.xformData.jointCount > 0)
        {
            DrawSkeleton(
                &app->controlledCharacter.xformData,
                app->config.drawEndSites,
                app->controlledCharacter.color,
                GRAY);
        }
    }

    // Draw joint velocities
    if (app->config.drawVelocities && app->animDatabase.valid)
    {
        const float velScale = 0.1f;  // scale factor for velocity visualization

        for (int c = 0; c < app->characterData.count; ++c)
        {
            if (c >= (int)app->animDatabase.clipStartFrame.size()) continue;

            // compute current motion frame index for this character
            const float frameTime = app->animDatabase.animFrameTime[c];
            const int clipStart = app->animDatabase.clipStartFrame[c];
            const int clipEnd = app->animDatabase.clipEndFrame[c];
            const int clipFrameCount = clipEnd - clipStart;
            if (clipFrameCount <= 0 || frameTime <= 0.0f) continue;

            int localFrame = (int)(app->scrubberSettings.playTime / frameTime);
            localFrame = ClampInt(localFrame, 0, clipFrameCount - 1);
            const int motionFrame = clipStart + localFrame;

            const int jointCount = app->animDatabase.jointCount;
            const TransformData& xform = app->characterData.xformData[c];
            span<const Vector3> velRow = app->animDatabase.jointVelocitiesRootSpace.row_view(motionFrame);

            for (int j = 0; j < jointCount && j < xform.jointCount; ++j)
            {
                if (xform.endSite[j]) continue;  // skip end sites

                const Vector3 pos = xform.globalPositions[j];
                const Vector3 endPos = Vector3Add(pos, Vector3Scale(velRow[j], velScale));

                DrawLine3D(pos, endPos, BLUE);
            }
        }
    }

    // Draw joint accelerations
    if (app->config.drawAccelerations && app->animDatabase.valid)
    {
        const float accScale = 0.01f;  // scale factor for acceleration visualization (smaller since acc is larger)

        for (int c = 0; c < app->characterData.count; ++c)
        {
            if (c >= (int)app->animDatabase.clipStartFrame.size()) continue;

            const float frameTime = app->animDatabase.animFrameTime[c];
            const int clipStart = app->animDatabase.clipStartFrame[c];
            const int clipEnd = app->animDatabase.clipEndFrame[c];
            const int clipFrameCount = clipEnd - clipStart;
            if (clipFrameCount <= 0 || frameTime <= 0.0f) continue;

            int localFrame = (int)(app->scrubberSettings.playTime / frameTime);
            localFrame = ClampInt(localFrame, 0, clipFrameCount - 1);
            const int motionFrame = clipStart + localFrame;

            const int jointCount = app->animDatabase.jointCount;
            const TransformData& xform = app->characterData.xformData[c];
            span<const Vector3> accRow = app->animDatabase.jointAccelerationsRootSpace.row_view(motionFrame);

            for (int j = 0; j < jointCount && j < xform.jointCount; ++j)
            {
                if (xform.endSite[j]) continue;  // skip end sites

                const Vector3 pos = xform.globalPositions[j];
                const Vector3 endPos = Vector3Add(pos, Vector3Scale(accRow[j], accScale));

                DrawLine3D(pos, endPos, RED);
            }
        }
    }

    // Shared colors for cursor visualization (used by root velocities and skeleton drawing)
    const Color cursorColors[ControlledCharacter::MAX_BLEND_CURSORS] = {
        RED, GREEN, BLUE, YELLOW, MAGENTA,
        ORANGE, PINK, SKYBLUE, LIME, VIOLET
    };

    // Draw root motion velocities from each cursor
    if (app->controlledCharacter.active && app->config.drawRootVelocities)
    {
        const ControlledCharacter& cc = app->controlledCharacter;
        const float velScale = 0.5f;  // scale for velocity visualization
        const float yOffset = 0.05f;  // slight vertical offset to avoid z-fighting

        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = cc.cursors[ci];
            if (!cur.active) continue;
            if (cur.normalizedWeight < 0.01f) continue;  // skip very low weight cursors

            // draw from character's world position (all cursors share same world pos)
            const Vector3 startPos = Vector3Add(cc.worldPosition, Vector3{ 0.0f, yOffset * (ci + 1), 0.0f });
            const Vector3 endPos = Vector3Add(startPos, Vector3Scale(cur.rootVelocityWorldForDisplayOnly, velScale));

            const Color col = cursorColors[ci];
            DrawLine3D(startPos, endPos, col);

            // draw small sphere at end to make it more visible
            DrawSphere(endPos, 0.02f, col);

            // also draw yaw rate as a small arc/line perpendicular to velocity
            // (positive yaw = counter-clockwise when viewed from above)
            //const float yawVis = cur.rootYawRate * velScale * 0.5f;
            //const Vector3 yawDir = Vector3{ -sinf(yawVis), 0.0f, cosf(yawVis) };
            //const Vector3 yawEnd = Vector3Add(startPos, Vector3Scale(yawDir, 0.3f));
            //DrawLine3D(startPos, yawEnd, col);
        }

        // also draw the smoothed velocity in white
        {
            const Vector3 startPos = Vector3Add(cc.worldPosition, Vector3{ 0.0f, yOffset * 0.5f, 0.0f });
            const Vector3 endPos = Vector3Add(startPos, Vector3Scale(cc.rootVelocityWorld, velScale));
            DrawLine3D(startPos, endPos, WHITE);
            DrawSphere(endPos, 0.025f, WHITE);
        }
    }

    // Draw toe velocities (actual vs blended)
    if (app->controlledCharacter.active && app->config.drawToeVelocities && app->animDatabase.valid)
    {
        const ControlledCharacter& cc = app->controlledCharacter;
        const AnimDatabase& db = app->animDatabase;
        const float velScale = 0.3f;  // scale for velocity visualization

        for (int side : sides)
        {
            const int toeIdx = db.toeIndices[side];
            if (toeIdx < 0 || toeIdx >= cc.xformData.jointCount) continue;

            const Vector3 toePos = cc.xformData.globalPositions[toeIdx];

            // Post-IK velocity (yellow) - computed from final pose
            const Vector3 actualEnd = Vector3Add(toePos, Vector3Scale(cc.toeVelocity[side], velScale));
            DrawLine3D(toePos, actualEnd, YELLOW);
            DrawSphere(actualEnd, 0.015f, YELLOW);

            // Blended velocity (cyan) - weighted average from cursors
            const Vector3 blendedEnd = Vector3Add(toePos, Vector3Scale(cc.toeBlendedVelocityWorld[side], velScale));
            DrawLine3D(toePos, blendedEnd, SKYBLUE);
            DrawSphere(blendedEnd, 0.015f, SKYBLUE);
        }
    }

    // Draw Magic anchor (from controlled character's world transform)
    if (app->controlledCharacter.active && app->config.drawMagicAnchor && app->animDatabase.valid)
    {
        const ControlledCharacter& cc = app->controlledCharacter;
        const AnimDatabase& db = app->animDatabase;

        // Use the character's world position and rotation directly
        const Vector3 magicPos = Vector3{ cc.worldPosition.x, 0.02f, cc.worldPosition.z };  // slight offset to avoid z-fighting
        const Quaternion magicRot = cc.worldRotation;

        // Draw transform axes (magenta-ish color scheme)
        const float axisLen = 0.3f;
        DrawLine3D(magicPos, Vector3Add(magicPos, Vector3RotateByQuaternion(Vector3{ axisLen, 0.0f, 0.0f }, magicRot)), MAGENTA);
        DrawLine3D(magicPos, Vector3Add(magicPos, Vector3RotateByQuaternion(Vector3{ 0.0f, axisLen, 0.0f }, magicRot)), Color{ 255, 100, 255, 255 });
        DrawLine3D(magicPos, Vector3Add(magicPos, Vector3RotateByQuaternion(Vector3{ 0.0f, 0.0f, axisLen }, magicRot)), Color{ 200, 0, 200, 255 });

        // Draw small sphere at anchor point
        DrawSphere(magicPos, 0.03f, MAGENTA);

        // Draw line from magic anchor to spine3 (vertical reference)
        if (db.spine3Index >= 0)
        {
            const Vector3 spine3Pos = cc.xformData.globalPositions[db.spine3Index];
            DrawLine3D(magicPos, spine3Pos, Color{ 255, 100, 255, 128 });
        }
    }

    // Draw Magic anchor for animated character (scrubber-controlled)
    if (app->config.drawMagicAnchor && app->animDatabase.valid && app->characterData.count > 0)
    {
        const TransformData& xform = app->characterData.xformData[app->characterData.active];
        const AnimDatabase& db = app->animDatabase;

        if (db.spine3Index >= 0 && db.headIndex >= 0 && db.handIndices[SIDE_RIGHT] >= 0 &&
            db.spine3Index < xform.jointCount && db.headIndex < xform.jointCount &&
            db.handIndices[SIDE_RIGHT] < xform.jointCount)
        {
            // Get joint positions from current pose
            const Vector3 spine3Pos = xform.globalPositions[db.spine3Index];
            const Vector3 headPos = xform.globalPositions[db.headIndex];
            const Vector3 rightHandPos = xform.globalPositions[db.handIndices[SIDE_RIGHT]];

            // Magic position = spine3 projected onto ground
            const Vector3 magicPos = Vector3{ spine3Pos.x, 0.02f, spine3Pos.z };

            // Magic yaw = direction from head to right hand (projected to XZ)
            const Vector3 headToHand = Vector3Subtract(rightHandPos, headPos);
            const float magicYaw = atan2f(headToHand.x, headToHand.z);
            const Quaternion magicRot = QuaternionFromAxisAngle(Vector3{ 0.0f, 1.0f, 0.0f }, magicYaw);

            // Draw transform axes (orange color scheme to match animated character)
            const float axisLen = 0.3f;
            DrawLine3D(magicPos, Vector3Add(magicPos, Vector3RotateByQuaternion(Vector3{ axisLen, 0.0f, 0.0f }, magicRot)), ORANGE);
            DrawLine3D(magicPos, Vector3Add(magicPos, Vector3RotateByQuaternion(Vector3{ 0.0f, axisLen, 0.0f }, magicRot)), Color{ 255, 200, 100, 255 });
            DrawLine3D(magicPos, Vector3Add(magicPos, Vector3RotateByQuaternion(Vector3{ 0.0f, 0.0f, axisLen }, magicRot)), Color{ 200, 100, 0, 255 });

            // Draw small sphere at anchor point
            DrawSphere(magicPos, 0.03f, ORANGE);

            // Draw line from magic anchor to spine3 (vertical reference)
            DrawLine3D(magicPos, spine3Pos, Color{ 255, 200, 100, 128 });

            // Draw line from head to right hand (orientation reference)
            DrawLine3D(headPos, rightHandPos, Color{ 255, 180, 100, 200 });
        }
    }

    // Draw Foot IK debug (virtual toe positions, FK foot positions before IK, full FK skeleton before IK)
    if (app->controlledCharacter.active &&
        app->config.drawFootIK &&
        app->config.enableFootIK &&
        app->animDatabase.valid)
    {
        const ControlledCharacter& cc = app->controlledCharacter;
        const AnimDatabase& db = app->animDatabase;

        // Draw full FK skeleton BEFORE IK (semi-transparent orange/yellow wireframe)
        if (app->config.drawSkeleton && cc.debugSaveBeforeIK && cc.xformBeforeIK.jointCount > 0)
        {
            const int jc = cc.xformBeforeIK.jointCount;
            for (int j = 0; j < jc; ++j)
            {
                // Draw joint sphere (semi-transparent yellow)
                if (!cc.xformBeforeIK.endSite[j])
                {
                    DrawSphereWires(cc.xformBeforeIK.globalPositions[j], 0.012f, 4, 6, Color{ 255, 200, 0, 128 });
                }

                // Draw bone to parent
                const int p = cc.xformBeforeIK.parents[j];
                if (p != -1 && !cc.xformBeforeIK.endSite[j])
                {
                    DrawLine3D(cc.xformBeforeIK.globalPositions[j], cc.xformBeforeIK.globalPositions[p],
                        Color{ 255, 180, 0, 128 });
                }
            }
        }

        for (int side : sides)
        {
            const int toeIdx = db.toeIndices[side];
            const int footIdx = db.footIndices[side];

            if (toeIdx < 0 || toeIdx >= cc.xformData.jointCount) continue;

            // Get FK blended toe position (weighted from cursors)
            Vector3 blendedFKToePos = Vector3Zero();
            for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
            {
                const BlendCursor& cur = cc.cursors[ci];
                if (!cur.active) continue;
                const float w = cur.normalizedWeight;
                if (w <= 1e-6f) continue;

                if (toeIdx < (int)cur.globalPositions.size())
                {
                    blendedFKToePos = Vector3Add(blendedFKToePos,
                        Vector3Scale(cur.globalPositions[toeIdx], w));
                }
            }

            // Draw timed unlock clamp radius sphere (if unlocked)
            if (app->config.enableTimedUnlocking && cc.virtualToeUnlockTimer[side] >= 0.0f)
            {
                const float clampRadius = cc.virtualToeUnlockClampRadius[side];
                if (clampRadius > 0.001f)
                {
                    // Draw wireframe sphere centered at INTERMEDIATE virtual toe (not FK blend!)
                    // Color changes based on unlock progress (red = just unlocked, fades to orange)
                    const float unlockProgress = cc.virtualToeUnlockTimer[side] / app->config.unlockDuration;
                    const unsigned char fadeAlpha = (unsigned char)(unlockProgress * 200.0f + 55.0f);
                    const Color clampColor = Color{ 255, (unsigned char)(unlockProgress * 128.0f), 0, fadeAlpha };

                    DrawSphereWires(cc.lookaheadDragToePosWorld[side], clampRadius, 8, 8, clampColor);
                }
            }

            // Intermediate virtual toe (cyan) - unconstrained natural motion
            DrawSphere(cc.lookaheadDragToePosWorld[side], 0.022f, LIME);

            // Final virtual toe (magenta) - constrained, this is the IK target
            DrawSphere(cc.virtualToePos[side], 0.025f, MAGENTA);

            // Draw line showing constraint between intermediate and final
            if (app->config.enableTimedUnlocking && cc.virtualToeUnlockTimer[side] >= 0.0f)
            {
                DrawLine3D(cc.lookaheadDragToePosWorld[side], cc.virtualToePos[side], Color{ 255, 128, 255, 180 });
            }

            // FK toe position BEFORE IK (yellow) - where the toe was before correction
            if (cc.debugSaveBeforeIK && toeIdx < cc.xformBeforeIK.jointCount)
            {
                DrawSphere(cc.xformBeforeIK.globalPositions[toeIdx], 0.020f, YELLOW);

                // Draw line from pre-IK toe to final virtual toe target to show the correction
                DrawLine3D(cc.xformBeforeIK.globalPositions[toeIdx], cc.virtualToePos[side], ORANGE);
            }

            // FK foot position BEFORE IK (green)
            if (cc.debugSaveBeforeIK && footIdx >= 0 && footIdx < cc.xformBeforeIK.jointCount)
            {
                DrawSphere(cc.xformBeforeIK.globalPositions[footIdx], 0.020f, GREEN);
            }

            // Current (post-IK) toe position (skyblue) - to compare with target
            DrawSphere(cc.xformData.globalPositions[toeIdx], 0.018f, SKYBLUE);
        }
    }

    // Draw basic blend skeleton (semi-transparent green wireframe) - independent of foot IK
    if (app->controlledCharacter.active &&
        app->config.drawSkeleton &&
        app->config.drawBasicBlend &&
        app->controlledCharacter.xformBasicBlend.jointCount > 0)
    {
        const ControlledCharacter& cc = app->controlledCharacter;
        const int jc = cc.xformBasicBlend.jointCount;
        for (int j = 0; j < jc; ++j)
        {
            // Draw joint sphere (semi-transparent green)
            if (!cc.xformBasicBlend.endSite[j])
            {
                DrawSphereWires(cc.xformBasicBlend.globalPositions[j], 0.012f, 4, 6, Color{ 100, 255, 100, 128 });
            }

            // Draw bone to parent
            const int p = cc.xformBasicBlend.parents[j];
            if (p != -1 && !cc.xformBasicBlend.endSite[j])
            {
                DrawLine3D(cc.xformBasicBlend.globalPositions[j], cc.xformBasicBlend.globalPositions[p],
                    Color{ 80, 200, 80, 128 });
            }
        }
    }

    // Draw Blend Cursor Skeletons (for debug visualization)
    if (app->controlledCharacter.active && app->config.drawBlendCursors)
    {
        for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
        {
            const BlendCursor& cur = app->controlledCharacter.cursors[ci];
            if (!cur.active) continue;

            const Color baseColor = cursorColors[ci];
            // Fade color based on normalized weight
            const unsigned char alpha = (unsigned char)(cur.normalizedWeight * 255.0f);
            const Color drawColor = Color{ baseColor.r, baseColor.g, baseColor.b, alpha };

            const int jc = app->controlledCharacter.xformData.jointCount;
            for (int j = 0; j < jc; ++j)
            {
                // Draw joint sphere
                if (!app->controlledCharacter.xformData.endSite[j])
                {
                    DrawSphereWires(cur.globalPositions[j], 0.015f, 4, 6, drawColor);
                }

                // Draw bone to parent
                const int p = app->controlledCharacter.xformData.parents[j];
                if (p != -1 && !app->controlledCharacter.xformData.endSite[j])
                {
                    DrawLine3D(cur.globalPositions[j], cur.globalPositions[p], drawColor);
                }
            }
        }
    }

    // Draw Joint Transforms

    if (app->config.drawTransforms)
    {
        for (int i = 0; i < app->characterData.count; i++)
        {
            DrawTransforms(&app->characterData.xformData[i]);
        }

        // Draw controlled character transforms
        if (app->controlledCharacter.active)
        {
            DrawTransforms(&app->controlledCharacter.xformData);
        }
    }

    // Draw Player Input Arrow
    if (app->controlledCharacter.active && app->config.drawPlayerInput)
    {
        const PlayerControlInput& input = app->controlledCharacter.playerInput;
        const float velMag = Vector3Length(input.desiredVelocity);

        if (velMag > 0.01f)
        {
            const Vector3 startPos = Vector3{
                app->controlledCharacter.worldPosition.x,
                0.05f,
                app->controlledCharacter.worldPosition.z
            };
            const Vector3 endPos = Vector3Add(startPos, input.desiredVelocity);

            // Draw arrow shaft
            DrawLine3D(startPos, endPos, PURPLE);

            // Draw arrowhead
            const Vector3 dir = Vector3Normalize(input.desiredVelocity);
            const Vector3 right = Vector3{ -dir.z, 0.0f, dir.x };
            const float arrowSize = 0.15f;
            const Vector3 arrowTip1 = Vector3Add(endPos, Vector3Scale(Vector3Subtract(Vector3Scale(dir, -1.0f), right), arrowSize));
            const Vector3 arrowTip2 = Vector3Add(endPos, Vector3Scale(Vector3Add(Vector3Scale(dir, -1.0f), right), arrowSize));
            DrawLine3D(endPos, arrowTip1, PURPLE);
            DrawLine3D(endPos, arrowTip2, PURPLE);
            DrawSphere(endPos, 0.03f, PURPLE);
        }

        // Draw aim direction arrow (orange, from magic anchor)
        {
            const Vector3 aimStart = Vector3{
                app->controlledCharacter.worldPosition.x,
                1.5f,  // draw at head height
                app->controlledCharacter.worldPosition.z
            };
            const Vector3 aimEnd = Vector3Add(aimStart, input.desiredAimDirection);

            DrawLine3D(aimStart, aimEnd, ORANGE);

            // Draw arrowhead
            const Vector3 aimDir = input.desiredAimDirection;
            const Vector3 aimRight = Vector3{ -aimDir.z, 0.0f, aimDir.x };
            const float arrowSize = 0.1f;
            const Vector3 aimTip1 = Vector3Add(aimEnd, Vector3Scale(Vector3Subtract(Vector3Scale(aimDir, -1.0f), aimRight), arrowSize));
            const Vector3 aimTip2 = Vector3Add(aimEnd, Vector3Scale(Vector3Add(Vector3Scale(aimDir, -1.0f), aimRight), arrowSize));
            DrawLine3D(aimEnd, aimTip1, ORANGE);
            DrawLine3D(aimEnd, aimTip2, ORANGE);
            DrawSphere(aimEnd, 0.025f, ORANGE);
        }
    }

    // Draw Past Position History
    if (app->controlledCharacter.active && app->config.drawPastHistory)
    {
        const ControlledCharacter& cc = app->controlledCharacter;

        if (cc.positionHistory.size() > 1)
        {
            // Find which history point is being used for motion matching (closest to pastTimeOffset)
            int mmUsedIdx = -1;
            if (!cc.positionHistory.empty())
            {
                const double currentTime = cc.positionHistory.empty() ? 0.0f : cc.positionHistory.back().timestamp;
                const float targetPastTime = (float)(currentTime - app->animDatabase.featuresConfig.pastTimeOffset);

                float bestTimeDiff = FLT_MAX;
                for (int i = 0; i < (int)cc.positionHistory.size(); ++i)
                {
                    const float timeDiff = (float)abs(cc.positionHistory[i].timestamp - targetPastTime);
                    if (timeDiff < bestTimeDiff)
                    {
                        bestTimeDiff = timeDiff;
                        mmUsedIdx = i;
                    }
                }
            }

            // Draw line segments connecting history points
            for (size_t i = 1; i < cc.positionHistory.size(); ++i)
            {
                const Vector3 p0 = cc.positionHistory[i - 1].position;
                const Vector3 p1 = cc.positionHistory[i].position;

                // Color fades from dark (old) to bright (recent)
                const float t = (float)i / (float)cc.positionHistory.size();
                const unsigned char alpha = (unsigned char)(t * 255.0f);
                const Color lineColor = Color{ 255, 255, 0, alpha };

                DrawLine3D(p0, p1, lineColor);
            }

            // Draw small spheres at each history point
            for (size_t i = 0; i < cc.positionHistory.size(); ++i)
            {
                const float t = (float)i / (float)cc.positionHistory.size();
                const unsigned char alpha = (unsigned char)(t * 200.0f + 55.0f);
                const Color sphereColor = Color{ 255, 255, 0, alpha };

                // Draw larger red sphere for the motion matching sample point
                if ((int)i == mmUsedIdx)
                {
                    DrawSphere(cc.positionHistory[i].position, 0.04f, RED);
                }
                else
                {
                    DrawSphere(cc.positionHistory[i].position, 0.015f, sphereColor);
                }
            }
        }
    }

    // Re-Enable Depth Test

    rlDrawRenderBatchActive();
    rlEnableDepthTest();



    // Rendering Done

    EndMode3D();

    PROFILE_END(Rendering);

    // Begin ImGui frame
    rlImGuiBegin();

    // Draw UI

    PROFILE_BEGIN(Gui);

    if (app->config.drawUI) {
        // Error Message (only show if there's an error)
        if (app->errMsg[0] != '\0') {
            ImGui::SetNextWindowPos(ImVec2(250, 20), ImGuiCond_FirstUseEver);
            ImGui::Begin("Error", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", app->errMsg);
            ImGui::End();
        }

        if (app->characterData.count == 0) {
            ImGui::SetNextWindowPos(ImVec2((float)app->screenWidth / 2 - 330, (float)app->screenHeight / 2 - 15), ImGuiCond_FirstUseEver);
            ImGui::Begin("Info", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("Drag and Drop .bvh or .fbx files to open them.");
            ImGui::End();
        }

        // Render Settings
        ImGuiRenderSettings(&app->config, &app->capsuleData, app->screenWidth, app->screenHeight,
            &app->genoRenderMode, app->genoModelLoaded);

        // FPS
        if (app->config.drawFPS) {
            ImGui::SetNextWindowPos(ImVec2(230, 10), ImGuiCond_FirstUseEver);
            ImGui::Begin("FPS", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("FPS: %d", GetFPS());
            ImGui::End();
        }

        // Camera Settings
        ImGuiCamera(&app->camera, &app->characterData, &app->controlledCharacter, &app->config, app->argc, app->argv);

        // Characters
        ImGuiCharacterData(&app->characterData,
            &app->camera, &app->controlledCharacter,
            app->errMsg, app->argc, app->argv);

        // Color Picker
        if (app->characterData.colorPickerActive) {
            ImGui::SetNextWindowPos(ImVec2((float)app->screenWidth - 180, 450), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(160, 140), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Color Picker", &app->characterData.colorPickerActive)) {
                float col[3] = { app->characterData.colors[app->characterData.active].r / 255.f,
                                 app->characterData.colors[app->characterData.active].g / 255.f,
                                 app->characterData.colors[app->characterData.active].b / 255.f };
                ImGui::ColorPicker3("##picker", col);
                app->characterData.colors[app->characterData.active] =
                    Color{ (unsigned char)(col[0] * 255), (unsigned char)(col[1] * 255), (unsigned char)(col[2] * 255), 255 };
            }
            ImGui::End();
        }

        // Scrubber
        ImGuiScrubberSettings(&app->scrubberSettings, &app->characterData, app->screenWidth, app->screenHeight);

        // Animation settings
        ImGuiAnimSettings(app);

        ImGuiPlayerControl(&app->controlledCharacter, &app->config);

        // File Dialog
        //ImGuiWindowFileDialog(&app->fileDialogState);
    }

    if (app->config.drawFeatures && app->animDatabase.motionFrameCount > 0)
    {
        ImGui::SetNextWindowPos(ImVec2(60, 60), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Features"))
        {
            // Show features based on what camera is tracking
            ImGui::Separator();

            const bool trackingControlled = app->camera.trackControlledCharacter &&
                                            app->controlledCharacter.active;

            if (trackingControlled && !app->controlledCharacter.mmQuery.empty())
            {
                // MM Query from controlled character
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "MM Query (Controlled Character)");

                const int fd = app->animDatabase.featureDim;
                const std::vector<float>& query = app->controlledCharacter.mmQuery;

                for (int fi = 0; fi < fd && fi < (int)query.size(); ++fi)
                {
                    const char* fname = (fi < (int)app->animDatabase.featureNames.size()) ?
                        app->animDatabase.featureNames[fi].c_str() : "Feature";
                    ImGui::Text("%s: % .6f", fname, query[fi]);
                }
            }
            else
            {
                // Database features from simple animated character
                const int activeChar = app->characterData.active;
                if (activeChar >= 0 && activeChar < app->animDatabase.animCount &&
                    activeChar < (int)app->animDatabase.clipStartFrame.size() &&
                    app->animDatabase.clipStartFrame[activeChar] < app->animDatabase.clipEndFrame[activeChar])
                {
                    const float playTime = app->scrubberSettings.playTime;
                    const float frameTime = app->animDatabase.animFrameTime[activeChar];
                    const int clipFrameCount = app->animDatabase.animFrameCount[activeChar];

                    int localFrame = 0;
                    if (frameTime > 0.0f)
                    {
                        localFrame = ClampInt((int)(playTime / frameTime + 0.5f), 0, clipFrameCount - 1);
                    }

                    const int motionIndex = app->animDatabase.clipStartFrame[activeChar] + localFrame;
                    if (motionIndex >= 0 && motionIndex < app->animDatabase.motionFrameCount)
                    {
                        ImGui::Text("Active: %s", app->characterData.names[activeChar].c_str());
                        ImGui::SameLine();
                        ImGui::Text("LocalFrame: %d  MotionIndex: %d", localFrame, motionIndex);

                        const int fd = app->animDatabase.featureDim;
                        span<const float> featRow = app->animDatabase.features.row_view(motionIndex);
                        for (int fi = 0; fi < fd; ++fi)
                        {
                            const char* fname = (fi < (int)app->animDatabase.featureNames.size()) ?
                                app->animDatabase.featureNames[fi].c_str() : "Feature";
                            ImGui::Text("%s: % .6f", fname, featRow[fi]);
                        }
                    }
                    else
                    {
                        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Active clip has no motion-DB frames.");
                    }
                }
                else
                {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Active character not present in motion DB.");
                }
            }
        }
        ImGui::End();
    }

    // Blend Stack Debug Window
    if (app->controlledCharacter.active && app->config.drawBlendCursors)
    {
        // Cursor colors matching the 3D visualization
        const ImVec4 cursorImColors[ControlledCharacter::MAX_BLEND_CURSORS] = {
            ImVec4(1.0f, 0.0f, 0.0f, 1.0f),    // RED
            ImVec4(0.0f, 1.0f, 0.0f, 1.0f),    // GREEN
            ImVec4(0.0f, 0.0f, 1.0f, 1.0f),    // BLUE
            ImVec4(1.0f, 1.0f, 0.0f, 1.0f),    // YELLOW
            ImVec4(1.0f, 0.0f, 1.0f, 1.0f),    // MAGENTA
            ImVec4(1.0f, 0.5f, 0.0f, 1.0f),    // ORANGE
            ImVec4(1.0f, 0.75f, 0.8f, 1.0f),   // PINK
            ImVec4(0.5f, 0.8f, 1.0f, 1.0f),    // SKYBLUE
            ImVec4(0.0f, 1.0f, 0.5f, 1.0f),    // LIME
            ImVec4(0.5f, 0.0f, 1.0f, 1.0f),    // VIOLET
        };

        ImGui::SetNextWindowPos(ImVec2(20, 400), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(320, 200), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Blend Stack"))
        {
            // Debug timescale display
            if (app->debugPaused)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "PAUSED");
                ImGui::SameLine();
                ImGui::Text("(hold * to advance)");
            }
            else
            {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "PLAYING");
            }
            ImGui::Text("Timescale: %.4f", app->debugTimescale);
            ImGui::Text("Switch Timer: %.2f", app->controlledCharacter.switchTimer);
            ImGui::Separator();

            int activeCursorCount = 0;
            for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
            {
                const BlendCursor& cur = app->controlledCharacter.cursors[ci];
                if (cur.active) activeCursorCount++;
            }
            ImGui::Text("Active Cursors: %d", activeCursorCount);

            for (int ci = 0; ci < ControlledCharacter::MAX_BLEND_CURSORS; ++ci)
            {
                const BlendCursor& cur = app->controlledCharacter.cursors[ci];
                if (!cur.active) continue;

                ImGui::PushID(ci);

                // Color indicator
                ImGui::TextColored(cursorImColors[ci], "[%d]", ci);
                ImGui::SameLine();

                // Anim name (if available)
                const char* animName = (cur.animIndex >= 0 && cur.animIndex < (int)app->characterData.names.size())
                    ? app->characterData.names[cur.animIndex].c_str()
                    : "???";
                ImGui::Text("%s", animName);

                // Weight bar
                ImGui::Text("  W: %.3f -> %.3f (norm: %.3f)",
                    cur.weightSpring.x, cur.targetWeight, cur.normalizedWeight);

                // Progress bar showing normalized weight
                ImGui::ProgressBar(cur.normalizedWeight, ImVec2(-1, 0), "");

                // Time info
                const float maxTime = (cur.animIndex >= 0 && cur.animIndex < app->characterData.count)
                    ? (app->characterData.bvhData[cur.animIndex].frameCount - 1) * app->characterData.bvhData[cur.animIndex].frameTime
                    : 0.0f;
                ImGui::Text("  Time: %.2f / %.2f", cur.animTime, maxTime);

                //// Root motion delta info
                //const float deltaLen = Vector3Length(cur.lastDeltaWorld);
                //ImGui::Text("  dPos: (%.4f, %.4f) len=%.4f",
                //    cur.lastDeltaWorld.x, cur.lastDeltaWorld.z, deltaLen);
                //ImGui::Text("  dYaw: %.4f rad (%.2f deg)",
                //    cur.lastDeltaYaw, cur.lastDeltaYaw * RAD2DEG);

                ImGui::PopID();
            }

            // Blended result
            ImGui::Separator();
            ImGui::Text("Blended Result:");
        }
        ImGui::End();
    }
    PROFILE_END(Gui);

#if defined(ENABLE_PROFILE) && defined(_WIN32)
    // Display Profile Records
    PROFILE_TICKERS_UPDATE();

    ImGui::SetNextWindowPos(ImVec2(260, 10), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Profile", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        for (int i = 0; i < globalProfileRecords.num; i++) {
            ImGui::Text("%s", globalProfileRecords.records[i]->name);
            ImGui::SameLine();
            ImGui::Text("%6.1f us", globalProfileTickers.times[i]);
            ImGui::SameLine();
            ImGui::Text("%i calls", globalProfileTickers.samples[i]);
        }
    }
    ImGui::End();
#endif




    PROFILE_END(Gui);

    // End ImGui frame and render
    rlImGuiEnd();

    // Done

    EndDrawing();
}

//----------------------------------------------------------------------------------
// Main
//----------------------------------------------------------------------------------

// Command-line FBX to BVH conversion (runs without GUI)
static int ConvertFBXtoBVH(const char* inputPath)
{
    // Generate output path: input.fbx -> input.fbx.bvh
    char outputPath[512];
    snprintf(outputPath, sizeof(outputPath), "%s.bvh", inputPath);

    printf("Converting: %s -> %s\n", inputPath, outputPath);

    // Load FBX
    BVHData bvh;
    BVHDataInit(&bvh);
    char errMsg[512];

    if (!FBXDataLoad(&bvh, inputPath, errMsg, sizeof(errMsg)))
    {
        fprintf(stderr, "Error loading FBX: %s\n", errMsg);
        return 1;
    }

    printf("Loaded: %d joints, %d frames\n", bvh.jointCount, bvh.frameCount);

    // Save as BVH
    if (!BVHDataSave(&bvh, outputPath, errMsg, sizeof(errMsg)))
    {
        fprintf(stderr, "Error saving BVH: %s\n", errMsg);
        BVHDataFree(&bvh);
        return 1;
    }

    BVHDataFree(&bvh);
    printf("Success: %s\n", outputPath);
    return 0;
}

int main(int argc, char** argv)
{
    TestCudaAndLibtorchAndTCN();
    //testLegIk();
    //TestBallTree();
    //if (true) return 0;

    // Handle command-line utilities (no GUI)
    if (argc >= 3 && strcmp(argv[1], "-fbx2bvh") == 0)
    {
        return ConvertFBXtoBVH(argv[2]);
    }

    // Set current working directory to source root for file access
    // This helps find shader files and other resources
#if defined(SOURCE_ROOT_PATH)
    // On Windows, use SetCurrentDirectory
#if defined(_WIN32)
    SetCurrentDirectory(SOURCE_ROOT_PATH);
#else
    chdir(SOURCE_ROOT_PATH);
#endif
    printf("Working directory set to: %s\n", SOURCE_ROOT_PATH);
#endif

    PROFILE_INIT();
    PROFILE_TICKERS_INIT();


    // Init Application State

    ApplicationState app;
    app.argc = argc;
    app.argv = argv;

    // Load saved window config
    app.config = LoadAppConfig(argc, argv);
    app.screenWidth = app.config.windowWidth;
    app.screenHeight = app.config.windowHeight;

    // Init Window
    SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(app.screenWidth, app.screenHeight, "Flomo");
    SetTargetFPS(60);

    // Restore window position if we have a valid config
    if (app.config.valid) {
        SetWindowPosition(app.config.windowX, app.config.windowY);
    }

    // Init Dear ImGui - scale based on monitor DPI
    const Vector2 dpiScale = GetWindowScaleDPI();
    rlImGuiBeginInitImGui();

    ImGui::GetIO().FontGlobalScale = dpiScale.x;
    //ImGui::GetIO().FontGlobalScale = 2.0f;
    ImGui::StyleColorsDark();
    rlImGuiEndInitImGui();

    SetTraceLogLevel(LOG_DEBUG);

    // Camera
    CameraSystemInit(&app.camera, argc, argv);

    // Restore camera state from config
    if (app.config.valid) {
        app.camera.unreal.position = Vector3{ app.config.cameraPosX, app.config.cameraPosY, app.config.cameraPosZ };
        app.camera.unreal.yaw = app.config.cameraYaw;
        app.camera.unreal.pitch = app.config.cameraPitch;
        app.camera.unreal.moveSpeed = app.config.cameraMoveSpeed;
        app.camera.mode = static_cast<FlomoCameraMode>(ClampInt(app.config.cameraMode, 0, 2));
        app.camera.trackHipsProjectedOnGround = app.config.trackHipsProjectedOnGround;
    }

    // Shader

    app.shader = LoadShader("shaders/shader.vert", "shaders/shader.frag");
    if (app.shader.id == 0 || app.shader.locs == NULL) {
        TraceLog(LOG_ERROR, "Failed to load shader!");
    }

    ShaderUniformsInit(&app.uniforms, app.shader);

    // Models

    app.groundPlaneMesh = GenMeshPlane(2.0f, 2.0f, 1, 1);
    app.groundPlaneModel = LoadModelFromMesh(app.groundPlaneMesh);
    app.groundPlaneModel.materials[0].shader = app.shader;

    app.capsuleModel = LoadCapsuleModel();
    app.capsuleModel.materials[0].shader = app.shader;

    // Character Data

    CharacterDataInit(&app.characterData, argc, argv);

    // Capsule Data

    CapsuleDataInit(&app.capsuleData);

    // Controlled Character (starts inactive until first animation is loaded)

    app.controlledCharacter.active = false;

    // Scrubber Settings

    ScrubberSettingsInit(&app.scrubberSettings, argc, argv);

    // Render Settings

    CapsuleDataUpdateShadowLookupTable(&app.capsuleData, app.config.sunLightConeAngle);

    // Geno Character Rendering

    app.genoRenderMode = false;
    app.genoModelLoaded = false;

    // Try to load Geno model
    app.genoModel = LoadGenoModel("data/Geno.bin");
    if (app.genoModel.meshCount > 0 && app.genoModel.boneCount > 0)
    {
        app.genoModelLoaded = true;
        app.genoAnimation = LoadEmptyModelAnimation(app.genoModel);

        // Load skinned shader for GPU skeletal animation
        app.genoBasicShader = LoadShader("shaders/skinnedBasic.vs", "shaders/skinnedForward.fs");
        if (app.genoBasicShader.id > 0)
        {
            app.genoModel.materials[0].shader = app.genoBasicShader;
            TraceLog(LOG_INFO, "GENO: Loaded skinned shader successfully");
        }
        else
        {
            TraceLog(LOG_WARNING, "GENO: Failed to load skinned shader, animation may not work");
        }

        TraceLog(LOG_INFO, "GENO: Model loaded successfully, %d bones", app.genoModel.boneCount);
    }
    else
    {
        TraceLog(LOG_WARNING, "GENO: Failed to load Geno model from data/Geno.bin");
    }

    // File Dialog

    //app.fileDialogState = InitGuiWindowFileDialog(GetWorkingDirectory());

    // Reset Error Message

    app.errMsg[0] = '\0';

    // Load any files given as command line arguments

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-') { continue; }

        CharacterDataLoadFromFile(&app.characterData, argv[i], app.errMsg, 512);
    }

    const bool loadDefaultFiles = true;
    if (loadDefaultFiles && app.characterData.count == 0)
    {
        app.errMsg[0] = '\0';

        // Auto-load a default scene on startup
        {
            vector<const char*> autoFiles =
            {
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-002.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-003.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-004.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-005.fbx"
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-006.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-007.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-008.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-009.fbx"
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-010.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-011.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-012.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-013.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-014.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-015.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-016.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-017.fbx"
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-018.fbx",
                "data\\timi\\xs_20251101_aleblanc_lantern_nav-019.fbx"
            };
            for (const char* file : autoFiles)
            {
                if (CharacterDataLoadFromFile(&app.characterData, file, app.errMsg, sizeof(app.errMsg)))
                {
                    // Make the auto-loaded file the active character (if loaded)
                    app.characterData.active = app.characterData.count - 1;
                    TraceLog(LOG_INFO, "Auto-loaded default file at startup: %s", file);
                }
            }
        }
    }
    else
    {
        TraceLog(LOG_INFO, "Loaded %d character(s) at startup.", app.characterData.count);
    }

    // If any characters loaded, update capsules and scrubber

    if (app.characterData.count > 0)
    {
        app.characterData.active = app.characterData.count - 1;

        CapsuleDataUpdateForCharacters(&app.capsuleData, &app.characterData);
        ScrubberSettingsRecomputeLimits(&app.scrubberSettings, &app.characterData);
        ScrubberSettingsInitMaxs(&app.scrubberSettings, &app.characterData);

        app.animDatabase.featuresConfig = app.config.mmConfigEditor;
        app.animDatabase.poseDragLookaheadTime = app.config.poseDragLookaheadTimeEditor;
        app.animDatabase.blendRootModePosition = app.config.blendRootModePositionEditor;
        app.animDatabase.blendRootModeRotation = app.config.blendRootModeRotationEditor;

        // Build animation database and initialize controlled character
        AnimDatabaseRebuild(&app.animDatabase, &app.characterData);
        if (!app.animDatabase.valid) {
            TraceLog(LOG_WARNING, "AnimDatabase invalid at startup - controlled character disabled.");
            app.controlledCharacter.active = false;
        }
        else {
            ControlledCharacterInit(
                &app.controlledCharacter,
                &app.characterData.bvhData[0],
                app.characterData.scales[0],
                app.config.switchInterval);
        }

        // Resize capsule buffer to include controlled character
        // Account for up to 3x joints: normal + footIK debug + basicBlend debug
        {
            const int totalJoints = (int)app.capsuleData.capsulePositions.size() +
                app.controlledCharacter.xformData.jointCount * 3;
            CapsuleDataResize(&app.capsuleData, totalJoints);
        }

        string windowTitle = app.characterData.filePaths[app.characterData.active] + " - BVHView";
        SetWindowTitle(windowTitle.c_str());
    }

    // Game Loop

#if defined(PLATFORM_WEB)
    emscripten_set_main_loop_arg(ApplicationUpdate, &app, 0, 1);
#else
    while (!WindowShouldClose())
    {
        ApplicationUpdate(&app);
    }
#endif

    // Unload and finish

    CapsuleDataFree(&app.capsuleData);

    // Unload Geno resources
    if (app.genoModelLoaded)
    {
        app.genoMappings.clear();
        UnloadModelAnimation(app.genoAnimation);
        UnloadModel(app.genoModel);
        if (app.genoBasicShader.id > 0)
        {
            UnloadShader(app.genoBasicShader);
        }
    }

    UnloadModel(app.capsuleModel);
    UnloadModel(app.groundPlaneModel);
    UnloadShader(app.shader);

    // Shutdown Dear ImGui
    rlImGuiShutdown();

    // Save config before closing (window + camera)
    const Vector2 windowPos = GetWindowPosition();
    app.config.windowX = (int)windowPos.x;
    app.config.windowY = (int)windowPos.y;
    app.config.windowWidth = app.screenWidth;
    app.config.windowHeight = app.screenHeight;

    app.config.cameraPosX = app.camera.unreal.position.x;
    app.config.cameraPosY = app.camera.unreal.position.y;
    app.config.cameraPosZ = app.camera.unreal.position.z;
    app.config.cameraYaw = app.camera.unreal.yaw;
    app.config.cameraPitch = app.camera.unreal.pitch;
    app.config.cameraMoveSpeed = app.camera.unreal.moveSpeed;
    app.config.cameraMode = static_cast<int>(app.camera.mode);
    app.config.trackHipsProjectedOnGround = app.camera.trackHipsProjectedOnGround;


    SaveAppConfig(app.config);

    CloseWindow();

    return 0;
}
