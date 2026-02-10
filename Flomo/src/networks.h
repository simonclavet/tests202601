#pragma once

#include "definitions.h"
#include <string>
#include <filesystem>
#include <unordered_map>

extern "C" void cuda_init_context();


// Bone importance weights from mesh skinning influence
// Paper: "Learned Motion Matching" (Holden et al.)
// https://theorangeduck.com/media/uploads/other_stuff/ControlOperators.pdf
// Used to weight pose feature loss - more important bones (hips, spine, legs) have higher weights
static inline float GetBoneWeight(const std::string& boneName)
{
    static const std::unordered_map<std::string, float> boneWeights = {
        {"Simulation",       0.00000000f},
        {"Hips",             0.27088639f},
        {"Spine",            0.12776886f},
        {"Spine1",           0.10730254f},
        {"Spine2",           0.08733685f},
        {"Spine3",           0.07508411f},
        {"Neck",             0.00838600f},
        {"Neck1",            0.00639638f},
        {"Head",             0.00515253f},
        {"HeadEnd",          0.00063045f},
        {"RightShoulder",    0.02654437f},
        {"RightArm",         0.02060832f},
        {"RightForeArm",     0.00825604f},
        {"RightHand",        0.00213240f},
        {"RightHandThumb1",  0.00073802f},
        {"RightHandThumb2",  0.00066565f},
        {"RightHandThumb3",  0.00063558f},
        {"RightHandThumb4",  0.00063045f},
        {"RightHandIndex1",  0.00070377f},
        {"RightHandIndex2",  0.00064898f},
        {"RightHandIndex3",  0.00063289f},
        {"RightHandIndex4",  0.00063045f},
        {"RightHandMiddle1", 0.00072178f},
        {"RightHandMiddle2", 0.00065547f},
        {"RightHandMiddle3", 0.00063321f},
        {"RightHandMiddle4", 0.00063045f},
        {"RightHandRing1",   0.00070793f},
        {"RightHandRing2",   0.00065231f},
        {"RightHandRing3",   0.00063322f},
        {"RightHandRing4",   0.00063045f},
        {"RightHandPinky1",  0.00067184f},
        {"RightHandPinky2",  0.00063829f},
        {"RightHandPinky3",  0.00063110f},
        {"RightHandPinky4",  0.00063045f},
        {"RightForeArmEnd",  0.00063045f},
        {"RightArmEnd",      0.00063045f},
        {"LeftShoulder",     0.02739252f},
        {"LeftArm",          0.02113067f},
        {"LeftForeArm",      0.00849728f},
        {"LeftHand",         0.00210641f},
        {"LeftHandThumb1",   0.00071845f},
        {"LeftHandThumb2",   0.00065790f},
        {"LeftHandThumb3",   0.00063489f},
        {"LeftHandThumb4",   0.00063045f},
        {"LeftHandIndex1",   0.00069211f},
        {"LeftHandIndex2",   0.00064446f},
        {"LeftHandIndex3",   0.00063293f},
        {"LeftHandIndex4",   0.00063045f},
        {"LeftHandMiddle1",  0.00071069f},
        {"LeftHandMiddle2",  0.00065042f},
        {"LeftHandMiddle3",  0.00063314f},
        {"LeftHandMiddle4",  0.00063045f},
        {"LeftHandRing1",    0.00070524f},
        {"LeftHandRing2",    0.00065236f},
        {"LeftHandRing3",    0.00063302f},
        {"LeftHandRing4",    0.00063045f},
        {"LeftHandPinky1",   0.00067250f},
        {"LeftHandPinky2",   0.00064092f},
        {"LeftHandPinky3",   0.00063160f},
        {"LeftHandPinky4",   0.00063045f},
        {"LeftForeArmEnd",   0.00063045f},
        {"LeftArmEnd",       0.00063045f},
        {"RightUpLeg",       0.05690333f},
        {"RightLeg",         0.02043630f},
        {"RightFoot",        0.00305942f},
        {"RightToeBase",     0.00080056f},
        {"RightToeBaseEnd",  0.00063045f},
        {"RightLegEnd",      0.00063045f},
        {"RightUpLegEnd",    0.00063045f},
        {"LeftUpLeg",        0.05668447f},
        {"LeftLeg",          0.02033588f},
        {"LeftFoot",         0.00289429f},
        {"LeftToeBase",      0.00078392f},
        {"LeftToeBaseEnd",   0.00063045f},
        {"LeftLegEnd",       0.00063045f},
        {"LeftUpLegEnd",     0.00063045f}
    };

    auto it = boneWeights.find(boneName);
    if (it != boneWeights.end())
    {
        return it->second;
    }

    // Default weight for unknown bones
    return 0.01f;
}

static inline void NetworkInitAutoEncoder(NetworkState* state, int inputDim, int latentDim, bool startTraining = true)
{
    if (inputDim <= 0) return;

    // 1. Explicit CUDA context initialization
    TraceLog(LOG_INFO, "Initializing CUDA context...");
    cuda_init_context();

    // 2. Detect device: default to GPU if available
    bool cuda_available = torch::cuda::is_available();
    state->device = cuda_available ? torch::kCUDA : torch::kCPU;
    
    TraceLog(LOG_INFO, "Neural Network Device Check:");
    TraceLog(LOG_INFO, "  CUDA available: %s", cuda_available ? "YES" : "NO");
    if (cuda_available) {
        TraceLog(LOG_INFO, "  CUDA device count: %d", (int)torch::cuda::device_count());
        
        // Trigger lazy initialization of CUDA kernels (like RNG and Math)
        try {
            torch::Tensor dummy = torch::ones({ 8, 8 }, torch::kCUDA);
            torch::Tensor dummy2 = dummy * 2.0f; // This will fail if kernels for the GPU architecture are missing
            TraceLog(LOG_INFO, "  CUDA kernels tested successfully.");
        }
        catch (const std::exception& e) {
            TraceLog(LOG_ERROR, "  CUDA kernel execution failed: %s", e.what());
            TraceLog(LOG_INFO, "  Your GPU architecture (Blackwell/SM 12.0?) might be too new for this LibTorch version (2.5.1+cu121).");
            TraceLog(LOG_INFO, "  Falling back to CPU training.");
            state->device = torch::kCPU;
        }
    }
    TraceLog(LOG_INFO, "  Selected device: %s", state->device.is_cuda() ? "CUDA" : "CPU");

    // Design: Denoising Autoencoder with bottleneck
    // Encoder: D -> 128 -> 64 -> L
    // Decoder: L -> 64 -> 128 -> D
    
    state->featuresAutoEncoder = torch::nn::Sequential(
        // Encoder
        torch::nn::Linear(inputDim, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, latentDim),
        // Decoder
        torch::nn::Linear(latentDim, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, inputDim)
    );

    // Move model to detected device
    state->featuresAutoEncoder->to(state->device);

    // Initialize optimizer with model parameters
    state->optimizer = std::make_shared<torch::optim::Adam>(
        state->featuresAutoEncoder->parameters(), torch::optim::AdamOptions(1e-3));
    
    state->isTraining = startTraining;
    state->currentLoss = 0.0f;
    state->iterations = 0;

    TraceLog(LOG_INFO, "Features AutoEncoder initialized: %d -> %d -> %d", inputDim, latentDim, inputDim);
}

static inline void NetworkSave(const NetworkState* state, const std::string& folderPath)
{
    if (!state->featuresAutoEncoder) return;

    const std::string aePath = folderPath + "/featureAutoEncoder.bin";
    try {
        torch::save(state->featuresAutoEncoder, aePath);
        TraceLog(LOG_INFO, "Saved AutoEncoder to: %s", aePath.c_str());
    }
    catch (const std::exception& e) {
        TraceLog(LOG_ERROR, "Failed to save AutoEncoder: %s", e.what());
    }
}

static inline void NetworkLoad(NetworkState* state, int inputDim, int latentDim, const std::string& folderPath)
{
    const std::string aePath = folderPath + "/featureAutoEncoder.bin";
    if (!std::filesystem::exists(aePath)) return;

    try {
        // Initialize model structure with correct dimensions, but don't start training
        NetworkInitAutoEncoder(state, inputDim, latentDim, false);
        torch::load(state->featuresAutoEncoder, aePath);
        state->featuresAutoEncoder->to(state->device);
        TraceLog(LOG_INFO, "Loaded AutoEncoder from: %s", aePath.c_str());
    }
    catch (const std::exception& e) {
        TraceLog(LOG_ERROR, "Failed to load AutoEncoder: %s", e.what());
    }
}

static inline void NetworkTrainAutoEncoder(NetworkState* state, const AnimDatabase* db)
{
    if (!state->isTraining || !state->featuresAutoEncoder || !state->optimizer || !db->valid) return;
    if (db->normalizedFeatures.empty() || db->featureDim <= 0) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int totalFrames = db->motionFrameCount;

    state->featuresAutoEncoder->train();

    try {
        for (int i = 0; i < 3; i++)
        {
            // 1. Sample random batch from database on CPU first
            torch::Tensor targetHost = torch::empty({ batchSize, featureDim });
            float* target_ptr = targetHost.data_ptr<float>();

            for (int b = 0; b < batchSize; b++)
            {
                int idx = rand() % totalFrames;
                auto row = db->normalizedFeatures.row_view(idx);
                std::copy(row.begin(), row.end(), target_ptr + b * featureDim);
            }

            // Move target to device (GPU or CPU)
            torch::Tensor target = targetHost.to(state->device);

            // 2. Add noise for Denoising AutoEncoder (DAE) 
            // Create noise on CPU then move to device to avoid GPU RNG issues during initial debugging
            torch::Tensor noise = torch::randn({ batchSize, featureDim }).to(state->device) * 0.05f;
            torch::Tensor input = target + noise;

            // 3. Optimization step
            state->optimizer->zero_grad();
            torch::Tensor output = state->featuresAutoEncoder->forward(input);
            torch::Tensor loss = torch::mse_loss(output, target);

            loss.backward();
            state->optimizer->step();

            state->currentLoss = loss.item<float>();
            state->iterations++;
        }
    }
    catch (const std::exception& e) {
        TraceLog(LOG_ERROR, "NN Training Error: %s", e.what());
        state->isTraining = false;
    }

    if (state->iterations % 300 == 0 && state->iterations > 0)
    {
        TraceLog(LOG_INFO, "AE Loss (iter %d): %.4f", state->iterations, state->currentLoss);
    }
}

// Keep these for compatibility if called, but they are now replaced by AE versions
static inline void NetworkInit(NetworkState* state) { (void)state; }
static inline void NetworkTrainStep(NetworkState* state) { (void)state; }