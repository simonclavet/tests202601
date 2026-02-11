#pragma once

#include "definitions.h"
#include <string>
#include <filesystem>
#include <unordered_map>
#include <chrono>

extern "C" void cuda_init_context();

constexpr int FEATURE_AE_LATENT_DIM = 16;
constexpr int SEGMENT_AE_LATENT_DIM = 128;
constexpr double HEADSTART_SECONDS = 10.0;
constexpr double AUTOSAVE_INTERVAL_SECONDS = 300.0;

// Bone importance weights from mesh skinning influence
// Paper: "Learned Motion Matching" (Holden et al.)
// https://theorangeduck.com/media/uploads/other_stuff/ControlOperators.pdf
// Used to weight pose feature loss
static inline float GetBoneWeight(const std::string& boneName)
{
    static const std::unordered_map<std::string, float>
        boneWeights = {
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

    return 0.01f;
}

//---------------------------------------------------------
// encoder / decoder helpers
//---------------------------------------------------------

// Both AEs have the same layer layout:
// [0] Linear  [1] ReLU  [2] Linear  [3] ReLU  [4] Linear
// (encoder bottleneck)
// [5] Linear  [6] ReLU  [7] Linear  [8] ReLU  [9] Linear

// runs only the encoder half (layers 0-4)
static inline torch::Tensor EncodeWithAE(
    torch::nn::Sequential& ae,
    const torch::Tensor& input)
{
    std::vector<std::shared_ptr<torch::nn::Module>>
        children = ae->children();
    torch::nn::LinearImpl* l0 =
        children[0]->as<torch::nn::LinearImpl>();
    torch::nn::LinearImpl* l2 =
        children[2]->as<torch::nn::LinearImpl>();
    torch::nn::LinearImpl* l4 =
        children[4]->as<torch::nn::LinearImpl>();
    assert(l0 && l2 && l4);

    torch::Tensor x = input;
    x = torch::relu(l0->forward(x));
    x = torch::relu(l2->forward(x));
    x = l4->forward(x);
    return x;
}

// runs only the decoder half (layers 5-9)
static inline torch::Tensor DecodeWithAE(
    torch::nn::Sequential& ae,
    const torch::Tensor& latent)
{
    std::vector<std::shared_ptr<torch::nn::Module>>
        children = ae->children();
    torch::nn::LinearImpl* l5 =
        children[5]->as<torch::nn::LinearImpl>();
    torch::nn::LinearImpl* l7 =
        children[7]->as<torch::nn::LinearImpl>();
    torch::nn::LinearImpl* l9 =
        children[9]->as<torch::nn::LinearImpl>();
    assert(l5 && l7 && l9);

    torch::Tensor x = latent;
    x = torch::relu(l5->forward(x));
    x = torch::relu(l7->forward(x));
    x = l9->forward(x);
    return x;
}

//---------------------------------------------------------
// CUDA device setup (shared by all networks)
//---------------------------------------------------------

static inline void NetworkEnsureDevice(NetworkState* state)
{
    if (state->device.is_cuda()) return;

    TraceLog(LOG_INFO, "Initializing CUDA context...");
    cuda_init_context();

    const bool cudaOk = torch::cuda::is_available();
    state->device = cudaOk ? torch::kCUDA : torch::kCPU;

    TraceLog(LOG_INFO, "  CUDA available: %s",
        cudaOk ? "YES" : "NO");

    if (cudaOk)
    {
        TraceLog(LOG_INFO, "  CUDA device count: %d",
            (int)torch::cuda::device_count());
        try
        {
            torch::Tensor dummy =
                torch::ones({ 8, 8 }, torch::kCUDA);
            torch::Tensor dummy2 = dummy * 2.0f;
            TraceLog(LOG_INFO,
                "  CUDA kernels tested successfully.");
        }
        catch (const std::exception& e)
        {
            TraceLog(LOG_ERROR,
                "  CUDA kernel execution failed: %s",
                e.what());
            TraceLog(LOG_INFO,
                "  Falling back to CPU training.");
            state->device = torch::kCPU;
        }
    }

    TraceLog(LOG_INFO, "  Selected device: %s",
        state->device.is_cuda() ? "CUDA" : "CPU");
}

//---------------------------------------------------------
// feature autoencoder
//---------------------------------------------------------

static inline void NetworkInitFeatureAE(
    NetworkState* state,
    int inputDim,
    int latentDim)
{
    if (inputDim <= 0) return;

    NetworkEnsureDevice(state);

    // encoder: D -> 128 -> 64 -> L
    // decoder: L -> 64 -> 128 -> D
    state->featuresAutoEncoder = torch::nn::Sequential(
        torch::nn::Linear(inputDim, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, latentDim),
        torch::nn::Linear(latentDim, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, inputDim)
    );

    state->featuresAutoEncoder->to(state->device);

    state->featureAEOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->featuresAutoEncoder->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->featureAELoss = 0.0f;
    state->featureAEIterations = 0;

    TraceLog(LOG_INFO,
        "Feature AE initialized: %d -> %d -> %d",
        inputDim, latentDim, inputDim);
}

static inline void NetworkSaveFeatureAE(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->featuresAutoEncoder) return;

    const std::string path =
        folderPath + "/featureAutoEncoder.bin";
    try
    {
        torch::save(state->featuresAutoEncoder, path);
        TraceLog(LOG_INFO,
            "Saved Feature AE to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to save Feature AE: %s", e.what());
    }
}

static inline void NetworkLoadFeatureAE(
    NetworkState* state,
    int inputDim,
    int latentDim,
    const std::string& folderPath)
{
    const std::string path =
        folderPath + "/featureAutoEncoder.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitFeatureAE(state, inputDim, latentDim);
        torch::load(state->featuresAutoEncoder, path);
        state->featuresAutoEncoder->to(state->device);
        TraceLog(LOG_INFO,
            "Loaded Feature AE from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to load Feature AE: %s", e.what());
    }
}

//---------------------------------------------------------
// segment autoencoder
//---------------------------------------------------------

static inline void NetworkInitSegmentAE(
    NetworkState* state,
    int flatDim,
    int latentDim)
{
    if (flatDim <= 0) return;

    NetworkEnsureDevice(state);

    // flatDim -> 512 -> 256 -> L -> 256 -> 512 -> flatDim
    state->segmentAutoEncoder = torch::nn::Sequential(
        torch::nn::Linear(flatDim, 512),
        torch::nn::ReLU(),
        torch::nn::Linear(512, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, latentDim),
        torch::nn::Linear(latentDim, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, 512),
        torch::nn::ReLU(),
        torch::nn::Linear(512, flatDim)
    );

    state->segmentAutoEncoder->to(state->device);

    state->segmentOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->segmentAutoEncoder->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->segmentAELoss = 0.0f;
    state->segmentAEIterations = 0;

    TraceLog(LOG_INFO,
        "Segment AE initialized: %d -> 512 -> 256 -> %d"
        " -> 256 -> 512 -> %d",
        flatDim, latentDim, flatDim);
}

static inline void NetworkSaveSegmentAE(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->segmentAutoEncoder) return;

    const std::string path =
        folderPath + "/segmentAutoEncoder.bin";
    try
    {
        torch::save(state->segmentAutoEncoder, path);
        TraceLog(LOG_INFO,
            "Saved Segment AE to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to save Segment AE: %s", e.what());
    }
}

static inline void NetworkLoadSegmentAE(
    NetworkState* state,
    int flatDim,
    int latentDim,
    const std::string& folderPath)
{
    const std::string path =
        folderPath + "/segmentAutoEncoder.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitSegmentAE(state, flatDim, latentDim);
        torch::load(state->segmentAutoEncoder, path);
        state->segmentAutoEncoder->to(state->device);
        TraceLog(LOG_INFO,
            "Loaded Segment AE from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to load Segment AE: %s", e.what());
    }
}

//---------------------------------------------------------
// segment latent average predictor
//---------------------------------------------------------

static inline void NetworkInitPredictor(
    NetworkState* state,
    int featureLatentDim,
    int segmentLatentDim)
{
    NetworkEnsureDevice(state);

    // featureLatent -> 128 -> 256 -> segmentLatent
    state->segmentLatentAveragePredictor =
        torch::nn::Sequential(
            torch::nn::Linear(featureLatentDim, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, 256),
            torch::nn::ReLU(),
            torch::nn::Linear(256, segmentLatentDim)
        );

    state->segmentLatentAveragePredictor->to(
        state->device);

    state->predictorOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->segmentLatentAveragePredictor
                ->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->predictorLoss = 0.0f;
    state->predictorIterations = 0;

    TraceLog(LOG_INFO,
        "Latent Predictor initialized: %d -> 128 -> 256"
        " -> %d",
        featureLatentDim, segmentLatentDim);
}

static inline void NetworkSavePredictor(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->segmentLatentAveragePredictor) return;

    const std::string path =
        folderPath + "/segmentLatentAveragePredictor.bin";
    try
    {
        torch::save(
            state->segmentLatentAveragePredictor, path);
        TraceLog(LOG_INFO,
            "Saved Latent Predictor to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to save Latent Predictor: %s",
            e.what());
    }
}

static inline void NetworkLoadPredictor(
    NetworkState* state,
    int featureLatentDim,
    int segmentLatentDim,
    const std::string& folderPath)
{
    const std::string path =
        folderPath + "/segmentLatentAveragePredictor.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitPredictor(
            state, featureLatentDim, segmentLatentDim);
        torch::load(
            state->segmentLatentAveragePredictor, path);
        state->segmentLatentAveragePredictor->to(
            state->device);
        TraceLog(LOG_INFO,
            "Loaded Latent Predictor from: %s",
            path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to load Latent Predictor: %s",
            e.what());
    }
}

//---------------------------------------------------------
// save/load all networks at once
//---------------------------------------------------------

static inline void NetworkSaveAll(
    const NetworkState* state,
    const std::string& folderPath)
{
    NetworkSaveFeatureAE(state, folderPath);
    NetworkSaveSegmentAE(state, folderPath);
    NetworkSavePredictor(state, folderPath);
}

static inline void NetworkLoadAll(
    NetworkState* state,
    int featureDim,
    int segmentFlatDim,
    const std::string& folderPath)
{
    NetworkLoadFeatureAE(
        state, featureDim,
        FEATURE_AE_LATENT_DIM, folderPath);
    if (segmentFlatDim > 0)
    {
        NetworkLoadSegmentAE(
            state, segmentFlatDim,
            SEGMENT_AE_LATENT_DIM, folderPath);
    }
    NetworkLoadPredictor(
        state, FEATURE_AE_LATENT_DIM,
        SEGMENT_AE_LATENT_DIM, folderPath);
}

//---------------------------------------------------------
// time-budgeted training functions
//---------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;

static inline double ElapsedSeconds(
    Clock::time_point start)
{
    return std::chrono::duration<double>(
        Clock::now() - start).count();
}

// train feature AE for up to budgetSeconds
static inline void NetworkTrainFeatureAEForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->featuresAutoEncoder) return;
    if (!state->featureAEOptimizer) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->featureDim <= 0) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const auto start = Clock::now();

    state->featuresAutoEncoder->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor targetHost =
                torch::empty({ batchSize, featureDim });
            float* ptr = targetHost.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int ri =
                    rand() % db->legalStartFrames.size();
                const int idx =
                    db->legalStartFrames[ri];
                std::span<const float> row =
                    db->normalizedFeatures.row_view(idx);
                std::copy(
                    row.begin(), row.end(),
                    ptr + b * featureDim);
            }

            torch::Tensor target =
                targetHost.to(state->device);

            // denoising: add gaussian noise
            torch::Tensor noise =
                torch::randn({ batchSize, featureDim })
                    .to(state->device) * 0.05f;
            torch::Tensor input = target + noise;

            state->featureAEOptimizer->zero_grad();
            torch::Tensor output =
                state->featuresAutoEncoder->forward(input);
            torch::Tensor loss =
                torch::mse_loss(output, target);

            loss.backward();
            state->featureAEOptimizer->step();

            state->featureAELoss = loss.item<float>();
            state->featureAEIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Feature AE Training Error: %s", e.what());
        state->isTraining = false;
    }

    if (state->featureAEIterations % 300 == 0
        && state->featureAEIterations > 0)
    {
        TraceLog(LOG_INFO,
            "Feature AE Loss (iter %d): %.6f",
            state->featureAEIterations,
            state->featureAELoss);
    }
}

// train segment AE for up to budgetSeconds
static inline void NetworkTrainSegmentAEForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->segmentAutoEncoder) return;
    if (!state->segmentOptimizer) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int flatDim = db->poseGenSegmentFlatDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const auto start = Clock::now();

    state->segmentAutoEncoder->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor targetHost =
                torch::empty({ batchSize, flatDim });
            float* ptr = targetHost.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int ri =
                    rand() % db->legalStartFrames.size();
                const int globalStart =
                    db->legalStartFrames[ri];

                assert(
                    (globalStart + segFrames)
                    <= db->clipEndFrame[
                        FindClipForMotionFrame(
                            db, globalStart)]);

                float* dst = ptr + b * flatDim;
                const float* src =
                    db->normalizedPoseGenFeatures.data()
                    + globalStart * pgDim;
                memcpy(dst, src,
                    (size_t)segFrames * pgDim
                    * sizeof(float));
            }

            torch::Tensor target =
                targetHost.to(state->device);

            torch::Tensor noise =
                torch::randn({ batchSize, flatDim })
                    .to(state->device) * 0.05f;
            torch::Tensor input = target + noise;

            state->segmentOptimizer->zero_grad();
            torch::Tensor output =
                state->segmentAutoEncoder->forward(input);
            torch::Tensor loss =
                torch::mse_loss(output, target);

            loss.backward();
            state->segmentOptimizer->step();

            state->segmentAELoss = loss.item<float>();
            state->segmentAEIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Segment AE Training Error: %s", e.what());
        state->isTraining = false;
    }

    if (state->segmentAEIterations % 300 == 0
        && state->segmentAEIterations > 0)
    {
        TraceLog(LOG_INFO,
            "Segment AE Loss (iter %d): %.6f",
            state->segmentAEIterations,
            state->segmentAELoss);
    }
}

// train predictor for up to budgetSeconds
// both AEs are frozen (eval + NoGrad)
static inline void NetworkTrainPredictorForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->segmentLatentAveragePredictor) return;
    if (!state->predictorOptimizer) return;
    if (!state->featuresAutoEncoder) return;
    if (!state->segmentAutoEncoder) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;
    const auto start = Clock::now();

    state->featuresAutoEncoder->eval();
    state->segmentAutoEncoder->eval();
    state->segmentLatentAveragePredictor->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            // build both batches on CPU
            torch::Tensor featureBatch =
                torch::empty({ batchSize, featureDim });
            torch::Tensor segmentBatch =
                torch::empty({ batchSize, flatDim });
            float* fPtr = featureBatch.data_ptr<float>();
            float* sPtr = segmentBatch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int ri =
                    rand() % db->legalStartFrames.size();
                const int frame =
                    db->legalStartFrames[ri];

                // feature for this frame
                std::span<const float> fRow =
                    db->normalizedFeatures.row_view(frame);
                std::copy(
                    fRow.begin(), fRow.end(),
                    fPtr + b * featureDim);

                // poseGen segment starting at this frame
                const float* src =
                    db->normalizedPoseGenFeatures.data()
                    + frame * pgDim;
                memcpy(
                    sPtr + b * flatDim, src,
                    (size_t)segFrames * pgDim
                    * sizeof(float));
            }

            featureBatch =
                featureBatch.to(state->device);
            segmentBatch =
                segmentBatch.to(state->device);

            // encode with both AEs (frozen)
            torch::Tensor featureLatent;
            torch::Tensor segmentLatent;
            {
                torch::NoGradGuard noGrad;
                featureLatent = EncodeWithAE(
                    state->featuresAutoEncoder,
                    featureBatch);
                segmentLatent = EncodeWithAE(
                    state->segmentAutoEncoder,
                    segmentBatch);
            }

            // predict and optimize
            state->predictorOptimizer->zero_grad();
            torch::Tensor predicted =
                state->segmentLatentAveragePredictor
                    ->forward(featureLatent);
            torch::Tensor loss =
                torch::mse_loss(predicted, segmentLatent);

            loss.backward();
            state->predictorOptimizer->step();

            state->predictorLoss = loss.item<float>();
            state->predictorIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Predictor Training Error: %s", e.what());
        state->isTraining = false;
    }

    if (state->predictorIterations % 300 == 0
        && state->predictorIterations > 0)
    {
        TraceLog(LOG_INFO,
            "Predictor Loss (iter %d): %.6f",
            state->predictorIterations,
            state->predictorLoss);
    }
}

//---------------------------------------------------------
// unified training: all networks, time-budgeted
//---------------------------------------------------------

// returns the wall time spent training (seconds)
static inline double NetworkTrainAll(
    NetworkState* state,
    const AnimDatabase* db,
    double totalBudgetSeconds)
{
    if (!state->isTraining || !db->valid) return 0.0;
    if (!state->featuresAutoEncoder) return 0.0;
    if (!state->segmentAutoEncoder) return 0.0;
    if (db->legalStartFrames.empty()) return 0.0;

    const auto wallStart = Clock::now();

    // during headstart, predictor doesn't exist yet
    const bool headstartActive =
        state->trainingElapsedSeconds < HEADSTART_SECONDS
        && !state->segmentLatentAveragePredictor;

    double featureBudget;
    double segmentBudget;
    double predictorBudget;

    if (headstartActive)
    {
        // 80% segment AE, 20% feature AE
        featureBudget = totalBudgetSeconds * 0.20;
        segmentBudget = totalBudgetSeconds * 0.80;
        predictorBudget = 0.0;
    }
    else
    {
        // 5% feature AE, 20% segment AE, 75% predictor
        featureBudget = totalBudgetSeconds * 0.05;
        segmentBudget = totalBudgetSeconds * 0.20;
        predictorBudget = totalBudgetSeconds * 0.75;
    }

    NetworkTrainFeatureAEForTime(
        state, db, featureBudget);
    NetworkTrainSegmentAEForTime(
        state, db, segmentBudget);

    // auto-init predictor when headstart expires
    if (!state->segmentLatentAveragePredictor
        && state->trainingElapsedSeconds
            >= HEADSTART_SECONDS)
    {
        NetworkInitPredictor(
            state,
            FEATURE_AE_LATENT_DIM,
            SEGMENT_AE_LATENT_DIM);
        TraceLog(LOG_INFO,
            "Headstart done, predictor initialized.");
    }

    if (state->segmentLatentAveragePredictor
        && predictorBudget > 0.0)
    {
        NetworkTrainPredictorForTime(
            state, db, predictorBudget);
    }

    const double wallElapsed =
        ElapsedSeconds(wallStart);
    state->trainingElapsedSeconds += wallElapsed;
    state->timeSinceLastAutoSave += wallElapsed;

    return wallElapsed;
}

//---------------------------------------------------------
// init all networks for a fresh training session
//---------------------------------------------------------

static inline void NetworkInitAllForTraining(
    NetworkState* state,
    const AnimDatabase* db)
{
    if (!db->valid) return;

    NetworkInitFeatureAE(
        state, db->featureDim, FEATURE_AE_LATENT_DIM);

    if (db->poseGenSegmentFlatDim > 0)
    {
        NetworkInitSegmentAE(
            state, db->poseGenSegmentFlatDim,
            SEGMENT_AE_LATENT_DIM);
    }

    // predictor is NOT initialized here,
    // it auto-inits after the headstart period

    state->trainingElapsedSeconds = 0.0;
    state->timeSinceLastAutoSave = 0.0;
    state->isTraining = true;

    TraceLog(LOG_INFO,
        "All networks initialized. "
        "Predictor will start after %.0f s headstart.",
        HEADSTART_SECONDS);
}

//---------------------------------------------------------
// predict a full segment from raw motion features
//---------------------------------------------------------

// features -> normalize -> encode -> predict latent ->
// decode -> denormalize -> segment of poseGenFeatures.
// Returns false if any required network is null.
static inline bool NetworkPredictSegment(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    Array2D<float>& /*out*/ segment)
{
    if (!state->featuresAutoEncoder) return false;
    if (!state->segmentLatentAveragePredictor) return false;
    if (!state->segmentAutoEncoder) return false;
    if ((int)rawQuery.size() != db->featureDim) return false;

    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;

    segment.resize(segFrames, pgDim);

    // normalize query (same way as MotionMatchingSearch)
    torch::Tensor queryTensor =
        torch::empty({ 1, featureDim });
    float* qPtr = queryTensor.data_ptr<float>();
    for (int d = 0; d < featureDim; ++d)
    {
        const FeatureType ft = db->featureTypes[d];
        const int ti = static_cast<int>(ft);
        const float norm =
            (rawQuery[d] - db->featuresMean[d])
            / db->featureTypesStd[ti];
        const float w =
            db->featuresConfig.featureTypeWeights[ti];
        qPtr[d] = norm * w;
    }

    state->featuresAutoEncoder->eval();
    state->segmentLatentAveragePredictor->eval();
    state->segmentAutoEncoder->eval();

    queryTensor = queryTensor.to(state->device);

    torch::Tensor flat;
    {
        torch::NoGradGuard noGrad;

        // encode features -> feature latent
        torch::Tensor featureLatent = EncodeWithAE(
            state->featuresAutoEncoder, queryTensor);

        // predict -> segment latent
        torch::Tensor segmentLatent =
            state->segmentLatentAveragePredictor
                ->forward(featureLatent);

        // decode -> flat normalized poseGen
        flat = DecodeWithAE(
            state->segmentAutoEncoder, segmentLatent);
    }

    flat = flat.to(torch::kCPU);
    const float* fPtr = flat.data_ptr<float>();

    // denormalize and write into segment rows
    for (int f = 0; f < segFrames; ++f)
    {
        std::span<float> dst = segment.row_view(f);
        for (int d = 0; d < pgDim; ++d)
        {
            const float w =
                db->poseGenFeaturesWeight[d];
            const float denorm = (w > 1e-10f)
                ? (fPtr[f * pgDim + d] / w
                    * db->poseGenFeaturesStd[d]
                    + db->poseGenFeaturesMean[d])
                : db->poseGenFeaturesMean[d];
            dst[d] = denorm;
        }
    }

    return true;
}

//---------------------------------------------------------
// segment AE apply (for visual evaluation)
//---------------------------------------------------------

// Pass a segment through the segment autoencoder
// in-place (normalize -> encode -> decode -> denormalize).
// Used to visually evaluate reconstruction quality.
static inline void NetworkApplySegmentAE(
    NetworkState* networkState,
    const AnimDatabase* db,
    Array2D<float>* segment)
{
    if (!networkState->segmentAutoEncoder) return;

    const int segFrameCount = segment->rows();
    const int dim = segment->cols();
    const int flatDim = segFrameCount * dim;

    // normalize: (raw - mean) / std * weight
    torch::Tensor normalized =
        torch::empty({ 1, flatDim });
    float* nPtr = normalized.data_ptr<float>();
    for (int f = 0; f < segFrameCount; ++f)
    {
        std::span<const float> row =
            segment->row_view(f);
        for (int d = 0; d < dim; ++d)
        {
            nPtr[f * dim + d] =
                (row[d] - db->poseGenFeaturesMean[d])
                / db->poseGenFeaturesStd[d]
                * db->poseGenFeaturesWeight[d];
        }
    }

    assertEvenInRelease(
        db->poseGenSegmentFlatDim == flatDim);

    // forward pass (eval mode)
    normalized = normalized.to(networkState->device);
    networkState->segmentAutoEncoder->eval();
    torch::Tensor reconstructed =
        networkState->segmentAutoEncoder
            ->forward(normalized);
    reconstructed = reconstructed.to(torch::kCPU);

    // denormalize: raw = output / weight * std + mean
    const float* rPtr = reconstructed.data_ptr<float>();
    for (int f = 0; f < segFrameCount; ++f)
    {
        std::span<float> dst = segment->row_view(f);
        for (int d = 0; d < dim; ++d)
        {
            const float w =
                db->poseGenFeaturesWeight[d];
            const float denorm = (w > 1e-10f)
                ? (rPtr[f * dim + d] / w
                    * db->poseGenFeaturesStd[d]
                    + db->poseGenFeaturesMean[d])
                : db->poseGenFeaturesMean[d];
            dst[d] = denorm;
        }
    }
}
