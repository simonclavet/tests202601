#pragma once

#include "definitions.h"
#include <string>
#include <filesystem>
#include <unordered_map>
#include <chrono>

extern "C" void cuda_init_context();

constexpr int FEATURE_AE_LATENT_DIM = 16;
constexpr int SEGMENT_AE_LATENT_DIM = 128;
constexpr double HEADSTART_SECONDS = 5.0;
constexpr double AUTOSAVE_INTERVAL_SECONDS = 300.0;
constexpr int FLOW_INFERENCE_STEPS = 8;
constexpr double LOSS_LOG_INTERVAL_SECONDS = 5.0;

// training time budget weights (relative, get normalized per phase)
constexpr float BUDGET_WEIGHT_FEATURE_AE = 1.0f;
constexpr float BUDGET_WEIGHT_SEGMENT_AE = 2.0f;
constexpr float BUDGET_WEIGHT_PREDICTOR  = 1.0f;
constexpr float BUDGET_WEIGHT_E2E        = 5.0f;
constexpr float BUDGET_WEIGHT_FLOW       = 5.0f;

// switch between ReLU and GELU for all networks
#define USE_GELU
#ifdef USE_GELU
    #define NN_ACTIVATION() torch::nn::GELU()
    #define nn_activate(x) torch::gelu(x)
#else
    #define NN_ACTIVATION() NN_ACTIVATION()
    #define nn_activate(x) nn_activate(x)
#endif

// Flow matching mode: if defined, predict delta from average latent (condition on avgLatent)
// if undefined, predict segment latent directly (don't condition on avgLatent)
//#define DO_DELTA_FLOWMATCHING

// Flow matching parameterization: if defined, predict endpoint x1 directly
// if undefined, predict velocity field v = (x1 - xt) / (1 - t)
//#define FLOW_PREDICT_X1

constexpr float LOSS_SMOOTHING_FACTOR = 0.1f;  // lerp factor: new loss weight (0=fully smoothed, 1=no smoothing)

// pick a random legal frame using cluster-stratified sampling if available,
// otherwise fall back to plain uniform
static inline int SampleLegalFrame(const AnimDatabase* db)
{
    if (db->clusterCount <= 0)
        return db->legalStartFrames[RandomInt((int)db->legalStartFrames.size())];

    const int c = RandomInt(db->clusterCount);
    const std::vector<int>& frames = db->clusterFrames[c];
    return frames[RandomInt((int)frames.size())];
}

//---------------------------------------------------------
// encoder / decoder helpers
//---------------------------------------------------------

// Both AEs have the same layer layout:
// [0] Linear  [1] Activation  [2] Linear  [3] Activation  [4] Linear
// (encoder bottleneck)
// [5] Linear  [6] Activation  [7] Linear  [8] Activation  [9] Linear

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
    x = nn_activate(l0->forward(x));
    x = nn_activate(l2->forward(x));
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
    x = nn_activate(l5->forward(x));
    x = nn_activate(l7->forward(x));
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
        NN_ACTIVATION(),
        torch::nn::Linear(128, 64),
        NN_ACTIVATION(),
        torch::nn::Linear(64, latentDim),
        torch::nn::Linear(latentDim, 64),
        NN_ACTIVATION(),
        torch::nn::Linear(64, 128),
        NN_ACTIVATION(),
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
        NN_ACTIVATION(),
        torch::nn::Linear(512, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, latentDim),
        torch::nn::Linear(latentDim, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, 512),
        NN_ACTIVATION(),
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
            torch::nn::Linear(featureLatentDim, 256),
            NN_ACTIVATION(),
            torch::nn::Linear(256, 256),
            NN_ACTIVATION(),
            torch::nn::Linear(256, 256),
            NN_ACTIVATION(),
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
// latent space statistics (for debugging/analysis)
//---------------------------------------------------------

static inline void NetworkAnalyzeLatentSpaceStatistics(
    NetworkState* state,
    const AnimDatabase* db)
{
    if (!state->featuresAutoEncoder) return;
    if (!state->segmentAutoEncoder) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    TraceLog(LOG_INFO, "=== Analyzing Latent Space Statistics ===");

    const int sampleCount = std::min(1000, (int)db->legalStartFrames.size());
    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;

    state->featuresAutoEncoder->eval();
    state->segmentAutoEncoder->eval();

    // accumulate latent values for statistics
    std::vector<float> featureLatentValues;
    std::vector<float> segmentLatentValues;
    featureLatentValues.reserve(sampleCount * FEATURE_AE_LATENT_DIM);
    segmentLatentValues.reserve(sampleCount * SEGMENT_AE_LATENT_DIM);

    try
    {
        for (int i = 0; i < sampleCount; ++i)
        {
            const int ri = RandomInt((int)db->legalStartFrames.size());
            const int frame = db->legalStartFrames[ri];

            // build feature batch
            torch::Tensor featureTensor = torch::empty({ 1, featureDim });
            float* fPtr = featureTensor.data_ptr<float>();
            std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
            std::copy(fRow.begin(), fRow.end(), fPtr);

            // build segment batch
            torch::Tensor segmentTensor = torch::empty({ 1, flatDim });
            float* sPtr = segmentTensor.data_ptr<float>();
            const float* src = db->normalizedPoseGenFeatures.data() + frame * pgDim;
            memcpy(sPtr, src, (size_t)segFrames * pgDim * sizeof(float));

            featureTensor = featureTensor.to(state->device);
            segmentTensor = segmentTensor.to(state->device);

            torch::Tensor featureLatent;
            torch::Tensor segmentLatent;
            {
                torch::NoGradGuard noGrad;
                featureLatent = EncodeWithAE(state->featuresAutoEncoder, featureTensor);
                segmentLatent = EncodeWithAE(state->segmentAutoEncoder, segmentTensor);
            }

            // copy to CPU and accumulate
            featureLatent = featureLatent.to(torch::kCPU);
            segmentLatent = segmentLatent.to(torch::kCPU);

            const float* fLatentPtr = featureLatent.data_ptr<float>();
            const float* sLatentPtr = segmentLatent.data_ptr<float>();

            for (int d = 0; d < FEATURE_AE_LATENT_DIM; ++d)
            {
                featureLatentValues.push_back(fLatentPtr[d]);
            }

            for (int d = 0; d < SEGMENT_AE_LATENT_DIM; ++d)
            {
                segmentLatentValues.push_back(sLatentPtr[d]);
            }
        }

        // compute mean
        float featureMean = 0.0f;
        for (float v : featureLatentValues) featureMean += v;
        featureMean /= (float)featureLatentValues.size();

        float segmentMean = 0.0f;
        for (float v : segmentLatentValues) segmentMean += v;
        segmentMean /= (float)segmentLatentValues.size();

        // compute variance and std
        float featureVariance = 0.0f;
        for (float v : featureLatentValues)
        {
            const float diff = v - featureMean;
            featureVariance += diff * diff;
        }
        featureVariance /= (float)featureLatentValues.size();
        const float featureStd = std::sqrt(featureVariance);

        float segmentVariance = 0.0f;
        for (float v : segmentLatentValues)
        {
            const float diff = v - segmentMean;
            segmentVariance += diff * diff;
        }
        segmentVariance /= (float)segmentLatentValues.size();
        const float segmentStd = std::sqrt(segmentVariance);

        // compute min/max
        const float featureMin = *std::min_element(
            featureLatentValues.begin(), featureLatentValues.end());
        const float featureMax = *std::max_element(
            featureLatentValues.begin(), featureLatentValues.end());

        const float segmentMin = *std::min_element(
            segmentLatentValues.begin(), segmentLatentValues.end());
        const float segmentMax = *std::max_element(
            segmentLatentValues.begin(), segmentLatentValues.end());

        TraceLog(LOG_INFO, "Feature AE Latent Space (dim=%d, samples=%d):",
            FEATURE_AE_LATENT_DIM, sampleCount);
        TraceLog(LOG_INFO, "  Mean: %.6f", featureMean);
        TraceLog(LOG_INFO, "  Std:  %.6f", featureStd);
        TraceLog(LOG_INFO, "  Min:  %.6f", featureMin);
        TraceLog(LOG_INFO, "  Max:  %.6f", featureMax);

        TraceLog(LOG_INFO, "Segment AE Latent Space (dim=%d, samples=%d):",
            SEGMENT_AE_LATENT_DIM, sampleCount);
        TraceLog(LOG_INFO, "  Mean: %.6f", segmentMean);
        TraceLog(LOG_INFO, "  Std:  %.6f", segmentStd);
        TraceLog(LOG_INFO, "  Min:  %.6f", segmentMin);
        TraceLog(LOG_INFO, "  Max:  %.6f", segmentMax);

        TraceLog(LOG_INFO, "=== Analysis Complete ===");
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Latent space analysis error: %s", e.what());
    }
}


//---------------------------------------------------------
// FlowModel implementation
//---------------------------------------------------------

inline FlowModelImpl::FlowModelImpl(int inputDim, int hiddenDim, int outputDim)
    : inputProj(register_module("inputProj", torch::nn::Linear(inputDim, hiddenDim))),
    res1a(register_module("res1a", torch::nn::Linear(hiddenDim, hiddenDim))),
    res1b(register_module("res1b", torch::nn::Linear(hiddenDim, hiddenDim))),
    res2a(register_module("res2a", torch::nn::Linear(hiddenDim, hiddenDim))),
    res2b(register_module("res2b", torch::nn::Linear(hiddenDim, hiddenDim))),
    res3a(register_module("res3a", torch::nn::Linear(hiddenDim, hiddenDim))),
    res3b(register_module("res3b", torch::nn::Linear(hiddenDim, hiddenDim))),
    outputProj(register_module("outputProj", torch::nn::Linear(hiddenDim, outputDim)))
{
}

inline torch::Tensor FlowModelImpl::forward(const torch::Tensor& x)
{
    // Input projection: inputDim -> hiddenDim
    torch::Tensor h = nn_activate(inputProj->forward(x));

    // Residual block 1: x = x + gelu(res1b(gelu(res1a(x))))
    h = h + nn_activate(res1b->forward(nn_activate(res1a->forward(h))));

    // Residual block 2
    h = h + nn_activate(res2b->forward(nn_activate(res2a->forward(h))));

    // Residual block 3
    h = h + nn_activate(res3b->forward(nn_activate(res3a->forward(h))));

    // Output projection: hiddenDim -> outputDim
    return outputProj->forward(h);
}

//---------------------------------------------------------
// latent flow matching model
//---------------------------------------------------------

static inline void NetworkInitFlow(
    NetworkState* state,
    int featureLatentDim,
    int segmentLatentDim)
{
    NetworkEnsureDevice(state);

    // input: condition + xt + time
    // condition = featureLatent + avgLatent (if DO_DELTA_FLOWMATCHING) or just featureLatent
#ifdef DO_DELTA_FLOWMATCHING
    const int inputDim = featureLatentDim + segmentLatentDim + segmentLatentDim + 1;
#else
    const int inputDim = featureLatentDim + segmentLatentDim + 1;
#endif

    state->latentFlowModel = FlowModel(inputDim, 256, segmentLatentDim);
    //state->latentFlowModel = torch::nn::Sequential(
    //    torch::nn::Linear(inputDim, 256),
    //    NN_ACTIVATION(),
    //    torch::nn::Linear(256, 256),
    //    NN_ACTIVATION(),
    //    torch::nn::Linear(256, 256),
    //    NN_ACTIVATION(),
    //    torch::nn::Linear(256, segmentLatentDim));

    state->latentFlowModel->to(state->device);

    state->flowOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->latentFlowModel->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->flowLoss = 0.0f;
    state->flowIterations = 0;

    TraceLog(LOG_INFO,
        "Latent Flow Model initialized: %d -> 256 (3 residual blocks) -> %d",
        inputDim, segmentLatentDim);
}

static inline void NetworkSaveFlow(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->latentFlowModel) return;

    const std::string path =
        folderPath + "/latentFlowModel.bin";
    try
    {
        torch::save(state->latentFlowModel, path);
        TraceLog(LOG_INFO,
            "Saved Latent Flow Model to: %s",
            path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to save Latent Flow Model: %s",
            e.what());
    }
}

static inline void NetworkLoadFlow(
    NetworkState* state,
    int featureLatentDim,
    int segmentLatentDim,
    const std::string& folderPath)
{
    const std::string path =
        folderPath + "/latentFlowModel.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitFlow(
            state, featureLatentDim, segmentLatentDim);
        torch::load(state->latentFlowModel, path);
        state->latentFlowModel->to(state->device);
        TraceLog(LOG_INFO,
            "Loaded Latent Flow Model from: %s",
            path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to load Latent Flow Model: %s",
            e.what());
    }
}

//---------------------------------------------------------
// loss history CSV save/load
//---------------------------------------------------------

static inline void LossHistorySave(
    const NetworkState* state,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/loss_history.csv";
    FILE* f = fopen(path.c_str(), "w");
    if (!f)
    {
        TraceLog(LOG_ERROR, "Failed to write loss history: %s", path.c_str());
        return;
    }

    fprintf(f, "time,featureAE,segmentAE,predictor,flow,e2e\n");
    const int n = (int)state->lossHistoryTime.size();
    for (int i = 0; i < n; ++i)
    {
        fprintf(f, "%.1f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
            state->lossHistoryTime[i],
            i < (int)state->featureAELossHistory.size() ? state->featureAELossHistory[i] : 0.0f,
            i < (int)state->segmentAELossHistory.size() ? state->segmentAELossHistory[i] : 0.0f,
            i < (int)state->predictorLossHistory.size() ? state->predictorLossHistory[i] : 0.0f,
            i < (int)state->flowLossHistory.size() ? state->flowLossHistory[i] : 0.0f,
            i < (int)state->e2eLossHistory.size() ? state->e2eLossHistory[i] : 0.0f);
    }

    fclose(f);
    TraceLog(LOG_INFO, "Saved loss history (%d points) to: %s", n, path.c_str());
}

static inline void LossHistoryLoad(
    NetworkState* state,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/loss_history.csv";
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return;

    state->lossHistoryTime.clear();
    state->featureAELossHistory.clear();
    state->segmentAELossHistory.clear();
    state->predictorLossHistory.clear();
    state->flowLossHistory.clear();
    state->e2eLossHistory.clear();

    // skip header
    char line[512];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return; }

    // read rows — supports both old (5 col) and new (6 col) formats
    while (fgets(line, sizeof(line), f))
    {
        float t = 0, fae = 0, sae = 0, pred = 0, flow = 0, e2e = 0;
        const int cols = sscanf(line, "%f,%f,%f,%f,%f,%f", &t, &fae, &sae, &pred, &flow, &e2e);
        if (cols < 5) break;
        state->lossHistoryTime.push_back(t);
        state->featureAELossHistory.push_back(fae);
        state->segmentAELossHistory.push_back(sae);
        state->predictorLossHistory.push_back(pred);
        state->flowLossHistory.push_back(flow);
        state->e2eLossHistory.push_back(cols >= 6 ? e2e : 0.0f);
    }

    fclose(f);
    if (!state->lossHistoryTime.empty())
    {
        TraceLog(LOG_INFO, "Loaded loss history (%d points) from: %s",
            (int)state->lossHistoryTime.size(), path.c_str());
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
    NetworkSaveFlow(state, folderPath);
    LossHistorySave(state, folderPath);
}

static inline void NetworkLoadAll(
    NetworkState* state,
    int featureDim,
    int segmentFlatDim,
    const std::string& folderPath)
{
    NetworkLoadFeatureAE(state, featureDim, FEATURE_AE_LATENT_DIM, folderPath);
    if (segmentFlatDim > 0)
    {
        NetworkLoadSegmentAE(state, segmentFlatDim, SEGMENT_AE_LATENT_DIM, folderPath);
    }
    NetworkLoadPredictor(state, FEATURE_AE_LATENT_DIM, SEGMENT_AE_LATENT_DIM, folderPath);
    NetworkLoadFlow(state, FEATURE_AE_LATENT_DIM, SEGMENT_AE_LATENT_DIM, folderPath);
    LossHistoryLoad(state, folderPath);
}

//---------------------------------------------------------
// time-budgeted training functions
//---------------------------------------------------------


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
    const Clock::time_point start = Clock::now();

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
                const int idx = SampleLegalFrame(db);
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
            state->featureAELossSmoothed = state->featureAELossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                + state->featureAELoss * LOSS_SMOOTHING_FACTOR;
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
    const Clock::time_point start = Clock::now();

    state->segmentAutoEncoder->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor targetHost = torch::empty({ batchSize, flatDim });
            float* ptr = targetHost.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int globalStart = SampleLegalFrame(db);

                assert(
                    (globalStart + segFrames)
                    <= db->clipEndFrame[
                        FindClipForMotionFrame(
                            db, globalStart)]);

                float* dst = ptr + b * flatDim;
                const float* src = db->normalizedPoseGenFeatures.data() + globalStart * pgDim;
                memcpy(dst, src,
                    (size_t)segFrames * pgDim * sizeof(float));
            }

            torch::Tensor target = targetHost.to(state->device);

            torch::Tensor noise =
                torch::randn({ batchSize, flatDim })
                    .to(state->device) * 0.05f;
            torch::Tensor input = target + noise;

            state->segmentOptimizer->zero_grad();
            torch::Tensor output =state->segmentAutoEncoder->forward(input);
            torch::Tensor loss = torch::mse_loss(output, target);

            loss.backward();
            state->segmentOptimizer->step();

            state->segmentAELoss = loss.item<float>();
            state->segmentAELossSmoothed = state->segmentAELossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                + state->segmentAELoss * LOSS_SMOOTHING_FACTOR;
            state->segmentAEIterations++;

            //if (state->segmentAEIterations % 1000 == 0)
            //{
            //    // analyze latent space statistics after a bit of AE training
            //    NetworkAnalyzeLatentSpaceStatistics(state, db);
            //}
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Segment AE Training Error: %s", e.what());
        state->isTraining = false;
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
    const Clock::time_point start = Clock::now();

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
                const int frame = SampleLegalFrame(db);

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
            state->predictorLossSmoothed = state->predictorLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                + state->predictorLoss * LOSS_SMOOTHING_FACTOR;
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
// latent flow matching training
// all upstream networks (AEs + predictor) are frozen
//---------------------------------------------------------

static inline void NetworkTrainFlowForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->latentFlowModel) return;
    if (!state->flowOptimizer) return;
    if (!state->featuresAutoEncoder) return;
    if (!state->segmentAutoEncoder) return;
    if (!state->segmentLatentAveragePredictor) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;
    const int segLatentDim = SEGMENT_AE_LATENT_DIM;
    const Clock::time_point start = Clock::now();

    // freeze everything upstream — we only train the
    // flow model, all the rest is locked in place
    state->featuresAutoEncoder->eval();
    state->segmentAutoEncoder->eval();
    state->segmentLatentAveragePredictor->eval();
    state->latentFlowModel->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            // grab random frames and build feature + segment batches on CPU,
            // same pattern as the other training loops
            torch::Tensor featureBatch = torch::empty({ batchSize, featureDim });
            torch::Tensor segmentBatch = torch::empty({ batchSize, flatDim });
            float* fPtr = featureBatch.data_ptr<float>();
            float* sPtr = segmentBatch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
                std::copy(fRow.begin(), fRow.end(), fPtr + b * featureDim);

                const float* src = db->normalizedPoseGenFeatures.data() + frame * pgDim;
                memcpy(sPtr + b * flatDim, src, (size_t)segFrames * pgDim * sizeof(float));
            }

            featureBatch = featureBatch.to(state->device);
            segmentBatch = segmentBatch.to(state->device);

            // run everything through the frozen networks to get latent representations
            torch::Tensor featureLatent;
            torch::Tensor trueSegLatent;
            torch::Tensor avgLatent;
            {
                torch::NoGradGuard noGrad;
                // features -> 16d latent
                featureLatent = EncodeWithAE(state->featuresAutoEncoder, featureBatch);
                // full pose segment -> 128d latent
                trueSegLatent = EncodeWithAE(state->segmentAutoEncoder, segmentBatch);
                // what the predictor thinks the average pose should be
                avgLatent = state->segmentLatentAveragePredictor->forward(featureLatent);
            }

#ifdef DO_DELTA_FLOWMATCHING
            // Delta flow matching: predict delta from average
            torch::Tensor delta = trueSegLatent - avgLatent;
            torch::Tensor x1 = delta;
#else
            // Direct flow matching: predict segment latent directly
            torch::Tensor x1 = trueSegLatent;
#endif



            // each sample gets its own random time in [0,1],
            // the "how far along the flow are we?" parameter
            torch::Tensor t = torch::rand(
                { batchSize, 1 }, torch::TensorOptions().device(state->device));

            // pure gaussian noise, the starting point of the flow at t=0
            torch::Tensor x0 = torch::randn(
                { batchSize, segLatentDim }, torch::TensorOptions().device(state->device));

            // linear interpolation between noise and the true target —
            // this is where we are at time t along the straight-line flow
            torch::Tensor xt = (1.0f - t) * x0 + t * x1;

#ifdef FLOW_PREDICT_X1
            // Endpoint prediction: network predicts x1 directly
            torch::Tensor target = x1;
#else
            // Velocity prediction: network predicts v = (x1 - xt) / (1 - t)
            // avoid division by zero at t=1
            torch::Tensor velocity = (x1 - xt) / (1.0f - t + 1e-8f);
            torch::Tensor target = velocity;
#endif

#ifdef DO_DELTA_FLOWMATCHING
            // the network sees: conditioning info (what motion features we want +
            // what the average looks like), the noisy sample xt, and time t
            torch::Tensor flowInput = torch::cat(
                { featureLatent, avgLatent, xt, t }, /*dim=*/1);
#else
            // the network sees: conditioning info (what motion features we want),
            // the noisy sample xt, and time t (no average conditioning)
            torch::Tensor flowInput = torch::cat(
                { featureLatent, xt, t }, /*dim=*/1);
#endif

            // endpoint prediction: the network tries to guess where the flow ends up
            state->flowOptimizer->zero_grad();
            torch::Tensor predicted = state->latentFlowModel->forward(flowInput);
            torch::Tensor loss = torch::mse_loss(predicted, target);
            loss.backward();
            state->flowOptimizer->step();

            state->flowLoss = loss.item<float>();
            state->flowLossSmoothed = state->flowLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                + state->flowLoss * LOSS_SMOOTHING_FACTOR;
            state->flowIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Flow Training Error: %s", e.what());
        state->isTraining = false;
    }

    if (state->flowIterations % 300 == 0
        && state->flowIterations > 0)
    {
        TraceLog(LOG_INFO,
            "Flow Loss (iter %d): %.6f",
            state->flowIterations,
            state->flowLoss);
    }
}

//---------------------------------------------------------
// end-to-end training: features → encode → predict → decode → segment
// gradients flow through all 3 networks simultaneously
//---------------------------------------------------------

static inline void NetworkTrainEndToEndForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->featuresAutoEncoder) return;
    if (!state->segmentAutoEncoder) return;
    if (!state->segmentLatentAveragePredictor) return;
    if (!state->featureAEOptimizer) return;
    if (!state->segmentOptimizer) return;
    if (!state->predictorOptimizer) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;
    const Clock::time_point start = Clock::now();

    // everything trains together — no frozen networks here
    state->featuresAutoEncoder->train();
    state->segmentAutoEncoder->train();
    state->segmentLatentAveragePredictor->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor featureBatch = torch::empty({ batchSize, featureDim });
            torch::Tensor segmentBatch = torch::empty({ batchSize, flatDim });
            float* fPtr = featureBatch.data_ptr<float>();
            float* sPtr = segmentBatch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
                std::copy(fRow.begin(), fRow.end(), fPtr + b * featureDim);

                const float* src = db->normalizedPoseGenFeatures.data() + frame * pgDim;
                memcpy(sPtr + b * flatDim, src, (size_t)segFrames * pgDim * sizeof(float));
            }

            featureBatch = featureBatch.to(state->device);
            segmentBatch = segmentBatch.to(state->device);

            // the full chain: encode features, predict segment latent, decode to segment
            // no NoGradGuard — gradients flow through everything
            torch::Tensor featureLatent = EncodeWithAE(state->featuresAutoEncoder, featureBatch);
            torch::Tensor predictedSegLatent = state->segmentLatentAveragePredictor->forward(featureLatent);
            torch::Tensor reconstructed = DecodeWithAE(state->segmentAutoEncoder, predictedSegLatent);

            // loss in output space: the network chain must reconstruct the true segment
            // this is in normalized bone-weighted space, so important joints matter more
            state->featureAEOptimizer->zero_grad();
            state->segmentOptimizer->zero_grad();
            state->predictorOptimizer->zero_grad();

            torch::Tensor loss = torch::mse_loss(reconstructed, segmentBatch);
            loss.backward();

            state->featureAEOptimizer->step();
            state->segmentOptimizer->step();
            state->predictorOptimizer->step();

            const float lossVal = loss.item<float>();
            state->e2eLoss = lossVal;
            state->e2eLossSmoothed = state->e2eIterations == 0
                ? lossVal
                : state->e2eLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR) + lossVal * LOSS_SMOOTHING_FACTOR;
            state->e2eIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "E2E Training Error: %s", e.what());
        state->isTraining = false;
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

    const Clock::time_point wallStart = Clock::now();

    const double elapsed = state->trainingElapsedSeconds;
    const bool hasPredictor =
        static_cast<bool>(
            state->segmentLatentAveragePredictor);
    const bool hasFlow =
        static_cast<bool>(state->latentFlowModel);

    // 2-phase training schedule using relative weights:
    // phase 1 (0-5s):  AEs only (headstart)
    // phase 2 (5s+):   everything
    const float wFeature = BUDGET_WEIGHT_FEATURE_AE;
    const float wSegment = BUDGET_WEIGHT_SEGMENT_AE;
    const float wPredictor = hasPredictor ? BUDGET_WEIGHT_PREDICTOR : 0.0f;
    const float wE2e = hasPredictor ? BUDGET_WEIGHT_E2E : 0.0f;
    const float wFlow = hasFlow ? BUDGET_WEIGHT_FLOW : 0.0f;
    const float totalWeight = wFeature + wSegment + wPredictor + wE2e + wFlow;

    const double featureBudget = totalBudgetSeconds * wFeature / totalWeight;
    const double segmentBudget = totalBudgetSeconds * wSegment / totalWeight;
    const double predictorBudget = totalBudgetSeconds * wPredictor / totalWeight;
    const double e2eBudget = totalBudgetSeconds * wE2e / totalWeight;
    const double flowBudget = totalBudgetSeconds * wFlow / totalWeight;

    NetworkTrainFeatureAEForTime(state, db, featureBudget);
    NetworkTrainSegmentAEForTime(state, db, segmentBudget);

    // auto-init predictor and flow after AE headstart
    if (!state->segmentLatentAveragePredictor && elapsed >= HEADSTART_SECONDS)
    {
        NetworkInitPredictor(state, FEATURE_AE_LATENT_DIM, SEGMENT_AE_LATENT_DIM);
        TraceLog(LOG_INFO, "AE headstart done, predictor initialized.");
    }

    if (!state->latentFlowModel && elapsed >= HEADSTART_SECONDS)
    {
        NetworkInitFlow(state, FEATURE_AE_LATENT_DIM, SEGMENT_AE_LATENT_DIM);
        TraceLog(LOG_INFO, "AE headstart done, flow model initialized.");
    }

    if (state->segmentLatentAveragePredictor && predictorBudget > 0.0)
        NetworkTrainPredictorForTime(state, db, predictorBudget);

    if (state->segmentLatentAveragePredictor && e2eBudget > 0.0)
        NetworkTrainEndToEndForTime(state, db, e2eBudget);

    if (state->latentFlowModel && flowBudget > 0.0)
        NetworkTrainFlowForTime(state, db, flowBudget);

    const double wallElapsed = ElapsedSeconds(wallStart);
    state->trainingElapsedSeconds += wallElapsed;
    state->timeSinceLastAutoSave += wallElapsed;
    state->timeSinceLastLossLog += wallElapsed;

    // log all losses at a fixed time interval so curves are directly comparable
    // use exponentially smoothed losses, record 0.0 for networks with < 100 iterations
    if (state->timeSinceLastLossLog >= LOSS_LOG_INTERVAL_SECONDS)
    {
        state->timeSinceLastLossLog = 0.0;
        state->lossHistoryTime.push_back((float)state->trainingElapsedSeconds);
        state->featureAELossHistory.push_back(state->featureAEIterations >= 100 ? state->featureAELossSmoothed : 0.0f);
        state->segmentAELossHistory.push_back(state->segmentAEIterations >= 100 ? state->segmentAELossSmoothed : 0.0f);
        state->predictorLossHistory.push_back(state->predictorIterations >= 100 ? state->predictorLossSmoothed : 0.0f);
        state->flowLossHistory.push_back(state->flowIterations >= 100 ? state->flowLossSmoothed : 0.0f);
        state->e2eLossHistory.push_back(state->e2eIterations >= 100 ? state->e2eLossSmoothed : 0.0f);
    }    return wallElapsed;
}

//---------------------------------------------------------
// init all networks for a fresh training session
//---------------------------------------------------------

static inline void NetworkInitAllForTraining(
    NetworkState* state,
    AnimDatabase* db)
{
    if (!db->valid) return;

    // cluster features for stratified training sampling (lazy, only once)
    if (db->clusterCount == 0)
        AnimDatabaseClusterFeatures(db);

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
        "Predictor + Flow at %.0fs headstart.",
        HEADSTART_SECONDS);
}

// reset all networks to null (e.g. after database rebuild
// which invalidates feature dimensions)
static inline void NetworkResetAll(NetworkState* state)
{
    state->isTraining = false;

    state->featuresAutoEncoder = nullptr;
    state->featureAEOptimizer = nullptr;
    state->featureAELoss = 0.0f;
    state->featureAEIterations = 0;

    state->segmentAutoEncoder = nullptr;
    state->segmentOptimizer = nullptr;
    state->segmentAELoss = 0.0f;
    state->segmentAEIterations = 0;

    state->segmentLatentAveragePredictor = nullptr;
    state->predictorOptimizer = nullptr;
    state->predictorLoss = 0.0f;
    state->predictorIterations = 0;

    state->latentFlowModel = nullptr;
    state->flowOptimizer = nullptr;
    state->flowLoss = 0.0f;
    state->flowIterations = 0;

    state->e2eLoss = 0.0f;
    state->e2eLossSmoothed = 0.0f;
    state->e2eIterations = 0;

    state->lossHistoryTime.clear();
    state->featureAELossHistory.clear();
    state->segmentAELossHistory.clear();
    state->predictorLossHistory.clear();
    state->flowLossHistory.clear();
    state->e2eLossHistory.clear();
    state->timeSinceLastLossLog = 0.0;

    state->trainingElapsedSeconds = 0.0;
    state->timeSinceLastAutoSave = 0.0;

    TraceLog(LOG_INFO, "All networks reset.");
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
// flow matching inference
// features -> encode -> predict average -> sample flow
// -> decode -> denormalize -> segment
//---------------------------------------------------------

static inline bool NetworkPredictSegmentFlow(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    Array2D<float>& /*out*/ segment)
{
    // need all 4 networks for this
    if (!state->featuresAutoEncoder) return false;
    if (!state->segmentLatentAveragePredictor) return false;
    if (!state->segmentAutoEncoder) return false;
    if (!state->latentFlowModel) return false;
    if ((int)rawQuery.size() != db->featureDim) return false;

    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int segLatentDim = SEGMENT_AE_LATENT_DIM;

    segment.resize(segFrames, pgDim);

    // normalize the raw motion matching query the same way we do for search and training
    torch::Tensor queryTensor = torch::empty({ 1, featureDim });
    float* qPtr = queryTensor.data_ptr<float>();
    for (int d = 0; d < featureDim; ++d)
    {
        const FeatureType ft = db->featureTypes[d];
        const int ti = static_cast<int>(ft);
        const float norm = (rawQuery[d] - db->featuresMean[d]) / db->featureTypesStd[ti];
        const float w = db->featuresConfig.featureTypeWeights[ti];
        qPtr[d] = norm * w;
    }

    // everything is inference-only, no gradients needed
    state->featuresAutoEncoder->eval();
    state->segmentLatentAveragePredictor->eval();
    state->latentFlowModel->eval();
    state->segmentAutoEncoder->eval();

    queryTensor = queryTensor.to(state->device);

    torch::Tensor flat;
    {
        torch::NoGradGuard noGrad;

        // compress features down to a 16d latent
        torch::Tensor featureLatent = EncodeWithAE(state->featuresAutoEncoder, queryTensor);

        // the predictor gives us the "average" pose for these features — good but boring
        torch::Tensor avgLatent = state->segmentLatentAveragePredictor->forward(featureLatent);

        // this is where the magic happens: we start from pure noise and let the flow
        // model gradually shape it into a plausible result

        // mode seek from zero: the mode is better than random noise - crisp, hopefully
        torch::Tensor x0 = torch::zeros(
            { 1, segLatentDim }, torch::TensorOptions().device(state->device));
        //torch::Tensor x0 = torch::randn(
        //    { 1, segLatentDim }, torch::TensorOptions().device(state->device));

        // euler integration with endpoint prediction: at each step the network predicts
        // where x1 should be, and we re-interpolate from our original noise x0 towards
        // that prediction. more steps = smoother trajectory through latent space
        torch::Tensor x = x0;
        torch::Tensor x1Hat;
        for (int step = 0; step < FLOW_INFERENCE_STEPS; ++step)
        {
            const float t = (float)step / FLOW_INFERENCE_STEPS;
            torch::Tensor tTensor = torch::full(
                { 1, 1 }, t, torch::TensorOptions().device(state->device));

#ifdef DO_DELTA_FLOWMATCHING
            // condition on: what features we want, what the average looks like,
            // where we currently are in the flow, and how far along we are (t)
            torch::Tensor flowInput = torch::cat(
                { featureLatent, avgLatent, x, tTensor }, /*dim=*/1);
#else
            // condition on: what features we want, where we currently are in the flow,
            // and how far along we are (t) - no average conditioning
            torch::Tensor flowInput = torch::cat(
                { featureLatent, x, tTensor }, /*dim=*/1);
#endif

#ifdef FLOW_PREDICT_X1
            // the network predicts the endpoint x1
            x1Hat = state->latentFlowModel->forward(flowInput);

            // jump to the interpolated position between our starting noise
            // and the predicted endpoint at the next timestep
            const float nextT = (float)(step + 1) / FLOW_INFERENCE_STEPS;
            x = (1.0f - nextT) * x0 + nextT * x1Hat;
#else
            // the network predicts the velocity field v
            torch::Tensor v = state->latentFlowModel->forward(flowInput);

            // integrate: x += v * dt
            const float dt = 1.0f / FLOW_INFERENCE_STEPS;
            x = x + v * dt;

            // cache the last prediction as x1Hat for final result
            if (step == FLOW_INFERENCE_STEPS - 1)
            {
                x1Hat = x;
            }
#endif
        }

#ifdef DO_DELTA_FLOWMATCHING
        // the flow gave us a delta — add it back to the average
        // to get a diverse but plausible segment latent
        torch::Tensor segmentLatent = avgLatent + x1Hat;
#else
        // the flow gave us the segment latent directly
        torch::Tensor segmentLatent = x1Hat;
#endif
        // decode back from 128d latent to the full flat pose segment
        flat = DecodeWithAE(state->segmentAutoEncoder, segmentLatent);
    }

    // bring it back to CPU for the denorm loop
    flat = flat.to(torch::kCPU);
    const float* fPtr = flat.data_ptr<float>();

    // undo the normalization so we get real-world pose values back
    for (int f = 0; f < segFrames; ++f)
    {
        std::span<float> dst = segment.row_view(f);
        for (int d = 0; d < pgDim; ++d)
        {
            const float w = db->poseGenFeaturesWeight[d];
            const float denorm = (w > 1e-10f)
                ? (fPtr[f * pgDim + d] / w * db->poseGenFeaturesStd[d] + db->poseGenFeaturesMean[d])
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
