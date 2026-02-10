#pragma once

#include "definitions.h"

extern "C" void cuda_init_context();

static inline void NetworkInitAutoEncoder(NetworkState* state, int inputDim, int latentDim)
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
    
    state->isTraining = true;
    state->currentLoss = 0.0f;
    state->iterations = 0;

    TraceLog(LOG_INFO, "Features AutoEncoder initialized: %d -> %d -> %d", inputDim, latentDim, inputDim);
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