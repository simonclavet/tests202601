                                                            2. Residual
connections in flow model (second biggest impact)                             
                                                                              
                                                                              
                                                       Problem: The flow model
is a plain 3-hidden-layer MLP. For approximating a vector field (flow
matching), deeper networks with skip connections           learn faster and
produce smoother outputs.                                                     
                                                                              
                                                                              
                                                 Change: Replace the
Sequential with a custom Module that has residual blocks.                     
                                                                              
                                                                              
                                               In src/definitions.h, define a
FlowModel struct (using TORCH_MODULE pattern, which is the standard libtorch
way):                                       struct FlowModelImpl :
torch::nn::Module                                                             
                                                  {                           
                                                                              
                                                torch::nn::Linear
inputProj{nullptr};                                                           
                                                       torch::nn::Linear
res1a{nullptr}, res1b{nullptr};                                               
                                                       torch::nn::Linear
res2a{nullptr}, res2b{nullptr};                                               
                                                       torch::nn::Linear
res3a{nullptr}, res3b{nullptr};                                               
                                                       torch::nn::Linear
outputProj{nullptr};                                                          
                                                                              
                                                                              
                                                 FlowModelImpl(int inputDim,
int hiddenDim, int outputDim);                                                
                                             torch::Tensor forward(const
torch::Tensor& x);                                                            
                                         };                                   
                                                                              
                                   TORCH_MODULE(FlowModel);                   
                                                                              
                                                                              
                                                                              
                       Forward: input projection → GELU → 3 residual blocks →
output projection.                                                            
                  Each residual block: x = x + gelu(resNb(gelu(resNa(x)))).   
                                                                              
                                                                              
                                                                              
      Change NetworkState::latentFlowModel from torch::nn::Sequential to
FlowModel.                                                                    
                                                                              
                                                                              
Keep hidden dim at 256 for now (we can try 512 later if needed). The 3
residual blocks give us effectively 7 linear layers deep with good            
  gradient flow, vs the current 4 without skip connections.                   
                                                                              
                                                                              
                                                                    
Save/load: torch::save/torch::load work with any Module, so
NetworkSaveFlow/NetworkLoadFlow need only the type change. NetworkInitFlow,   
             NetworkTrainFlowForTime, and NetworkPredictSegmentFlow just use
->forward() which stays the same.                                             
                                                  
