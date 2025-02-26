import torch as t
import torch.nn as nn
from torch import Tensor
from torch.autograd.functional import jacobian
import os

import einops

from models.gpt import GPT
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput

from config.gpt.training import options
from config.sae.models import sae_options



from data.tokenizers import ASCIITokenizer, TikTokenTokenizer

from models.sae import SparseAutoencoder
from typing import Callable
from datasets import Dataset
from data.dataloaders import TrainingDataLoader
TensorFunction = Callable[[Tensor], Tensor]

def all_ig_attributions(model: SparsifiedGPT, ds: TrainingDataLoader, nbatches: int = 32):
    """
    Returns a dict of all consecutive integrated gradient attributions for a model
    :param model: SparsifiedGPT model
    :param ds: Dataloader
    :param nbatches: How many batches of data to aggregate into attributions
    :return: a dict where key 'i-i+1' (e.g. '0-1' or '1-2' is the 2d tensor of attributions between layers)
    """
    layers = model.gpt.config.n_layer
    attributions = {}
    for i in range(layers - 1):
        attributions[f'{i}-{i+1}'] = ig_attributions(model, i, i+1, ds, nbatches)
        ds.reset()
    return attributions

def ig_attributions(model: SparsifiedGPT, layer0: int, layer1: int, ds: TrainingDataLoader, nbatches: int=32):
    """
    Computes integrated gradient attribution for a model between two layers
    :param model: SparsifiedGPT model
    :param layer0: index of layer of source sae
    :param layer1: index of layer of target sae
    :param ds: Dataloader
    :param nbatches: How many batches of data to aggregate into attributions
    :return: a tensor of shape (source_size, target_size) where the dimensions are the sizes of the hidden layers of the source and target sae respectively
    """
    assert layer0 < layer1
    assert layer0 >= 0
    assert layer1 <= model.gpt.config.n_layer
    sae0 = model.saes[f'{layer0}']
    sae1 = model.saes[f'{layer1}']
    

    #define function that goes from feature magnitudes in layer0 to feature magnitudes in layer1
    #Q: Is this good form? I need it to make my Sequential object below
    class Sae0Decode(nn.Module):
        def forward(self, x):
            return sae0.decode(x)
        
    class Sae1Encode(nn.Module):
        def forward(self, x):
            return sae1.encode(x)
    
    #construct function from Sae0 to Sae1
    forward_list = [Sae0Decode()] + [model.gpt.transformer.h[i] for i in range(layer0, layer1)] + [Sae1Encode()]
    forward = t.nn.Sequential(*forward_list)

    source_size, _ = sae0.W_dec.shape
    target_size, _ = sae1.W_dec.shape
    
    attributions = t.zeros((source_size, target_size), device = model.gpt.config.device)
    for _ in range(nbatches):

        input, _ = ds.next_batch(model.gpt.config.device) #get batch of inputs 
        output = model.forward(input, targets=None, is_eval=True)
        feature_magnitudes0 = output.feature_magnitudes[layer0] #feature magnitudes at source layer (batchsize, seqlen, source_size)
        
        #loop over fms in target sae, and find attributions from all fms in source layer.
        #aggregate these by root mean square, following https://www.lesswrong.com/posts/Rv6ba3CMhZGZzNH7x/interpretability-integrated-gradients-is-a-decent
        for fm_i in range(target_size):
            y_i = t.zeros(target_size, device = model.gpt.config.device)
            y_i[fm_i] = 1
            #Once again I make a custom class to make the sequential work
            class dot_i(nn.Module):
                def forward(self, x):
                    return einops.einsum(x, y_i, '... target_size, target_size -> ...').sum()
            
            eval_i = nn.Sequential(forward, dot_i())
        
            gradient = integrate_gradient(feature_magnitudes0, None, eval_i) #(batch seq source_size)
            #sum over batch and position
            #Q: Does this make sense? It might be incorrect to sum over position
            attributions[:,fm_i] = attributions[:,fm_i] + (gradient **2).sum(dim=[0,1]) 
                
    attributions = t.sqrt(attributions)
    return attributions

def integrate_gradient(x: Tensor, x_i: Tensor | None, fun: TensorFunction, base:int= 0, steps:int=10):
    """
    Approximates int_C d/dx_i y(z) dz where C is a linear path from base to x
    :param x: End of path. In practice it is a tensor of feature magnitudes generated by the data, 
        or a one hot encoding of a feature magnitude.
        Q: Should I only use certain x to compute attributions for a given x_i?.
        Shape (batchsize, seq_len, encoding)
    :param x_i: Direction of partial derivative. It is often a one hot encoding of a given feature magnitude. 
        If none, function computes and returns the attributions for all feature magnitudes 
        Shape (encoding)
    :param fun: Scalar valued function with signature (batchsize, seq_len, encoding) -> []. 
        In practice, it is the value of the jth feature magnitude after passing the feature magnitudes in the input layer through the model to the target layer
    :param base: Start of path. Default to 0
    :param steps: Number of steps to use to approximate the integral
    :return: Tensor of shape x.shape if x_i = None else shape (batchsize, seq_len)

    Q: I do a lot to make this process memory light, because otherwise it crashes my pod. 
        Some of it is probably redundant/bad form

    """
    
    #compute a linear path from base to x
    if base == 0:
        base = t.zeros_like(x)
    path = t.linspace(0, 1, steps)
    steplength = t.linalg.norm(x - base, dim = -1, keepdim = True)/steps

    integral = 0
    for alpha in path:
        point = (alpha*x + (1-alpha)*base).detach() #Find point on path
        point.requires_grad_()
        y = fun(point)  #compute gradient of y wrt x, scale by length of a step
        
        y.backward(retain_graph=False) #we only need to do a backward pass once, this saves memory

        g = point.grad
        with t.no_grad(): #once again, no_grad is required to keep memory usage down
            if x_i == None:
                integral += g.detach() * steplength
                
            else:
                integral += (g.detach() * x_i).sum(dim=-1) * steplength


    return integral


if __name__ == "__main__":
   #This code loads a model and data, and computes the attrbution from layers 0 to 1
    c_name = 'standardx8.shakespeare_64x4'
    name = 'standard.shakespeare_64x4'
    data_dir = 'data/shakespeare'
    batch_size = 32
    config = sae_options[c_name]

    model = SparsifiedGPT(config)
    model_path = os.path.join("checkpoints", name)
    model = model.load(model_path, device=config.device)
    model.to(config.device) #for some reason, when I do this the model starts on the cpu, and I have to move it


    #copied from Peter's training code
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = t.device(f"cuda:{ddp_local_rank}")

        assert t.cuda.is_available()
        t.cuda.set_device(device)
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = config.device

    dataloader = TrainingDataLoader(
        dir_path=data_dir,
        B= batch_size,
        T=model.config.block_size,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
    )
    x,_ = dataloader.next_batch(device)

    
    layer0 = 0
    layer1 = 1

    attributions = all_ig_attributions(model, dataloader, nbatches=1)

   
    
 




   






