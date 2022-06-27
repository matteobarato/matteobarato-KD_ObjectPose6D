import random

import torch
import torch.nn as nn
import torch_pruning as tp

class NASGatedConv:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.gate_layers = []
        self.apply_to_layer_types = nn.Conv2d
        self.max_recursion_level = 25
        self.gating_threshold = 1e-4
        self.ready_to_prune = []

    def factory_inject_layer(self, module):
        n_channels = module.out_channels
        gate = GatedConvolution(n_channels)
        return gate, nn.Sequential(module,gate)

    def apply_gates_to_model(self, model, apply_to_layer_types=[], _level=0): 
        if _level >= self.max_recursion_level: 
            print("|ERR| Maximum level of recursion reached ", self.max_recursion_level )
            return []
        if hasattr(model, 'gating_layers'): print("|WARNING| NAS already applied to model!")
        if self.verbose: print("-"*50, " Level ",_level)
        # if self.verbose: print("Step model", model)
        modules = list(model.named_children())
        self.gate_layers = []
        for i, module in enumerate(modules):
            module_name, module_obj = module
            module_children = list(module_obj.children())
        
            if any([isinstance(module_obj, x) for x in apply_to_layer_types]):
                gate, gated_module = self.factory_inject_layer(module_obj)
                if isinstance(model, nn.Sequential):
                    model[i] = gated_module
                else:
                    setattr(model, module_name, gated_module)
                setattr(gated_module, '_is_gated', True)
                if self.verbose: print("Added new Layer to module ", module_name)
                self.gate_layers.append((gate, module_name, model))
            elif len(module_children) and not hasattr(module_obj, '_is_gated'):
                self.gate_layers += self.apply_gates_to_model(module_obj, apply_to_layer_types=apply_to_layer_types, _level=_level+1)   
        return self.gate_layers

    def estimate_required_channels(self, use_mean = .8 ):
        gates_zeros_idxs = []
        with torch.no_grad():
            for i, layer in enumerate(self.gate_layers):
                gate, _, _ = layer
                w = gate.transformed_weight()
                threshold = torch.mean(w) if use_mean else self.gating_threshold
                are_zeros = torch.where(w < threshold, 0., 1.)
                zeros_idxs = torch.squeeze((are_zeros-1).nonzero())[:,0]
                gates_zeros_idxs.append(zeros_idxs)
                zeros = torch.squeeze(torch.count_nonzero(are_zeros -1 , dim=0)).item()
                if self.verbose: print(i, "Layer index ", i ," | Zeros", zeros , "/", w.shape[0], " | Mean", torch.mean(w, dim=0).item() )
        return gates_zeros_idxs

    def project_gates_on_model(self, use_mean=.8):
        gates_zeros_idxs = self.estimate_required_channels(use_mean=use_mean)
        with torch.no_grad():
            for i, item in enumerate(zip(self.gate_layers, gates_zeros_idxs)):
                layer, zeros_idxs = item
                gate, module_name, model = layer
                submodules = [x for x in model.named_children() if hasattr(x[1], "_is_gated")]
                for submodule in submodules:
                    submodule_name, submodule_obj = submodule
                    if isinstance(submodule_obj, nn.Conv2d) and submodule_name == module_name:
                        target_module = submodule_obj
                        for i in zeros_idxs:
                            target_module.weight[i,:,:] = 0
    
    def remove_gates_from_model(self):
        with torch.no_grad():
            for i, layer in enumerate(self.gate_layers):
                gate, module_name, model = layer
                submodules = [x for x in model.named_children() if hasattr(x[1], "_is_gated")]
                for submodule in submodules:
                    submodule_name, submodule_obj = submodule
                    if isinstance(submodule_obj[0], nn.Conv2d) and submodule_name == module_name:
                        target_module = submodule_obj[0]
                        print("Replacing ", module_name, " with ", target_module)
                        setattr(model, module_name, target_module)
                        self.ready_to_prune.append((module_name, target_module))

    def optimize(self, model, use_mean=0.8):
        gates_zeros_idxs = self.estimate_required_channels(use_mean=use_mean)
        self.project_gates_on_model(use_mean=use_mean)
        self.remove_gates_from_model()
        self.prune_model_channels(model, amount=0.5, pruning_idxs=gates_zeros_idxs)
        return gates_zeros_idxs

    def calc_improvement(self, base_model , gated_model):
        base_model_parameters = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        gated_model_parameters = sum(p.numel() for p in gated_model.parameters() if p.requires_grad)
        return gated_model_parameters / base_model_parameters

    def set_zeros(self, amount=0.5):
        with torch.no_grad():
            for i, layer in enumerate(self.gate_layers):
                gate, _, _ = layer
                w = gate.weight
                size = int(w.shape[0]*amount)
                idxs_samples = random.sample(range(0, w.shape[0]), size)
                for i in idxs_samples:
                    w[i,:,:] = 0

    def prune_model_channels(self, model, amount=0.4, pruning_idxs=None, ):
        for i, module in enumerate(self.ready_to_prune):
            _, module_obj = module
            DG = tp.DependencyGraph()
            DG.build_dependency(model, example_inputs=torch.randn(1,3,32,32))
            
            if pruning_idxs is None:
                strategy = tp.strategy.L1Strategy() 
                idxs = strategy(module_obj.weight, amount=0.4)
            else: idxs = pruning_idxs[i]

            pruning_plan = DG.get_pruning_plan( module_obj, tp.prune_conv, idxs=idxs )
            if self.verbose:
                print(pruning_plan)
            pruning_plan.exec()



# GatedConvolution Layer
class GatedConvolution(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, n_channels, channel_first=True):
        super().__init__()
        self.n_channels = n_channels
        self.out_channels = self.n_channels
        if channel_first:
            weight = torch.ones(self.n_channels, 1, 1)
        else:
            weight = torch.ones(1, 1, self.n_channels)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        # initialize weight
        nn.init.ones_(self.weight) # weight init
        self.weight_transformation = lambda x:x #torch.sigmoid

    def transformed_weight(self):
        with torch.no_grad():
            return self.weight_transformation(self.weight) 

    def forward(self, x):
        a = torch.mul(x, self.weight_transformation(self.weight))  # w times x + b
        return a

    def __repr__(self):
        return f'GatedConvolution(n_channels={self.n_channels})'