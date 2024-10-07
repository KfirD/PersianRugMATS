from dataclasses import dataclass
import pathlib
import dill as pickle
import dill # not sure why someone else make import dill as pickle ...
import data
import torch as t
from jaxtyping import Float
import numpy as np
from typing import List, Callable, Tuple, Union
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    from IPython.display import clear_output # type: ignore
except ImportError:
    def clear_output(*args, **kwargs):
        pass
import json
import os
import uuid

@dataclass
class Config:
    
    # architecture parameters
    n_sparse: int
    n_dense: int
    n_hidden_layers: int = 0
    
    init_layer_bias: bool = False
    hidden_layers_bias: bool = False
    final_layer_bias: bool = False
    
    init_layer_act_func: str = "identity"
    hidden_layers_act_func: str = "identity"
    final_layer_act_func: str = "relu"
    
    tie_dec_enc_weights: bool = False
    
    # training parameters
    data_size: int = 10_000
    batch_size: int = 1024
    
    max_epochs: int = 10
    min_epochs: int = 0
    loss_window: int = 1000

    lr: float = 1e-5
    convergence_tolerance: float = 1e-1
    
    update_times: int = 10
    residual: bool = False
    importance: int = 0

class Model(t.nn.Module):
    
    # ============================================================
    # Model Architecture
    # ============================================================
    
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.layers = nn.ModuleList()
        
        self.p_feat = None
        self.losses = None
                
        # encoding layer (n_sparse -> n_dense)
        self.initial_layer = nn.Linear(cfg.n_sparse, cfg. n_dense, bias = cfg.init_layer_bias)
        #self.layers.append(initial_layer)
        
        for _ in range(cfg.n_hidden_layers):
            self.layers.append(nn.Linear(cfg.n_dense, cfg.n_dense, bias = cfg.hidden_layers_bias))
            
        # decoding layer (n_dense -> n_sparse)
        self.final_layer = nn.Linear(cfg.n_dense, cfg.n_sparse, bias = cfg.final_layer_bias)
        
        if cfg.tie_dec_enc_weights:
            self.final_layer.weight.data = self.initial_layer.weight.data.T
        #self.layers.append(final_layer)
        
            
    def forward(self, x: Float[Tensor, "batch n_sparse"]): 
        cache = {}
        y = x.clone()
        
        
        act_funcs = {"identity": lambda x: x, "relu": F.relu, "tanh": F.tanh, "leaky_relu": F.leaky_relu, "gelu":F.gelu}
        
        x = self.initial_layer(x)
        x = act_funcs[self.cfg.init_layer_act_func](x)
                
        cache["activations"] = [x]
        
        for l in self.layers:
            if self.cfg.residual:
                x = act_funcs[self.cfg.hidden_layers_act_func](l(x)) + x
            else:
                x = act_funcs[self.cfg.hidden_layers_act_func](l(x))
            cache["activations"].append(x)            
            
        x = self.final_layer(x)
        cache["activations"].append(x)
        x = act_funcs[self.cfg.final_layer_act_func](x)
        
        
        return x, cache
        
    # ============================================================
    # Model Training
    # ============================================================
    
    def plot_live_loss(self, step_log, losses, sf_losses, steps):
            clear_output(wait=True)
            _, ax = plt.subplots(ncols=2, figsize=(10, 3))
            
            # graphs
                        
            ax[0].plot(step_log, losses)
            ax[0].set_title(f"loss")
            ax[0].set_xlabel("log steps")
            ax[0].set_ylabel("log loss")
            ax[0].set_xlim(0,steps)
            
            ax[0].text(0.5, -0.2, self.metadata, ha="center", va="top", transform=ax[0].transAxes, fontsize=12, color="gray")

            
            ax[1].plot(step_log, sf_losses)
            ax[1].set_title("On loss")
            ax[1].set_xlabel("log steps")
            ax[1].set_ylabel("log on loss")
            ax[1].set_xlim(0,steps)
            
            plt.show()

    def compute_loss(self, data_factory: data.DataFactory, batch_size = 1000, device = "cpu"):
        data = data_factory.generate_data_loader(
            data_size = batch_size, # single batch for now
            batch_size = batch_size,
            device = device
        )
        
        loss_function = F.mse_loss
        importance_weights = t.pow(t.arange(self.cfg.n_sparse) + 1, -self.cfg.importance / 2).reshape((1, -1))
        losses = []
        
        for batch, labels in data:
            batch = batch.to(device)
            labels = labels.to(device)
            importance_weights = importance_weights.to(device)
            predictions, _ = self(batch)
            
            losses.append(loss_function(predictions * importance_weights, labels * importance_weights))            
        
        return np.mean(losses[0].item())

    
    def optimize(self, data_factory: data.DataFactory, plot=False, logging=True, device='cpu') -> List[float]:
            
        optimizer = t.optim.Adam([param for param in self.parameters() if param.requires_grad], lr = self.cfg.lr, eps=1e-7)
        loss_function = F.mse_loss
        
        losses = []
        on_losses = []
        step_log = []
        
        step = 0
        epoch = 0
        tot_steps = self.cfg.max_epochs*self.cfg.data_size//self.cfg.batch_size + 1

        loss_change = float("inf")
        import time
        stime = time.time()

        sum_t = 0.0
        sum_sq_t = 0.0
        tot_n = 0
        
        importance_weights = t.pow(t.arange(self.cfg.n_sparse) + 1, -self.cfg.importance / 2).reshape((1, -1))
        
        while (loss_change > self.cfg.convergence_tolerance or epoch < self.cfg.min_epochs) and epoch < self.cfg.max_epochs:
            # deltat = time.time() - stime
            # sum_t += deltat
            # sum_sq_t += deltat**2
            # tot_n += 1
            # print(f'{tot_n}: t = {sum_t/tot_n:.2f} +/- {(sum_sq_t/tot_n - (sum_t/tot_n)**2)**.5:.2f}')
            
            stime = time.time()
            if len(losses) > 2 * self.cfg.loss_window:
                loss_change = np.log10(np.mean(losses[-2 * self.cfg.loss_window:-self.cfg.loss_window])) - np.log10(np.mean(losses[-self.cfg.loss_window:]))
            
            # create data
            data = data_factory.generate_data_loader(
                data_size=self.cfg.data_size, 
                batch_size=self.cfg.batch_size,
                device=device
            )

            if plot:
                iterator = iter(data)
            else:
                iterator = iter(data)

            for batch, labels in iterator:
                            
                # compute outputs
                batch = batch.to(device)
                labels = labels.to(device)
                importance_weights = importance_weights.to(device)
                predictions, _ = self(batch)
                
                # compute loss, on_loss, and loss_change
                
                loss = loss_function(predictions * importance_weights, labels * importance_weights)
                # on_loss = self.test_on_loss( # on_loss is loss for features that are turned on
                #     data_factory=data_factory,
                #     device=device, 
                #     batch_size=self.cfg.batch_size
                #     ) 
                on_loss = t.tensor(1.0)
                
                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step += 1
                # print if needed
                update_times = min(tot_steps, self.cfg.update_times)
                time_to_log = step % (tot_steps//update_times) == 0 or step == 1
                if time_to_log or not plot:
                    step_log.append(step)
                    losses.append(loss.item())
                    on_losses.append(on_loss.item())
                
                if time_to_log and logging:
                    print(f'{loss_change=}')
                    print(f'{loss.item() = }')
                    print(f'{epoch=} {len(losses) = }')
                    if plot:
                        self.plot_live_loss(step_log, losses, on_losses, steps = len(step_log))
                    
                    
            epoch += 1
                    
        self.losses = losses
        self.p_feat = data_factory.cfg.p_feature
        return 
    
    # ============================================================
    # Helper functions
    # ============================================================
    
    def ratio(self):
        return self.cfg.n_dense/self.cfg.n_sparse
    
    def final_loss(self):
        return self.losses[-1]
    
    def W_matrix(self):
        return self.final_layer.weight.data @ self.initial_layer.weight.data
    
    
    def save(self, path: str):
        save_data = dict(cfg=self.cfg, state_dict = self.state_dict(), losses=self.losses, p_feat=self.p_feat)
        t.save(save_data, path, pickle_module=dill)
        path = pathlib.Path(path)

        with path.with_suffix('.modelinfo').open('wb') as f:
            pickle.dump(dict(cfg=self.cfg,
                             losses=self.losses,
                             p_feat=self.p_feat,
                             modelpth=path), f)
    
    @classmethod
    def load(cls, path: Union[str, pathlib.Path], map_location="cpu"):
        model_data = t.load(path, map_location=map_location)
        model = cls(model_data["cfg"])
        model.load_state_dict(model_data["state_dict"])
        model.losses = model_data["losses"]
        model.p_feat = model_data["p_feat"]
        return model
    

    
    # ============================================================
    # Analytics
    # ============================================================
        
    def test_random_input(self, data_factory: data.DataFactory):
        input, target = data_factory.generate_data(batch_size = 1)
        pred, _ = self(input)
        print(f'{input=}')
        print(f'{target=}')
        print(f'{pred=}')
        
    def test_first_feat(self):
        input, label = self.generate_first_order_data(batch_size = 1)
        pred, id_pred = self(input)
        print(f'{input[0,0]=}')
        print(f'{label[0,0]=}')
        print(f'{pred[0,0]=}')
     
    # returns average loss of only the single feature turned on
    def test_on_loss(self, data_factory: data.DataFactory, device, batch_size = 1):
        input, target = data_factory.generate_first_order_data(batch_size = batch_size)
        input = input.to(device)
        target = target.to(device)
        prediction, _ = self(input)
        
        num_eyes = (batch_size//self.cfg.n_sparse) + 1
        target = t.diagonal(target.reshape(self.cfg.n_sparse,self.cfg.n_sparse, num_eyes), dim1=0,dim2=1)
        prediction = t.diagonal(prediction.reshape(self.cfg.n_sparse,self.cfg.n_sparse, num_eyes), dim1=0,dim2=1)
        
        loss = F.mse_loss(target,prediction)
        return loss
        
            
def main():
    pass


if __name__ == "__main__": main()

