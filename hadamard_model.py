from typing import List
import numpy as np
import torch as t
from scipy.linalg import hadamard
import scipy.optimize as optimize
import random
import data
from dataclasses import dataclass 
import dill
import pathlib
from typing import Union
import NonLinSAE.data as scd

@dataclass
class HadamardConfig:
    
    # architecture parameters
    n_sparse: int
    n_dense: int
    n_hidden_layers: int = 0
    
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
    scalar_of_Identity: bool = True

class HadamardModel(t.nn.Module):
    def __init__(self, cfg: HadamardConfig, device='cpu', **optim_kwargs):
        super().__init__()
    
        self.cfg = cfg

        self.n_sparse = cfg.n_sparse
        self.n_dense = cfg.n_dense

        H = hadamard(cfg.n_sparse, dtype=np.float32)

        rand_rows = random.sample(range(cfg.n_sparse), cfg.n_dense)

        self.Win = H[rand_rows]
        self.Wout = self.Win.T / cfg.n_dense

        self.Win = t.tensor(self.Win).to(device)
        self.Wout = t.tensor(self.Wout).to(device)

        self.mean = t.nn.Parameter(t.tensor(0.0).to(device))
        self.Win = self.Win.to(device)
        self.Wout = self.Wout.to(device)
        self.mean = self.mean.to(device)

        self._final_loss = float('inf')

        self.scalar_of_Identity = cfg.scalar_of_Identity
        if self.scalar_of_Identity:
            self.Adiag = t.nn.Parameter(t.tensor(1.0))
        else:
            self.Adiag = t.nn.Parameter(t.ones(self.n_sparse, device=device))
        
        # self.optimize(df, device=device, **optim_kwargs)

    def set_mean(self, mean):
        """
        This is the part which actually conditions the epsilons properly
        """
        device = 'cpu' if self.Win is None else self.Win.device

        self.mean = t.nn.Parameter(t.tensor(mean)).to(device)

    def forward(self, x):
        preactivs = (x @ self.Win.T) @ self.Wout.T
        preactivs = self.Adiag.reshape(-1, 1) * (preactivs + self.mean)
        return t.nn.functional.relu(preactivs)
    
    def compute_loss(self, x, labels):
        device = self.Win.device

        x = x.to(device)
        labels = labels.to(device)
        y = self(x)
        return t.nn.functional.mse_loss(y, labels).item()

    def brute_force_optimize(self, x, labels, range_of_A: np.ndarray = np.linspace(0,3,100)):
        device = self.Win.device

        x = x.to(device)
        labels = labels.to(device)

        if self.scalar_of_Identity:
            with t.no_grad():
                losses = []
                for A in range_of_A:
                    self.set_A(float(A))
                    losses.append(self.compute_loss(x, labels))
                minA = range_of_A[np.argmin(losses)]
                self.set_A(float(minA))
                return minA, losses
        else: 
            with t.no_grad():
                losses = t.zeros(len(range_of_A), x.shape[1], device=device)
                for (i,A) in enumerate(range_of_A):
                    self.set_A(t.ones(self.Adiag.data.shape[0], device=device) * A)
                    squared_errors = (self(x) - labels)**2
                    component_wise_loss = t.mean(squared_errors, axis=0)
                    losses[i,:] = component_wise_loss

                (minvals, minindices) = t.min(losses, axis=0)
                self.set_A(t.tensor(range_of_A[minindices], dtype=t.float, device=device))
                
            return self.Adiag

    def set_A(self, A):
            device = self.Win.device

            if self.scalar_of_Identity:
                assert type(A) == float, "A must be a scalar"
                self.Adiag.data = t.tensor(A, device=device)
            else:
                self.Adiag.data = A.to(device)
            return self

    def ratio(self):
        return self.n_dense / self.n_sparse
    
    def final_loss(self):
        return self._final_loss
    
    def W_matrix(self):
        return self.Wout @ self.Win

    def optimize(self, data_factory: data.DataFactory, plot=False, logging=True, device='cpu') -> List[float]:
            
        optimizer = t.optim.Adam(self.parameters(), lr = self.cfg.lr, eps=1e-7)
        loss_function = t.nn.functional.mse_loss
        
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
                predictions = self(batch)
                
                # compute loss, on_loss, and loss_change
                
                loss = loss_function(predictions * importance_weights, labels * importance_weights)
                # on_loss = self.test_on_loss( # on_loss is loss for features that are turned on
                #     data_factory=data_factory,
                #     device=device, 
                #     batch_size=self.cfg.batch_size
                #     ) 
                
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
                    # on_losses.append(on_loss.item())
                
                if time_to_log and logging:
                    print(f'{loss_change=}')
                    print(f'{loss.item() = }')
                    print(f'{epoch=} {len(losses) = }')
                    if plot:
                        self.plot_live_loss(step_log, losses, steps = len(step_log))
                    
                    
            epoch += 1
                    
        self.losses = losses
        self.p_feat = data_factory.cfg.p_feature
        self._final_loss = losses[-1]
        return 
    
    def save(self, path: str):
        save_data = dict(cfg=self.cfg, state_dict = self.state_dict(), losses=self.losses, p_feat=self.p_feat)
        t.save(save_data, path, pickle_module=dill)
        path = pathlib.Path(path)

        with path.with_suffix('.modelinfo').open('wb') as f:
            dill.dump(dict(cfg=self.cfg,
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


def optimize_over_mean(n_s, n_d, p, inputs, labels, device='cpu') -> optimize.OptimizeResult:
    
    model = HadamardModel(n_s, n_d, p, 0.0, device=device)

    def loss_to_minimize(mu):
        model.set_mean(mu)
        model.brute_force_optimize(inputs, labels)
        return model.compute_loss(inputs, labels)

    return optimize.minimize_scalar(loss_to_minimize, bounds=(-1, 0), tol=.01)

def loss_estimator(hmodel: HadamardModel, batch_size = 10_000):
    datafactory = scd.DataFactory(scd.Config(p_feature=hmodel.p_feat, n_input=hmodel.n_sparse, domain=(0,1)))
    data, labels = datafactory.generate_data(batch_size=batch_size)
    return hmodel.compute_loss(data, labels)


