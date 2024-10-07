import data 
import torch as t
import numpy as np
import random
from matplotlib import pyplot as plt
from torch import Tensor
from jaxtyping import Float 
from synthetic_model import optimal_A, gen_matrices
from tqdm import tqdm

try:
    from IPython.display import clear_output # type: ignore
except ImportError:
    def clear_output(*args, **kwargs):
        pass

class DiagAModel (t.nn.Module):
    def __init__(self, Win: t.tensor, Wout: t.tensor, scalar_of_Identity: bool= False, p_feat: float = None):
        super(DiagAModel, self).__init__()
        self.register_buffer('Win', Win)
        self.register_buffer('Wout', Wout)
        self.relu = t.nn.ReLU()
        self.scalar_of_Identity = scalar_of_Identity
        if scalar_of_Identity:
            self.Adiag = t.nn.Parameter(t.rand(1))
        else:
            self.Adiag = t.nn.Parameter(t.rand(Wout.shape[0]))

        self.ratio = Win.shape[0] / Win.shape[1]
        self.n_sparse = Win.shape[1]
        self.n_dense  = Win.shape[0]
        self.p_feat = p_feat
    
    def forward(self, x: Float[Tensor, "batch n_sparse"]): # type: ignore
        y = self.Win @ x.T
        y = self.Adiag.reshape(-1,1) * (self.Wout @ y)
        return self.relu(y.T)
    
    def preactivations(self, x: Float[Tensor, "batch n_sparse"]): # type: ignore
        y = self.Win @ x.T
        y = self.Adiag.reshape(-1,1) * (self.Wout @ y)
        return y.T
    
    def compute_loss(self, df:data.DataFactory, batch_size = 10000):
        x, labels = df.generate_data(batch_size=batch_size)

        device = self.Win.device

        x = x.to(device)
        labels = labels.to(device)
        y = self(x)
        return t.nn.functional.mse_loss(y, labels).item()
    
    # def optimize(self, df: data.DataFactory, learning_rate: float = 2*1e-3, n_iter: int = 10000, batch_size: int = 1000, title=''):
    #     optimizer = t.optim.Adam(self.parameters(), lr=learning_rate)
    #     loss_function = t.nn.functional.mse_loss
    #     losses = []
    #     A_means = []
    #     step_log = []
    #     epoch = 0 
    #     loss_change = float("inf")
    #     while (loss_change > self.cfg.convergence_tolerance or epoch < self.cfg.min_epochs) and epoch < self.cfg.max_epoch:
    #         optimizer.zero_grad()
    #         x, labels = df.generate_data(batch_size=batch_size)
    #         y = self(x)
    #         loss = loss_function(y, labels)
    #         loss.backward()
    #         optimizer.step()
    #         if i % 100 == 0:
    #             step_log.append(i)
    #             # print(f"""iteration: {i}:   , 
    #             #       Loss: {loss.item()},
    #             #       mean A: {t.mean(self.Adiag).item()},
    #             #       variance A: {t.var(self.Adiag).item()},
    #             #       min A: {t.min(self.Adiag).item()},
    #             #       max A: {t.max(self.Adiag).item()}
    #             #       """)
    #             losses.append(loss.item())
    #             A_means.append(t.mean(self.Adiag).item())
    #             self.plot_live_loss_and_mean_A(step_log, losses, A_means, n_iter, title=title)
    #     return losses

    def optimize(self, df: data.DataFactory, learning_rate: float = 1e-3, n_iter: int = 1000, batch_size: int = 100, title=''):
        optimizer = t.optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = t.nn.functional.mse_loss
        losses = []
        A_means = []
        step_log = []

        device = self.Win.device

        for i in range(n_iter):
            optimizer.zero_grad()
            x, labels = df.generate_data(batch_size=batch_size)
            x = x.to(device)
            labels = labels.to(device)

            y = self(x)
            loss = loss_function(y, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                step_log.append(i)
                # print(f"""iteration: {i}:   , 
                #       Loss: {loss.item()},
                #       mean A: {t.mean(self.Adiag).item()},
                #       variance A: {t.var(self.Adiag).item()},
                #       min A: {t.min(self.Adiag).item()},
                #       max A: {t.max(self.Adiag).item()}
                #       """)
                losses.append(loss.item())
                A_means.append(t.mean(self.Adiag).item())
                self.plot_live_loss_and_mean_A(step_log, losses, A_means, n_iter, title=title)
        return losses
    
    def brute_force_optimize(self, df: data.DataFactory, batch_size:int = 10000, range_of_A: np.ndarray = np.linspace(0,3,100)):
        device = self.Win.device

        if self.scalar_of_Identity:
            with t.no_grad():
                losses = []
                for A in range_of_A:
                    self.set_A(float(A))
                    losses.append(self.compute_loss(df, batch_size))
                minA = range_of_A[np.argmin(losses)]
                self.set_A(float(minA))
                return minA, losses
        else: 
            with t.no_grad():
                x, labels = df.generate_data(batch_size=batch_size)
                x = x.to(device)
                labels = labels.to(device)

                losses = t.zeros(len(range_of_A), x.shape[1], device=device)
                for (i,A) in enumerate(range_of_A):
                    self.set_A(t.ones(self.Adiag.data.shape[0], device=device) * A)
                    squared_errors = (self(x) - labels)**2
                    component_wise_loss = t.mean(squared_errors, axis=0)
                    losses[i,:] = component_wise_loss

                (minvals, minindices) = t.min(losses, axis=0)
                self.set_A(t.tensor(range_of_A[minindices], dtype=t.float, device=device))
                
            return self.Adiag
            
    def plot_live_loss_and_mean_A(self, step_log, losses, A_means, steps, title=''):
            clear_output(wait=True)
            _, ax = plt.subplots(ncols=2, figsize=(10, 3))
            
            # graphs
                        
            ax[0].plot(step_log, losses)
            ax[0].set_title(f"loss")
            ax[0].set_xlabel("steps")
            ax[0].set_ylabel("loss")
            ax[0].set_xlim(0,steps)
            
            ax[0].text(0.5, -0.2, None, ha="center", va="top", transform=ax[0].transAxes, fontsize=12, color="gray")

            
            ax[1].plot(step_log, A_means)
            ax[1].set_title("mean(A)" + title)
            ax[1].set_xlabel("steps")
            ax[1].set_ylabel("mean(A)")
            ax[1].set_xlim(0,steps)
            
            plt.show()

    def set_A(self, A):
            device = self.Win.device

            if self.scalar_of_Identity:
                assert type(A) == float, "A must be a scalar"
                self.Adiag.data = t.tensor(A, device=device)
            else:
                self.Adiag.data = A.to(device)
            return self

    def analytically_set_A(self, feature_probability):
        assert not self.scalar_of_Identity, "Only works per row now"

        As = []

        for i in range(self.W.shape[0]):
            mean = (feature_probability/3) * (self.W[i].sum() - self.W[i,i]).item()
            var = (feature_probability / 3 - feature_probability**2 / 4) * (self.W[i].square().sum() - self.W[i,i].square()).item()


            As.append(optimal_A(mean, var**.5, feature_probability)[0])
        
        print(self.Adiag.data.shape, t.tensor(As, device=self.Adiag.device).float().shape)
        self.set_A(t.tensor(As, device=self.Adiag.device).float())


def get_optimal_model(n_dense: int, n_sparse: int, p: float, device="cpu") -> DiagAModel:
    dataconfig = data.Config(p, n_sparse, (0, 1))
    datafactory = data.DataFactory(dataconfig)
    k = int(n_dense/2)
    Win, Wout = gen_matrices(n_sparse, n_dense, k, return_torch_tensor=True)
    model = DiagAModel(Win, Wout)
    current_loss = model.compute_loss(datafactory)
    k = k-1
    while k>0:
        prev_loss = current_loss
        Win, Wout = gen_matrices(n_sparse, n_dense, k, return_torch_tensor=True)

        model = DiagAModel(Win=Win, Wout=Wout)
        model = model.to(device)
        model.brute_force_optimize(datafactory)
        current_loss = model.compute_loss(datafactory)
        if current_loss < prev_loss:
            #print(f"{k+1} to {k}: Loss decreased from {prev_loss} to {current_loss}")
            k -= 1
        else:
            break
    return model





    

    
