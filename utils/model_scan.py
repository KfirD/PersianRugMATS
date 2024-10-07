from model import *
import data
import itertools
import pathlib
from tqdm import tqdm
import pandas as pd
import uuid
from NonLinSAE.model import Model
from NonLinSAE.utils import synthetic_model as sm
from NonLinSAE import hadamard_model as hm
from NonLinSAE.utils import model_measurement as mm
from typing import Literal, Iterable, Union, Optional

"""
this is a file to create, save, and load models as data points for a 
loss vs (ratio, p_feat) graph
"""
# Todo: 
# 1. All of this code should be written with an iterator over model configs. 

# create/save models
def train_model(n_dense, n_sparse, p, activation = "relu", final_layer_bias = False, 
                tie_dec_enc_weights: bool = False, device="cpu",
                train_win=True):        
    # training variables for both models
    dataconfig = data.Config(p, n_sparse, (0, 1))
    datafactory = data.DataFactory(dataconfig)

    cfg = Config(    
        # architecture parameters
        n_sparse=n_sparse, n_dense=n_dense, 
        final_layer_bias=final_layer_bias, tie_dec_enc_weights=tie_dec_enc_weights,
        
        # training parameters
        data_size = 30_000, batch_size = 1024, max_epochs = 1500, lr = (3e-3)/np.sqrt(n_sparse), update_times = 3000, convergence_tolerance = 0,
        loss_window=200
    )

    cfg.final_layer_act_func = activation

    model = Model(cfg)
    model = model.to(device)
    if not train_win:
        model.initial_layer.weight.requires_grad_(False)

        if cfg.init_layer_bias:
            model.initial_layer.bias.requires_grad_(False)

    
    model.optimize(datafactory, device=device, plot=False, logging=False)        
    return model


def train_multiple_models(n_sparses, ratios, p_feats, experiment_name: str, 
                          activations = ["relu"], final_layer_biases = [False],
                          tie_dec_enc_weights_list = [False], device="cpu",
                          overwrite=False, DiagAModel_flag: bool = False,
                          overwrite_experiments=False, train_win=True):
    
    path = pathlib.Path(f"saved_models/{experiment_name}/")
    print(path, path.exists())
    if not overwrite: assert not path.exists()
    path.mkdir(parents=True, exist_ok=True)
        
    total = len(n_sparses) * len(ratios) * len(p_feats) * len(activations) * len(final_layer_biases) * len(tie_dec_enc_weights_list) 
    pbar = tqdm(itertools.product(n_sparses, ratios, p_feats, activations, final_layer_biases, tie_dec_enc_weights_list), total=total)
    pbar.set_description(f'ratio: N/A, p_feat: N/A, non_linearity: N/A, final_layer_biases: N/A, tie_dec_enc_weights: N/A,  loss: N/A')
    for model_idx, (n_sparse, ratio, p_feat, activation, final_layer_bias, tie_dec_enc_weights) in enumerate(pbar):
        if (path / str(model_idx)).exists() and not overwrite_experiments:
            continue
        model = train_model(max(int(n_sparse * ratio),1), n_sparse, p_feat, activation=activation, final_layer_bias=final_layer_bias, tie_dec_enc_weights=tie_dec_enc_weights, device=device, DiagAModel_flag=DiagAModel_flag, train_win=train_win)
        # models.append(model)
        pbar.set_description(f'ratio: {model.ratio():.2f}, p: {p_feat:.4f}, non_linearity: {activation}, bias: {final_layer_bias}, tie_weights: {tie_dec_enc_weights}, loss: {model.final_loss():.4f}')
        model.save(path / str(model_idx))
    # return models



def train_hadamard_model(
        n_dense, 
        n_sparse, 
        p,  
        device="cpu", 
        n_tries: int = 1,
        batch_size: int = 1024, 
        max_epochs: int = 100, 
        loss_window: int = 50,
        update_times: int = 3000,
        data_size: int = 10_000,
        convergence_tolerance: float = 0.0)->hm.HadamardModel:

    dataconfig = data.Config(p, n_sparse, (0, 1))
    datafactory = data.DataFactory(dataconfig)

    cfg = hm.HadamardConfig(
        n_sparse = n_sparse,
        n_dense = n_dense,
        data_size = data_size, 
        batch_size = batch_size, 
        max_epochs = max_epochs,
        lr = (3e-1)/np.sqrt(n_sparse),
        update_times = update_times, 
        convergence_tolerance = convergence_tolerance,
        loss_window=loss_window)
    
    models = []
    for n in range(n_tries):
        model = hm.HadamardModel(cfg, device=device)
        model = model.to(device)
        model.optimize(datafactory, device=device, plot=False, logging=False)        
        models.append(model)

    losses =  [model.losses[-1] for model in models]
    model = sorted(models, key=lambda x: x.losses[-1])[0]
    return model, losses


def train_multiple_hadamard_models(
        n_sparses: Iterable[int], 
        ratios: Iterable[float],
        p_feats: Iterable[float], 
        experiment_name: str,
        num_samples: int = 40, # deprecated for now
        overwrite: bool = False,
        overwrite_experiments: bool = False,
        device: str = "cpu",
        n_tries:int = 1,
        batch_size: int = 1024, 
        max_epochs: int = 100, 
        loss_window: int = 50,
        update_times: int = 3000,
        data_size: int = 10_000
        ):
        
    path = pathlib.Path(f"saved_models/{experiment_name}/")
    # if not overwrite: assert not path.exists()
    path.mkdir(parents=True, exist_ok=True)

    losses = []
    pbar = tqdm(itertools.product(n_sparses, ratios, p_feats), total=len(n_sparses)* len(ratios) * len(p_feats))
    pbar.set_description(f'n_sparse: N/A, Ratio: N/A, p_feat: N/A, Loss: N/A')
    for model_idx, (n_sparse, ratio, p_feat) in enumerate(pbar):
        if (path / str(model_idx)).exists() and not overwrite_experiments:
            continue
        n_dense = max(1,int(n_sparse*ratio))
        dataconfig = data.Config(p_feat, n_sparse, (0, 1))
        datafactory = data.DataFactory(dataconfig)
        model, loss_list = train_hadamard_model(n_dense, n_sparse, p_feat, device=device, n_tries=n_tries, batch_size=batch_size, max_epochs=max_epochs, loss_window=loss_window, update_times=update_times, data_size=data_size)
        losses.append(loss_list)
        pbar.set_description(f'n_sparse: {model.cfg.n_sparse}, Ratio: {model.ratio():.2f}, p_feat: {p_feat}, Loss: {model.final_loss():.4f}')
        model.save(path / str(model_idx))
        with open(path / f'{model_idx}_losses.dill', 'wb') as file:
            pickle.dump(losses, file)
    return


def train_multiple_synthetic_models(D, ratios, p_feats, abstol=None, reltol=None):        
    models = []
    i=0
    pbar = tqdm(itertools.product(ratios, p_feats), total=len(ratios) * len(p_feats))
    pbar.set_description(f'Ratio: N/A, p_feat: N/A, Loss: N/A')
    for (ratio, p_feat) in pbar:
        model = sm.OptimalSyntheticModel(max(int(D * ratio),1), D, p_feat, abstol=abstol, reltol=reltol)
        models.append(model)
        pbar.set_description(f'Ratio: {model.ratio():.2f}, p_feat: {p_feat}, Loss: {model.final_loss():.4f}')
    return models
    
# load models
def load_trained_models(experiment_name, device="cpu"):
    
    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir()
    models = []
    for file in path.iterdir():
        if file.is_file() and not file.name.endswith(('.modelinfo', '.dill', '.DS_Store')): 
            new_model = Model.load(file, map_location=t.device('cpu'))
            models.append(new_model)
    return models      


def load_hadamard_losses(experiment_name, device='cpu'):
    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir()
    losses = []
    for file in path.iterdir():
        if file.is_file() and not file.name.endswith(('.modelinfo', '.dill','.dill_spec', '.DS_Store')): 
            new_model = hm.HadamardModel.load(file, map_location=t.device('cpu'))
            losses.append(new_model._final_loss)
    return losses



def load_hadamard_models(experiment_name, device="cpu"):
    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir()
    models = []
    names = []
    for file in path.iterdir():
        if file.is_file() and not file.name.endswith(('.modelinfo', '.dill','.dill_spec', '.DS_Store')): 
            try:
                new_model = hm.HadamardModel.load(file, map_location=t.device('cpu'))
            except:
                print(file)
            models.append(new_model)
            names.append(uuid.uuid4())


    model_dict = {name: model for (name,model) in zip(names,models)}
    df = pd.DataFrame([
        {'model_id': model_id,  # Use a unique identifier
        'p_feat': model.p_feat,
        'ratio': model.ratio(),
        'n_dense': model.cfg.n_dense,
        'n_sparse': model.cfg.n_sparse,
        'final_loss': model.losses[-1]
        }
        for (model_id, model) in model_dict.items()]
        )
    return df, model_dict

def models_to_dataframe(models): 
    model_dict = {uuid.uuid4(): model for model in models}
    df = pd.DataFrame([
        {'model_id': model_id,  # Use a unique identifier
        'p_feat': model.p_feat,
        'ratio': model.ratio(),
        'n_dense': model.cfg.n_dense,
        'n_sparse': model.cfg.n_sparse,
        'final_loss': model.losses[-1]
        }
        for (model_id, model) in model_dict.items()]
        )
    return df, model_dict

def load_model_measurements_from_models(experiment_name, device="cpu"):
    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir()
    model_measurements = []
    for file in path.iterdir():
        if file.is_file() and not file.name.endswith(('.modelinfo','.dill', '.DS_Store')): 
            new_model = Model.load(file, map_location=t.device('cpu'))
            model_measurements.append(mm.ModelMeasurement(new_model, uuid.uuid4()))
    return model_measurements 

def load_hadamard_model_measurements(experiment_name, device="cpu", overwrite=False):    
    path = pathlib.Path(f"saved_models/{experiment_name}/")
    assert path.is_dir()
    if not overwrite: 
        df, mm_dict = load_df_from_file(experiment_name=experiment_name)
    else:
        print('here')
        model_measurements = []
        names = []
        for file in tqdm(path.iterdir()):
            if file.is_file() and not file.name.endswith(('.modelinfo','.dill', '.dill_spec', '.DS_Store')): 
                new_model = hm.HadamardModel.load(file, map_location=t.device('cpu'))
                new_model._final_loss = new_model.losses[-1]
                model_measurements.append(mm.HadamardModelMeasurement(new_model, file.name))
                names.append(uuid.uuid4())
            
        mm_dict = {name: model for (name, model) in zip(names,model_measurements)}
        df = pd.DataFrame([
            {'model_id': model_id,  # Use a unique identifier
            'p_feat': model.p_feat,
            'ratio': model.ratio(),
            'n_dense': model.cfg.n_dense,
            'n_sparse': model.cfg.n_sparse,
            'final_loss': model.final_loss
            }
            for (model_id, model) in mm_dict.items()]
            )
        file = path / 'df_and_dict.dill'
        with file.open('wb') as f:
            dill.dump((df, mm_dict), f)        
    return df, mm_dict



def hadamard_models_to_dataframe(models): 
    model_dict = {uuid.uuid4(): model for model in models}
    df = pd.DataFrame([
        {'model_id': model_id,  # Use a unique identifier
        'p_feat': model.p_feat,
        'ratio': model.ratio(),
        'n_dense': model.cfg.n_dense,
        'n_sparse': model.cfg.n_sparse,
        'final_loss': model.losses[-1]
        }
        for (model_id, model) in model_dict.items()]
        )
    return df, model_dict

def model_measurements_to_dataframe(model_measurements: List[mm.ModelMeasurement],
                                    experiment_name: Optional[str] = None):
    mm_dict = {uuid.uuid4(): model_meas for model_meas in model_measurements}
    df = pd.DataFrame([
        {'model_id': model_id,  # Use a unique identifier
        'p_feat': model_meas.p_feat,
        'ratio': model_meas.ratio(),
        'n_dense': model_meas.cfg.n_dense,
        'n_sparse': model_meas.cfg.n_sparse,
        'final_loss': model_meas.final_loss,
        'final_layer_bias': model_meas.cfg.final_layer_bias,
        'tie_dec_enc_weights': model_meas.cfg.tie_dec_enc_weights,
        'nonlinearity': model_meas.cfg.final_layer_act_func,
        'chi_pval': model_meas.chi_pval,
        'chi_varvar': model_meas.chi_varvar,
        'chi_meanvar': model_meas.chi_meanvar,
        'bias_var': model_meas.bias_var,
        'diag_mean': model_meas.diag_mean,
        'diag_var': model_meas.diag_var        
        }
        for (model_id, model_meas) in mm_dict.items()
        ])
    if experiment_name:
        path = pathlib.Path(f"saved_models/{experiment_name}/")
        assert path.is_dir()
        file = path / 'df_and_dict.dill'
        with file.open('wb') as f:
            dill.dump((df, mm_dict), f)
    return df, mm_dict

def load_df_from_file(experiment_name:str):
    path = pathlib.Path(f"saved_models/{experiment_name}/df_and_dict.dill")
    assert path.is_file()
    with path.open('rb') as f:
        df, mm_dict = dill.load(f)
    return df, mm_dict

# Estimate the size of a model (for doing large runs)
def size_of_model_estimate(n_sparse, ratio):
    param = t.ones(n_sparse, int(np.ceil(n_sparse*ratio)),dtype=t.float)
    return 2*param.numel()* param.element_size()

# Estimate the size of a run
def size_of_run(n_sparses, ratios, p_feats, final_layer_biases, tie_dec_enc_weights):
    sum = 0.0
    for n_sparse in n_sparses:
        for ratio in ratios:
            sum += size_of_model_estimate(n_sparse, ratio)
    return sum * len(p_feats) * len(final_layer_biases) * len(tie_dec_enc_weights)

def bytes_to_gb(bytes_value):
    return bytes_value / (1024 ** 3)

def get_PyTorch_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all = (param_size + buffer_size) 
    return size_all

# def get_model_size(models, model_id):
#     m = get_model(models, model_id)
#     return get_PyTorch_model_size(m)

# Deprecated until/if we need much bigger models.
# def load_model_measurements(experiment_name):
#     path = pathlib.Path(f"saved_models/{experiment_name}/")
#     assert path.is_dir()
#     models = []
#     for file in path.iterdir():
#         if file.is_file() and not file.name.endswith(('.dill_spec', '.pickle_dict')): 
#             new_model = model_measurement.ModelMeasurement.load(file)
#             models.append(new_model)
#     return models 

# def model_measurements_to_dataframe(models):
#     model_dict = {uuid.uuid4(): model for model in models}
#     df = pd.DataFrame([
#         {'model_id': model_id,  # Use a unique identifier
#         'p_feat': model.p_feat,
#         'ratio': model.ratio(),
#         'n_dense': model.cfg.n_dense,
#         'n_sparse': model.cfg.n_sparse,
#         'final_layer_bias': model.cfg.final_layer_bias,
#         'tie_dec_enc_weights': model.cfg.tie_dec_enc_weights,
#         'nonlinearity': model.cfg.final_layer_act_func
#         }
#         for (model_id, model) in model_dict.items()
#         ])
#     return df, model_dict
# def train_multiple_model_measurements(n_sparses, ratios, p_feats, experiment_name: str, 
#                           activations = ["relu"], final_layer_biases = [False],
#                           tie_dec_enc_weights_list = [False], device="cpu",
#                           overwrite=False, DiagAModel_flag: bool = False, 
#                           num_measurements: float = 1000):
    
#     path = pathlib.Path(f"saved_models/{experiment_name}/")
#     path.mkdir(parents=True, exist_ok=True)
    
#     # to make it backwards compatible
#     n_sparses = [n_sparses] if isinstance(n_sparses, int) else n_sparses
#     ratios = [ratios] if isinstance(ratios, int) else ratios
#     p_feats = [p_feats] if isinstance(p_feats, int) else p_feats
#     activations = [activations] if isinstance(activations, bool) else activations
        
#     models = []
#     W_matrices = {}
#     total = len(n_sparses) * len(ratios) * len(p_feats) * len(activations) * len(final_layer_biases) * len(tie_dec_enc_weights_list) 
#     pbar = tqdm(itertools.product(n_sparses, ratios, p_feats, activations, final_layer_biases, tie_dec_enc_weights_list), total=total)
#     pbar.set_description(f'ratio: N/A, p_feat: N/A, non_linearity: N/A, final_layer_biases: N/A, tie_dec_enc_weights: N/A,  loss: N/A')
#     for (n_sparse, ratio, p_feat, activation, final_layer_bias, tie_dec_enc_weights) in pbar:
#         W_matrices_for_iter = np.zeros([num_measurements, n_sparse, n_sparse])
#         for i in tqdm(range(num_measurements), leave=False):
#             # t_start_training = time.time()
#             model = train_model(max(int(n_sparse * ratio),1), n_sparse, p_feat, activation=activation, final_layer_bias=final_layer_bias, tie_dec_enc_weights=tie_dec_enc_weights, device=device, DiagAModel_flag=DiagAModel_flag)
#             model_id = str(uuid.uuid4())
#             # t_finish_training = time.time()
#             # print(f"training time: {t_finish_training-t_start_training}")
#             model_m = model_measurement.ModelMeasurement(model, model_id)
#             # t_finish_mm = time.time()
#             # print(f"time to make model measurement: {t_finish_mm - t_finish_training} ")
#             models.append(model_m)
#             pbar.set_description(f'ratio: {model.ratio():.2f}, p: {p_feat:.4f}, non_linearity: {activation}, bias: {final_layer_bias}, tie_weights: {tie_dec_enc_weights}, loss: {model.final_loss():.4f}')        
#             model_m.save(path / f'{model_id}.dill')
#             # t_save_mm = time.time()
#             # print(f"time to save model measurement: {t_save_mm-t_finish_mm}")
#             W_matrices_for_iter[i][:,:] = model.W_matrix().detach().cpu()
#             # t_detach = time.time()
#             # print(f"detach and move matrix time: {t_detach - t_save_mm}" )
            
#         W_matrices[(n_sparse, ratio, p_feat, activation, final_layer_bias, tie_dec_enc_weights)] = np.mean(W_matrices_for_iter, axis=0)
    
    
#     with open(path / "mean_W_mat.pickle_dict", 'wb') as file:
#         pickle.dump(W_matrices, file)

#     return models


# Deprecated: Not training on multiple denses, and when we do, we should just iterate over configs
# def train_multiple_models_n_dense(n_sparses, n_denses, p_feats, experiment_name: str, non_linearity = True, overwrite=False, DiagAModel_flag: bool = False):
    
#     path = pathlib.Path(f"saved_models/{experiment_name}/")

#     if not overwrite: assert not path.exists()
#     path.mkdir(parents=True, exist_ok=True)
    
#     # to make it packwards compatible
#     n_sparses = [n_sparses] if isinstance(n_sparses, int) else n_sparses
#     n_denses = [n_denses] if isinstance(n_denses, int) else n_denses
#     p_feats = [p_feats] if isinstance(p_feats, int) else p_feats    
        
#     models = []
#     total = len(n_sparses) * len(n_denses) * len(p_feats)
#     pbar = tqdm(itertools.product(n_sparses, n_denses, p_feats), total=total)
#     pbar.set_description(f'Ratio: N/A, p_feat: N/A, Loss: N/A')
#     for model_idx, (n_sparse, n_dense, p_feat) in enumerate(pbar):
#         model = train_model(n_dense, n_sparse, p_feat, non_linearity, DiagAModel_flag=DiagAModel_flag)
#         models.append(model)
#         pbar.set_description(f'Ratio: {model.ratio():.2f}, p_feat: {p_feat}, Loss: {model.final_loss():.4f}')
#         model.save(path / str(model_idx))
#     return models




    
