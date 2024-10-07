import sys
sys.path.insert(0, '/home/alex/InterpretabilityResearch/')
sys.path.insert(0, '/home/alex/InterpretabilityResearch/NonLinSAE')
sys.path.insert(0, '/home/alex/InterpretabilityResearch/NonLinSAE/utils/')
sys.path.insert(0, '/home/alex/InterpretabilityResearch/NonLinSAE/utils/plot/')
print(sys.path)

import utils.model_scan
import numpy as np
from dataclasses import dataclass
from typing import List
import dill
import os
import pathlib
from typing import Optional

@dataclass 
class ExperimentSpec:
    experiment_name: str
    n_sparses: np.ndarray
    grid_width: int
    ratios: np.ndarray
    p_feats: np.ndarray
    final_layer_biases: Optional[List]
    tie_dec_enc_weights: Optional[List]
    num_measurements: Optional[1]
    train_win: Optional[bool]
    num_samples: Optional[int]
    n_tries: Optional[int]
    batch_size: Optional[int]
    max_epochs: Optional[int]
    loss_window: Optional[int]
    update_times: Optional[int]
    data_size: Optional[int]



    def save(self):
        """
        Save the experiment specification using dill.

        Args:
            directory (str): The directory to save the file. Defaults to the current directory.

        Returns:
            str: The path to the saved file.
        """
        # Create a filename based on the experiment name
        path = pathlib.Path(f"saved_models/{self.experiment_name}/")
        # assert not path.exists()
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{self.experiment_name}.dill_spec"
        filepath = path / filename

        # Save the entire object using dill
        with open(filepath, 'wb') as f:
            dill.dump(self, f)

        return filepath

    @classmethod
    def load(cls, filepath: str):
        """
        Load an experiment specification from a dill file.

        Args:
            filepath (str): The path to the dill file.

        Returns:
            experiment_spec: The loaded experiment specification object.
        """
        with open(filepath, 'rb') as f:
            return dill.load(f)

def main():

#     # # grid_width = 20
#     # # spec = ExperimentSpec(
#     # #     experiment_name = "hadamard_128_num_tries_5_better_spec",
#     # #     n_sparses = np.array([128]),
#     # #     grid_width = grid_width,
#     # #     ratios = np.linspace(0.002001, 0.8, grid_width),
#     # #     p_feats = np.linspace(0.002001, 0.8, grid_width),
#     # #     final_layer_biases = None,
#     # #     tie_dec_enc_weights = None,
#     # #     num_measurements = None,
#     # #     train_win=False,
#     # #     num_samples=1,
#     # #     n_tries=20,
#     # #     batch_size=128,
#     # #     max_epochs=100,
#     # #     loss_window=50,
#     # #     update_times=3000,
#     # #     data_size=10_000
#     # #     )    
    
#     # # hadamard_models_128 = utils.model_scan.train_multiple_hadamard_models(
#     # #     spec.n_sparses,
#     # #     spec.ratios,
#     # #     spec.p_feats,
#     # #     spec.experiment_name,
#     # #     n_tries = spec.n_tries,
#     # #     batch_size = spec.batch_size,
#     # #     max_epochs = spec.max_epochs,
#     # #     loss_window = spec.loss_window,
#     # #     update_times = spec.update_times,
#     # #     data_size = spec.data_size
#     # #     )
#     # # spec.save()

#     # # grid_width = 20
#     # # spec = ExperimentSpec(
#     # #     experiment_name = "hadamard_1024_num_tries_5_better_spec",
#     # #     n_sparses = np.array([1024]),
#     # #     grid_width = grid_width,
#     # #     ratios = np.linspace(0.002001, 0.8, grid_width),
#     # #     p_feats = np.linspace(0.002001, 0.8, grid_width),
#     # #     final_layer_biases = None,
#     # #     tie_dec_enc_weights = None,
#     # #     num_measurements = None,
#     # #     train_win=False,
#     # #     num_samples=1,
#     # #     n_tries=20,
#     # #     batch_size=128,
#     # #     max_epochs=100,
#     # #     loss_window=50,
#     # #     update_times=3000,
#     # #     data_size=10_000
#     # #     )    
    
#     # hadamard_models_1024 = utils.model_scan.train_multiple_hadamard_models(
#     #     spec.n_sparses,
#     #     spec.ratios,
#     #     spec.p_feats,
#     #     spec.experiment_name,
#         # n_tries = spec.n_tries,
#         # batch_size = spec.batch_size,
#         # max_epochs = spec.max_epochs,
#         # loss_window = spec.loss_window,
#         # update_times = spec.update_times,
#         # data_size = spec.data_size
#     #     )

#     # spec.save()

    grid_width = 20
    spec = ExperimentSpec(
        experiment_name = "8192_overnight",
        n_sparses = np.array([8192]),
        grid_width = grid_width,
        ratios = np.linspace(0.002001, 0.8, grid_width),
        # ratios= [0.8],
        p_feats = np.linspace(0.002001, 0.8, grid_width),
        final_layer_biases = None,
        tie_dec_enc_weights = None,
        num_measurements = None,
        train_win=False,
        num_samples=1,
        n_tries=5,
        batch_size=512,
        max_epochs=100,
        loss_window=100,
        update_times=3000,
        data_size=10_000
        )
    
    hadamard_models_8192 = utils.model_scan.train_multiple_hadamard_models(
        spec.n_sparses,
        spec.ratios,
        spec.p_feats,
        spec.experiment_name,
        n_tries = spec.n_tries,
        batch_size = spec.batch_size,
        max_epochs = spec.max_epochs,
        loss_window = spec.loss_window,
        update_times = spec.update_times,
        data_size = spec.data_size,
        device='cuda'
        )
    spec.save()

#     grid_width = 20
#     spec = ExperimentSpec(
#         experiment_name = "8192_small_data",
#         n_sparses = np.array([8192]),
#         grid_width = grid_width,
#         ratios = np.linspace(0.002001, 0.8, grid_width),
#         # ratios= [0.8],
#         p_feats = np.linspace(0.002001, 0.8, grid_width),
#         final_layer_biases = None,
#         tie_dec_enc_weights = None,
#         num_measurements = None,
#         train_win=False,
#         num_samples=1,
#         n_tries=5,
#         batch_size=512,
#         max_epochs=100,
#         loss_window=100,
#         update_times=3000,
#         data_size=1_000
#         )
    
#     hadamard_models_8192 = utils.model_scan.train_multiple_hadamard_models(
#         spec.n_sparses,
#         spec.ratios,
#         spec.p_feats,
#         spec.experiment_name,
#         n_tries = spec.n_tries,
#         batch_size = spec.batch_size,
#         max_epochs = spec.max_epochs,
#         loss_window = spec.loss_window,
#         update_times = spec.update_times,
#         data_size = spec.data_size,
#         device='cuda'
#         )
#     spec.save()
    


if __name__ == "__main__":
    main()