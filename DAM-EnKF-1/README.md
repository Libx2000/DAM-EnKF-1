# DAM-EnKF Local Analysis-Weight Network

This repository contains the PyTorch implementation of the neural network used to estimate structured, same-grid local analysis weights for DAM-EnKF.

The model maps ensemble perturbation statistics, the local observation operator (`H`), observation-error covariance (`R`), and an explicit observation mask to a grid of local Kalman-gain blocks. It does **not** construct or predict a dense global state-by-observation gain matrix.

## Main features

- Permutation-invariant ensemble statistics: mean, variance, skewness, and kurtosis.
- Convolutional spatial encoder-decoder.
- Optional spatial self-attention.
- Explicit conditioning on local `H`, `R`, and observation availability.
- Missing observations produce exactly zero gain columns.
- No symmetry constraint is imposed on the Kalman gain.
- A synthetic data generator is included for demonstration and basic testing.

## Repository structure

```text
DAM-EnKF-1/
├── deepk.py                  # Model, synthetic data generator, loss, and training loop
├── examples/
│   └── quickstart.py         # Lightweight CPU/GPU smoke test
├── docs/
│   └── deepk.drawio          # Editable model-framework diagram
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later
- NumPy
- SciPy

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build appropriate for your CUDA version by following the official PyTorch installation instructions.

## Quick start

Run the lightweight example:

```bash
python examples/quickstart.py
```

The example uses a `16 x 32` grid with spatial attention disabled, so it can be used as a basic installation and interface check on either CPU or GPU.

## Model interface

Let:

- `B`: batch size
- `N`: ensemble size
- `H_g`, `W_g`: grid height and width
- `S`: state-variable dimension
- `O`: observation-variable dimension

The primary tensors are:

| Tensor | Shape | Description |
|---|---:|---|
| `ensemble_perturbations` | `(B, N, H_g, W_g, S)` | Ensemble anomalies or perturbations |
| `background` | `(B, H_g, W_g, S)` | Background state |
| `observations` | `(B, H_g, W_g, O)` or `(B, O, H_g, W_g)` | Same-grid observations |
| `obs_operator` | `(O, S)`, `(H_g, W_g, O, S)`, or `(B, H_g, W_g, O, S)` | Local observation operator `H` |
| `obs_error` | scalar, diagonal, `(O, O)`, or supported gridded form | Local observation-error covariance `R` |
| `obs_mask` | `(H_g, W_g, O)` or `(B, H_g, W_g, O)` | Explicit observation mask |
| model output | `(B, H_g, W_g, S, O)` | Grid of local gain blocks |

Minimal use:

```python
import torch
from deepk import KalmanGainLearner

batch_size = 1
ensemble_size = 20
height, width = 64, 128
state_dim = 1
obs_dim = 1

model = KalmanGainLearner(
    state_dim=state_dim,
    obs_dim=obs_dim,
    ensemble_size=ensemble_size,
    hidden_channels=32,
    num_encoder_layers=2,
    use_attention=True,
)

ensemble_perturbations = torch.randn(
    batch_size, ensemble_size, height, width, state_dim
)
obs_operator = torch.eye(obs_dim, state_dim)
obs_error = torch.eye(obs_dim) * 0.01
obs_mask = torch.ones(batch_size, height, width, obs_dim, dtype=torch.bool)

with torch.no_grad():
    local_gain = model(
        ensemble_perturbations,
        obs_operator=obs_operator,
        obs_error=obs_error,
        obs_mask=obs_mask,
    )

print(local_gain.shape)
```

## Analysis update

The convenience method `compute_kalman_update` applies the predicted local gain to the masked innovation:

```python
analysis = model.compute_kalman_update(
    background,
    observations,
    ensemble_perturbations,
    obs_operator=obs_operator,
    obs_error=obs_error,
    obs_mask=obs_mask,
)
```

At each grid point, the update has the form

```text
x_a = x_b + K_local * mask * (y - H x_b).
```

The synthetic target generator computes

```text
K_local = P_local H^T (H P_local H^T + R)^(-1)
```

independently at each grid point. Consequently, the implementation scales with local state and observation dimensions rather than with a dense global gain matrix.

## Training demonstration

`deepk.py` includes `DataGenerator`, `KalmanLoss`, and `train_kalman_model`. The synthetic generator is intended to demonstrate the software interface and is not a replacement for application-specific atmospheric training data.

Example:

```python
from deepk import DataGenerator, KalmanGainLearner, train_kalman_model

generator = DataGenerator(
    grid_size=(64, 128),
    state_dim=1,
    obs_dim=1,
    ensemble_size=20,
)

model = KalmanGainLearner(
    state_dim=1,
    obs_dim=1,
    ensemble_size=20,
    hidden_channels=32,
    num_encoder_layers=2,
    use_attention=True,
)

trained_model = train_kalman_model(
    model,
    generator,
    num_epochs=50,
    batch_size=16,
    lr=1e-3,
)
```

## Implementation notes

- With `use_attention=True`, the current positional encoding is configured for the paper grid of `64 x 128`; inputs must use that grid size.
- Global self-attention has quadratic memory growth in the number of grid points. Disable attention for lightweight tests or memory-limited hardware.
- The supplied observation interface is same-grid and local. Off-grid or finite-radius observation operators require an additional mapping step.
- `R` must be symmetric. Its positive definiteness remains the responsibility of the caller.
- The `ensemble_size` constructor argument documents the configured ensemble, while the encoder aggregates members through statistics and does not use member order.

## Reproducibility

Record the Python, PyTorch, CUDA, GPU, precision, random seed, batch size, and checkpoint used in each experiment. Large benchmark or training data should not be committed directly to Git; publish them through an archival data repository and link them from this README.

## Publish on GitHub

After creating an empty GitHub repository, run the following commands from this directory. Replace `<repository-url>` with the HTTPS or SSH address shown by GitHub.

```bash
git init
git add .
git commit -m "Initial release of the DAM-EnKF local-gain network"
git branch -M main
git remote add origin <repository-url>
git push -u origin main
```

Alternatively, extract the release archive and upload its contents through GitHub's **Add file > Upload files** interface. Keep `README.md`, `requirements.txt`, `deepk.py`, `examples`, and `docs` at the repository root.

## License

No license has been selected in this package. Before making the repository public, add a license approved by all copyright holders. Without a license, the default copyright restrictions apply.

## Citation

If you use this code in academic work, please cite the associated DAM-EnKF manuscript and the repository release. Add the final article citation and DOI here after publication.
