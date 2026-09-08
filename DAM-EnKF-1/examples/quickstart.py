"""Lightweight interface check for the DAM-EnKF local-gain network."""

from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from deepk import DataGenerator, KalmanGainLearner  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    grid_size = (16, 32)
    state_dim = 2
    obs_dim = 1
    ensemble_size = 8

    model = KalmanGainLearner(
        state_dim=state_dim,
        obs_dim=obs_dim,
        ensemble_size=ensemble_size,
        hidden_channels=16,
        num_encoder_layers=2,
        use_attention=False,
    )
    model.eval()

    generator = DataGenerator(
        grid_size=grid_size,
        state_dim=state_dim,
        obs_dim=obs_dim,
        ensemble_size=ensemble_size,
    )
    batch = generator.generate_batch(batch_size=1)

    with torch.no_grad():
        gain = model(
            batch["ensemble_perturbations"],
            obs_operator=batch["obs_operator"],
            obs_error=batch["obs_error"],
            obs_mask=batch["obs_mask"],
        )
        analysis = model.compute_kalman_update(
            batch["background"],
            batch["observations"],
            batch["ensemble_perturbations"],
            obs_operator=batch["obs_operator"],
            obs_error=batch["obs_error"],
            obs_mask=batch["obs_mask"],
        )

    expected_gain_shape = (1, *grid_size, state_dim, obs_dim)
    expected_analysis_shape = (1, *grid_size, state_dim)
    assert tuple(gain.shape) == expected_gain_shape
    assert tuple(analysis.shape) == expected_analysis_shape

    missing = ~batch["obs_mask"]
    missing_gain = gain.masked_select(missing.unsqueeze(-2))
    if missing_gain.numel():
        assert torch.count_nonzero(missing_gain).item() == 0

    print(f"Device: {next(model.parameters()).device}")
    print(f"Local gain shape: {tuple(gain.shape)}")
    print(f"Analysis shape: {tuple(analysis.shape)}")
    print("Quick-start check passed.")


if __name__ == "__main__":
    main()
