"""Small smoke test for the standalone EEG presentation demo."""

import os
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from eeg_demo.quantum_eeg_demo import (  # noqa: E402
    N_COMPONENTS,
    evaluate_on_target,
    make_synthetic_dataset,
    quantum_inspired_search,
    select_source_only_candidate,
)


def main() -> None:
    dataset = make_synthetic_dataset(subjects=4, trials_per_class=3, seed=11)
    assert len(dataset) == 4
    assert dataset[0].tf_trials.shape == (6, 8, 9)

    source = dataset[1:]
    archive, trace = quantum_inspired_search(
        source,
        population=4,
        generations=3,
        archive_limit=8,
        rotation_step=0.05 * np.pi,
        seed=12,
        verbose=False,
    )
    assert archive
    assert len(trace) == 3
    assert all(candidate.mask.shape == (N_COMPONENTS,) for candidate in archive)

    selected = select_source_only_candidate(archive)
    score = evaluate_on_target(source, dataset[0], selected.mask)
    assert 0.0 <= score <= 1.0
    print("EEG demo smoke test passed")


if __name__ == "__main__":
    main()
