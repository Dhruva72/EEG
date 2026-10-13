"""Presentation-sized implementation of the paper's core algorithm.

This demo intentionally uses synthetic motor-imagery EEG so it can run without
downloading OpenBMI before a presentation. The research idea is still present:

1. Create a time-frequency (TF) representation for every EEG trial.
2. Treat the 8 x 9 TF cells as 72 binary decisions.
3. Search masks with quantum-inspired probabilities (Q-bit angles).
4. Keep non-dominated masks on a Pareto archive using three objectives:
   mean source-subject accuracy, source-subject consistency, and compactness.
5. Select a representation using source subjects only and test it on a held-out
   subject in a leave-one-subject-out (LOSO) evaluation.

The code is a teaching/demo implementation, not a claim of experimental
performance on OpenBMI. It uses a fast nearest-centroid proxy in the search;
the paper proposes passing the selected representation to a fixed deep decoder
after the search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


N_WINDOWS = 8
N_BANDS = 9
N_COMPONENTS = N_WINDOWS * N_BANDS
FREQUENCY_BANDS = tuple((4.0 + 4.0 * i, 8.0 + 4.0 * i) for i in range(N_BANDS))


@dataclass(frozen=True)
class SubjectData:
    """Precomputed TF trials for one subject."""

    subject_id: int
    tf_trials: np.ndarray  # [trials, time_windows, frequency_bands]
    labels: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """A mask and its three source-side objective values."""

    mask: np.ndarray
    mean_accuracy: float
    consistency: float  # negative standard deviation, so larger is better
    compactness: float
    subject_scores: tuple[float, ...]
    generation: int

    @property
    def retained(self) -> int:
        return int(self.mask.sum())


def make_synthetic_dataset(
    *,
    subjects: int,
    trials_per_class: int,
    seed: int,
) -> list[SubjectData]:
    """Create a small, reproducible two-class motor-imagery EEG dataset.

    Class 0 has stronger mu-band activity and class 1 has stronger beta-band
    activity during the middle of the trial. Subject-specific amplitude,
    frequency, and noise changes make cross-subject transfer non-trivial.
    """

    rng = np.random.default_rng(seed)
    sample_rate = 128
    samples = sample_rate * 4
    time = np.arange(samples) / sample_rate
    event = ((time >= 1.0) & (time < 3.0)).astype(float)
    window_size = samples // N_WINDOWS
    taper = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(window_size) / (window_size - 1))
    frequencies = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)

    data: list[SubjectData] = []
    for subject_id in range(subjects):
        # These shifts are deliberately small but different for each person.
        subject_frequency_shift = rng.normal(0.0, 1.15)
        subject_amplitude = rng.uniform(0.82, 1.18)
        # Make the proxy work for its score: informative rhythms are present,
        # but weak noisy TF cells should still be removable.
        subject_noise = rng.uniform(0.55, 0.78)
        trials: list[np.ndarray] = []
        labels: list[int] = []

        for label in (0, 1):
            for _ in range(trials_per_class):
                raw = rng.normal(0.0, subject_noise, size=(4, samples))
                phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
                class_frequency = (10.0 if label == 0 else 20.0) + subject_frequency_shift
                # A weaker shared rhythm prevents the task from becoming a
                # single-feature toy problem.
                shared = 0.20 * np.sin(2.0 * np.pi * 14.0 * time + phase[0])
                for channel in range(raw.shape[0]):
                    informative = (
                        subject_amplitude
                        * (0.16 + 0.025 * channel)
                        * np.sin(2.0 * np.pi * class_frequency * time + phase[channel])
                    )
                    nuisance = 0.28 * np.sin(
                        2.0 * np.pi * (7.0 + 0.3 * channel) * time + phase[channel]
                    )
                    raw[channel] += event * (informative + nuisance + shared)

                trials.append(_tf_representation(raw, taper, frequencies, window_size))
                labels.append(label)

        data.append(
            SubjectData(
                subject_id=subject_id,
                tf_trials=np.asarray(trials, dtype=np.float64),
                labels=np.asarray(labels, dtype=np.int64),
            )
        )

    return data


def _tf_representation(
    raw: np.ndarray,
    taper: np.ndarray,
    frequencies: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Compute a compact STFT-like log-power grid shared across channels."""

    grid = np.zeros((N_WINDOWS, N_BANDS), dtype=np.float64)
    for window_index in range(N_WINDOWS):
        start = window_index * window_size
        segment = raw[:, start : start + window_size] * taper
        spectrum = np.abs(np.fft.rfft(segment, axis=1)) ** 2
        for band_index, (low, high) in enumerate(FREQUENCY_BANDS):
            selected = (frequencies >= low) & (frequencies < high)
            # Mean over channels and frequency bins: this is the shared TF
            # mask from the paper's first-study design.
            grid[window_index, band_index] = np.log1p(spectrum[:, selected].mean())
    return grid


def _selected_features(subjects: Sequence[SubjectData], mask: np.ndarray) -> np.ndarray:
    """Gather only selected TF cells, preserving trial rows."""

    cells = [subject.tf_trials.reshape(len(subject.tf_trials), -1)[:, mask] for subject in subjects]
    return np.concatenate(cells, axis=0)


def _fit_proxy(subjects: Sequence[SubjectData], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a standardized nearest-centroid proxy on source fitting subjects."""

    features = _selected_features(subjects, mask)
    labels = np.concatenate([subject.labels for subject in subjects])
    if features.shape[1] == 0:
        raise ValueError("A candidate mask must retain at least one TF component")

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (features - mean) / scale
    centroids = np.vstack([normalized[labels == label].mean(axis=0) for label in (0, 1)])
    return mean, scale, centroids


def _proxy_accuracy(
    subject: SubjectData,
    mask: np.ndarray,
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    mean, scale, centroids = normalization
    features = subject.tf_trials.reshape(len(subject.tf_trials), -1)[:, mask]
    normalized = (features - mean) / scale
    distances = ((normalized[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
    predictions = distances.argmin(axis=1)
    return float(np.mean(predictions == subject.labels))


def _evaluate_candidate(
    candidate_mask: np.ndarray,
    fitting_subjects: Sequence[SubjectData],
    validation_subjects: Sequence[SubjectData],
    generation: int,
) -> Candidate:
    normalization = _fit_proxy(fitting_subjects, candidate_mask)
    scores = tuple(_proxy_accuracy(subject, candidate_mask, normalization) for subject in validation_subjects)
    mean_accuracy = float(np.mean(scores))
    consistency = -float(np.std(scores))
    compactness = 1.0 - float(candidate_mask.sum()) / N_COMPONENTS
    return Candidate(
        mask=candidate_mask.copy(),
        mean_accuracy=mean_accuracy,
        consistency=consistency,
        compactness=compactness,
        subject_scores=scores,
        generation=generation,
    )


def _dominates(left: Candidate, right: Candidate) -> bool:
    left_objectives = np.array([left.mean_accuracy, left.consistency, left.compactness])
    right_objectives = np.array([right.mean_accuracy, right.consistency, right.compactness])
    return bool(np.all(left_objectives >= right_objectives) and np.any(left_objectives > right_objectives))


def _pareto_front(candidates: Iterable[Candidate]) -> list[Candidate]:
    unique: dict[bytes, Candidate] = {}
    for candidate in candidates:
        key = candidate.mask.tobytes()
        previous = unique.get(key)
        if previous is None or candidate.mean_accuracy > previous.mean_accuracy:
            unique[key] = candidate

    front: list[Candidate] = []
    for candidate in unique.values():
        if not any(_dominates(other, candidate) for other in unique.values() if other is not candidate):
            front.append(candidate)
    return front


def _prune_archive(archive: list[Candidate], limit: int) -> list[Candidate]:
    if len(archive) <= limit:
        return archive

    # Keep extremes first so the demonstration preserves the visible
    # accuracy-versus-compression trade-off, then fill by objective score.
    chosen: list[Candidate] = []
    chosen_keys: set[bytes] = set()
    for attribute in ("mean_accuracy", "consistency", "compactness"):
        extreme = max(archive, key=lambda item: getattr(item, attribute))
        key = extreme.mask.tobytes()
        if key not in chosen_keys:
            chosen.append(extreme)
            chosen_keys.add(key)

    remaining = [item for item in archive if item.mask.tobytes() not in chosen_keys]
    remaining.sort(key=lambda item: item.mean_accuracy + item.compactness + item.consistency)
    chosen.extend(reversed(remaining))
    return chosen[:limit]


def quantum_inspired_search(
    source_subjects: Sequence[SubjectData],
    *,
    population: int,
    generations: int,
    archive_limit: int,
    rotation_step: float,
    seed: int,
    verbose: bool = True,
) -> tuple[list[Candidate], list[dict[str, float]]]:
    """Search TF masks using the paper's Q-bit angle update."""

    if len(source_subjects) < 3:
        raise ValueError("At least three source subjects are needed for fitting and validation splits")

    rng = np.random.default_rng(seed)
    epsilon = 0.05
    theta = np.full(N_COMPONENTS, np.pi / 4.0)
    archive: list[Candidate] = []
    trace: list[dict[str, float]] = []

    for generation in range(1, generations + 1):
        order = rng.permutation(len(source_subjects))
        validation_count = max(2, len(source_subjects) // 3)
        validation_indices = order[:validation_count]
        fitting_indices = order[validation_count:]
        fitting = [source_subjects[index] for index in fitting_indices]
        validation = [source_subjects[index] for index in validation_indices]

        probability = np.sin(theta) ** 2
        population_candidates: list[Candidate] = []
        sampled_masks: list[np.ndarray] = []
        for _ in range(population):
            mask = rng.random(N_COMPONENTS) < probability
            if not np.any(mask):
                mask[rng.integers(0, N_COMPONENTS)] = True
            sampled_masks.append(mask)
            population_candidates.append(_evaluate_candidate(mask, fitting, validation, generation))

        archive = _prune_archive(_pareto_front([*archive, *population_candidates]), archive_limit)
        reference = archive[int(rng.integers(0, len(archive)))].mask
        for mask in sampled_masks:
            theta += rotation_step * (reference.astype(float) - mask.astype(float))
        theta = np.clip(theta, epsilon, np.pi / 2.0 - epsilon)

        best = max(archive, key=lambda item: item.mean_accuracy)
        row = {
            "generation": float(generation),
            "archive_size": float(len(archive)),
            "best_mean_accuracy": best.mean_accuracy,
            "best_compactness": best.compactness,
            "mean_selection_probability": float(probability.mean()),
        }
        trace.append(row)
        if verbose:
            print(
                f"    generation {generation:02d}/{generations} | "
                f"archive={len(archive):02d} | "
                f"best source accuracy={best.mean_accuracy:.1%} | "
                f"best compression={best.compactness:.1%} | "
                f"mean P(select)={probability.mean():.2f}"
            )

    return archive, trace


def select_source_only_candidate(archive: Sequence[Candidate]) -> Candidate:
    """Choose a compact high-performing point without looking at the target."""

    if not archive:
        raise ValueError("Cannot select a representation from an empty archive")

    compact_candidates = [candidate for candidate in archive if candidate.compactness >= 0.50]
    pool = compact_candidates or list(archive)
    # Accuracy is primary; compactness breaks near-ties. This policy is fixed
    # before the held-out subject is evaluated.
    return max(pool, key=lambda item: (item.mean_accuracy, item.compactness, item.consistency))


def evaluate_on_target(
    source_subjects: Sequence[SubjectData],
    target: SubjectData,
    mask: np.ndarray,
) -> float:
    normalization = _fit_proxy(source_subjects, mask)
    return _proxy_accuracy(target, mask, normalization)


def print_mask(mask: np.ndarray) -> None:
    """Print a readable 8 x 9 selected-cell mask for the presentation."""

    grid = mask.reshape(N_WINDOWS, N_BANDS)
    print("    TF mask: rows=time windows, columns=frequency bands 4-40 Hz")
    print("             " + " ".join(f"{int(low):02d}" for low, _ in FREQUENCY_BANDS))
    for row_index, row in enumerate(grid, start=1):
        print(f"      t{row_index:02d}       " + " ".join("##" if selected else ".." for selected in row))


def run_demo(args: argparse.Namespace) -> None:
    print("=" * 78)
    print("QUANTUM-INSPIRED EEG TIME-FREQUENCY SEARCH - CORE DEMO")
    print("=" * 78)
    print("This run uses synthetic two-class motor-imagery EEG for presentation.")
    print("It demonstrates the algorithm; it is not an OpenBMI result.\n")

    dataset = make_synthetic_dataset(
        subjects=args.subjects,
        trials_per_class=args.trials_per_class,
        seed=args.seed,
    )
    print(
        f"[1/4] Generated {len(dataset)} subjects x "
        f"{args.trials_per_class * 2} trials/subject."
    )
    print(
        f"      TF representation: {N_WINDOWS} time windows x {N_BANDS} bands "
        f"= {N_COMPONENTS} binary components."
    )
    print("      Frequency bands: 4-8, 8-12, ..., 36-40 Hz.\n")

    fixed_mask = np.ones(N_COMPONENTS, dtype=bool)
    searched_scores: list[float] = []
    fixed_scores: list[float] = []
    selected_masks: list[np.ndarray] = []
    fold_limit = min(args.folds, len(dataset)) if args.folds is not None else len(dataset)

    print(f"[2/4] LOSO evaluation: {fold_limit} held-out subject(s).")
    for held_out_index in range(fold_limit):
        target = dataset[held_out_index]
        source = [subject for index, subject in enumerate(dataset) if index != held_out_index]
        print(f"\n  Fold {held_out_index + 1}/{fold_limit}: subject {target.subject_id} is held out")
        print("  Search sees source subjects only:", ", ".join(str(subject.subject_id) for subject in source))
        archive, _ = quantum_inspired_search(
            source,
            population=args.population,
            generations=args.generations,
            archive_limit=args.archive,
            rotation_step=args.rotation_step,
            seed=args.seed + held_out_index + 1,
            verbose=not args.quiet,
        )
        selected = select_source_only_candidate(archive)
        searched_accuracy = evaluate_on_target(source, target, selected.mask)
        fixed_accuracy = evaluate_on_target(source, target, fixed_mask)
        searched_scores.append(searched_accuracy)
        fixed_scores.append(fixed_accuracy)
        selected_masks.append(selected.mask)
        print(
            f"  Source-only choice: {selected.retained}/{N_COMPONENTS} cells retained "
            f"({selected.compactness:.1%} compression), "
            f"source validation={selected.mean_accuracy:.1%}"
        )
        print(
            f"  Held-out result: searched={searched_accuracy:.1%} | "
            f"fixed-all-cells baseline={fixed_accuracy:.1%}"
        )
        if held_out_index == 0:
            print_mask(selected.mask)

    print("\n[3/4] Aggregate held-out results")
    print(
        f"  Searched representation: mean={np.mean(searched_scores):.1%}, "
        f"std={np.std(searched_scores):.1%}"
    )
    print(
        f"  Fixed representation:    mean={np.mean(fixed_scores):.1%}, "
        f"std={np.std(fixed_scores):.1%}"
    )
    retained = [int(mask.sum()) for mask in selected_masks]
    print(
        f"  Selected cells:           mean={np.mean(retained):.1f}/{N_COMPONENTS} "
        f"({1.0 - np.mean(retained) / N_COMPONENTS:.1%} mean compression)"
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the quantum-inspired EEG paper core demo")
    parser.add_argument("--subjects", type=int, default=5, help="Synthetic subjects (default: 5)")
    parser.add_argument("--trials-per-class", type=int, default=10, help="Trials per class and subject (default: 10)")
    parser.add_argument("--population", type=int, default=10, help="Masks sampled per generation (default: 10)")
    parser.add_argument("--generations", type=int, default=10, help="Search generations (default: 10)")
    parser.add_argument("--archive", type=int, default=20, help="Pareto archive limit (default: 20)")
    parser.add_argument("--rotation-step", type=float, default=0.05 * np.pi, help="Q-bit angle update step")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument("--folds", type=int, default=None, help="Run only the first N LOSO folds")
    parser.add_argument("--quiet", action="store_true", help="Hide per-generation progress")
    return parser


if __name__ == "__main__":
    run_demo(build_parser().parse_args())
