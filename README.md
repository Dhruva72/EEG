# EEG

A presentation-sized Python demo of quantum-inspired time-frequency feature selection for cross-subject EEG decoding.

The implementation follows the core idea from *Quantum-Inspired Multi-Objective Optimization of EEG Time-Frequency Representations for Cross-Subject BCI Decoding*:

- Generate a reproducible synthetic two-class motor-imagery EEG dataset.
- Convert each trial into an 8 × 9 time-frequency log-power grid spanning 4–40 Hz.
- Search over the 72 grid cells with quantum-inspired Q-bit probabilities.
- Maintain a Pareto archive using source-subject accuracy, cross-subject consistency, and feature compactness.
- Select a representation using source subjects only, then evaluate it on held-out subjects with leave-one-subject-out (LOSO) evaluation.
- Compare the searched representation with a fixed all-cells baseline.

This is a teaching and presentation demo. It uses a fast standardized nearest-centroid proxy and synthetic data; it does not claim performance on OpenBMI or any other real EEG dataset. No frontend or dataset download is required.

## Repository layout

```text
.
├── eeg_demo/
│   ├── quantum_eeg_demo.py   # Demo implementation and command-line interface
│   └── README.md             # Detailed explanation of the demo
├── requirements-eeg-demo.txt # Python dependencies
└── test_eeg_demo.py           # Smoke test for the main workflow
```

## Requirements

- Python 3.9 or newer
- NumPy 1.24 or newer

## Installation

Clone the repository, create an isolated environment, and install the dependency:

```bash
git clone https://github.com/Dhruva72/EEG.git
cd EEG

python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Then install the requirements:

```bash
python -m pip install -r requirements-eeg-demo.txt
```

## Run the demo

Run the default experiment from the repository root:

```bash
python eeg_demo/quantum_eeg_demo.py
```

The default run creates five synthetic subjects, performs a search for each held-out subject, and reports searched versus fixed all-cells accuracy, compression, and the selected time-frequency mask.

For a shorter classroom or presentation run:

```bash
python eeg_demo/quantum_eeg_demo.py --folds 1 --population 8 --generations 8
```

Hide per-generation search progress with `--quiet`:

```bash
python eeg_demo/quantum_eeg_demo.py --folds 1 --quiet
```

### Useful options

| Option | Default | Description |
| --- | ---: | --- |
| `--subjects` | `5` | Number of synthetic subjects |
| `--trials-per-class` | `10` | Trials per class for each subject |
| `--population` | `10` | Masks sampled per search generation |
| `--generations` | `10` | Number of search generations |
| `--archive` | `20` | Maximum Pareto archive size |
| `--rotation-step` | `0.05π` | Q-bit angle update step |
| `--seed` | `7` | Random seed for reproducible data and search |
| `--folds` | all | Run only the first N LOSO folds |
| `--quiet` | off | Suppress per-generation progress output |

See all options with:

```bash
python eeg_demo/quantum_eeg_demo.py --help
```

## Run the smoke test

From the repository root:

```bash
python test_eeg_demo.py
```

A successful run prints:

```text
EEG demo smoke test passed
```

## How the demo works

1. **Synthetic EEG generation** — Four-channel, four-second trials are generated for two classes. Subject-specific amplitude, frequency, and noise variations make transfer between subjects non-trivial.
2. **Time-frequency representation** — Each trial is divided into eight time windows. Mean channel power is aggregated into nine frequency bands: 4–8, 8–12, ..., 36–40 Hz.
3. **Quantum-inspired search** — Each of the 72 cells is represented by a Q-bit angle. The probability of selecting a cell is `sin(theta)^2`; sampled masks are updated toward archive solutions.
4. **Multi-objective evaluation** — Candidate masks are scored on source subjects by mean accuracy, consistency (negative score standard deviation), and compactness (the fraction of cells removed).
5. **LOSO evaluation** — One subject is held out at a time. The mask is chosen without using that subject, then evaluated with a nearest-centroid classifier trained on the source subjects.
6. **Baseline comparison** — The selected mask is compared with a fixed mask containing all 72 cells.

For the detailed algorithm notes and limitations, see [eeg_demo/README.md](eeg_demo/README.md).
