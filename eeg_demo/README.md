# Quantum-inspired EEG presentation demo

This folder is a standalone core implementation of the research paper
`Quantum-Inspired Multi-Objective Optimization of EEG Time-Frequency
Representations for Cross-Subject BCI Decoding`.

It is intentionally separate from the existing IPsec analyzer in this
repository. There is no frontend and no external dataset download required.

## Run it

From the repository root:

```powershell
python eeg_demo\quantum_eeg_demo.py
```

For a faster classroom run with one LOSO fold:

```powershell
python eeg_demo\quantum_eeg_demo.py --folds 1 --population 8 --generations 8
```

If the active Python environment does not already have NumPy:

```powershell
python -m pip install numpy
```

## What the code demonstrates

1. Synthetic two-class motor-imagery EEG is generated for several subjects.
2. Each trial is converted to an 8×9 STFT-like log-power grid covering 4–40 Hz.
3. A mask over the 72 cells is sampled from Q-bit probabilities
   `P(selected) = sin(theta)^2`.
4. Every mask is scored on source subjects using three maximization objectives:
   mean accuracy, negative subject-to-subject standard deviation, and
   compactness.
5. Non-dominated candidates form a Pareto archive. Q-bit angles are rotated
   toward archive solutions.
6. A source-only candidate is evaluated on the held-out subject, alongside a
   fixed all-cells baseline.

The synthetic dataset makes the run reproducible for a presentation. It is not
evidence of an improvement on OpenBMI. The paper's full experiment should use
the real dataset, a fixed deep decoder such as EEGNet, multiple random seeds,
and the same budget for all baselines.
