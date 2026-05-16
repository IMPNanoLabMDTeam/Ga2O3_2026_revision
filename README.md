# Ga2O3 2026 Revision Tests

This repository contains the additional validation tests prepared for the
second revision of the manuscript

**A Neuroevolution Potential for Gallium Oxide: Accurate and Efficient
Modeling of Polymorphism and Swift Heavy-Ion Irradiation**.

The repository was organized to make the technical checks behind the revised
equation-of-state comparison and force-field benchmarks transparent and
reproducible. The main conclusions from these repository tests have also been
included in the Supplementary Material of the revised manuscript.

## Repository Structure

- `forcefield/`
  - Contains the NEP model and the published tabGAP potential used for the
    comparison.
  - The tabGAP potential was used in its published form without retraining or
    modification.

- `model/`
  - Contains representative configurations used in the additional checks,
    including structures before and after geometry optimization.

- `scripts/`
  - Contains LAMMPS input files and analysis scripts for running the NEP/tabGAP
    comparisons and post-processing the energy-volume data.

- `reviewer_tests/test_01_relax_prepost_ev/`
  - Tests whether internal atomic relaxation of the strained configurations
    affects the equation-of-state comparison in Fig. 5.
  - All configurations used in Fig. 5 were further optimized using VASP and then
    recalculated with single-point DFT energies.
  - `results/1.png` compares the optimized and unoptimized DFT
    energy-volume curves and shows that the differences are limited.
  - `results/2.png` shows the tabGAP and NEP predictions on the optimized
    configurations. The same conclusions as in the revised manuscript are
    obtained.

- `reviewer_tests/test_02_lammps_version_ev/`
  - Tests whether the LAMMPS version affects the energy-volume comparison.
  - `results/3.png` shows that using the LAMMPS version corresponding to the
    original GAP workflow and the LAMMPS version used in our formal comparison
    gives no observable difference in the energy-volume curves.

## Notes on the Gamma-Phase Equation-of-State Data

The DFT data used in the original energy-volume curves were taken from the
original GAP dataset, except for the gamma-phase configurations. The original
GAP dataset does not contain enough gamma-phase configurations to construct a
complete energy-volume curve. Therefore, we selected the lowest-energy
gamma-phase configuration and generated additional structures by isotropic
compression and expansion using the same protocol adopted for the GAP dataset.

To address the reviewer's concern about possible missing internal relaxation,
we additionally optimized all configurations shown in Fig. 5 using VASP and
repeated the DFT, tabGAP, and NEP energy-volume comparisons. These additional
tests show that geometry optimization does not alter the main conclusions.

## Relation to the Revised Supplementary Material

The key repository results, including the optimized/unoptimized DFT
energy-volume comparison and the NEP/tabGAP predictions on optimized
configurations, have been added to the Supplementary Material of the revised
manuscript so that the validation is visible directly in the submitted revision
as well as reproducible through this repository.
