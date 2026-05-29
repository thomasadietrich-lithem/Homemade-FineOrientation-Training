# Homemade Fine Orientation Training Protocol
(Huxlin Lab Inspired)

## Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Why Fine Orientation Training?](#why-fine-orientation-training)
- [Protocol Overview](#protocol-overview)
- [Protocol Fidelity](#protocol-fidelity)
- [Stimulus Characteristics](#stimulus-characteristics)
- [Adaptive Staircases](#adaptive-staircases)
- [Training Modes](#training-modes)
- [Monitor Calibration](#monitor-calibration)
- [Data Outputs](#data-outputs)
- [Analysis Dashboard](#analysis-dashboard)
- [Repository Structure](#repository-structure)
- [Known Differences vs Rochester Implementation](#known-differences-vs-rochester-implementation)
- [References](#references)
- [Disclaimer](#disclaimer)

================================================================================
OVERVIEW
================================================================================

This repository contains a PsychoPy implementation of the Fine Orientation
Discrimination paradigm derived from the Huxlin Laboratory MATLAB task
FineOrientationDiscrimination_MetaCog.m.

The objective is to reproduce the experimental logic of the Rochester protocol
while providing a practical home-based training and analysis environment.

================================================================================
SCIENTIFIC BACKGROUND
================================================================================

Orientation discrimination tasks are widely used in perceptual learning and
neurovisual rehabilitation research. Unlike motion discrimination paradigms,
they target sensitivity to fine orientation differences using Gabor stimuli.

The implementation follows the structure of the Rochester Fine Orientation
training task and its associated psychometric analysis conventions.

================================================================================
WHY FINE ORIENTATION TRAINING?
================================================================================

Fine orientation discrimination probes neural mechanisms involved in early
visual processing and perceptual learning.

The protocol uses adaptive thresholds to continuously challenge performance,
allowing longitudinal monitoring of visual sensitivity changes at specific
retinotopic locations.

================================================================================
PROTOCOL OVERVIEW
================================================================================

1. Fixation is maintained.
2. A Gabor stimulus is presented.
3. A mask follows stimulus presentation.
4. The participant reports LEFT or RIGHT orientation.
5. Difficulty is adjusted by adaptive staircases.

================================================================================
PROTOCOL FIDELITY
================================================================================

The PsychoPy implementation was designed to preserve:

- Trial structure
- Difficulty scale
- Staircase rules
- Response logic
- Stimulus geometry
- Temporal envelope
- Mask generation
- MATLAB-compatible outputs

The reference source is:

TestingCodes/FineOrientation/FineOrientationDiscrimination_MetaCog.m

================================================================================
STIMULUS CHARACTERISTICS
================================================================================

Stimulus Type:
Static Gabor Patch

Spatial Frequency:
1 cycle / degree

Radius:
2.5° visual angle

Contrast:
100%

Envelope:
Gaussian

Mask:
Band-limited noise mask

================================================================================
ADAPTIVE STAIRCASES
================================================================================

Difficulty Levels:

53.1°
33.2°
20.75°
12.97°
8.1°
5.1°
3.2°
2.0°
1.2°
0.8°
0.5°
0.1°

Three independent staircases are used:

Staircase 1 → 4 correct responses required to advance
Staircase 2 → 3 correct responses required to advance
Staircase 3 → 2 correct responses required to advance

================================================================================
TRAINING MODES
================================================================================

DEFAULT MODE

LEFT ARROW  = Left Tilt
RIGHT ARROW = Right Tilt

MATLAB ISO METACOGNITIVE MODE

1 = Confident Left
2 = Uncertain Left
9 = Uncertain Right
0 = Confident Right

================================================================================
MONITOR CALIBRATION
================================================================================

The protocol operates in visual-angle coordinates.

Users must calibrate:

- Screen width
- Viewing distance

For longitudinal studies, the same monitor, resolution and viewing distance
should be maintained across sessions.

================================================================================
DATA OUTPUTS
================================================================================

CSV
- Trial-by-trial behavioural data

JSON
- Session summaries
- Protocol metadata
- Staircase statistics

MAT
- MATLAB-compatible exports

================================================================================
ANALYSIS DASHBOARD
================================================================================

The companion analytics application provides:

Psychometric Analysis
- Rochester-compatible Weibull fitting
- Direct 72.5% threshold estimation
- Fit quality metrics
- AIC / BIC statistics

Longitudinal Analysis
- Threshold evolution
- Rolling averages
- Session variability
- Pooled thresholds

Session Analysis
- Staircase traces
- Stimulus distributions
- Response-time trends
- Block-by-block performance

Laterality Analysis
- LEFT orientation accuracy
- RIGHT orientation accuracy

Interactive Dashboard
- HTML report generation
- Session summaries
- Longitudinal plots
- Psychometric curves

================================================================================
REPOSITORY STRUCTURE
================================================================================

FineOrientation_Training.py

DataFineOrientation/
└── SubjectID/
    ├── *_results.csv
    ├── *_summary.json
    └── *.mat

FineOrientation_Analytics.py

analysis_results/
├── dashboard.html
├── psychometric plots
├── staircase plots
├── longitudinal reports
└── summary tables

================================================================================
KNOWN DIFFERENCES VS ROCHESTER IMPLEMENTATION
================================================================================

Not reproduced:

- EyeLink integration
- Laboratory gamma calibration tables
- Dedicated fixation monitoring hardware
- Rochester display hardware environment

================================================================================
REFERENCES
================================================================================

Huxlin KR et al. (2009)
Perceptual Relearning of Complex Visual Motion after V1 Damage in Humans.

Cavanaugh MR & Huxlin KR (2017)
Visual Discrimination Training Improves Humphrey Perimetry in Chronic Cortical
Blindness.

Cavanaugh MR et al. (2019)
Predicting Visual Recovery Following Training in Chronic Cortical Blindness.

Cavanaugh MR et al. (2021)
Training-Induced Recovery of Visual Functions after Occipital Stroke.

================================================================================
DISCLAIMER
================================================================================

This software is provided for research and educational purposes only.

It is not a medical device and does not claim to reproduce the exact laboratory
conditions or clinical outcomes reported in the scientific literature.
