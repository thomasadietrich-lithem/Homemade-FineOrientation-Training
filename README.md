# Fine Orientation Gabor Patch Training

Python/PsychoPy implementation of a Huxlin-lab-inspired **Fine Orientation Discrimination** home-training task using a briefly presented static **Gabor patch**, followed by a visual mask, plus a dedicated analytics dashboard based on **Rochester/DataFitting-style Weibull psychometric fitting**.

This repository contains two applications:

```text
FineOrientation_Training.py
FineOrientation_Analytics.py
```

---

## Table of contents

- [Repository structure](#repository-structure)
- [Application 1 — FineOrientation_Training.py](#application-1--fineorientation_trainingpy)
- [Application 2 — FineOrientation_Analytics.py](#application-2--fineorientation_analyticspy)
- [Paradigm summary](#paradigm-summary)
- [Stimulus and mask generation](#stimulus-and-mask-generation)
- [Adaptive staircase](#adaptive-staircase)
- [Response mapping](#response-mapping)
- [Output data structure](#output-data-structure)
- [Weibull psychometric analysis](#weibull-psychometric-analysis)
- [Longitudinal dashboard](#longitudinal-dashboard)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Research references](#research-references)
- [Acknowledgement and disclaimer](#acknowledgement-and-disclaimer)

---

## Repository structure

The repository is expected to contain the following two Python files:

```text
.
├── FineOrientation_Training.py
└── FineOrientation_Analytics.py
```

After running training sessions and analyses, the following folders and files are created automatically:

```text
.
├── FineOrientation_Training.py
├── FineOrientation_Analytics.py
├── monitor_profiles.json
└── DataFineOrientation/
    ├── <subject>/
    │   ├── <subject>_OriMetaCog_<H>_<V>_<timestamp>_results.csv
    │   ├── <subject>_OriMetaCog_<H>_<V>_<timestamp>_summary.json
    │   └── <subject>_OriMetaCog_<H>_<V>_<timestamp>.mat
    └── analysis_results/
        ├── fine_orientation_dashboard.html
        ├── fine_orientation_session_summary_table.csv
        ├── fine_orientation_analysis_dashboard_summary.json
        ├── *_psychometric_weibull.png
        ├── *_session_summary.json
        ├── *_staircase_trace.png
        ├── *_stimulus_distribution.png
        ├── *_rt_and_block_accuracy.png
        ├── longitudinal_*_orientation_accuracy_threshold_overlay.png
        ├── recent_*_orientation_block_accuracy.png
        └── longitudinal_*_left_right_accuracy.png
```

Important distinction:

| Application | Role |
|---|---|
| `FineOrientation_Training.py` | Runs the PsychoPy training session and creates `DataFineOrientation/<subject>/` outputs |
| `FineOrientation_Analytics.py` | Reads `*_results.csv` files and writes dashboard outputs to `analysis_results/` |

The analytics application first searches recursively inside:

```text
DataFineOrientation/**/_results.csv
```

If no files are found there, it also searches for `*_results.csv` files next to the script or in the current working directory.

---

## Application 1 — `FineOrientation_Training.py`

### Purpose

`FineOrientation_Training.py` runs the actual PsychoPy home-training session.

It ports the Huxlin Lab MATLAB Fine Orientation Discrimination MetaCog task:

```text
TestingCodes/FineOrientation/FineOrientationDiscrimination_MetaCog.m
```

The task is a fine orientation discrimination paradigm:

| Step | Description |
|---|---|
| 1 | The participant fixates centrally |
| 2 | A static Gabor patch appears briefly at a chosen visual-field location |
| 3 | A noise mask appears after the stimulus |
| 4 | The participant reports whether the Gabor was tilted left or right |
| 5 | One of three adaptive staircases is updated |


---

### Training lineage

The training script follows a single-source lineage:

```text
TestingCodes/FineOrientation/FineOrientationDiscrimination_MetaCog.m
```

The implementation intentionally preserves:

| Component | Status |
|---|---|
| Fine orientation task structure | Preserved |
|  `baseAngle` difficulty scale | Preserved |
| Three interleaved staircases | Preserved |
| Catch / bonus trials | Preserved |
| Post-stimulus masking | Preserved |
| Response after mask only | Preserved |
| Optional metacognitive response mode | Preserved |
| EyeLink integration | Not implemented |
| Gamma-table loading from Rochester lab computer | Not reproduced |

The script uses PsychoPy/Python rendering and stores additional CSV/JSON metadata to make home-based session auditing easier.

---

### User inputs

At launch, the script opens a PsychoPy dialog asking for:

| Input | Meaning |
|---|---|
| `Subject ID` | Subject identifier used in output folder and filenames |
| `Stimulus horizontal eccentricity X` | Horizontal stimulus position in degrees, Left(-) to Right(+) |
| `Stimulus vertical eccentricity Y` | Vertical stimulus position in degrees, Down(-) to Up(+) |
| `Number of trials` | Number of trials requested for the session |
| `Activate metacognition?` | Enables the original MATLAB-style confidence response mode |

Default values:

| Input | Default |
|---|---:|
| Subject ID | `0000` |
| Viewing distance | `42.0 cm` |
| Screen width | `54.3 cm` |
| Horizontal stimulus eccentricity | `0.0°` |
| Vertical stimulus eccentricity | `0.0°` |
| Number of trials | `300` |
| Metacognition | Disabled |


Stimulus location is user-defined.

---

### Monitor calibration

The task is expressed in degrees of visual angle.

The script therefore requires a monitor profile containing:

| Parameter | Use |
|---|---|
| Screen width in centimeters | Converts degrees into pixels |
| Viewing distance in centimeters | Converts degrees into pixels |
| Screen resolution | Defines active PsychoPy window size |
| Refresh rate | Defines temporal envelope frames |

The monitor profile is stored in:

```text
monitor_profiles.json
```

The profile key depends on:

```text
operating system + detected screen resolution
```

If screen-size detection fails before the PsychoPy window opens, the script falls back to:

```text
1920 × 1080 px
```

The script shows the active screen profile at every launch. The user can:

```text
Continue with this profile
Recalibrate
Cancel
```

If the current resolution differs from the stored profile, the script displays a warning and recommends recalibration.
This matters because screen width, viewing distance and resolution affect:

| Affected quantity | Why it matters |
|---|---|
| Stimulus eccentricity | Retinotopic placement changes if calibration is wrong |
| Gabor radius | A 2.5° stimulus must be converted correctly into pixels |
| Spatial frequency | 1 cycle/degree depends on accurate visual-angle scaling |
| Fixation/stimulus geometry | Longitudinal comparison requires stable setup |

---

### Screen requirements for longitudinal comparability

For repeated sessions, use the same setup whenever possible:

| Requirement | Reason |
|---|---|
| Same physical monitor | Avoids changes in size, luminance and pixel pitch |
| Same screen resolution | Prevents spatial scaling changes |
| Same OS display scaling | Prevents hidden pixel/degree changes |
| Same viewing distance | Preserves visual-angle geometry |
| Fixed brightness and contrast | Reduces luminance variability |
| SDR mode rather than HDR | Avoids dynamic contrast behavior |
| Disable Night Light / Night Shift / True Tone | Avoids color and luminance shifts |
| Stable room lighting | Reduces adaptation and reflection effects |
| Stable head/chin position | Preserves eccentricity and stimulus geometry |

Do not compare thresholds across different screens unless this is explicitly marked as a new setup.

---

## Application 2 — `FineOrientation_Analytics.py`

### Purpose

`FineOrientation_Analytics.py` is the analysis and dashboard application.

It performs:

| Analysis step | Description |
|---|---|
| CSV discovery | Finds `*_results.csv` files |
| Data cleaning | Keeps valid trial rows and required columns |
| Session fitting | Fits one psychometric Weibull per session when possible |
| Location grouping | Groups sessions by trained retinotopic location |
| Longitudinal analysis | Tracks thresholds and accuracy over sessions |
| Staircase plotting | Shows trial-by-trial difficulty evolution |
| Distribution plotting | Shows how many trials occurred at each difficulty |
| RT / block analysis | Tracks response time and within-session accuracy |
| LEFT / RIGHT analysis | Separates performance by signed orientation class |
| HTML generation | Builds an interactive browser dashboard |

The analytics application is explicitly descriptive:

```text
No automatic clinical decision rule is implemented.
```

Confidence ratings and score are treated as optional metadata because the default practical training mode disables metacognition.
Bonus/catch trials are displayed descriptively but excluded from threshold fitting by default because they do not update the three adaptive staircases.

---

### Output folder

If CSV files are found inside `DataFineOrientation/`, the analytics tool writes outputs to:

```text
DataFineOrientation/analysis_results/
```

Example:

```text
project/
├── DataFineOrientation/
│   ├── <subject>/
│   │   └── *_results.csv
│   └── analysis_results/
│       └── fine_orientation_dashboard.html
```

If CSV files are found outside `DataFineOrientation/`, the analytics application creates `analysis_results/` next to the input files.

---

### Analytics outputs

For each session, the analytics application can generate:

| Output | Description |
|---|---|
| `*_psychometric_weibull.png` | Session psychometric curve with Rochester-compatible and direct 72.5% Weibull fits |
| `*_session_summary.json` | Session-level analysis summary |
| `*_staircase_trace.png` | Trial-by-trial staircase trace |
| `*_stimulus_distribution.png` | Distribution of trials across orientation-difference levels |
| `*_rt_and_block_accuracy.png` | Within-session accuracy and median response-time plot |

Across sessions and locations, it generates:

| Output | Description |
|---|---|
| `fine_orientation_session_summary_table.csv` | Summary table of all analyzed sessions |
| `fine_orientation_analysis_dashboard_summary.json` | Structured dashboard-level JSON summary |
| `fine_orientation_dashboard.html` | Main navigable HTML dashboard |
| `longitudinal_*_orientation_accuracy_threshold_overlay.png` | Accuracy and threshold over sessions |
| `recent_*_orientation_block_accuracy.png` | Within-session block accuracy for recent sessions |
| `longitudinal_*_left_right_accuracy.png` | LEFT vs RIGHT orientation accuracy comparison |

The HTML dashboard includes:

| Section | Content |
|---|---|
| Summary | Global analysis constants and link to summary CSV |
| Per-location dashboard | Longitudinal threshold and accuracy charts |
| Session table | Trial counts, accuracy, thresholds and warnings |
| Session-level plots | Psychometric curve, staircase trace, stimulus distribution, RT/block plot |
| Interactive controls | Toggle Rochester threshold, direct 72.5% threshold, pooled threshold, accuracy, RT and threshold delta |
| Export | Browser print/export through `window.print()` |

---

## Paradigm summary

This repository implements a static Gabor fine orientation discrimination paradigm.
The participant fixates centrally while a Gabor patch is presented at a user-specified visual-field location.
The Gabor does not move.
The participant judges whether the orientation is tilted:

```text
LEFT
RIGHT
```

The sign of the trial angle determines the correct response:

| Signed angle | Target side |
|---:|---|
| Negative angle | LEFT |
| Positive angle | RIGHT |

The task is adaptive. Correct and incorrect responses update one of three interleaved staircases.
The default training mode is simplified for home use:

```text
LEFT arrow  = left tilt
RIGHT arrow = right tilt
```

The original MATLAB metacognitive response mode remains available as an option.

---

## Stimulus and mask generation

### Gabor stimulus

The current implementation uses the following fixed stimulus parameters:

| Parameter | Value |
|---|---:|
| Stimulus type | Static Gabor patch |
| Spatial frequency | 1 cycle/degree |
| Stimulus radius | 2.5° |
| Contrast | 100% |
| Spatial envelope | Gaussian |
| Background luminance | 128 in MATLAB-style 0–255 space |
| Black | 0 in MATLAB-style 0–255 space |
| White | 255 in MATLAB-style 0–255 space |
| PsychoPy color domain | Converted to RGB -1..1 |
| Interpolation | Disabled |

The stimulus patch size is computed from the calibrated monitor geometry:

```text
stimulus_radius_px = round(60 × 2.5 / arcmin_per_pixel)
patch_size_px = 2 × stimulus_radius_px + 1
```

The Gaussian standard deviation is computed as:

```text
gaussian_stdev_px = round(stimulus_radius_px / 1.5)
```

The Gabor frame is generated in MATLAB-like 0–255 grayscale, then converted to a PsychoPy-compatible RGB texture.
This conversion is intentional. The code uses PIL RGB images to avoid backend-dependent black-square artifacts that can occur with some PsychoPy/numpy texture paths.

---

### Temporal envelope

The Gabor is not shown as a single static image. It is drawn frame-by-frame with a sinusoidal onset/offset temporal envelope.

Current timing constants:

| Phase | Value |
|---|---:|
| Onset ramp | 0.25 s |
| Plateau duration | 0.00 s |
| Offset ramp | 0.25 s |
| Total nominal stimulus duration | Approximately 0.50 s |
| Mask delay | 0.25 s |
| Inter-trial interval | 1.00 s |

The number of temporal-envelope frames depends on the detected refresh rate.
At 60 Hz, the onset and offset ramps each contain approximately 15 frames, with inclusive MATLAB-style indexing.

---

### Mask generation

The mask is generated as a single-frame post-stimulus noise mask.

Current mask parameters:

| Parameter | Value |
|---|---:|
| Mask width | 151 px |
| Mask orientation | 30° |
| Mask coherence | 0.0 |
| Mask spatial frequency | 0.001 cycles/pixel |
| Mask spatial-frequency standard deviation | 0.01 cycles/pixel |
| Mask aperture | Raised-cosine circular aperture |

Because mask coherence is set to `0.0`, the nominal mask orientation is not behaviorally meaningful.

The mask is generated using frequency-domain filtering and then center-cropped or padded to match the Gabor patch destination size. This mirrors the MATLAB logic in which the mask texture is drawn through the same stimulus rectangle.

---

### Trial timing

Each trial follows this sequence:

| Phase | Description |
|---|---|
| Pre-stimulus display | Trial counter, optional score and fixation are drawn |
| Start beep | 1000 Hz beep immediately before the stimulus when PsychoPy sound is available |
| Gabor presentation | Frame-by-frame temporal envelope |
| Post-stimulus display | Fixation / counter display |
| Mask | Noise mask shown for 250 ms |
| Response phase | Response collected only after the mask |
| Feedback sound | Correct / incorrect beep when sound is available |
| ITI | 1.0 s pause before next trial |

The CSV stores actual stimulus timing:

```text
stimulus_start_s
stimulus_end_s
stimulus_duration_actual_s
response_phase
```

The response phase is recorded as:

```text
post_mask_only
```

---

## Adaptive staircase

The current version uses:

```text
3 fixed interleaved adaptive staircases
```

The MATLAB-compatible staircase identities are:

```text
1, 2, 3
```

The catch/bonus condition is encoded as:

```text
0
```

### Trial structure

Default session:

| Trial type | Count |
|---|---:|
| Bonus/catch trials | 30 |
| Staircase 1 | 90 |
| Staircase 2 | 90 |
| Staircase 3 | 90 |
| Total | 300 |

The default trial structure is:

```text
[0 × 30, 1 × 90, 2 × 90, 3 × 90]
```

The full trial list is shuffled before the session.

If the user changes the total number of trials, the script preserves the approximate ratio:

```text
10% bonus/catch trials
90% adaptive trials split across 3 staircases
```

The actual planned trial count is recorded in the session JSON.

---

### Difficulty scale

The difficulty scale is the MATLAB `baseAngle` vector:

```text
[53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5, 0.1]
```

Because the list is descending:

| Stair step | Meaning |
|---|---|
| Lower step | Larger orientation difference, easier trial |
| Higher step | Smaller orientation difference, harder trial |

### Initial staircase positions

The three staircases start at different 1-based MATLAB steps:

| Staircase | Initial step | Initial orientation difference |
|---|---:|---:|
| 1 | 1 | 53.1° |
| 2 | 4 | 12.97° |
| 3 | 9 | 1.2° |

### Update rule

Each staircase has its own correct-response requirement:

| Staircase | Rule |
|---|---|
| 1 | Step down after 4 correct responses |
| 2 | Step down after 3 correct responses |
| 3 | Step down after 2 correct responses |

In this code, “step down” means moving to the next entry in the `baseAngle` vector, therefore to a smaller orientation difference and a harder trial.

On incorrect responses:

| Condition | Effect |
|---|---|
| Incorrect response on staircase 1, 2 or 3 | Staircase moves one step easier if possible |
| Incorrect response on bonus/catch trial | No staircase update |
| Any incorrect response | Consecutive correct counter for that staircase resets |

Staircase indices are clipped to remain inside the available difficulty scale.

### Bonus/catch trials

Bonus/catch trials use staircase identity `0`.
They are included in the behavioral record and descriptive accuracy but do not update the adaptive staircases.
Their difficulty is derived from staircase 1 according to the MATLAB-inspired rule:

```text
if stair_step[1] < 5:
    angle = baseAngle[0]
else:
    angle = baseAngle[stair_step[1] - 4]
```

This keeps bonus/catch trial difficulty linked to current staircase progression without treating those trials as staircase-updating trials.

---

## Response mapping

### Default practical training mode

Default mode disables metacognition and confidence-weighted score.

The participant responds with arrows:

```text
LEFT arrow  = left tilt
RIGHT arrow = right tilt
```

Correctness rule:

| Signed angle | Correct key |
|---:|---|
| Negative | LEFT arrow |
| Positive | RIGHT arrow |

In this mode:

| Field | Value |
|---|---|
| `confidence` | `-1` |
| `points_delta` | `0` |
| Score visible during task | No |
| Threshold estimation | Performed later in the analysis dashboard |

---

### MATLAB ISO metacognitive mode

If metacognition is activated at launch, the original MATLAB-like response keys are used:

```text
1 = confident LEFT
2 = uncertain LEFT
9 = uncertain RIGHT
0 = confident RIGHT
```

Scoring:

| Response type | Score |
|---|---:|
| Confident correct | +50 |
| Uncertain correct | +20 |
| Uncertain incorrect | 0 |
| Confident incorrect | -50 |

In this mode:

| Field | Meaning |
|---|---|
| `confidence = 1` | Confident response |
| `confidence = 0` | Uncertain response |
| `running_score` | Updated trial by trial |
| Score visible during task | Yes |

---

### Escape behavior

Pressing `ESC` aborts the session.

If the session is interrupted, completed trials are still saved. The JSON summary records:

```text
aborted = true
session_complete = false
n_trials_completed
```

---

## Output data structure

### CSV output

Each session creates:

```text
DataFineOrientation/<subject>/<subject>_OriMetaCog_<H>_<V>_<timestamp>_results.csv
```

The CSV contains one row per completed trial.

Columns:

| Column | Meaning |
|---|---|
| `trial` | Trial number |
| `difficulty_abs_deg` | Absolute orientation difference |
| `response_time_s` | Response time after response phase begins |
| `signed_angle_deg` | Signed orientation difference; negative = LEFT, positive = RIGHT |
| `correct` | 1 = correct, 0 = incorrect |
| `confidence` | 1/0 in metacognition mode, -1 in default mode |
| `staircase` | 0 = bonus/catch, 1–3 = adaptive staircases |
| `response_key` | Raw PsychoPy key pressed |
| `running_score` | Current score, mainly meaningful in metacognition mode |
| `is_bonus_trial` | 1 for staircase 0, otherwise 0 |
| `stair_step_pre` | Staircase step before trial update |
| `stair_step_post` | Staircase step after trial update |
| `stair_count_pre` | Consecutive-correct counter before update |
| `stair_count_post` | Consecutive-correct counter after update |
| `target_side` | Correct orientation class: left or right |
| `response_side` | Participant response class: left or right |
| `points_delta` | Trial score delta |
| `stimulus_start_s` | Stimulus start time relative to session start |
| `stimulus_end_s` | Stimulus end time relative to session start |
| `stimulus_duration_actual_s` | Actual measured stimulus duration |
| `response_phase` | Recorded as `post_mask_only` |

---

### JSON summary output

Each session creates:

```text
*_summary.json
```

The JSON summary contains:

| Section | Examples |
|---|---|
| Session metadata | subject, task name, timestamp, source lineage |
| Completion status | requested trials, completed trials, aborted/error flags |
| Performance | overall accuracy, accuracy by staircase, target/response side counts |
| Staircase state | final steps, final values in degrees, mean and SD |
| Timing | response phase, actual stimulus duration summary |
| Monitor geometry | screen width, distance, resolution, refresh, arcmin/pixel |
| Protocol constants | baseAngle, mask delay, ITI, stimulus radius, response keys |
| Output filenames | CSV and MAT filename references |

The JSON summary is used by the analytics script to infer retinotopic location and to flag protocol warnings.

---



## Weibull psychometric analysis

The analytics application estimates psychometric thresholds from trial-level CSV data.

### Data cleaning

The analyzer requires these columns:

```text
trial
difficulty_abs_deg
correct
staircase
```

It also uses optional enriched columns when available:

```text
response_time_s
signed_angle_deg
confidence
response_key
running_score
is_bonus_trial
target_side
response_side
points_delta
stair_step_pre
stair_step_post
response_phase
stimulus_duration_actual_s
```

Rows are removed if they have invalid or missing values for trial number, difficulty, correctness or staircase identity.

Bonus/catch trials are excluded from threshold fitting by default:

```text
FIT_EXCLUDES_BONUS_TRIALS = True
```

They remain visible in descriptive analyses.

---

### Fit requirements

A session is fitted only if it meets minimum requirements:

| Requirement | Value |
|---|---:|
| Minimum fit trials | 80 |
| Minimum unique difficulty levels | 4 |

If these criteria are not met, the session is kept in the dashboard but marked as invalid for threshold fitting.

---

### Fitted stimulus space

The psychometric fit uses log-transformed stimulus values:

```text
x = log10(difficulty_abs_deg + 1)
```

This avoids fitting directly on the raw degree scale and follows the DataFitting-style conventions used by the analysis script.

---

### Constants

Current constants:

| Constant | Value |
|---|---:|
| Chance | 0.5 |
| Lapse rate | 0.05 |
| Threshold target | 0.725 |
| Within-session blocks | 6 |
| Recent-session window | 5 |
| Pooled-threshold window | 3 |

---

### Rochester-compatible Weibull

The main threshold reported by the dashboard is the Rochester/DataFitting-compatible threshold.

The function used is:

```text
p(correct) =
1 - lapse - (1 - chance - lapse) × exp(-((k × x / threshold)^beta))

k =
(-log((1 - target) / (1 - chance)))^(1 / beta)
```

The resulting threshold is stored as:

```text
threshold_rochester_exact_deg
threshold_rochester_exact_log10_plus1
beta_rochester_exact
```

For backward compatibility, the dashboard also maps this value to:

```text
threshold_deg
```

---

### Direct 72.5% threshold

The dashboard also computes a mathematically direct 72.5% threshold.

The direct function is:

```text
p(correct) =
chance + (1 - chance - lapse) × (1 - exp(-(x / alpha)^beta))
```

The resulting threshold is stored as:

```text
threshold_python_true_725_deg
threshold_python_true_725_log10_plus1
beta_python
```

The dashboard reports the difference between both conventions:

```text
threshold_delta_python_minus_rochester_deg
threshold_delta_python_minus_rochester_percent
```

This distinction matters because the Rochester/DataFitting-compatible convention and the direct 72.5% extraction are not always numerically identical when lapse is included.

---

### Fit validation

The analyzer rejects non-finite or unreasonable thresholds.
A fitted threshold is rejected if it falls outside the reasonable session range:

```text
threshold > max(observed_max × 1.5, observed_max + 5)
```

This prevents the dashboard from presenting numerically divergent fits as meaningful thresholds.

---

### Fit-quality metrics

The dashboard computes descriptive fit-quality metrics:

```text
fit_log_likelihood
null_log_likelihood
fit_pseudo_r2
fit_aic
fit_bic
```

It computes these for both the Rochester-compatible fit and the direct 72.5% fit.
These metrics are descriptive only. They do not trigger a clinical interpretation.

---

## Longitudinal dashboard

The dashboard groups sessions by trained location.
Location is inferred from the companion JSON summary when possible:

```text
stimulus_location_deg = {"H": ..., "V": ...}
```

The location ID is formatted as:

```text
H+<horizontal>_V+<vertical>
```

Example:

```text
H-7.00_V-4.50
```

### Longitudinal metrics

For each location, the dashboard tracks:

| Metric | Meaning |
|---|---|
| Session threshold | Per-session Rochester-compatible Weibull threshold |
| Direct 72.5% threshold | Per-session direct threshold |
| Raw fit-trial accuracy | Accuracy excluding bonus/catch trials |
| 3-session rolling threshold | Smoothed longitudinal estimate |
| 3-session pooled threshold | Weibull fit after pooling recent session trials |
| Threshold variability | Standard deviation across valid sessions |
| Accuracy variability | Standard deviation across sessions |

---

### Within-session block analysis

Each session is split into:

```text
6 equal trial blocks
```

The dashboard plots accuracy across blocks.

For recent sessions, it shows the last:

```text
5 sessions
```

This helps identify within-session fatigue, adaptation or instability.

---

### Response-time analysis

If response-time data are present, the dashboard computes:

```text
mean RT
median RT
standard deviation RT
minimum RT
maximum RT
median RT first block
median RT last block
last-minus-first median RT
```

It also generates an RT/block plot combining:

```text
block accuracy
median response time
```

---

### LEFT vs RIGHT analysis

The analyzer classifies trials by orientation sign:

| Signed angle | Class |
|---:|---|
| Negative | LEFT |
| Positive | RIGHT |

It then computes separate accuracy for:

```text
LEFT
RIGHT
```

This helps detect asymmetric performance across orientation classes.

---

### Protocol warnings

The dashboard can flag:

```text
session_aborted
session_error
partial_session
screen_resolution_changed
analysis_exception
```

These warnings are shown in the HTML dashboard and the session summary table.

---

## Installation

Recommended Python version:

```text
Python 3.10 – 3.12
```

Required Python packages:

```text
psychopy
numpy
pandas
matplotlib
scipy
pillow
```

The training application requires PsychoPy.

The analytics application requires:

```text
numpy
pandas
matplotlib
scipy
```

`scipy` is also used by the training application to create MATLAB-compatible `.mat` files when available.

---

## Tutorial

### 1. Clone or download the repository

The repository should contain:

```text
FineOrientation_Training.py
FineOrientation_Analytics.py
```

### 2. Install dependencies

Using pip:

```bash
pip install numpy pandas matplotlib scipy pillow psychopy
```

PsychoPy can also be installed through the official PsychoPy distribution.

### 3. Run the training application

From a terminal:

```bash
python FineOrientation_Training.py
```

Or open the file in PsychoPy and run it from the PsychoPy interface.

### 4. Complete the setup dialog

You will be asked for:

```text
Subject ID
Stimulus horizontal eccentricity X
Stimulus vertical eccentricity Y
Number of trials
Activate metacognition?
```

If no monitor profile exists, or if recalibration is selected, you will also be asked for:

```text
screen width in cm
viewing distance in cm
```

### 5. Confirm the monitor profile

At every launch, the program shows the active screen profile.

Choose:

```text
Continue with this profile
```

only if the same monitor, resolution and viewing distance are being used.

Choose:

```text
Recalibrate
```

if the setup changed.

### 6. Read the instruction screen

The instruction screen summarizes:

```text
stimulus location
trial count
viewing distance
screen width
detected refresh rate
difficulty scale
stimulus radius
timing constants
response keys
```

Press:

```text
SPACE
```

to start.

Press:

```text
ESC
```

to quit.

### 7. Complete the task

During the task:

| Instruction | Reason |
|---|---|
| Keep fixation on the central dot | Preserves retinotopic training location |
| Do not follow the peripheral stimulus with the eyes | The task assumes fixation |
| Respond after the mask | Responses are collected post-mask |
| Use only the configured response keys | Other keys are ignored |
| Press ESC if needed | Completed trials are still saved |

Default keys:

```text
LEFT arrow  = left tilt
RIGHT arrow = right tilt
```

### 8. Locate saved data

After the session, outputs are saved in:

```text
DataFineOrientation/<subject>/
```

You should see:

```text
*_results.csv
*_summary.json
*.mat
```

The `.mat` file is created only if `scipy.io.savemat` is available.

### 9. Run the analytics application

From the same project folder:

```bash
python FineOrientation_Analytics.py
```

The script searches for:

```text
DataFineOrientation/**/_results.csv
```

Then creates:

```text
DataFineOrientation/analysis_results/
```

and opens the HTML dashboard in the default browser.

### 10. Interpret results cautiously

The most useful longitudinal markers are:

| Marker | Interpretation |
|---|---|
| Rochester-compatible threshold | Main threshold estimate; lower is better |
| Direct 72.5% threshold | Secondary threshold convention |
| Fit-trial accuracy | Descriptive performance measure |
| Staircase trace | Shows adaptive progression |
| Stimulus distribution | Shows which difficulty levels were sampled |
| LEFT vs RIGHT accuracy | Shows possible response-class asymmetry |
| RT/block plot | Shows speed and fatigue-related patterns |

Do not interpret a single session in isolation as clinical evidence.

Prefer repeated sessions at the same calibrated location.

---

## Research references

This repository is inspired by the Huxlin Lab visual rehabilitation work and public code lineage.

Recommended references:

1. Huxlin, K. R., Martin, T., Kelly, K., Riley, M., Friedman, D. I., Burgin, W. S., & Hayhoe, M. (2009).  
   **Perceptual Relearning of Complex Visual Motion after V1 Damage in Humans.**  
   *Journal of Neuroscience, 29(13), 3981–3991.*  
   https://www.jneurosci.org/content/29/13/3981

2. Das, A., Tadin, D., & Huxlin, K. R. (2014).  
   **Beyond Blindsight: Properties of Visual Relearning in Cortically Blind Fields.**  
   *Journal of Neuroscience, 34(35), 11652–11664.*  
   https://www.jneurosci.org/content/34/35/11652

3. Cavanaugh, M. R., Barbot, A., Carrasco, M., & Huxlin, K. R. (2019).  
   **Feature-based attention potentiates recovery of fine direction discrimination in cortically blind patients.**  
   *Neuropsychologia, 128, 315–324.*  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC5994362/

4. Cavanaugh, M. R., Huxlin, K. R., and collaborators.  
   **Visual discrimination training and recovery in chronic cortical blindness.**  
   Relevant Huxlin Lab publications on perceptual learning, Humphrey perimetry and chronic cortical blindness.

Public Huxlin Lab GitHub organization:

```text
https://github.com/huxlinlab
```

Reference source code lineage:

```text
TestingCodes/FineOrientation/FineOrientationDiscrimination_MetaCog.m
DataFitting-style psychometric fitting conventions
```

---

## Acknowledgement and disclaimer

This project is a Python/PsychoPy reinterpretation of the structure and logic of the Fine Orientation Discrimination tools shared by the Huxlin Lab / University of Rochester ecosystem.

It is not an official Huxlin Lab release.

It is not a medical device.

It should not be used as a substitute for medical advice, clinical examination, neuro-ophthalmological follow-up or supervised rehabilitation.

Any use for self-training should be discussed with a qualified clinician.

Home-based deployment differs from laboratory deployment in important ways:

| Laboratory factor | Home implementation status |
|---|---|
| Eye tracking | Not implemented |
| Gamma calibration | Not reproduced |
| Controlled luminance | Not guaranteed |
| Fixation monitoring | User-dependent |
| Display hardware | User-dependent |
| Viewing distance | User-dependent unless externally controlled |

Results should therefore be interpreted as experimental/descriptive data, not as clinical proof of recovery.

Feedback or corrections from the original authors, from clinicians, or from researchers familiar with the protocol are welcome.

Contact: Thomas Dietrich – thomas.a.dietrich@gmail.com
