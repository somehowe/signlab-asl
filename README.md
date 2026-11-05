# SignLab ASL

SignLab is a local American Sign Language recognition and evaluation workbench.
It classifies seven pre-segmented ASL glosses from hand and upper-body
landmarks. Explore held-out examples, replay their motion, upload a compatible
sequence, compare scores, and inspect which signs the classifier confuses.

## Run the demo

Use Python 3.10 or later. From the unzipped project folder:

```sh
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8765**. On Windows, use `py -3` in place of `python`
if that is how you run Python. The packaged demo has a trained model and
28 held-out landmark sequences. Inference runs on your machine without an
account or network connection. The web server binds to loopback.
For a GitHub checkout, run the reproducibility commands below to generate
the model and samples before starting the app.

1. Choose a sample and play its landmark sequence. Switch signs or filter the
   library to see examples from both test signers.
2. Inspect the prediction, reference gloss, and ranked scores. A prediction
   below the selected cutoff is shown as uncertain.
3. Open **Evaluation** to compare models and inspect the confusion matrix.
   A reproducible miss is `clerc_v04_0077`, which the model calls WANT with
   a high score even though the annotated gloss is WHAT.
4. Export a preview as JSON to see the input schema. Upload your own array or
   an object with a `frames` field. Supply up to 240 frames, with 48 points
   per frame and three coordinates per point. Use null for missing coordinates.

For your own recording, the optional offline video tool converts a trimmed
MP4 into importable JSON. See the [video setup](video_tools/README.md).

## What the results show

The dataset preparation script extracts 309 annotated clips from eight signers
for CAN, HAVE, NEED, WANT, WHAT, YOU, and YOUR. The model discarded one
low-signal clip. Whole signers are assigned to training, validation, or test.
The normalization scaler and classifier are fitted on the four training
signers. Model selection and confidence cutoff use the two validation signers.
The 74 clips from the remaining two signers are held out for the main test.

| Model | Test macro F1 | Test top-1 accuracy | Pooled macro F1, eight held-out signer folds |
| --- | ---: | ---: | ---: |
| Mean pose logistic regression | 0.684 | 68.9% | 0.746 |
| Temporal landmark logistic regression | 0.731 | 75.7% | 0.786 |

The main test rows use raw top predictions before abstention. The separately
reported eight-fold audit retrains fixed classifiers with one signer held
out per fold and pools the out-of-fold predictions. The temporal model's
per-signer macro F1 ranges from 0.633 to 0.932.

On the main test, the validation-selected cutoff accepts 48 of 74 clips
and 38 of those 48 decisions are correct. Validation accuracy among accepted
clips was 89.6% against the predeclared 90% target. Model scores are softmax
scores. Their magnitudes can be unreliable, especially on unfamiliar input.
The raw top prediction for 11 of 12 WHAT clips is WANT, and 9 of those errors
pass the cutoff.

The benchmark starts with annotated segment boundaries and covers seven glosses
in a small corpus. Some prompts recur across signers. Results describe
held-out signers in this subset. Performance on new phrases, continuous
signing, arbitrary cameras, and out-of-vocabulary signs requires separate
evaluation.

## Reproduce the benchmark
Use Python 3.11 or later for training.

```sh
python -m pip install -r requirements-train.txt
python scripts/prepare_epee.py --out data/epee_glosses.npz
python -m model.train --data data/epee_glosses.npz --output artifacts --audit-signer-cv
python scripts/prepare_demo.py --data data/epee_glosses.npz
python -m unittest model.test_model -v
python -m unittest discover -s tests -v
```

The first acquisition downloads annotations and landmarks for the selected
clips from a pinned public source revision. Later runs reuse the local cache.
Training compares frame-averaged pose features with time-ordered landmark
features under the same split. Features normalize global translation and
scale, include local hand shape and body-relative positions, and retain
missing-point indicators. The exported model uses numerical arrays. Serving
requires NumPy and Python's standard library.

| Path | Contents |
| --- | --- |
| `scripts/` | Reproducible data preparation and held-out demo selection |
| `model/` | Features, grouped splits, training, audit, and NumPy inference |
| `artifacts/` | Trained weights, demo examples, main report, and separate signer audit |
| `web/` | Local review interface for landmark playback, predictions, and evaluation |

**Data credit.** Florian Méloux and CLERC, *CLERC Épée v0.4: Sign Language
Data Layer* (2026). The adapted samples retain the CC BY-NC-SA 4.0 license.
Pseudonymous landmark sequences can still contain identifying body geometry.
See the data provenance file for the source's terms and evaluation scope.

[Dataset card](https://huggingface.co/datasets/CLERC-DATA/epee) ·
[Data provenance](DATA_PROVENANCE.md) ·
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
