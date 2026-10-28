# phi-guard-v7-final (ONNX) — weights phi-guard-masked-3m-v7.21

Byte-level model for clinical text with **four outputs in one pass**: PHI left in a de-identified note, whether that leak is harmful, how much PHI a
note exposes (level 0-6), and PHI in a raw (not yet de-identified) note. 3,459,253 parameters, CPU only, any note length; about 6.4 ms per 1 KB note
on one CPU thread (Apple M-series, onnxruntime), all four outputs together. Internal research model for evaluation; please do not redistribute.

Training data: our own de-identified notes, US-government text published with no licence attached (VAERS, FDA device reports and warning letters, ClinicalTrials.gov) and JSL-owned data; no CC / CC0 text and no evaluation datasets.

## Quick start
```
pip install numpy onnxruntime
python score.py notes.csv scores.csv        # notes.csv needs a "text" column; an "id" column is kept
python score.py --text "Pt [NAME] seen on [DATE] for chest pain."
```
In Python: `from score import PhiGuard; g = PhiGuard(); g.scores(text)` (all 9 probabilities), `g.score(text)` (leak), `g.level(text)` (richness).

## What it outputs (scores.csv)
- `score`, `flag`: P(PHI left in a de-identified note); flag when score >= **0.5**. `logit` is the same before the sigmoid: rank with it (PR-AUC / ROC).
- `harmful_score`, `harmful_flag`: P(a harmful leak), never above `score`. Harmless alone (not flagged): clinician names, dates other than date of birth,
  unless 3 or more different kinds appear together; ages over 89 are harmful. Flag when >= 0.5.
- `level`, `level_name`, `score_1`..`score_6`: exposed PHI spans per 100 words: 0 none, 1 trace (under 1), 2 low (1-2), 3 moderate (2-3), 4 high (3-5),
  5 very high (5-8), 6 saturated (8 or more); `score_k` = P(level >= k), never rising with k; level = how many are >= 0.5.
- `raw_score`, `raw_flag`: P(a raw note contains PHI); flag when >= 0.5. Use it on text before de-identification, e.g. to skip PHI-free notes.

## Thresholds
`score` is calibrated so 0.5 flags as many clean validation notes as phi-guard-v6 does at its 0.5; `harmful_score` so 0.5 keeps 98% of harmful
validation leaks; richness and raw are fixed at 0.5. On our held-out de-identified test sets (notes never used in training) at 0.5:

| test set | notes | leaks | leak recall @0.5 | clean notes flagged @0.5 |
|---|---|---|---|---|
| main (pipeline output) | 3,755 | 1,841 | 0.926 | 91 of 1,914 |
| heldout (unseen sources) | 1,571 | 757 | 0.950 | 89 of 814 |
| long notes (1-6 KB) | 538 | 274 | 0.901 | 53 of 264 |

## Leak detection compared (same notes; PR-AUC, clean notes flagged at 98% / 99% leak recall)
| model | main PR-AUC | main FA@R98 | main FA@R99 | heldout PR-AUC | long PR-AUC |
|---|---|---|---|---|---|
| **this model** | 0.9861 | 375 | 577 | 0.9821 | 0.9335 |
| phi-guard-v6 | 0.9854 | 380 | 585 | 0.9818 | 0.9290 |
| phi-guard-v6-fast | 0.9599 | 848 | 1,081 | 0.9590 | 0.8929 |
| phi-guard v5-final | 0.9728 | 726 | 991 | 0.9388 | 0.9135 |
| kev-phi-4b-v5 (4B LLM) | 0.9862 | 405 | 1,152 | – | – |
| StanfordAIMI deidentifier-base | 0.9341 | 1,072 | 1,296 | 0.9408 | 0.8884 |
| obi/deid_roberta_i2b2 | 0.9412 | 1,176 | 1,342 | 0.9652 | 0.8747 |

kev-phi-4b-v5 was scored on the main set only.

## Harmful leaks compared (PR-AUC on harmful-leak labels; harmless-alone leaks count as negatives)
| model | main | heldout | long | harmless-alone leaks flagged @0.5 (main) |
|---|---|---|---|---|
| **this model** | 0.9800 | 0.8828 | 0.9281 | 38% |
| phi-guard-v6 | 0.9617 | 0.7794 | 0.9038 | 70% |

## Richness compared with phi-guard-richness-v6 (level error MAE / ranking Spearman; this model / richness-v6)
| test set | level error (MAE) | ranking (Spearman) |
|---|---|---|
| de-identified: main (3,746) | 0.140 / 0.137 | 0.866 / 0.859 |
| de-identified: test (13,575) | 0.156 / 0.151 | 0.941 / 0.945 |
| de-identified: heldout (2,928) | 0.303 / 0.325 | 0.909 / 0.902 |
| de-identified: long notes (538) | 0.238 / 0.245 | 0.755 / 0.773 |
| raw notes: test (4,720) | 0.183 / 0.192 | 0.963 / 0.963 |
| raw notes: heldout (1,074) | 0.238 / 0.265 | 0.956 / 0.954 |

## Raw notes compared (PR-AUC; clean notes flagged / notes with PHI missed at 0.5)
| model | test PR-AUC | test: flagged / missed | heldout PR-AUC | heldout: flagged / missed |
|---|---|---|---|---|
| **this model** | 0.9992 | 28 / 42 | 0.9986 | 5 / 17 |
| phi-guard-raw-v6 | 0.9993 | 29 / 38 | 0.9984 | 3 / 18 |
| phi-guard-v6 | 0.9993 | 107 / 16 | 0.9989 | 20 / 6 |
| phi-guard-v6-fast | 0.9981 | 273 / 22 | 0.9972 | 95 / 2 |

## How it works
`model.onnx` takes `input_ids` int64 [windows, 1024] (UTF-8 bytes as ids 0-255, padded with 256) and returns window logits `logit` [windows, 9] and a
window vector `vec` [windows, 2d]. `note.json` holds the note combiners (standard and harmful read the first half of `vec`, richness and raw the
second) that turn all windows of one note into 9 note logits; `score.py` implements them in numpy. `config.json` is the training record (leak logit offset +1.1485, harmful +0.9898, fitted on jev_phi_v5 validation + validation_pipeline (7074 docs, labels v5_final)).

## Known limits
- It says whether, and how much, PHI a note holds; it does not find or remove PHI and is not a de-identifier.
- The harmful policy above is a default; tell us if your rules differ.
- Richness levels 4-6 are rare in de-identified notes, so high levels there are less exact than levels 0-3.
