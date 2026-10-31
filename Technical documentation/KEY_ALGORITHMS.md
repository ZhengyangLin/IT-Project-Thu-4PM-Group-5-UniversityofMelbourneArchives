## Key Algorithms

### 1 Preprocessing and Readability

`preprocess.run()` decodes the image as grayscale, estimates small skew from thresholded text regions, rotates valid estimates outside the configured deadzone, and applies CLAHE contrast enhancement. It retains a plain deskewed image for geometric analysis and an enhanced version for OCR.

`triage.assess()` estimates character height from connected components. Poor readability produces a warning rather than automatically stopping processing. The pipeline stops if coarse OCR produces no usable tokens. It also stops when over 75% of recognised strings have at most three trimmed characters and no digits, no probe anchors are present, and `Pipeline.force` is false.

### 2 Title-Block Localisation

`locate.locate()` tries these methods in order:

1. **Local table detection:** `table_regions.find_table()` combines line geometry, metadata-label groups, content coverage, and boundary checks. A reliable proposal is accepted with score `0.85`.
2. **Keyword anchors:** exact/fuzzy label matching and spatial relationships.
3. **Frame cells:** frame geometry, text density, and position.
4. **Text density:** a coarse OCR heat map with a lower-right preference.
5. **Fallback:** region `(0.60W, 0.55H, W, H)`, score `0.15`, marked degraded and requiring review.

These scores are heuristic signals rather than measured accuracy probabilities.

### 3 OCR Passes and Fusion

The pipeline runs full-page coarse OCR, crops the detected title block, and runs fine OCR at an adjusted scale. Resizing and crop offsets are reversed into the full preprocessed image's coordinates; there is no inverse deskew transform back to the unrotated source.

Within a pass, `fusion.fuse()` filters weak observations, groups overlapping boxes, chooses the highest-confidence reading, and retains candidates and agreement. The adapter layer supports multiple engines, but the current web worker uses only PaddleOCR.

`merge_passes()` matches fine and coarse evidence using overlap and text similarity. Different readings remain as `@coarse` alternatives. Unmatched coarse tokens covering at least 70% of a fine-token box are suppressed; remaining coarse-only tokens are restored with agreement capped at `0.5`.

### 4 Layout and Field Extraction

`layout.group_rows()` groups upright tokens by vertical proximity and horizontal gaps; tall vertical tokens are handled separately. `views.build_views()` creates a spatial text grid, an indexed token view, and a flattened token list.

Rules use value patterns, fuzzy labels, architect-name matching, and spatial anchor/value scores. SciPy's `linear_sum_assignment()` selects global anchor/value matches. `rules.extract()` returns locked fields, pending fields, and rule-deferral reasons. Locked values take precedence over model results, but still undergo final scoring.

The eight result fields are `project_name`, `drawing_title`, `architect`, `draughtsperson`, `drawing_number`, `date`, `scale`, and `drawing_type`. Specialised rules handle selected fields; unresolved fields are passed to the LLM.

### 5 LLM Completion and Evidence

`assign_fields()` sends the text grid, numbered tokens, alternatives, locked values, and deferral reasons when pending fields exist, LLM assistance is enabled, and the OpenAI SDK can be imported. Drawing image pixels are not included.

For `drawing_type`, the prompt requests classification from the supplied text with `tokens: []`; this is a prompt instruction, not a schema-enforced constraint.

Returned transcription is compared with cited OCR readings. Supported missing token references can be repaired, but repaired fields are prevented from automatic acceptance. Failed evidence receives zero confidence and requires review. Parsing and evidence checks do not fully validate every possible model response: missing citations are not automatically rejected, and malformed values can still cause processing errors.

Handled model-request or JSON-parsing failures preserve available rule results and mark processing failed. An enabled LLM whose SDK cannot be imported also causes failure when fields remain pending; disabling the LLM leaves them unresolved without itself causing failure. Corrections, original transcription, correction reasons, and token evidence are retained separately.

### 6 Confidence Scoring

`verify.fuse()` calculates a weighted mean using only available signals:

```text
confidence = round(sum(weight * signal) / sum(included weights), 3)
```

Current `config.yaml` weights are agreement `0.18`, OCR confidence `0.18`, format validation `0.18`, and LLM confidence `0.15`. Model labels map to high `1.0`, medium `0.7`, and low `0.35`; an unrecognised non-empty label uses `0.5`. Format checks apply to scale, date, drawing number, and draughtsperson. Stored rule/localisation scores are not extra weighted terms.

Where supplied, token-based confidence and agreement use the minimum values across cited tokens. A model note, correction basis, or corrected value can mark a field adjudicated, raising agreement below `1.0` to at least `0.85`.

| Confidence | Normal field status |
| --- | --- |
| `>= 0.75` | `auto_accept` |
| `>= 0.45` and `< 0.75` | `needs_review` |
| `< 0.45` | `rejected` |

Overrides apply: failed evidence or unavailable scoring signals require review; corrections that change letters or digits (ignoring case) and include `correction_basis` are capped at `0.85`, or `0.45` when their format check fails; repaired references remain below automatic acceptance. Localisation below `0.5` or other degraded conditions cap confidence at `0.45` and force review. Missing transcription with an `illegible` or `low_resolution` reason receives zero confidence and is rejected unless an evidence failure or later pipeline override requires review. These internal scores do not guarantee correctness.

