# CLERC Épée v0.4 data

The benchmark uses public landmark arrays and time-aligned gloss annotations
from Florian Méloux and CLERC, *CLERC Épée v0.4: Sign Language Data Layer*
(2026), DOI [10.5281/zenodo.23587708](https://doi.org/10.5281/zenodo.23587708).
See the [dataset card](https://huggingface.co/datasets/CLERC-DATA/epee) and
[CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The data preparation script pins Hugging Face revision
`9037c3f0278a900ac26eb4ba065a44d791d28f98` and downloads only the files used
for this benchmark.

Run `python scripts/prepare_epee.py --out data/epee_glosses.npz` with NumPy installed.
The initial run requires internet access and stores the downloaded files in
`.epee-cache/`. Subsequent runs can reuse that cache. The source contains
landmarks and annotations, with no raw video.

The script selects up to six clips per signer and gloss from eight signers for
`CAN`, `HAVE`, `NEED`, `WANT`, `WHAT`, `YOU`, and `YOUR`. It takes one
annotated gloss segment per source clip, using the longest segment if the
same gloss occurs more than once. Each segment becomes 16 uniformly sampled
frames. The output retains both hands, shoulders, elbows, and hips and
converts missing all-zero landmark vectors to NaN. It includes source clip IDs
and annotation timestamps so results can be traced to the input.

Gloss labels identify individual annotated segments cut out of signed phrases.
Some source prompts recur across signers, and some signers follow different
prompt sets. A split by signer tests generalization to unseen signers in this
subset. Prompt and signer variation can both affect the result. The benchmark
begins with known segment boundaries and addresses seven labels.

CLERC publishes these body landmarks with pseudonymous signer IDs. Body
geometry can still carry identifying information. Use the data for sign
recognition research and evaluation within its license. CLERC prohibits
identification, profiling, and surveillance. Carry these limitations forward
with any redistribution of derivatives.

## Bundled evaluation examples

The demo archive includes a deterministic subset of held-out landmark
sequences adapted from Épée v0.4. It includes 16-frame resampled segments with
hands and six body anchors and omits face landmarks and video. These adapted
samples are distributed under the source's CC BY-NC-SA 4.0 license. Model
weights are included under the same terms for this educational release.
Use the GitHub source to reproduce the evaluation from the pinned source.
