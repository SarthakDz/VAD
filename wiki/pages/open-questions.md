# Open questions and blocked items

Things that cannot be resolved from inside a session. Questions are written in
full prose so they can be pasted to the organisers directly.

## Blocking

**1. Arena site URL and login.**
> "What's the arena site URL, and can you confirm which Google account is
> registered for me?"

Not present in either PDF. A deep scan of both files — plain text, link
annotations, and decompressed streams — found zero URLs in the submission format
document. Without this, nothing can be submitted at all.

**2. `manifest.json` shape.**
> "Can you share a sample `manifest.json` now, or describe its fields? I want to
> test my submission writer against the real shape before the first run."

`src/submit.py::load_manifest` accepts four plausible encodings (list of objects,
`{videos:[...]}`, `{predictions:[...]}`, flat `{id: level}`) and fails loudly
otherwise. It has never seen the real file. A wrong level parse means every event
for that video is rejected. Save the real file to `data/manifest.json`.

**3. Run limit.**
> "Is there a cap on the number of benchmark runs per team?"

"Each upload costs one run" implies a budget. With no best-of, the run count
decides how aggressively thresholds can be tuned against the live scorer.

## Scoring — changes tuning directly

**4. Level 2/3 component weights.** Highest-value question after the URL.
> "For Levels 2 and 3, how are the alert / matched / timing components weighted?
> The format doc says timing weighs more at Level 3 but doesn't give numbers."

Currently assumed L2 `(0.2, 0.5, 0.3)` and L3 `(0.2, 0.4, 0.4)`. See [[scoring]].

**5. Cross-level aggregation.**
> "Are Level 1, 2 and 3 combined into a single ranking number? If so, weighted
> how — equally, or by video count?"

The public split is 24/6/4. Equal weight per level makes the 4 Level-3 videos
worth as much as 24 Level-1 ones, which decides where the afternoon goes.

**6. Level-1 multi-event handling.**
> "On a Level-1 video, if I emit multiple events with different class names, how
> is class accuracy computed — first event, best match, or is any mismatch
> penalised?"

The PDF says repeating a class earns nothing extra, but is silent on *different*
classes. If any-match counts, emitting a top-2 shortlist is free score. Current
code emits exactly one event at Level 1.

**7. Latency bonus weighting.**
> "How is the latency bonus weighted against accuracy, and what ratio counts as
> good? Is it reported-time divided by video-duration, lower being better?"

Currently measuring 0.0242. Decides whether to push VLM coverage below 5% or
spend the compute on accuracy instead.

**8. Private set composition.**
> "Does the private set follow the same level proportions as the public 24/6/4,
> and is it also 28 videos?"

## Data

**9. Train schema.**
> "`train/<class>/ground_truth.csv` has no `level` column, but
> `test/ground_truth.csv` does. Intentional?"

Confirms the two-loader design in `io_dataset.py` is right and not compensating
for a corrupt download.

**10. Label noise.**
> "Some `fighting_or_violence` rows have descriptions of non-fighting events —
> one describes a truck rollover spilling cargo, another an SUV hitting a tree.
> Is there known label noise, and should we trust the folder label or the
> description?"

Examples are in [[dataset-audit]].

**11. Is `explanation` scored on content?**
> "Is `explanation` scored on content quality, or only on presence and length? If
> content, judged by what — a model, or human review?"

`description_summary` in train is heavily templated — one unique string for all
300 loitering videos. If content is scored, that training signal is nearly
worthless and better text should be generated instead. If presence only, emit
templates and spend the time on localisation.

## Rules

**12. Runtime model size.**
> "To confirm the runtime constraint: a locally-run 4B-parameter VLM is
> acceptable, correct? Is there a hard parameter or VRAM ceiling?"

**13. Pre-trained weights and licences.**
> "Are pre-trained public weights allowed at runtime — SigLIP as a frozen
> encoder, Qwen3-VL as the reasoning stage? Any licence restrictions we should
> avoid, e.g. AGPL?"

The entire Stage A is a frozen public encoder, so this matters.

**14. May we train on the public test set?**
> "May we train or tune on the public test set, or must it stay held out?"

34 labelled videos including 10 long multi-event ones. That is the **only**
long-form supervised data in existence for this task — the train split has none
(see [[dataset-audit]]). Materially changes what M2 can do.

## Manual tasks for the user

Not questions, but things no session can do:

- Kaggle account phone-verified — GPU stays locked until then, needed for M4
- Pull `Qwen/Qwen3-VL-4B-Instruct` weights to local disk before they are needed
- All arena uploads. **Never upload without the local scorer passing first** —
  there is no best-of and a worse run permanently replaces a better score
- The 2-slide PPT and architecture write-up at the end
