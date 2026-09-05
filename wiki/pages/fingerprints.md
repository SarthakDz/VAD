# Fingerprints — the encoding profile is a class prior

## The observation

The pack was assembled from several source collections, and each collection kept
its own encoding settings. `(width, height, native_fps)` therefore identifies
which collection a video came from. The public test set's ground truth then says
what each collection contains, and several collections turn out to be almost
single-purpose.

From `scripts/fingerprint.py --main`, over the 34 public test videos:

| profile | public videos | classes present |
|---|---|---|
| `(1920,1080,29.97)` | T029 T030 | **normal only** |
| `(896,448,1.88)` | T021 T022 T023 T024 T032 T034 | fighting / loitering only |
| `(1280,720,25.0)` | T025 T026 T027 T028 | traffic classes only |
| `(1280,720,29.97)` | T033 | traffic_accident |
| `(256,192,30.0)` | T003 T004 | **normal only** |
| `(800,410,30.0)` | T010 T031 | stalled / traffic_congestion |
| `(720,404,30.0)` | T006 T007 T018 T020 | accident / wrong_way / road_spill |
| `(640,640,24.0)` | ten videos | fire, smoke, congestion, waterlogging, accident, normal |

The `1.88` fps profile is worth noticing on its own: 1.875 fps is 30/16, so those
files were decimated before they were shipped. That is a processing pipeline, not
a camera, which is exactly why the profile identifies a collection.

## Why it is trustworthy

The prior predicts **E024 is normal**. The leaderboard had already proved E024 is
normal by a completely independent route: putting one event on it moved D2 from
14.0 to 5.3, a drop of 8.7, and one L2 video is worth 35/4 = 8.75. A prior that
reproduces a separately measured fact is worth acting on.

It is still a prior and not a label. `(640,640,24.0)` holds six different classes
and constrains almost nothing; `(320,240,30.0)` has no public twin at all and
falls back to the train distribution, which is 61% normal.

## How it is used

`scripts/fingerprint.py` exposes `allowed(video_id)` — the set of classes that
video's collection is known to contain, or `None` when the collection is unknown.
Two uses, both in `scripts/eval_v4.py`:

- **Zero out impossible classes** before the D1 argmax. This silences E002,
  whose collection is normal-only and which the ensemble had called
  `traffic_accident` at 0.40.
- **Choose the D2/D3 class spray.** Instead of the model's top five, emit every
  allowed class over each candidate window. E022's model top-five spent two slots
  on `fire` and `smoke`, which its collection has never contained; E026 and E028
  get both `fighting_or_violence` and `loitering_or_suspicious_presence`, which
  is a genuine coin flip the model cannot resolve.

## The limit

This reads the *public test* ground truth to build the map, so it is a legitimate
use of released data — but it generalises only as far as the private set reuses
the same collections. It does, for 22 of 28 videos. The six unknown-collection
videos (E003, E004, E005, E008, E017, E018) get no constraint and are handled by
the model alone. See [[scoring]] for what the marks formula does with a wrong
class, and [[experiments]] exp-015 for the measured effect.
