# Changelog

## v0.7.1 (2026-09-08)

## What's Changed
* build: revert pip wheel-vendoring on dev (forward-port of #103) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/104
* feat(recognition): branch-scope the embedding register (SO_EDGE_BRANCH_CODE) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/105
* Promote dev → main: branch-scoped register + Dockerfile revert (→ 0.7.1) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/106


**Full Changelog**: https://github.com/lumiohub-ai/lum-model-face-recognition/compare/v0.7.0...v0.7.1

## v0.7.0 (2026-09-08)

## What's Changed
* fix(attendance): dedup IN/OUT records across camera-worker processes by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/100
* feat(workers): self-assigning camera slot routing (LSO-186) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/101
* Migration from single proccess to delery worked based pipeline by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/98
* Promote dev → main: LSO-67 Celery-split AI pipeline by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/102
* build: revert pip wheel-vendoring, restore pre-vendor network install by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/103


**Full Changelog**: https://github.com/lumiohub-ai/lum-model-face-recognition/compare/v0.6.1...v0.7.0

## v0.6.1 (2026-08-26)

## What's Changed
* chore(deps): bump lum-model-vision to v0.3.0 (LSO-137 locking fix) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/96
* chore(deps): bump lum-model-vision to v0.3.0 (LSO-137 GlobalTrackMana… by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/97


**Full Changelog**: https://github.com/lumiohub-ai/lum-model-face-recognition/compare/v0.6.0...v0.6.1

## v0.6.0 (2026-08-26)

## What's Changed
* Bench/celery face detection throughput by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/94
* fix(gpu): correlate GPU requests with their responses (LSO-138) by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/93
* Promote dev → main: GPU request/response correlation (LSO-138) + Celery face-detection benchmark by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/95


**Full Changelog**: https://github.com/lumiohub-ai/lum-model-face-recognition/compare/v0.5.0...v0.6.0

## v0.5.0 (2026-08-25)

## What's Changed
* LSO-100: Celery worker-per-model benchmark harness by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/87
* feat(action-recognition): skip crops too small for the VLM to read (LSO-68) by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/89
* fix(calibration): honor the undistort flag in CaptureFrame by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/88
* fix(detection): derive branch_id on every DetectionRepository write by @azamjon-xusanov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/91
* Promote dev → main: Celery benchmark (LSO-100), undistort fixes (LSO-26), action crop gating (LSO-68), branch_id on detection writes (LSO-145) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/92


**Full Changelog**: https://github.com/lumiohub-ai/lum-model-face-recognition/compare/v0.4.0...v0.5.0

## v0.4.0 (2026-08-24)

## What's Changed
* Docs/lumiohub docs submodule bump by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/71
* Key cameras by DB id instead of list position (LSO-130 phase 1) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/83
* Scope AI camera loading to one branch (LSO-133) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/84
* Promote dev → main: LSO-130 phase 1 + LSO-133 branch scoping by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/85


**Full Changelog**: https://github.com/lumiohub-ai/lum-model-face-recognition/compare/v0.3.1...v0.4.0

## v0.3.1 (2026-08-22)

## What's Changed
* Add process/pipeline/decode visibility to metrics by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/76
* Batch ArcFace embedding across all ROIs in a cycle by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/77
* Embed one face per ArcFace call, not one batch per cycle (LSO-117) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/81
* Fix ArcFace embedding perf regression shipped in 0.3.0 (LSO-117) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/82


**Full Changelog**: https://github.com/lumiohub-ai/lum-model-face-recognition/compare/v0.3.0...v0.3.1

## v0.3.0 (2026-08-20)

## What's Changed
* Add process/pipeline/decode visibility to metrics (LSO-115) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/79
* Batch ArcFace embedding across all ROIs in a cycle (LSO-117) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/80


**Full Changelog**: https://github.com/lumiohub-ai/lum-model-face-recognition/compare/v0.2.3...v0.3.0

## v0.2.3 (2026-08-19)

## What's Changed
* refactor(compose): pull lumiohub-model image instead of building (LSO-49) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/55
* release: promote dev → main (AI compose pull) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/56
* refactor(models)!: extract ML models into lum-model-vision, own repo by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/57
* fix: reload listeners never subscribe (set _running before starting them) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/58
* docs: add lumiohub-docs product docs as a submodule at docs/lumiohub-docs by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/59
* Refactor and extract ML models to lum-model-vision package by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/60
* Doc/modals by @inokov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/61
* Add Claude Code GitHub Workflow by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/63
* feat: frontality/pitch gate for unrecognized cases (LSO-7) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/62
* fix: pick best gate-passing frame for unrecognized cases (LSO-7) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/65
* Release dev → main: LSO-7 unrecognized-case gate + frame undistortion + model docs by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/64
* fix(docker): install cuDNN >=9.8 + put it on LD_LIBRARY_PATH for onnx… by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/67
* fix(docker): install cuDNN >=9.8 + put it on LD_LIBRARY_PATH for onnx… by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/68
* Fix/lso 13 ignore inactive users by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/69
* fix(recognition): ignore inactive users' embeddings (LSO-13) (#69) by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/70
* Add 4-step version-file release pipeline by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/72
* Add 4-step version-file release pipeline by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/73
* Fix ActionRecognitionWorker never getting its AsyncLogger wired in by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/74
* fix(pipeline): wire AsyncLogger into ActionRecognitionWorker so activ… by @dilshod-obidov in https://github.com/lumiohub-ai/lum-model-face-recognition/pull/75


**Full Changelog**: https://github.com/lumiohub-ai/lum-model-face-recognition/compare/v0.1.0...v0.2.3
