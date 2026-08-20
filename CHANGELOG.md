# Changelog

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


