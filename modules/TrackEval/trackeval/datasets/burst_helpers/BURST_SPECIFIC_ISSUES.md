---
noteId: "3b6c59f3af8211ee96778d7d2a8d6094"
tags: []

---

The track ids in both ground truth and predictions are not globally unique, but
start from 1 for each video. At the moment when converting from Ali format to
TAO format, we remap the ids to be globally unique. It would be better to
directly have this in the data though.


Improve setting of EXEMPLAR_GUIDED flag, maybe this can be done automatically.
