---
title: "DINO-GS-SLAM: A Unified Semantic SLAM with DINO-Embedded 3D Gaussian Splatting"
authors:
- Tian Lv
- Ruoyu Fan
- Ho-Jui Fang
- Yihan Li
- Xiaotong Zhou
- Yu-Hui Wen
- Yong-Jin Liu
highlight_authors:
- Ho-Jui Fang

publication_types: ["1"]
publication: "*The 14th International Conference on Image and Graphics*"
publication_short: "ICIG"

abstract: "We present DINO-GS-SLAM, a unified indoor SLAM system that integrates vision-foundation-model semantics into a 3D Gaussian Splatting (3DGS) representation. By embedding DINO features directly into a Gaussian scene model, our approach yields a unified representation that jointly supports camera tracking, high-quality novel view synthesis, and dense semantic feature rendering. To address the high dimensionality of DINO features under SLAM constraints, we propose a codebook-based encoding–decoding strategy that reconstructs a compact semantic feature field while preserving end-to-end differentiability. This representation enables multiple downstream tasks, including open-set semantic segmentation, segmentation tracking, and image captioning, without task-specific retraining. In addition, we incorporate a keyframe-guided diffusion-based refinement module to enhance the perceptual quality of rendered images. Extensive experiments on the TUM RGB-D and Replica datasets demonstrate that our system achieves reliable tracking accuracy across both RGB-D and monocular settings, competitive perceptual rendering quality, and strong open-set semantic segmentation performance, establishing DINO-GS-SLAM as a versatile platform for semantic scene understanding."

date: 2026-10-01
publishDate: 2026-10-01

featured: true

url_pdf: ""
url_code: ""
url_dataset: ""
url_project: ""
url_slides: ""
url_video: ""

# Optional external publication link
url_source: ""

# Optional DOI
doi: ""
---

We present DINO-GS-SLAM, a unified indoor SLAM system that integrates vision-foundation-model semantics into a 3D Gaussian Splatting (3DGS) representation. By embedding DINO features directly into a Gaussian scene model, our approach yields a unified representation that jointly supports camera tracking, high-quality novel view synthesis, and dense semantic feature rendering. Our system is built around three key designs.

1. Codebook-Based Feature Compression: to address the high dimensionality of DINO features under SLAM constraints, we propose a codebook-based encoding–decoding strategy that reconstructs a compact semantic feature field while preserving end-to-end differentiability.
2. Task-Agnostic Semantic Rendering: the unified representation enables multiple downstream tasks, including open-set semantic segmentation, segmentation tracking, and image captioning, without task-specific retraining.
3. Keyframe-Guided Diffusion Refinement: we incorporate a keyframe-guided diffusion-based refinement module to enhance the perceptual quality of rendered images.

Extensive experiments on the TUM RGB-D and Replica datasets demonstrate that our system achieves reliable tracking accuracy across both RGB-D and monocular settings, competitive perceptual rendering quality, and strong open-set semantic segmentation performance, establishing DINO-GS-SLAM as a versatile platform for semantic scene understanding.