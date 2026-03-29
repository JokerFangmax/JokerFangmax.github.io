---
title: "A Text-to-3D Framework for Joint Generation of CG-Ready Humans and Compatible Garments"
authors:
- Zhiyao Sun
- Yu-Hui Wen
- Ho-Jui Fang
- Sheng Ye
- Matthieu Lin
- Tian Lv
- Yong-Jin Liu
highlight_authors:
- Ho-Jui Fang

publication_types: ["2"]
publication: "*IEEE Transactions on Visualization and Computer Graphics*"
publication_short: "TVCG"

abstract: "An Integrated Text-Driven CG-Ready Human and Garment Generation System."

date: 2026-01-01
publishDate: 2026-01-01

featured: true

url_pdf: "https://arxiv.org/pdf/2503.12052"
url_code: "https://human-tailor.github.io/"
url_dataset: ""
url_project: "https://human-tailor.github.io/"
url_slides: ""
url_video: ""

# Optional external publication link
url_source: ""

# Optional DOI
doi: "10.48550/arXiv.2503.12052"
---

Creating detailed 3D human avatars with fitted garments traditionally requires specialized expertise and labor-intensive workflows. While recent advances in generative AI have enabled text-to-3D human and clothing synthesis, existing methods fall short in offering accessible, integrated pipelines for generating CG-ready 3D avatars with physically compatible outfits; here we use the term CG-ready for models following a technical aesthetic common in computer graphics (CG) and adopt standard CG polygonal meshes and strands representations (rather than neural representations like NeRF and 3DGS) that can be directly integrated into conventional CG pipelines and support downstream tasks such as physical simulation. 

To bridge this gap, we introduce Tailor, an integrated text-to-3D framework that generates high-fidelity, customizable 3D avatars dressed in simulation-ready garments. Tailor consists of three stages. 
1. Semantic Parsing: we employ a large language model to interpret textual descriptions and translate them into parameterized human avatars and semantically matched garment templates. 
2. Geometry-Aware Garment Generation: we propose topology-preserving deformation with novel geometric losses to generate body-aligned garments under text control. 
3. Consistent Texture Synthesis: we propose a novel multi-view diffusion process optimized for garment texturing, which enforces view consistency, preserves photorealistic details, and optionally supports symmetric texture generation common in garments. 

Through comprehensive quantitative and qualitative evaluations, we demonstrate that Tailor outperforms state-of-the-art methods in fidelity, usability, and diversity.