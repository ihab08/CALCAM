CALCAM

Official repository associated with the manuscript:

CALCAM: Cross Attention based on Linear Central Alignment for Multimodal Learning

CALCAM is a multimodal fusion framework that uses Centered Kernel Alignment (CKA) to identify the most aligned modality pair and guide targeted cross-attention.

Supplementary Material

Additional details on the datasets, experimental protocol, CKA analysis, and experimental results are available in:

[CALCAM ESANN 2027 - Supplementary Material](CALCAM_ESANN2027_Supplementary_Material.pdf)

Datasets

CALCAM is evaluated on three multimodal datasets:

•	C-MAPSS — industrial predictive maintenance

•	Fracture — multimodal medical diagnosis

•	CMU-MOSEI — multimodal sentiment analysis

The datasets are not redistributed in this repository.


Linear CKA Analysis
The repository provides the Linear Centered Kernel Alignment (CKA) implementation used to determine the modality pair selected by CALCAM.
The implementation is available in:
src/cka.py
The experimental script for the three datasets is available in:
experiments/run_cka.py
Expected embedding structure
The CKA experiment expects precomputed modality representations with the following local structure:
data/
└── embeddings/
    ├── cmapss/
    │   ├── Z1_tr.npy
    │   ├── Z1_va.npy
    │   ├── Z2_tr.npy
    │   ├── Z2_va.npy
    │   ├── Z3_tr.npy
    │   └── Z3_va.npy
    │
    ├── fracture/
    │   ├── Zimg_train.npy
    │   ├── Zimg_val.npy
    │   ├── Zimg_test.npy
    │   ├── Ztxt_train.npy
    │   ├── Ztxt_val.npy
    │   ├── Ztxt_test.npy
    │   ├── Ztab_train.npy
    │   ├── Ztab_val.npy
    │   └── Ztab_test.npy
    │
    └── cmu_mosei/
        ├── X_visual_openface.npy
        ├── X_audio_covarep.npy
        └── X_text_wordvec.npy
