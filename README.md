# CALCAM
 
Official repository associated with the manuscript:

## CALCAM: Cross Attention based on Linear Central Alignment for Multimodal Learning

CALCAM is a multimodal fusion framework that uses Centered Kernel Alignment (CKA) to identify the most aligned modality pair and guide targeted cross-attention.

## Supplementary Material

Additional details on the datasets, experimental protocol, CKA analysis, and experimental results are available in:

[CALCAM ESANN 2027 - Supplementary Material](CALCAM_ESANN2027_Supplementary_Material.pdf)

## Datasets

CALCAM is evaluated on three multimodal datasets:

• **C-MAPSS** — industrial predictive maintenance

• **Fracture** — multimodal medical diagnosis

• **CMU-MOSEI** — multimodal sentiment analysis

The datasets are not redistributed in this repository.

## Linear CKA Analysis
The repository provides the Linear Centered Kernel Alignment (CKA) implementation used to determine the modality pair selected by CALCAM.

CKA implementation:

`src/cka.py`

Experimental script:

`experiments/run_cka.py`


### Expected embedding files

The CKA experiments require precomputed modality representations.

**C-MAPSS**

```text
Z1_tr.npy
Z1_va.npy
Z2_tr.npy
Z2_va.npy
Z3_tr.npy
Z3_va.npy
```

**Fracture**

```text
Zimg_train.npy
Zimg_val.npy
Zimg_test.npy

Ztxt_train.npy
Ztxt_val.npy
Ztxt_test.npy

Ztab_train.npy
Ztab_val.npy
Ztab_test.npy
```

**CMU-MOSEI**

```text
X_visual_openface.npy
X_audio_covarep.npy
X_text_wordvec.npy
sentiment.npy
```

The embedding files are not included in this repository.

### Running the CKA analysis

Run the scripts from the repository root.

**C-MAPSS**

```bash
python experiments/run_cka.py --dataset cmapss --data-dir data/embeddings/cmapss
```

**Fracture**

```bash
python experiments/run_cka.py --dataset fracture --data-dir data/embeddings/fracture
```

**CMU-MOSEI**

```bash
python experiments/run_cka.py --dataset cmu-mosei --data-dir data/embeddings/cmu_mosei
```

### CKA modality selection

| Dataset | Selected pair | Linear CKA |
|---|---|---:|
| C-MAPSS | M2--M3 | 0.943 |
| Fracture | Image--Tabular | 0.025 |
| CMU-MOSEI | Visual--Text | 0.048 |

The modality pair with the highest Linear CKA score is selected for cross-attention in CALCAM.

## CALCAM Implementation

The generic CALCAM architecture is available in:

`src/calcam.py`

A minimal forward-pass test is available in:

`experiments/test_calcam.py`

Dataset-specific training and evaluation scripts are provided for:

- **C-MAPSS** — `experiments/run_calcam_cmapss.py`
- **Fracture** — `experiments/run_calcam_fracture.py`
- **CMU-MOSEI** — `experiments/run_calcam_mosei.py`

CALCAM applies bidirectional cross-attention to the modality pair selected by Linear CKA, processes the remaining modality independently, and integrates the resulting representations through a shared Transformer.


## Installation

```bash
pip install -r requirements.txt
```
## Running CALCAM

The experiments assume that the precomputed modality embeddings are available locally as described above.

### C-MAPSS

```bash
python experiments/run_calcam_cmapss.py \
    --data-dir data/embeddings/cmapss
```

### Fracture

```bash
python experiments/run_calcam_fracture.py \
    --data-dir data/embeddings/fracture
```

### CMU-MOSEI

```bash
python experiments/run_calcam_mosei.py \
    --data-dir data/embeddings/cmu_mosei
```

### Architecture test

A minimal forward-pass test can be executed with:

```bash
python experiments/test_calcam.py
```
