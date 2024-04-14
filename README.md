# CS598 Deep Learning for Healthcare Final Project
In this project, we aim to replicate the study in the paper [MoleculeSTM: Multi-modal Molecule Structure-text Model for Text-based Editing and Retrieval](https://www.nature.com/articles/s42256-023-00759-6), focusing on the multi-modal molecule structure–text model (MoleculeSTM) which 
integrates molecular chemical structures and textual descriptions through a contrastive learning strategy. The original repository for the paper is at [MoleculeSTM repo](chao1224.github.io/MoleculeSTM).

## Environment Setup
```
!pip install rdkit
!pip install torch torchvision
!pip install requests tqdm matplotlib spacy Levenshtein boto3 deepspeed
!pip install ogb==1.2.0
!pip install transformers==4.30.2

!pip install torch_geometric
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
!pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
!pip install git+https://github.com/MolecularAI/pysmilesutils.git

# install metagron
git clone https://github.com/MolecularAI/MolBART.git --branch megatron-molbart-with-zinc
cd MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism
pip install .
cd ../../..

# install apex
git clone https://github.com/chao1224/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

## Dataset
We can use the following python script to download the pretraining dataset and downstream datasets.
```
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
snapshot_download(repo_id="chao1224/MoleculeSTM", repo_type="dataset", local_dir='/content/drive/MyDrive/MoleculeSTM/data')
```
The data folder will include:
```
data
├── PubChemSTM_data/
│   └── raw
│        └── CID2SMILES.csv
│        └── CID2name.json
│        └── CID2name_raw.json
│        └── molecules.sdf
│   └── processed/
├── pretrained_SciBERT/
├── pretrained_MegaMolBART/
├── pretrained_KV-PLM/
├── pretrained_GraphMVP/
├── pretrained_MoleculeSTM_Raw/
├── pretrained_MoleculeSTM/
├── DrugBank_data/
├── ZINC250K_data/
├── Editing_data/
│   └── single_multi_property_SMILES.txt
│   └── neighbor2drug/
│   └── ChEMBL_data/
└── MoleculeNet_data/
```
The preprocessing code can be found at preprocessing/PubchemSTM folder.

## Checkpoints
### SciBERT
This can be done by calling the following for SciBERT:
```
SciBERT_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
SciBERT_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(device)
```
### MegaMolBART
Run download_MegaMolBART.sh, and the output structure is like:
```
├── bart_vocab.txt
└── checkpoints
    ├── iter_0134000
    │   ├── mp_rank_00
    │   │   └── model_optim_rng.pt
    │   ├── mp_rank_00_model_states.pt
    │   ├── zero_pp_rank_0_mp_rank_00optim_states.pt
    │   ├── zero_pp_rank_1_mp_rank_00optim_states.pt
    │   ├── zero_pp_rank_2_mp_rank_00optim_states.pt
    │   ├── zero_pp_rank_3_mp_rank_00optim_states.pt
    │   ├── zero_pp_rank_4_mp_rank_00optim_states.pt
    │   ├── zero_pp_rank_5_mp_rank_00optim_states.pt
    │   ├── zero_pp_rank_6_mp_rank_00optim_states.pt
    │   └── zero_pp_rank_7_mp_rank_00optim_states.pt
    └── latest_checkpointed_iteration.txt
```
### GNN and GraphMVP
For GraphMVP, check this [repo](https://github.com/chao1224/GraphMVP), and the checkpoints on [Google Drive link](https://drive.google.com/drive/u/1/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6).

### Baseline KV-PLM
For KV-PLM, check this [repo](https://github.com/thunlp/KV-PLM) and checkpoints on [Google Drive link](https://drive.google.com/drive/folders/1xig3-3JG63kR-Xqj1b9wkPEdxtfD_4IX).

### Checkpoints for MeleculeSTM
It can be downloaded by using the following python script:
```
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
snapshot_download(repo_id="chao1224/MoleculeSTM", repo_type="model", cache_dir='.')
```
### Citation
```
@article{liu2023moleculestm,
    title={Multi-modal molecule structure-text model for text-based retrieval and editing},
    author={Liu, Shengchao and Nie, Weili and Wang, Chengpeng and Lu, Jiarui and Qiao, Zhuoran and Liu, Ling and Tang, Jian and Xiao, Chaowei and Anandkumar, Anima},
    title={Multi-modal molecule structure--text model for text-based retrieval and editing},
    journal={Nature Machine Intelligence},
    year={2023},
    month={Dec},
    day={01},
    volume={5},
    number={12},
    pages={1447-1457},
    issn={2522-5839},
    doi={10.1038/s42256-023-00759-6},
    url={https://doi.org/10.1038/s42256-023-00759-6}
}
```

