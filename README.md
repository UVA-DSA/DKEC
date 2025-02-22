# DKEC: Domain Knowledge Enhanced Multi-Label Classification for Diagnosis Prediction

code repository for EMNLP 2024 main conference paper [DKEC: Domain Knowledge Enhanced Multi-Label Classification for Diagnosis Prediction](https://arxiv.org/pdf/2310.07059)

## Introduction
Multi-label text classification (MLTC) tasks in the medical domain often face the long-tail label distribution problem. 
Prior works have explored hierarchical label structures to find relevant information for few-shot classes, 
but mostly neglected to incorporate external knowledge from medical guidelines. 
This paper presents DKEC, **D**omain **K**nowledge **E**nhanced **C**lassification for diagnosis prediction 
with two innovations: 
- (1) automated construction of heterogeneous knowledge graphs from external 
sources to capture semantic relations among diverse medical entities.

![Architecture](<figure/KG.png>)

- (2) incorporating the heterogeneous knowledge graphs in few-shot classification using a label-wise attention mechanism.

![Architecture](<figure/Pipeline.png>)


[//]: # (## Presentation)

[//]: # (One presentation ppt is available from [here]&#40;https://docs.google.com/presentation/d/1UDghDmYYrFjqUCDl9Q_15gfOCsv00Yur/edit#slide=id.p1&#41;)

## Dataset
* EMS dataset
  - The EMS dataset is restrictedly available due to patient privacy concerns.
* MIMIC-III dataset
  - MIMIC-III dataset is publicly available. Refer to [this page](https://physionet.org/content/mimiciii/1.4/) to apply online.
  - The created adjacent matrix for ICD-9 codes are stored in corresponding [MIMIC-III data folder](https://github.com/UVA-DSA/DKEC/tree/main/dataset)
    - **SYMPTOM**: icd9code2symptom.json, symptom2icd9code.json
    - **TREATMENT**: icd9code2treatment.json, treatment2icd9.json
* Web Annotation
  -  To evaluate the accuracy of different methods for constructing knowledge graphs, we evenly sampled
50 codes from head, middle, and tail classes and manually annotated symptoms and treatments from
Wikipedia and Mayo Clinic website contents for ICD-9 diagnosis codes. For EMS protocols, we manually annotated all 43 protocols in ODEMSA documents.  


## Environment
Run the following commands to get an anaconda environment `DKEC`
```
chmod +x install.sh
./install.sh
```
[//]: # (* A pre-built docker image is available in docker hub [repo]&#40;https://hub.docker.com/repository/docker/masqueraderx/emnlp_2023/general&#41;,)


[//]: # (* Creating docker images:)

[//]: # (Rivanna has their own "pre-built" docker images in their [git repo]&#40;https://github.com/uvarc/rivanna-docker&#41;)

[//]: # (you can download one of them according to [link]&#40;https://www.rc.virginia.edu/userinfo/howtos/rivanna/docker-images-on-rivanna/&#41;. )

[//]: # (I used pytorch 1.12.0 which already had cuda and pytorch installed.)

[//]: # ()
[//]: # (* Install all dependencies: )

[//]: # (The basic command is **singularity exec <container.sif> python -m pip install --user <package>**)

[//]: # (Check the [link]&#40;https://www.rc.virginia.edu/userinfo/howtos/rivanna/add-packages-to-container/&#41; for more details.)

[//]: # (it has installed CUDA, pytorch and all dependencies for this work.)

[//]: # ()
[//]: # (* Run with bash file:)

[//]: # (See an example in **run.slurm**, more details can be seen from this [link]&#40;https://www.rc.virginia.edu/userinfo/rivanna/slurm/&#41;)

[//]: # (Run with `sbatch run.slurm*`)

[//]: # ()
[//]: # (* request a node)

[//]: # (ijob -A uva-dsa -w udc-an34-1 -p gpu --gres=gpu -c 8 -t 01:00:00)

## Steps to run the code

### Generate train / val / test:
- download code from [CAML](https://github.com/jamesmullenbach/caml-mimic) and run notebook `dataproc_mimic_III.ipynb`, you need to download pre-trained embeddings
`BioWordVec_PubMed_MIMICIII_d200.vec.bin` from [link](https://github.com/ncbi-nlp/BioSentVec).
- run `mimic_iii_6668.ipynb`, `mimic_iii_3737.ipynb` and `mimic_iii_1000.ipynb` in sequence.
  - You need specify the root for CAML code and BioWordVec_PubMed_MIMICIII_d200.vec.bin in the `mimic_iii_6668.ipynb`

### Generate pre-trained embedding
- First specify config files well for every backbone
- run `python Heterogeneous_graph.py config/whichname.json` to generate embedding for different backbones
  - We suggest to change `dataset` to `MIMIC3-6668` since it will generate the initial node embedding for all 6668 ICD-9 codes.

### Config
This section specifies some parameters that can be changed in config file
  - train
    - dataset: MIMIC3-3737 or MIMIC3-1000 or MIMIC3-6668
    - root_dir: the absolute path the DKEC directory
    - topk: 8 (MIMIC3-3737); 6 (MIMIC3-1000); 12 (MIMIC3-6668)
    - seed: 3407 or 1234 or 42 or 0 or 1 
  - test
    - epoch: you need to select the model of the epoch has the best performance
    - seed: change the seed based on the seed set in train
    - is_test: True when testing, False when training
  - wandb
    - enable: True if you use wandb to check training curves
    - entity: your wandb account name

### Slurm
This section specifies the terminal commands
- Cluster: you can run the project with slurm, go to the slurm folder and run with `sbatch whichname.slurm`
- Local machine: `python main.py config/whichname.json`

### Reproduce Experimental Results
The following tables specify how to reproduce main experimental results in Table 4 by using slurm.
You can also find corresponding json file in config to run on local machine.
For ISD, we directly use their github code. 

| Model          |               Slurm script or URL               |
|:----: |:-----------------------------------------------:|
| CAML           |               `sbatch CAML.slurm`               |
 | ZAGCNN         |              `sbatch ZAGCNN.slurm`              |
 | MultiResCNN    |           `sbatch MultiResCNN.slurm`            |
 | ISD            |  https://github.com/tongzhou21/ISD/tree/master  |
 | DKEC-M-CNN     |             `sbatch DKEC_CNN.slurm`             |
 | DKEC-GatirTron |           `sbatch DKEC_GatorTron.slurm`           |

### Citation
If you find this work helpful, please cite,
```
@inproceedings{ge-etal-2024-dkec,
    title = "{DKEC}: Domain Knowledge Enhanced Multi-Label Classification for Diagnosis Prediction",
    author = "Ge, Xueren  and
      Satpathy, Abhishek  and
      Williams, Ronald Dean  and
      Stankovic, John  and
      Alemzadeh, Homa",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.712",
    pages = "12798--12813",
    abstract = "Multi-label text classification (MLTC) tasks in the medical domain often face the long-tail label distribution problem. Prior works have explored hierarchical label structures to find relevant information for few-shot classes, but mostly neglected to incorporate external knowledge from medical guidelines. This paper presents DKEC, Domain Knowledge Enhanced Classification for diagnosis prediction with two innovations: (1) automated construction of heterogeneous knowledge graphs from external sources to capture semantic relations among diverse medical entities, (2) incorporating the heterogeneous knowledge graphs in few-shot classification using a label-wise attention mechanism. We construct DKEC using three online medical knowledge sources and evaluate it on a real-world Emergency Medical Services (EMS) dataset and a public electronic health record (EHR) dataset. Results show that DKEC outperforms the state-of-the-art label-wise attention networks and transformer models of different sizes, particularly for the few-shot classes. More importantly, it helps the smaller language models achieve comparable performance to large language models.",
}
```