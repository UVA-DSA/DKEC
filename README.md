## DKEC: Domain Knowledge Enhanced Multi-Label Classification for Diagnosis Prediction
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


## Presentation
One presentation ppt is available from [here](https://docs.google.com/presentation/d/1UDghDmYYrFjqUCDl9Q_15gfOCsv00Yur/edit#slide=id.p1)

## Dataset
* EMS dataset
  - EMS dataset is not publicly available.
* MIMIC-III dataset
  - MIMIC-III dataset is publicly available. Refer to [this page](https://physionet.org/content/mimiciii/1.4/) to apply online.
  - The created adjacent matrix for ICD-9 codes are stored in corresponding MIMIC-III data folder
    - **SYMPTOM**: icd9code2symptom.json, symptom2icd9code.json
    - **TREATMENT**: icd9code2treatment.json, treatment2icd9.json
* Web Annotation
  -  To evaluate the accuracy of different methods for constructing knowledge graphs, we evenly sampled
50 codes from head, middle, and tail classes and manually annotated symptoms and treatments from
Wikipedia and Mayo Clinic website contents for ICD-9 diagnosis codes. For EMS protocols, we manually annotated all 43 protocols in ODEMSA documents.  


## Environment
* run with `conda env create --file=environment.yaml`
* A pre-built docker image is available in docker hub [repo](https://hub.docker.com/repository/docker/masqueraderx/emnlp_2023/general),


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

### Generate pre-trained embedding 
- First specify config files well for every backbone
- run `python Heterogeneous_graph.py` to generate embedding for different backbones

### Config
This section specifies some parameters that can be changed in config file
  - train
    - dataset: MIMIC3-3737 or MIMIC3-1000 or MIMIC3-6668
    - root_dir: the root the DKEC directory
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
- Local machine: `python main.py ./config/whichname.json`

### Citation
**[DKEC: Domain knowledge enhanced multi-label classification for electronic health records](https://arxiv.org/pdf/2310.07059)**
```
@article{ge2023dkec,
  title={Dkec: Domain knowledge enhanced multi-label classification for electronic health records},
  author={Ge, Xueren and Williams, Ronald Dean and Stankovic, John A and Alemzadeh, Homa},
  journal={arXiv preprint arXiv:2310.07059},
  year={2023}
}
```