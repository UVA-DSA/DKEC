## SBATCH
to request how much memory you want for a100 machine
#SBATCH --constraint=a100_80gb

request a node foe debug
ijob -A uva-dsa -p gpu --gres=gpu:a6000:4 -c 16 -t 70:00:00

check the long slurm name
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me

## Heterogeneous_graph.py
- bertEmbedding: 
- HeteroGraph: GNN architectures that contains knowledge information from protocol information


## data_utils.py
- EMSDataPipeline: wrapper class to tokenize narratives and create data loader for training/testing
    - .build_single_dataloader() return a dataloader that streams samples for training/testing
- EMSDataset: wrapper class that tokenizes narrative and meta knowledge text. Used in EMSDataPipeline to create a data_loader object.


## model.py
- HGT (Heterogeneous Graph Transformer) that operates on graph object passed as argument
- EMSMultiModel: This class integrates the label-wise attention with the BERT model. Below I have summarized the initialization fields (to the best of my understanding):
    -   Backbone: backbone for BERT model to process narrative text
    -   dp: dropout (not used anymore)
    -   max_len: max sentence length (not used anymore, I think)
    -   attn: specifies what type of attention to use in knowledge fusion
        - 'qkv-la': label-wise attention qkv
        - 'la': label-wise attention
        - 'sa': self-attention
    -   cluster: whether the model is outputting grouped or ungrouped protocols
    -   cls: whether or not the output of fusion through a fully connected layer (does this stand for classifier?)
    -   graph: the GNN architecture (such as Heterograph) used for knowledge

- classifier: MLP with 3 hidden layers. I believe this inputs the embeddings after they have been attended to


## loss_fn.py
- SimCLR
- ResampleLoss
- FocalLoss


## eval_metrics.py
- evaluate model performance and generate report json file


## bad_case_analysis.py
- an interface for evaluate bad cases

## psuedo_label_generator.py
- dataAug
- knowledgeModel
- EMSBertModel
- comb_pipeline
- dm_pipeline


## counterfactual_reasoning.py
- Doing counterfactual reasoning

## trainer
- trainer
- tester
- validator
- pseudolabeler


## visualize
- Desription: a bunch of graphing code


## TF-IDF.ipynb
- generate a vector representation for protocols by calculate TF-IDF of signs & symptoms 

## Metamap Extractor.ipynb
-  extract signs & symptoms by using Metamap

## GT-Label Extraction.ipynb
- extract gt labels from RAA data
- split into 3-folds train-val-test

## Model Explanation.ipynb
- explain model's prediction by using Integrated Gradient

## N-gram for Text Classification.ipynb
- use one-gram as features and train machine learning models
- used as baseline