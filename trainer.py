import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from data_utils import EMSDataPipeline, MIMIC3DataPipeline
from eval_metrics import log_metrics
from loss_fn import ClassificationLoss
from model import pick_model
from optimizer import ems_optimizer
from train_one_epoch import train_fn, eval_fn
from Heterogeneous_graph import HeteroGraph, HierarchyGraph, SematicGraph, CooccurGraph, HeteroGraphwoHier
from default_sets import *
from utils import cnt_instance_per_label, reduce_tensor, fpSixteen, bfSixteen
from logger import setup_wandb, log_dict_to_wandb, is_main_process
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.megatron_bert.modeling_megatron_bert import MegatronBertLayer
from vocab import WordVocab
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
import torch.distributed._shard.checkpoint as dist_cp
import functools
from torch.distributed import init_process_group, destroy_process_group
import psutil

from pkg_resources import packaging
import torch.cuda.nccl as nccl


def ddp_setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)

def bfloat_support():
    return (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

def get_policies(cfg, rank):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.FSDP.mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not cfg.FSDP.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.FSDP.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )
    else:
        mixed_precision_policy = None

    if cfg.train.backbone == 'stanford-crfm/BioMedLM':
        wrapping_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                GPT2Block,
            },
        )
    elif cfg.train.backbone in ['UFNLP/gatortron-large', 'UFNLP/gatortron-medium', 'UFNLP/gatortron-base']:
        wrapping_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                MegatronBertLayer,
            },
        )
    elif cfg.train.backbone == 'nlpie/tiny-clinicalbert':
        wrapping_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                BertLayer,
            },
        )

    return mixed_precision_policy, wrapping_policy

class Trainer:
    def __init__(self, config, model, graph, data, optimizer, scheduler, loss_fn, gpu_id, save_every, path):
        self.config = config
        self.gpu_id = gpu_id

        if graph:
            if isinstance(graph, list):
                self.graph = [g.to(self.gpu_id) for g in graph]
            else:
                self.graph = graph.to(self.gpu_id)
        else:
            self.graph = None
        self.train_data = data['train']
        self.val_data = data['val']
        self.test_data = data['test']
        self.tr_label_cnt = data['tr_label_cnt']
        self.val_label_cnt = data['val_label_cnt']
        self.test_label_cnt = data['test_label_cnt']
        self.loss_fn = loss_fn
        self.save_every = save_every
        self.epochs_run = 0
        self.result_path = path["result"]
        self.model_path = path["model"]
        self.snapshot_path = os.path.join(path["model"], 'snapshot.pt')

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # print(torch.cuda.mem_get_info())
        print('RAM memory % used:', psutil.virtual_memory()[2])
        print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

        model_memory_bytes = torch.cuda.memory_allocated(self.gpu_id)
        model_memory_mb = model_memory_bytes / (1024 ** 2)
        print("Model memory usage:", model_memory_mb, "MB")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _distributed_concat(self, tensor):
        output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat

    def _train_fn(self, epoch):
        # train_loss = 0.0
        train_loss = torch.zeros(2).to(self.gpu_id)
        self.model.train()
        # self.model.zero_grad()
        self.train_data.sampler.set_epoch(epoch)
        train_bar = tqdm(self.train_data, desc='Epoch {:1d}/{:1d}'.format(epoch, num_train_epochs))
        for source in train_bar:
            ids = source["ids"].to(self.gpu_id)
            mask = source["mask"].to(self.gpu_id)
            if self.config.FSDP.enable and self.config.FSDP.mixed_precision and not self.config.FSDP.use_fp16:
                targets = source["labels"].to(self.gpu_id, dtype=torch.bfloat16)
            else:
                targets = source["labels"].to(self.gpu_id, dtype=torch.float)
            self.optimizer.zero_grad()
            outputs = self.model(ids, mask, self.graph, target=targets)

            if isinstance(outputs, tuple):
                outputs, loss = outputs[0], outputs[1]
            else:
                loss = self.loss_fn(outputs, targets, is_multi=True)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            train_loss[0] += loss.item()
            train_loss[1] += len(source)
            self.optimizer.step()
            self.scheduler.step()

        if self.config.DDP.enable or self.config.FSDP.enable:
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        tr_loss = train_loss[0] / train_loss[1]
        return tr_loss

    def _eval_fn(self, isTest):
        eval_loss = torch.zeros(2).to(self.gpu_id)
        ev_loss = None
        self.model.eval()
        target_list = []
        logit_list = []
        data = self.val_data if not isTest else self.test_data
        desc = 'val' if not isTest else 'test'
        with torch.no_grad():
            for bi, d in enumerate(tqdm(data, desc=desc)):
                ids = d["ids"].to(self.gpu_id)
                mask = d["mask"].to(self.gpu_id)

                if self.config.FSDP.enable and self.config.FSDP.mixed_precision and not self.config.FSDP.use_fp16:
                    targets = d["labels"].to(self.gpu_id, dtype=torch.bfloat16)
                else:
                    targets = d["labels"].to(self.gpu_id, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask, G=self.graph, target=targets)
                if isinstance(outputs, tuple):
                    outputs, loss = outputs[0], outputs[1]
                if self.loss_fn != None:
                    loss = self.loss_fn(outputs, targets, is_multi=True)
                    eval_loss[0] += loss.item()
                    eval_loss[1] += len(d)
                target_list.append(targets)
                logit_list.append(torch.sigmoid(outputs))

        pred_logits = self._distributed_concat(torch.concat(logit_list, dim=0))
        labels = self._distributed_concat(torch.concat(target_list, dim=0))
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        if self.loss_fn != None:
            ev_loss = eval_loss[0] / eval_loss[1]
        return ev_loss, pred_logits, labels

    def _save_snapshot(self, epoch):
        if self.config.DDP.enable:
            if self.gpu_id == 0:
                snapshot = {
                    "MODEL_STATE": self.model.module.state_dict(),
                    "OPTIMIZER": self.optimizer.state_dict(),
                    "SCHEDULER": self.scheduler.state_dict(),
                    "EPOCHS_RUN": epoch,
                }
                torch.save(snapshot, self.snapshot_path)
        elif self.config.FSDP.enable:
            pass
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, config):
        if is_main_process():
            performance_list = []
            stop_cnt = 0
        print("======begin training======")
        for epoch in range(config.train.epochs):
            train_loss = self._train_fn(epoch)
            eval_loss, preds, labels = self._eval_fn(isTest=False)

            if is_main_process():
                # ####### stopping criteria #######
                performance = log_metrics(preds, labels, self.tr_label_cnt, self.val_label_cnt, self.test_label_cnt)
                if isinstance(self.config.train.topk, int):
                    if epoch != 0:
                        if stop_cnt > 10:
                            print(f"R-Precision@{self.config.train.topk} does not increase for 10 times, stop training")
                            break
                        else:
                            if abs(performance[f'R-Precision@{self.config.train.topk}'] - performance_list[-1]) < 1e-3:
                                stop_cnt += 1
                            else:
                                stop_cnt = 0
                    performance_list.append(performance[f'R-Precision@{self.config.train.topk}'])

                for param_group in self.optimizer.param_groups:
                    print("Learning rate: {}".format(param_group["lr"]))
                print(f"EPOCH{epoch}: train loss: {train_loss.cpu().item()} | val loss: {eval_loss.cpu().item()}")
                if config.wandb.enable:
                    log_dict_to_wandb(performance, train_loss.cpu().item(), eval_loss.cpu().item())

                with open(os.path.join(self.result_path, 'val_ep{}.json'.format(epoch)), 'w') as f:
                    json.dump(performance, f, indent=4)

            # self.scheduler.step(performance[f'R-Precision@{self.config.train.topk}'])

            if self.config.FSDP.enable:
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    cpu_state = self.model.state_dict()

            if is_main_process():
                if self.config.DDP.enable:
                    torch.save(self.model.module.state_dict(),
                               os.path.join(self.model_path, 'model_ep{}.pt'.format(epoch)),
                               _use_new_zipfile_serialization=False)
                elif self.config.FSDP.enable:
                    torch.save(cpu_state, os.path.join(self.model_path, 'model_ep{}.pt'.format(epoch)),
                               _use_new_zipfile_serialization=False)
                print(f"model saved for epoch {epoch} at {self.model_path}")

        if self.config.FSDP.enable:
            dist.barrier()

    def test(self, config):
        eval_loss, preds, labels = self._eval_fn(isTest=True)


        if is_main_process():
            test_path = os.path.join(ROOT, f'results/{date}/test/lr:{config.test.learning_rate}_bs:{config.test.batch_size}/seed:{config.test.seed}')
            if not os.path.exists(test_path):
                os.makedirs(test_path)

            np.save(os.path.join(test_path, "Pred.npy"), preds.to(torch.float).cpu().numpy())
            np.save(os.path.join(test_path, "GT.npy"), labels.to(torch.float).cpu().numpy())
            performance = log_metrics(preds, labels, self.tr_label_cnt, self.val_label_cnt, self.test_label_cnt)
            with open(os.path.join(test_path, f'test_ep{config.test.epoch}.json'), 'w') as f:
                json.dump(performance, f, indent=4)

def load_train_objs(config, dataset, rank, world_size):
    vocab = WordVocab.load_vocab(os.path.join(DIR, 'vocab_{}.pkl'.format(dataset)))
    if dataset == 'RAA':
        train = pd.read_csv(os.path.join(DIR, 'train.csv'), index_col=0)
        val = pd.read_csv(os.path.join(DIR, 'val.csv'), index_col=0)
        test = pd.read_csv(os.path.join(DIR, 'test.csv'), index_col=0)
        data_pipeline = EMSDataPipeline(config, vocab, rank, world_size)
    elif 'MIMIC3' in dataset:
        train = pd.read_csv(os.path.join(DIR, 'clean_train.csv'))
        val = pd.read_csv(os.path.join(DIR, 'clean_val.csv'))
        test = pd.read_csv(os.path.join(DIR, 'clean_test.csv'))
        data_pipeline = MIMIC3DataPipeline(config, vocab, rank, world_size)
    else:
        raise Exception('check dataset in default_sets.py')
    train_label_cnt = cnt_instance_per_label(train)
    val_label_cnt = cnt_instance_per_label(val)
    test_label_cnt = cnt_instance_per_label(test)
    train_dataloader, val_dataloader, test_dataloader = data_pipeline(train, val, test)

    data = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
        "tr_label_cnt": train_label_cnt,
        "val_label_cnt": val_label_cnt,
        "test_label_cnt": test_label_cnt
    }
    if is_main_process():
        print("Length of Train Dataloader: ", len(train_dataloader))
        print("Length of Valid Dataloader: ", len(val_dataloader))

    if config.train.graph == 'HGT':
        if config.train.with_hier:
            HGraph = HeteroGraph(config)
        else:
            HGraph = HeteroGraphwoHier(config)
        if dataset == 'RAA':
            signs_df = pd.read_excel(os.path.join(DIR, 'All Protocols Mapping.xlsx'))
            graph = HGraph(signs_df)
        elif 'MIMIC3' in dataset:
            graph = HGraph()
        else:
            raise Exception('check dataset in default_sets.py')
        print('Graph built')
    elif config.train.graph == 'GCN':
        if dataset == 'RAA':
            df = pd.read_excel(os.path.join(DIR, 'All Protocols Mapping.xlsx'))
        else:
            df = ICD9_description
        if config.model == 'KAMG':
            Graph1 = HierarchyGraph(config)
            G1 = Graph1(df, label2hier)
            Graph2 = SematicGraph(config)
            G2 = Graph2(df)
            Graph3 = CooccurGraph(config)
            G3 = Graph3(df)
            graph = [G1, G2, G3]
        else:
            Graph = HierarchyGraph(config)
            graph = Graph(df, label2hier)
        print('Graph built')
    else:
        graph = None

    model = pick_model(config, vocab, rank)

    if not config.test.is_test:
        if config.DDP.enable:
            print('configuring DDP setup for model...')
            model = model.to(rank)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            optimizer = ems_optimizer(model, config)
            if config.train.backbone == 'CNN' or config.train.backbone == 'RNN':
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=len(train_dataloader) * config.train.epochs * 0.1,
                    num_training_steps=len(train_dataloader) * config.train.epochs
                )

            else:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=len(train_dataloader) * config.train.epochs
                    # num_training_steps=len(train) / config.train.batch_size * config.train.epochs
                )
        elif config.FSDP.enable:
            print('configuring FSDP setup for model...')
            # model = model.to(torch.bfloat16)
            torch.cuda.set_device(rank)
            mixed_precision_policy, my_auto_wrap_policy = get_policies(config, rank)
            sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
            model = FSDP(model,
                         auto_wrap_policy=my_auto_wrap_policy,
                         mixed_precision=mixed_precision_policy,
                         sharding_strategy=sharding_strategy,
                         device_id=torch.cuda.current_device(),
                         limit_all_gathers=True)
            # print(len(train) / config.train.batch_size, len(train_dataloader))
            optimizer = ems_optimizer(model, config)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(train) / config.train.batch_size * config.train.epochs
            )

        else:
            raise Exception('check DDP or FSDP')
        loss_func = ClassificationLoss(label_size=len(label), loss_type=config.train.loss_type)
    else:
        save_model_root = os.path.join(ROOT, 'models/{}/lr:{}_bs:{}/seed:{}'.format(date,
                                                                                    config.test.learning_rate,
                                                                                    config.test.batch_size,
                                                                                    config.test.seed))
        model_path = os.path.join(save_model_root, 'model_ep{}.pt'.format(config.test.epoch))
        print(model_path)
        if graph:
            graph.to(rank)
        model = model.to(rank)
        model.load_state_dict(torch.load(model_path, map_location=f"cuda:{rank}"))
        model_memory_bytes = torch.cuda.memory_allocated(rank)
        model_memory_mb = model_memory_bytes / (1024 ** 2)
        print("Model memory usage:", model_memory_mb, "MB")

        scheduler = None
        optimizer = None
        loss_func = None
    return data, model, graph, optimizer, scheduler, loss_func

