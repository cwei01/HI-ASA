# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SemEval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import pickle
import json
import argparse
import collections

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import bert.tokenization as tokenization
from bert.modeling import BertConfig
from bert.sentiment_modeling import BertForSpanAspectClassification

from squad.squad_evaluate import exact_match_score
from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features, \
    RawFinalResult, wrapped_get_final_text, id_to_label
from absa.run_base import copy_optimizer_params_to_model, set_optimizer_params_grad, prepare_optimizer, post_process_loss, bert_load_state_dict

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def read_train_data(args, tokenizer, logger):
    if args.debug:
        args.train_batch_size = 8

    train_path = os.path.join(args.data_dir, args.train_file)
    train_set = read_absa_data(train_path)
    train_examples = convert_absa_data(dataset=train_set, verbose_logging=args.verbose_logging)
    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length,
                                                  args.verbose_logging, logger)

    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    logger.info("Batch size = %d", args.train_batch_size)
    logger.info("Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_span_starts = torch.tensor([f.start_indexes for f in train_features], dtype=torch.long)
    all_span_ends = torch.tensor([f.end_indexes for f in train_features], dtype=torch.long)
    all_labels = torch.tensor([f.polarity_labels for f in train_features], dtype=torch.long)
    all_label_masks = torch.tensor([f.label_masks for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_starts, all_span_ends,
                               all_labels, all_label_masks)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader, num_train_steps

def read_eval_data(args, tokenizer, logger):
    if args.debug:
        args.predict_batch_size = 8

    eval_path = os.path.join(args.data_dir, args.predict_file)
    eval_set = read_absa_data(eval_path)
    eval_examples = convert_absa_data(dataset=eval_set, verbose_logging=args.verbose_logging)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,
                                                 args.verbose_logging, logger)

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    logger.info("Batch size = %d", args.predict_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_span_starts = torch.tensor([f.start_indexes for f in eval_features], dtype=torch.long)
    all_span_ends = torch.tensor([f.end_indexes for f in eval_features], dtype=torch.long)
    all_label_masks = torch.tensor([f.label_masks for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_starts, all_span_ends,
                              all_label_masks, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    return eval_examples, eval_features, eval_dataloader

def pipeline_eval_data(args, tokenizer, logger):
    if args.debug:
        args.predict_batch_size = 8

    eval_path = os.path.join(args.data_dir, args.predict_file)
    eval_set = read_absa_data(eval_path)
    eval_examples = convert_absa_data(dataset=eval_set, verbose_logging=args.verbose_logging)

    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,
                                                 args.verbose_logging, logger)

    assert args.extraction_file is not None
    eval_extract_preds = []
    extract_predictions = pickle.load(open(args.extraction_file, 'rb'))
    extract_dict = {}
    for pred in extract_predictions:
        extract_dict[pred.unique_id] = pred
    for eval_feature in eval_features:
        eval_extract_preds.append(extract_dict[eval_feature.unique_id])
    assert len(eval_extract_preds) == len(eval_features)

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    logger.info("Batch size = %d", args.predict_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_span_starts = torch.tensor([f.start_indexes for f in eval_extract_preds], dtype=torch.long)
    all_span_ends = torch.tensor([f.end_indexes for f in eval_extract_preds], dtype=torch.long)
    all_label_masks = torch.tensor([f.span_masks for f in eval_extract_preds], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_starts, all_span_ends,
                              all_label_masks, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    return eval_examples, eval_features, eval_dataloader

def run_train_epoch(args, global_step, model, param_optimizer, train_dataloader,
                    eval_examples, eval_features, eval_dataloader,
                    optimizer, n_gpu, device, logger, log_path, save_path,
                    save_checkpoints_steps, start_save_steps, best_f1):
    running_loss, count = 0.0, 0
    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
        input_ids, input_mask, segment_ids, span_starts, span_ends, labels, label_masks = batch
        loss = model('train', input_mask, input_ids=input_ids, token_type_ids=segment_ids,
                     span_starts=span_starts, span_ends=span_ends, labels=labels, label_masks=label_masks)
        loss = post_process_loss(args, n_gpu, loss)
        loss.backward()
        running_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16 or args.optimize_on_cpu:
                if args.fp16 and args.loss_scale != 1.0:
                    # scale down gradients for fp16 training
                    for param in model.parameters():
                        param.grad.data = param.grad.data / args.loss_scale
                is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                if is_nan:
                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                    args.loss_scale = args.loss_scale / 2
                    model.zero_grad()
                    continue
                optimizer.step()
                copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
            else:
                optimizer.step()
            model.zero_grad()
            global_step += 1
            count += 1

            if global_step % save_checkpoints_steps == 0 and count != 0:
                logger.info("step: {}, loss: {:.4f}".format(global_step, running_loss / count))

            if global_step % save_checkpoints_steps == 0 and global_step > start_save_steps and count != 0:  # eval & save model
                logger.info("***** Running evaluation *****")
                model.eval()
                metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger)
                f = open(log_path, "a")
                print("step: {}, loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f} "
                      "(common: {}, retrieved: {}, relevant: {})"
                      .format(global_step, running_loss / count, metrics['p'], metrics['r'],
                              metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant']), file=f)
                print(" ", file=f)
                f.close()
                running_loss, count = 0.0, 0
                model.train()
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': global_step
                    }, save_path)
                if args.debug:
                    break
    return global_step, model, best_f1

def metric_max_over_ground_truths_all(metric_fn, term, polarity, gold_terms, gold_polarities):
    hit = 0
    for gold_term, gold_polarity in zip(gold_terms, gold_polarities):
        score = metric_fn(term, gold_term)
        if score and polarity == gold_polarity:
            hit = 1
    return hit

def metric_max_over_ground_truths_ae(metric_fn,term, gold_terms):
    scores_for_ground_truths = []
    for ground_truth in gold_terms:
        score = metric_fn(term, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def metric_max_over_ground_truths_ac( polarity, gold_polarities):
    hit = 0
    for gold_polarity in gold_polarities:
        if polarity == gold_polarity:
            hit = 1
    return hit

def eval_absa(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    common_ae, relevant_ae, retrieved_ae = 0., 0., 0.
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []
        pred_polarities = []
        for start_index, end_index, cls_pred, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.cls_pred, result.span_masks):
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                pred_terms.append(final_text)
                pred_polarities.append(id_to_label[cls_pred])

        prediction = {'pred': pred_terms, 'gold terms': example.term_texts,
                      'pred_polarities': pred_polarities, 'gold_polarities': example.polarities}
        all_nbest_json[example.example_id] = prediction

        for term, polarity in zip(pred_terms, pred_polarities):
            common += metric_max_over_ground_truths_all(exact_match_score, term, polarity, example.term_texts,example.polarities)
            common_ae += metric_max_over_ground_truths_ae(exact_match_score, term, example.term_texts)
            #common += metric_max_over_ground_truths(exact_match_score, term, polarity, example.term_texts, example.polarities)
        retrieved += len(pred_terms)
        relevant += len(example.term_texts)
        retrieved_ae += len(pred_terms)
        relevant_ae += len(example.term_texts)

    p_all = common / retrieved if retrieved > 0 else 0.
    r_all = common / relevant
    f1_all = (2 * p_all * r_all) / (p_all + r_all) if p_all > 0 and r_all > 0 else 0.

    p_ae = common_ae / retrieved_ae if retrieved_ae > 0 else 0.
    r_ae = common_ae / relevant_ae
    f1_ae = (2 * p_ae* r_ae) / (p_ae + r_ae) if p_ae > 0 and r_ae > 0 else 0.

    return {'p_all': p_all, 'r_all': r_all, 'f1_all': f1_all,
            'p_ae': p_ae, 'r_ae': r_ae, 'f1_ae': f1_ae}, all_nbest_json

def eval_ac(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common_ac, all_sent= 0., 0.
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []
        pred_polarities = []
        for start_index, end_index, cls_pred, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.cls_pred, result.span_masks):
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                pred_terms.append(final_text)
                pred_polarities.append(id_to_label[cls_pred])

        prediction = {'pred': pred_terms, 'gold terms': example.term_texts,
                      'pred_polarities': pred_polarities, 'gold_polarities': example.polarities}
        all_nbest_json[example.example_id] = prediction

        for term, polarity in zip(pred_terms, pred_polarities):
            common_ac += metric_max_over_ground_truths_ac(polarity, example.polarities)

        all_sent+= len(pred_terms)

    Acc = common_ac / all_sent if all_sent > 0 else 0.

    return {'Acc_ac': Acc}, all_nbest_json

def evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=False):
    all_results = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, span_starts, span_ends, label_masks, example_indices = batch
        with torch.no_grad():
            cls_logits = model('inference', input_mask, input_ids=input_ids, token_type_ids=segment_ids,
                               span_starts=span_starts, span_ends=span_ends)

        for j, example_index in enumerate(example_indices):
            cls_pred = cls_logits[j].detach().cpu().numpy().argmax(axis=1).tolist()
            start_indexes = span_starts[j].detach().cpu().tolist()
            end_indexes = span_ends[j].detach().cpu().tolist()
            span_masks = label_masks[j].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                              end_indexes=end_indexes, cls_pred=cls_pred, span_masks=span_masks))

    metrics, all_nbest_json = eval_absa(eval_examples, eval_features, all_results,
                                        args.do_lower_case, args.verbose_logging, logger)

    if write_pred:
        output_file = os.path.join(args.output_dir, "predictions.json")
        with open(output_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_file))
    return metrics

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--debug", default=False, action='store_true', help="Whether to run in debug mode.")
    parser.add_argument("--data_dir", default='data/semeval_14', type=str, help="SemEval data dir")
    parser.add_argument("--train_file", default=None, type=str, help="SemEval xml for training")
    parser.add_argument("--predict_file", default=None, type=str, help="SemEval csv for prediction")
    parser.add_argument("--extraction_file", default=None, type=str, help="pkl file for extraction")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=96, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pipeline", default=False, action='store_true', help="Whether to run pipeline on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_proportion", default=0.5, type=float,
                        help="Proportion of steps to save models for. E.g., 0.5 = 50% "
                             "of training.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    if not args.do_train and not args.do_predict and not args.do_pipeline:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train and not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict and not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")
    if args.do_pipeline and not args.extraction_file:
            raise ValueError(
                "If `do_pipeline` is True, then `extraction_file` must be specified.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("torch_version: {} device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        torch.__version__, device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info('output_dir: {}'.format(args.output_dir))
    save_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    log_path = os.path.join(args.output_dir, 'performance.txt')
    network_path = os.path.join(args.output_dir, 'network.txt')
    parameter_path = os.path.join(args.output_dir, 'parameter.txt')

    f = open(parameter_path, "w")
    for arg in sorted(vars(args)):
        print("{}: {}".format(arg, getattr(args, arg)), file=f)
    f.close()

    logger.info("***** Preparing model *****")
    model = BertForSpanAspectClassification(bert_config)
    if args.init_checkpoint is not None and not os.path.isfile(save_path):
        model = bert_load_state_dict(model, torch.load(args.init_checkpoint, map_location='cpu'))
        logger.info("Loading model from pretrained checkpoint: {}".format(args.init_checkpoint))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'])
        step = checkpoint['step']
        logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                    .format(save_path, step))

    f = open(network_path, "w")
    for n, param in model.named_parameters():
        print("name: {}, size: {}, dtype: {}, requires_grad: {}"
              .format(n, param.size(), param.dtype, param.requires_grad), file=f)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: {}".format(total_trainable_params), file=f)
    print("Total parameters: {}".format(total_params), file=f)
    f.close()

    logger.info("***** Preparing data *****")
    train_dataloader, num_train_steps = None, None
    eval_examples, eval_features, eval_dataloader = None, None, None
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    if args.do_train:
        logger.info("***** Preparing training *****")
        train_dataloader, num_train_steps = read_train_data(args, tokenizer, logger)
        logger.info("***** Preparing evaluation *****")
        eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger)

    logger.info("***** Preparing optimizer *****")
    optimizer, param_optimizer = prepare_optimizer(args, model, num_train_steps)

    global_step = 0
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        logger.info("Loading optimizer from finetuned checkpoint: '{}' (step {})".format(save_path, step))
        global_step = step

    if args.do_train:
        logger.info("***** Running training *****")
        best_f1 = 0
        save_checkpoints_steps = int(num_train_steps / (5 * args.num_train_epochs))
        start_save_steps = int(num_train_steps * args.save_proportion)
        if args.debug:
            args.num_train_epochs = 1
            save_checkpoints_steps = 20
            start_save_steps = 0
        model.train()
        for epoch in range(int(args.num_train_epochs)):
            logger.info("***** Epoch: {} *****".format(epoch+1))
            global_step, model, best_f1 = run_train_epoch(args, global_step, model, param_optimizer,
                                                          train_dataloader, eval_examples, eval_features,
                                                          eval_dataloader,
                                                          optimizer, n_gpu, device, logger, log_path, save_path,
                                                          save_checkpoints_steps, start_save_steps, best_f1)

    if args.do_predict:
        logger.info("***** Running prediction *****")
        if eval_dataloader is None:
            eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger)

        # restore from best checkpoint
        if save_path and os.path.isfile(save_path) and args.do_train:
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model'])
            step = checkpoint['step']
            logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                        .format(save_path, step))

        model.eval()
        metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=True)
        print("step: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {})"
              .format(global_step, metrics['p'], metrics['r'],
                      metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant']))

    if args.do_pipeline:
        logger.info("***** Running prediction *****")
        eval_examples, eval_features, eval_dataloader = pipeline_eval_data(args, tokenizer, logger)

        # restore from best checkpoint
        if save_path and os.path.isfile(save_path) and args.do_train:
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model'])
            step = checkpoint['step']
            logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                        .format(save_path, step))

        model.eval()
        metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=True)
        f = open(log_path, "a")
        print("pipeline, step: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {})"
              .format(global_step, metrics['p'], metrics['r'],
                      metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant']), file=f)
        print(" ", file=f)
        f.close()


if __name__=='__main__':
    main()
