
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import pickle
import argparse
import collections

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import bert.tokenization as tokenization
from bert.modeling import BertConfig
from bert.sentiment_modeling import BertForSpanAspectExtraction

from squad.squad_evaluate import exact_match_score, metric_max_over_ground_truths
from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features, RawFinalResult, RawSpanResult, \
    span_annotate_candidates, wrapped_get_final_text
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
    all_start_positions = torch.tensor([f.start_positions for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_positions for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader, num_train_steps

def read_eval_data(args, tokenizer, logger):
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
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
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
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
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
                print("step: {}, loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {})"
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

def eval_aspect_extract(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []
        for start_index, end_index, span_mask in zip(result.start_indexes, result.end_indexes, result.span_masks):
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                pred_terms.append(final_text)

        prediction = {'pred': pred_terms, 'gold': example.term_texts}
        all_nbest_json[example.example_id] = prediction

        for pred_term in pred_terms:
            common += metric_max_over_ground_truths(exact_match_score, pred_term, example.term_texts)
        retrieved += len(pred_terms)
        relevant += len(example.term_texts)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.

    return {'p': p, 'r': r, 'f1': f1, 'common': common, 'retrieved': retrieved, 'relevant': relevant}, all_nbest_json

def evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=False, do_pipeline=False):
    all_results = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, example_indices = batch
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        batch_features, batch_results = [], []
        for j, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[j].detach().cpu().tolist()
            end_logits = batch_end_logits[j].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            batch_features.append(eval_feature)
            batch_results.append(RawSpanResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

        span_starts, span_ends, _, label_masks = span_annotate_candidates(eval_examples, batch_features, batch_results,
                                                                          args.filter_type, False,
                                                                          args.use_heuristics, args.use_nms,
                                                                          args.logit_threshold, args.n_best_size,
                                                                          args.max_answer_length, args.do_lower_case,
                                                                          args.verbose_logging, logger)

        for j, example_index in enumerate(example_indices):
            start_indexes = span_starts[j]
            end_indexes = span_ends[j]
            span_masks = label_masks[j]
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                              end_indexes=end_indexes, cls_pred=None, span_masks=span_masks))

    metrics, all_nbest_json = eval_aspect_extract(eval_examples, eval_features, all_results,
                                                  args.do_lower_case, args.verbose_logging, logger)
    if write_pred:
        output_file = os.path.join(args.output_dir, "predictions.json")
        with open(output_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_file))
    if do_pipeline:
        output_file = os.path.join(args.output_dir, "extraction_results.pkl")
        pickle.dump(all_results, open(output_file, 'wb'))
    return metrics


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_config_file", default='bert-base-uncased/bert_config.json', type=str,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default='bert-base-uncased/vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default='out/extract/lap', type=str, 
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--debug", default=False, action='store_true', help="Whether to run in debug mode.")
    parser.add_argument("--data_dir", default='data/absa', type=str, help="SemEval data dir")
    parser.add_argument("--train_file", default='laptop14_train.txt', type=str, help="SemEval xml for training")
    parser.add_argument("--predict_file", default='laptop14_test.txt', type=str, help="SemEval csv for prediction")
    parser.add_argument("--init_checkpoint", default='bert-base-uncased/pytorch_model.bin', type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=96, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=True, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pipeline", default=False, action='store_true', help="Whether to run pipeline on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_proportion", default=0.5, type=float,
                        help="Proportion of steps to save models for. E.g., 0.5 = 50% of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=12, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--logit_threshold", default=7.5, type=float,
                        help="Logit threshold for annotating labels.")
    parser.add_argument("--filter_type", default="f1", type=str, help="Which filter type to use")
    parser.add_argument("--use_heuristics", default=True, action='store_true',
                        help="If true, use heuristic regularization on span length")
    parser.add_argument("--use_nms", default=True, action='store_true',
                        help="If true, use nms to prune redundant spans")
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
    model = BertForSpanAspectExtraction(bert_config)
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
                                                           train_dataloader, eval_examples, eval_features, eval_dataloader,
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
        f = open(log_path, "a")
        print("threshold: {}, step: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {})"
              .format(args.logit_threshold, global_step, metrics['p'], metrics['r'],
                      metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant']), file=f)
        print(" ", file=f)
        f.close()

    if args.do_pipeline:
        logger.info("***** Running pipeline *****")
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
        metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=True,
                           do_pipeline=True)
        print("step: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {})"
              .format(global_step, metrics['p'], metrics['r'],
                      metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant']))


if __name__=='__main__':
    main()











