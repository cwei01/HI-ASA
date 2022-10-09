from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import math
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from bert.modeling import BertModel, BERTLayerNorm
from bert.dynamic_rnn import DynamicLSTM
import torch.nn.functional as F
cuda=torch.device('cuda')

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()

def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.view([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.view([batch_size, turn_num, sequence_length])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def get_span_representation(span_starts, span_ends, input, input_mask):
    '''
    :param span_starts: [N, M]
    :param span_ends: [N, M]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N*M, JR, D], [N*M, JR]
    '''
    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask

def get_self_att_representation(input, input_score, input_mask):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output

def js_div(p_output, q_output, get_softmax=True):
        """
        Function that measures JS divergence between target and output logits:
        """
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = F.softmax(p_output)
            q_output = F.softmax(q_output)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

def distant_cross_entropy(logits, positions, mask=None, threshold=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    sigmoid = nn.Sigmoid()
    positions = positions.to(dtype=logits.dtype)
    mask = mask.to(dtype=logits.dtype)
    probs = (1 - positions) + (2 * positions - 1) * sigmoid(logits)
    log_probs = torch.log(probs) * mask
    loss = -1 * torch.mean(torch.sum(log_probs, dim=-1) / torch.sum(mask, dim=-1))
    aspect_num = torch.sum((sigmoid(logits) > threshold ).to(dtype=logits.dtype) * mask, -1)
    return loss, aspect_num

def pad_sequence(sequence, length):
    while len(sequence) < length:
        sequence.append(0)
    return sequence
class BertForSpanAspectClassification(nn.Module):
    def __init__(self, config):
        super(BertForSpanAspectClassification, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.affine = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, 5)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, span_starts=None, span_ends=None,
                labels=None, label_masks=None):
        '''
        :param input_ids: [N, L]
        :param token_type_ids: [N, L]
        :param attention_mask: [N, L]
        :param span_starts: [N, M]
        :param span_ends: [N, M]
        :param labels: [N, M]
        '''
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert span_starts is not None and span_ends is not None and labels is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            cls_logits = self.classifier(span_pooled_output)  # [N*M, 4]

            cls_loss_fct = CrossEntropyLoss(reduction='none')
            flat_cls_labels = flatten(labels)
            flat_label_masks = flatten(label_masks)
            loss = cls_loss_fct(cls_logits, flat_cls_labels)
            mean_loss = torch.sum(loss * flat_label_masks.to(dtype=loss.dtype)) / torch.sum(flat_label_masks.to(dtype=loss.dtype))
            return mean_loss

        elif mode == 'inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert span_starts is not None and span_ends is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            cls_logits = self.classifier(span_pooled_output)  # [N*M, 4]
            return reconstruct(cls_logits, span_starts)

        else:
            raise Exception
class HI_ASA(nn.Module):
    def __init__(self, config,args):
        super(HI_ASA, self).__init__()
        self.bert = BertModel(config)
        self.ate = DynamicLSTM(config.hidden_size, args.hidden_size, num_layers=1, batch_first=True, bidirectional=True, rnn_type='GRU')
        self.atc= DynamicLSTM(config.hidden_size, args.hidden_size, num_layers=1, batch_first=True, bidirectional=True, rnn_type='GRU')
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.unary_affine = nn.Linear(2*args.hidden_size, 1)
        self.start_outputs = nn.Linear(2*args.hidden_size, 1)
        self.end_outputs = nn.Linear(2*args.hidden_size, 1)
        self.dense = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.activation = nn.Tanh()
        self.activation_sigmoid = nn.Sigmoid()
        self.classifier = nn.Linear(2*args.hidden_size, 5)

        self.share_weight = args.shared_weight


        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)



    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, start_positions=None, end_positions=None, aspect_num=None,
                span_aspect_num=None, span_starts=None, span_ends=None, polarity_labels=None, label_masks=None, sequence_input=None,
                weight_kl=None, n_best_size = None, logit_threshold =None ):
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]
            batch_size, seq_len, _ = sequence_output.size()

            assert start_positions is not None and end_positions is not None  and n_best_size is not  None \
                   and logit_threshold  is not None
            temp = torch.ones(batch_size,dtype=torch.int64,device=cuda)
            L_tensor = seq_len*temp  #
            sequence_output_ate_init, (_, _) = self.ate(sequence_output, L_tensor )  # [batch_size,seq_len, d]
            sequence_output_atc_init, (_, _) = self.atc(sequence_output, L_tensor )  # [batch_size,seq_len,d]
           # shallow interactive
            sequence_output_ate  = self.share_weight* sequence_output_ate_init+ (1-self.share_weight) * sequence_output_atc_init
            sequence_output_atc  = self.share_weight* sequence_output_atc_init+ (1-self.share_weight) * sequence_output_ate_init

            start_logits = self.start_outputs(sequence_output_ate)   # [batch_size, seq_len, 1]
            start_logits = start_logits.squeeze(-1)
            start_loss ,start_aspect_num = distant_cross_entropy(start_logits, start_positions,attention_mask,logit_threshold )

            end_logits = self.end_outputs(sequence_output_ate)  # [batch_size, seq_len ,1]
            end_logits = end_logits.squeeze(-1)
            end_loss,end_aspect_num = distant_cross_entropy(end_logits, end_positions,attention_mask, logit_threshold )


            ae_loss = (start_loss + end_loss) / 2
            start_logits_dim =  F.pad(start_logits.unsqueeze(0).expand(1, -1, -1), (1,1,0,0))
            start_logits_score = F.avg_pool1d(start_logits_dim, kernel_size=3, stride=1)
            end_logits_dim = F.pad(end_logits.unsqueeze(0).expand(1,-1,-1),(1,1,0,0))
            end_logits_score = F.avg_pool1d(end_logits_dim,kernel_size=3,stride=1)
            aspect_score = (start_logits_score + end_logits_score)/2
            aspect_score = aspect_score.squeeze(0)
            aspect_score = aspect_score.unsqueeze(1).expand(-1, n_best_size, -1).reshape(n_best_size* batch_size, -1)

            assert span_starts is not None and span_ends is not None and polarity_labels is not None \
                   and label_masks is not None and  weight_kl is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output_atc,
                                                             attention_mask)  # [N*M, JR, d], [N*M, JR]

            sequence_output_atc = sequence_output_atc.unsqueeze(1).expand(-1, span_starts.size(1), -1, -1)
            sequence_output_atc = sequence_output_atc.reshape(span_output.size(0), seq_len, -1)  # [N*M, L,  d]
            interaction_mat = torch.matmul(sequence_output_atc, torch.transpose(span_output, 1, 2))  #[N*M, L, JR]
            alpha = torch.nn.functional.softmax(interaction_mat, dim=1)  #[N*M, L, JR]
            beta = torch.nn.functional.softmax(interaction_mat, dim=2) #[N*M, L, JR]
            beta_avg = beta.mean(dim=1, keepdim=True)   #[N*M, 1, JR]
            gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # [N*M, L, 1]
            span_pooled_output = torch.matmul(torch.transpose(sequence_output_atc, 1, 2), gamma).squeeze(-1)  # [N*M, d]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            ac_loss_fct = CrossEntropyLoss(reduction='none')
            flat_polarity_labels = flatten(polarity_labels)
            flat_label_masks = flatten(label_masks).to(dtype=ac_logits.dtype)
            ac_loss = ac_loss_fct(ac_logits, flat_polarity_labels)
            ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()

            #deep interaction
            #1.cosine
            sum_aspect = aspect_score.contiguous().view(-1).norm()
            sum_attention = gamma.contiguous().view(-1).norm()
            #mutual_loss = - torch.abs(torch.dot(aspect_score.contiguous().view(-1),gamma.contiguous().view(-1)))/(sum_aspect**2 * sum_attention**2)
            #2.JS
            mutual_loss = - js_div(aspect_score,gamma.squeeze(-1) )

            return ae_loss + ac_loss + weight_kl * mutual_loss

        elif mode == 'extract_inference':
            assert input_ids is not None and token_type_ids is not None  and logit_threshold is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]
            batch_size, seq_len, hid_size = sequence_output.size()

            temp = torch.ones(batch_size, dtype=torch.int64, device=cuda)
            L_tensor = seq_len * temp  #
            sequence_output_ate_init, (_, _) = self.ate(sequence_output, L_tensor)  # [batch_size,seq_len, d]
            sequence_output_atc_init, (_, _) = self.atc(sequence_output, L_tensor)  # [batch_size,seq_len,d]
            # shallow interactive
            sequence_output_ate = self.share_weight * sequence_output_ate_init + (
                        1 - self.share_weight) * sequence_output_atc_init
            sequence_output_atc = self.share_weight * sequence_output_atc_init + (
                        1 - self.share_weight) * sequence_output_ate_init

            start_logits = self.start_outputs(sequence_output_ate)  # [batch_size, seq_len, 1]
            start_logits = start_logits.squeeze(-1)

            end_logits = self.end_outputs(sequence_output_ate)  # [batch_size, seq_len ,1]
            end_logits = end_logits.squeeze(-1)

            sigmoid = torch.nn.Sigmoid()
            start_logits = sigmoid(start_logits)
            end_logits = sigmoid(end_logits)
            start_target_num = torch.sum((start_logits> logit_threshold).to(dtype=start_logits.dtype) * attention_mask.to(dtype=start_logits.dtype), -1)
            end_target_num = torch.sum((end_logits> logit_threshold).to(dtype=end_logits.dtype) * attention_mask.to(dtype=end_logits.dtype), -1)
            target_num_prediction = (start_target_num + end_target_num) / 2
            return start_logits, end_logits, target_num_prediction, sequence_output_atc

        elif mode == 'classify_inference':
            assert span_starts is not None and span_ends is not None and sequence_input is not None
            batch_size, seq_len, hid_size = sequence_input .size()
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                             attention_mask)  # [N*M, JR, d], [N*M, JR]

            sequence_output_atc = sequence_input.unsqueeze(1).expand(-1, span_starts.size(1), -1, -1)
            sequence_output_atc = sequence_output_atc.reshape(-1, seq_len, span_output.size(2))  # [N*M, L,  d]
            interaction_mat = torch.matmul(sequence_output_atc, torch.transpose(span_output, 1, 2))  # [N*M, L, JR]
            alpha = torch.nn.functional.softmax(interaction_mat, dim=1)  # [N*M, L, JR]
            beta = torch.nn.functional.softmax(interaction_mat, dim=2) #[N*M, L, JR]
            beta_avg = beta.mean(dim=1, keepdim=True)   #[N*M, 1, JR]
            gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # [N*M, L, 1]
            span_pooled_output = torch.matmul(torch.transpose(sequence_output_atc, 1, 2), gamma).squeeze(-1)  # [N*M, d]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            return reconstruct(ac_logits, span_starts)


