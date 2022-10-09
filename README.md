#A Hierarchical Interactive Network for Joint Span-based Aspect-Sentiment Analysis [paper][https://arxiv.org/pdf/2208.11283.pdf]

A PyTorch implementation of Joint Span-based Aspect-Sentiment Analysis

This repo contains the code and data:

In this paper, we novelly propose a hierarchical interactive network (HI-ASA) to model two-way interactions between two tasks appropriately, where the hierarchical interactions involve two steps: shallow-level interaction and deep-level interaction. First, we utilize cross-stitch mechanism to combine the different task-specific features selectively as the input to ensure proper two-way interactions. Second, the mutual information technique is applied to mutually constrain learning between two tasks in the output layer, thus the aspect input and the sentiment input is capable of encoding features of the other task via backpropagation. Extensive experiments on three real-world datasets demonstrate HI-ASA's superiority over baselines. 



This framework consists of two components:

- Aspect Extraction

- Sentiment Classification

Both of two components utilize [BERT](https://github.com/huggingface/pytorch-pretrained-BERT) as backbone network. The aspect extraction aims to propose one or multiple candidate targets based on the probabilities of the start and end positions. The polarity classifier predicts the sentiment polarity using the span representation of the given target.

## Usage
1. Install required packages：

      Python 3.6

      [Pytorch 1.1](https://pytorch.org/)

      [Allennlp](https://allennlp.org/)

2. Download pre-train models used in the paper unzip it in the current directory

    uncased [BERT-Large](https://drive.google.com/file/d/13I0Gj7v8lYhW5Hwmp5kxm3CTlzWZuok2/view?usp=sharing) model
3. train the MIM for aspect term-polarity co-extraction and the results are in /result like this:
```shell
python -m main.run_joint_span \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --data_dir $DATA_DIR \
  --train_file rest_total_train.txt \
  --predict_file rest_total_test.txt \
  --train_batch_size 32 \
  --output_dir out/01
```
## Detailed information

The range of parameter in this paper is as follows:

```
the learning rate:[0.1,0.01,0.001,0.0001]
the batch size:[16,32,64,128,256]
the shared_weight:[0,0.1,0.2,0.3,0.4,0.5]
the weight_kl:[0.0,10^-7,10^-5,10^-3,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
the num_train_epochs:[10,20,30,40,50,60,70,80,90,100]
```

## Acknowledgements
We sincerely thank Xin Li for releasing the [datasets](https://github.com/lixin4ever/E2E-TBSA).

## Citation

```
@article{chen2022hierarchical,
  title={A Hierarchical Interactive Network for Joint Span-based Aspect-Sentiment Analysis},
  author={Chen, Wei and Du, Jinglong and Zhang, Zhao and Zhuang, Fuzhen and He, Zhongshi},
  journal={arXiv preprint arXiv:2208.11283},
  year={2022}
}
```
