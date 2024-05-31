# From_Transformer_to_GPTs

一文详解从Tokenization到LLaMA模型的实现。

## Tokenizers

1. 阅读 [Tokenizer](https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html) 这篇文章，掌握不同算法的理论基础和实现方法。
2. 克隆 [minbpe](https://github.com/karpathy/minbpe) 的GitHub仓库，并仔细阅读代码，理解其实现细节。
3. 参考 [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)，利用tokenizers库和sentencepiece库，训练一个BPE模型。

## Transformers

1. 参考 [Transformer GitHub](https://github.com/hyunwoongko/transformer) 和 [Harvard NLP Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)，理解Transformer的架构和实现。 使用`torch`实现一个基础的Transformer模型，包括Encoder和Decoder。
2. 参考 [PyTorch Translation Tutorial](https://pytorch.org/tutorials/beginner/translation_transformer.html)，理解如何使用Transformer模型进行翻译任务。 用已训练好的BPE模型替换教程中的spacy Tokenization。

## BERT and GPT

1. 参考 [Trainer](https://huggingface.co/docs/transformers/model_memory_anatomy)、[Masked language modeling](https://huggingface.co/docs/transformers/tasks/masked_language_modeling)、[Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)，掌握`transformers`库中的Trainer的使用方法。
2. 利用预训练模型 [roberta](https://huggingface.co/FacebookAI/roberta-base) + decoder，实现一个生成任务。

## LLaMa

1. 参考 [PyTorch LLaMa](https://github.com/hkproj/pytorch-llama)、[Meta LLaMa](https://github.com/meta-llama/llama) 了解LLaMa模型的实现。 在本地环境中实现模型的推理。
2. 参考 [Baby LLaMa-Chinese](https://github.com/DLLXW/baby-llama2-chinese) ，了解如何训练一个小型的中文LLaMa模型，并进行监督微调（SFT）和强化学习人类反馈（RLHF）。

## CodeLLaMA

1. 参考 [CodeLLaMA GitHub](https://github.com/meta-llama/codellama/tree/main)，了解CodeLLaMA的实现和应用。

## Distributed Training

1. 参考 [ddp-tutorial](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/single_gpu.py)，学习分布式数据并行（DDP）的实现和应用。

