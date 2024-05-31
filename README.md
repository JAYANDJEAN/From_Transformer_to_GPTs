# From_Transformer_to_GPTs



## Tokenizers

1. 参考 [Tokenizer](https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html) 学习多种Tokenization算法。
2. 重点学习下 [minbpe GitHub](https://github.com/karpathy/minbpe) 代码。
3. 参考 [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)，基于`tokenizers`和`sentencepiece`在翻译数据上训练一个BPE模型。

## Transformers

1. 参考 [Transformer GitHub](https://github.com/hyunwoongko/transformer)、[Harvard NLP Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)，基于`torch`从头实现一个Transformer。
2. 参考 [PyTorch Translation Tutorial](https://pytorch.org/tutorials/beginner/translation_transformer.html)，学习Transformer模型在翻译任务中的应用。
3. 教程中是用`spacy`来做Tokenization，现利用上述已训练好的BPE模型做Tokenization，实现翻译任务。

## BERT and GPT

1. 参考 [Trainer](https://huggingface.co/docs/transformers/model_memory_anatomy)、[Masked language modeling](https://huggingface.co/docs/transformers/tasks/masked_language_modeling)、[Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)，学习`transformers`下的trainer。
2. 利用预训练模型 [roberta](https://huggingface.co/FacebookAI/roberta-base) + decoder，实现一个生成任务。

## LLaMa

1. 参考 [PyTorch LLaMa](https://github.com/hkproj/pytorch-llama)、[Meta LLaMa](https://github.com/meta-llama/llama) 实现一个单机版的llama，并实现inference。
2. 参考 [Baby LLaMa-Chinese](https://github.com/DLLXW/baby-llama2-chinese) 训练一个tiny版本的chinese-llama，并在该模型上实现sft和rlhf。

## CodeLLaMA

1. 参考 [CodeLLaMA GitHub](https://github.com/meta-llama/codellama/tree/main)，了解一下 CodeLLaMA。

## DDP

1. 参考 [ddp-tutorial](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/single_gpu.py)，了解DDP机制。

