## From_Transformer_to_GPTs

### 1、Tokenizer

1. 了解多种LLM的tokenizer
2. 利用`tokenizers`训练一个BPE
3. 从头训练一个BPE
4. 参考：
   1. https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html

### 2、Transformer

1. 用PyTorch从头实现Transformer：`TransformerScratch`
2. 针对德语到英语的翻译任务，用PyTorch自带的Transformer和手写的Transformer分别跑训练过程，并翻译具体句子。
3. 参考：
   1. https://github.com/hyunwoongko/transformer 
   2. https://pytorch.org/tutorials/beginner/translation_transformer.html


### 3、T5

1. 了解encoder-decoder架构的T5，并尝试fine-tune。

### 4、CodeLLaMA

1. 了解代码数据对LLM训练的重要性。

### 5、LLaMa

1. 参考meta-llama代码，实现一个非分布式llama2，并单机跑推理代码。
2. 下载中文数据，训练一个tiny版本的llama-chinese，优化参数很随意，主要是让流程跑起来。
2. fine-tune 训练好的tiny-llama-chinese
3. 参考：
   1. https://github.com/hkproj/pytorch-llama
   2. https://github.com/meta-llama/llama
   3. https://github.com/DLLXW/baby-llama2-chinese
