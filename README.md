## EasyJudge: an Easy-to-use Tool for Comprehensive Response Evaluation of LLMs

- **Lightweight Usage Model**: EasyJudge is built to minimize dependency requirements, offering a simple installation process and precise documentation. Users can initiate the evaluation interface with only a few basic commands.

- **Comprehensive Evaluation Tool**: EasyJudge offers a highly customizable interface, allowing users to select evaluation scenarios and flexibly combine evaluation criteria based on their needs. The visualization interface has been carefully designed to provide users with an intuitive view of various aspects of the evaluation results.

- **Efficient Inference Engine**: EasyJudge employs model quantization, memory management optimization, and hardware acceleration support to enable efficient inference. As a result, EasyJudge can run seamlessly on consumer-grade GPUs and even CPUs.

### System Overview
![Example Image](picture/screenshot.png)

### Model

EasyJudge is now available on huggingface-hub:
[🤗 4real/EasyJudge_gguf](https://huggingface.co/4real/EasyJudge_gguf)

### Quick Start

以autodl云服务器部署为例

#### 部署ollama

##### 1. 在autodl安装软件启动
```bash
export OLLAMA_HOST="0.0.0.0:6006"
export OLLAMA_MODELS=/root/autodl-tmp/models
curl -fsSL https://ollama.com/install.sh | sh
```

##### 2. 启动服务
```bash
ollama serve
```

##### 3. 导入EasyJudge模型
需要将 Modelfile 每个模型文件中的 `from` 后的路径，修改为从 huggingface 下载模型的本地路径。
```bash
export OLLAMA_HOST="0.0.0.0:6006"
ollama create PAIRWISE -f /root/autodl-tmp/Modelfile/PAIRWISE.Modelfile
ollama create POINTWISE -f /root/autodl-tmp/Modelfile/POINTWISE.Modelfile
```


### Acknowledge
