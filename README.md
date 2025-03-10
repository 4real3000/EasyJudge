# 🕵️EasyJudge: an Easy-to-use Tool for Comprehensive Response Evaluation of LLMs

- **Lightweight Usage Model**:EasyJudge is built to minimize dependency requirements, offering a simple installation process and precise documentation. Users can initiate the evaluation interface with only a few basic commands.

- **Comprehensive Evaluation Tool**: EasyJudge offers a highly customizable interface, allowing users to select evaluation scenarios and flexibly combine evaluation criteria based on their needs. The visualization interface has been carefully designed to provide users with an intuitive view of various aspects of the evaluation results.

- **Efficient Inference Engine**: EasyJudge employs model quantization, memory management optimization, and hardware acceleration support to enable efficient inference. As a result, EasyJudge can run seamlessly on consumer-grade GPUs and even CPUs.

## System Overview
![Example Image](picture/screenshot.png)

## Model

EasyJudge is now available on huggingface-hub:
[🤗 4real/EasyJudge_gguf](https://huggingface.co/4real/EasyJudge_gguf)

## Quick Start

(Example of Deploying on [AutoDL](https://www.autodl.com/home) Cloud Server)

#### Deploy ollama

##### 1. Start the installation software on autodl
```bash
export OLLAMA_MODELS=/root/autodl-tmp/models
curl -fsSL https://ollama.com/install.sh | sh
```

##### 2. Start the service
```bash
ollama serve
```

##### 3. Import EasyJudge models
Modify the path after `from` in each Modelfile to the local path where the model is downloaded from huggingface.
```bash
ollama create PAIRWISE -f /root/autodl-tmp/EasyJudge/Modelfile/PAIRWISE.Modelfile
ollama create POINTWISE -f /root/autodl-tmp/EasyJudge/Modelfile/POINTWISE.Modelfile
```

#### Environment Configuration

(EasyJudge uses the environment PyTorch 2.3.0, Python 3.12 (ubuntu22.04), and Cuda 12.1.)

##### 1. Create conda environment
```bash
conda create -n EasyJudge
conda init
conda activate EasyJudge
```

##### 2. Install specified Python packages in bulk
```bash
pip install -r requirements.txt
```
#### Run the Program

To start the application, use the following command to run `main.py` with specific server configurations:

```bash
streamlit run main.py --server.address=127.0.0.1 --server.port=6006 --server.enableXsrfProtection=false
```

## Citation

If you use our dataset or model, please cite our paper:

```bibtex
@inproceedings{li2025easyjudge,
  title={EasyJudge: an Easy-to-use Tool for Comprehensive Response Evaluation of LLMs},
  author={Li, Yijie and Sun, Yuan},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics: System Demonstrations},
  pages={91--103},
  year={2025}
}
```

## Acknowledge❤️
We acknowledge these works for their public codes: 
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), 
[llama.cpp](https://github.com/ggerganov/llama.cpp), 
[ollama](https://github.com/ollama/ollama), 
[auto-j](https://github.com/GAIR-NLP/auto-j/tree/main),
[JudgeLM](https://github.com/baaivision/JudgeLM).
