<div align="center">

# Baichuan-M2-32B

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/baichuan-inc/Baichuan-M2-32B)
[![M2 GPTQ-4bit](https://img.shields.io/badge/🤗%20M2%20GPTQ--4bit-Model-orange)](https://huggingface.co/baichuan-inc/Baichuan-M2-32B-GPTQ-Int4)
[![Huawei Ascend 8bit](https://img.shields.io/badge/✨%20Huawei%20Ascend%208bit-Model-green)](https://modelers.cn/models/Baichuan/Baichuan-M2-32B-W8A8)
[![Tech Report](https://img.shields.io/badge/📜%20TechReport-Arxiv-red)](https://arxiv.org/abs/2509.02208)

<h4 align="center">
    <p>
        <a href="https://github.com/baichuan-inc/Baichuan-M2-32B/blob/main/README.md">中文</a> |
        <b>English</b>
    <p>
</h4>

</div>

## 🌟 Model Overview

Baichuan-M2-32B is Baichuan AI's medically enhanced reasoning model, the second medical model released by Baichuan. Designed for real-world medical reasoning tasks, this model builds upon Qwen2.5-32B with an innovative Large Verifier System. Through domain-specific fine-tuning on real-world medical questions, it achieves breakthrough medical performance while maintaining strong general capabilities.

**Model Features:**

Baichuan-M2 incorporates three core technical innovations: First, through the **Large Verifier System**, it combines medical scenario characteristics to design a comprehensive medical verification framework, including patient simulators and multi-dimensional verification mechanisms; second, through **medical domain adaptation enhancement** via Mid-Training, it achieves lightweight and efficient medical domain adaptation while preserving general capabilities; finally, it employs a **multi-stage reinforcement learning** strategy, decomposing complex RL tasks into hierarchical training stages to progressively enhance the model's medical knowledge, reasoning, and patient interaction capabilities.

**Core Highlights:**
- 🏆 **World's Leading Open-Source Medical Model**: Outperforms all open-source models and many proprietary models on HealthBench, achieving medical capabilities closest to GPT-5
- 🧠 **Doctor-Thinking Alignment**: Trained on real clinical cases and patient simulators, with clinical diagnostic thinking and robust patient interaction capabilities
- ⚡ **Efficient Deployment**: Supports 4-bit quantization for single-RTX4090 deployment, with 58.5% higher token throughput in MTP version for single-user scenarios

<div align="center">
  <img src="images/m2-framework.png" alt="Beyond the Model: Scaling Medical Capability with a Large Verifier System" width="80%">
</div>

## 📊 Performance Metrics

### HealthBench Scores

| Model Name | HealthBench | HealthBench-Hard | HealthBench-Consensus |
|------------|-------------|------------------|-----------------------|
| Baichuan-M2 | 60.1 | 34.7 | 91.5 |
| gpt-oss-120b | 57.6 | 30 | 90 |
| Qwen3-235B-A22B-Thinking-2507 | 55.2 | 25.9 | 90.6 |
| Deepseek-R1-0528 | 53.6 | 22.6 | 91.5 |
| GLM-4.5 | 47.8 | 18.7 | 85.3 |
| Kimi-K2 | 43 | 10.7 | 90.9 |
| gpt-oss-20b | 42.5 | 10.8 | 82.6 |

### General Performance

| Benchmark | Baichuan-M2-32B | Qwen3-32B (thinking) |
|-----------|-----------------|-----------|
| AIME24 | 83.4 | 81.4 |
| AIME25 | 72.9 | 72.9 |
| Arena-Hard-v2.0 | 45.8 | 44.5 |
| CFBench | 77.6 | 75.7 |
| WritingBench | 8.56 | 7.90 |

*Note: AIME uses max_tokens=64k, others use 32k; temperature=0.6 for all tests.*

## 🔧 Technical Features

[📗 **Technical Blog**](https://www.baichuan-ai.com/blog/baichuan-M2)
[📜 **Technical Report**](https://arxiv.org/abs/2509.02208)

### Large Verifier System
- **Patient Simulator**: Virtual patient system based on real clinical cases
- **Multi-Dimensional Verification**: 8 dimensions including medical accuracy, response completeness, and follow-up awareness
- **Dynamic Scoring**: Real-time generation of adaptive evaluation criteria for complex clinical scenarios
### Medical Domain Adaptation
- **Mid-Training**: Medical knowledge injection while preserving general capabilities
- **Reinforcement Learning**: Multi-stage RL strategy optimization
- **General-Specialized Balance**: Carefully balanced medical, general, and mathematical composite training data

## ⚙️ Quick Start

```python
# 1. load model
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-M2-32B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-M2-32B")
# 2. Input prompt text
prompt = "Got a big swelling after a bug bite. Need help reducing it."
# 3. Encode the input text for the model
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    thinking_mode='on' # on/off/auto
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# 4. Generate text
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=4096
)
output_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
][0].tolist()
# 5. parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

```

For deployment, you can use `sglang>=0.4.6.post1` or `vllm>=0.9.0` or to create an OpenAI-compatible API endpoint:
- SGLang:
    ```shell
    python -m sglang.launch_server --model-path baichuan-inc/Baichuan-M2-32B --reasoning-parser qwen3
    ```
- vLLM:
    ```shell
    vllm serve baichuan-inc/Baichuan-M2-32B  --reasoning-parser qwen3
    ```

## MTP inference with SGLang

1. Replace the qwen2.py file in the sglang installation directory with draft/qwen2.py.
2. Launch sglang:
```
python3 -m sglang.launch_server \
--model Baichuan-M2-32B \
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path Baichuan-M2-32B/draft \
--speculative-num-steps 6 \
--speculative-eagle-topk 10 \
--speculative-num-draft-tokens 32 \
--mem-fraction 0.9 \
--cuda-graph-max-bs 2 \
--reasoning-parser qwen3 \
--dtype bfloat16
```

## ⚠️ Usage Notices
1. **Medical Disclaimer**: For research and reference only; cannot replace professional medical diagnosis or treatment
2. **Intended Use Cases**: Medical education, health consultation, clinical decision support
3. **Safe Use**: Recommended under guidance of medical professionals

## 📄 License
Licensed under the [Apache License 2.0](LICENSE). Research and commercial use permitted.

## 🤝 Acknowledgements
- Base Model: Qwen2.5-32B
- Training Framework: verl
- Inference Engines: vLLM, SGLang
- Quantization: AutoRound, GPTQ
Thank you to the open-source community. We commit to continuous contribution and advancement of healthcare AI.

## 📞 Contact Us
- Resources: [Baichuan AI Website](https://www.baichuan-ai.com)
- Technical Support: [GitHub](https://github.com/baichuan-inc)

---
<div align="center">

**Empowering Healthcare with AI, Making Health Accessible to All**

</div>

## 📚 Citation

```bibtex
@misc{baichuan-m2,
    title={Baichuan-M2: Scaling Medical Capability with Large Verifier System},
    author={M2 Team and Chengfeng Dou and Chong Liu and Fan Yang and Fei Li and Jiyuan Jia and Mingyang Chen and Qiang Ju and Shuai Wang and Shunya Dang and Tianpeng Li and Xiangrong Zeng and Yijie Zhou, et al.},
    year={2025},
    eprint={2509.02208},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2509.02208},
}
```
