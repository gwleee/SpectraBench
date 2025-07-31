---
license: other
license_name: hyperclovax-seed
license_link: LICENSE
pipeline_tag: text-generation
library_name: transformers
---

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6512d9827fccffe1e9e28fa7/UjERLVyWYYPVacsN03Rqi.png)

## Overview

HyperCLOVAX-SEED-Text-Instruct-0.5B is a Text-to-Text model with instruction-following capabilities that excels in understanding Korean language and culture. Compared to external competitors of similar scale, it demonstrates improved mathematical performance and a substantial enhancement in Korean language capability. The HyperCLOVAX-SEED-Text-Instruct-0.5B is currently the smallest model released by the HyperCLOVAX, representing a lightweight solution suitable for deployment in resource‑constrained environments such as edge devices. It supports a maximum context length of 4K and functions as a versatile small model applicable to a wide range of tasks. The total cost of a single training run for HyperCLOVAX-SEED-Text-Instruct-0.5B was 4.358K A100 GPU hours (approximately USD 6.537K), which is 39 times lower than the cost of training the `QWEN2.5‑0.5B‑instruct` model.


## Basic Information

- **Architecture**: Transformer‑based (Dense Model)  
- **Parameters**: 0.57 B (total); 0.45 B (excluding token embeddings, tied embeddings)  
- **Input/Output Format**: Text / Text  
- **Maximum Context Length**: 4 K tokens  
- **Knowledge Cutoff Date**: Trained on data up to January 2025


## Training and Data

The training dataset for HyperCLOVAX-SEED-Text-Instruct-0.5B consists of diverse sources, including the high‑quality data accumulated during the development of HyperCLOVAX-SEED-Text-Instruct-0.5B. Training was conducted in three main stages:
1. **Pretraining**: Knowledge acquisition using high‑quality data and a high‑performance pretrained model.  
2. **Rejection Sampling Fine‑Tuning (RFT)**: Enhancement of multi‑domain knowledge and complex reasoning capabilities.  
3. **Supervised Fine‑Tuning (SFT)**: Improvement of instruction‑following proficiency.


## Training Cost

HyperCLOVAX-SEED-Text-Instruct-0.5B leveraged HyperCLOVA X’s lightweight training process and high‑quality data to achieve significantly lower training costs compared to industry‑leading competitors of similar scale. Excluding the SFT stage, a single pretraining run incurred:  

| Pretraining Cost Category       | HyperCLOVAX-SEED-Text-Instruct-0.5B  | QWEN2.5‑0.5B‑instruct  |
|---------------------------------|-----------------------------------------------|-------------------------------------|
| **A100 GPU Hours**              | 4.358 K                                       | 169.257 K                           |
| **Cost (USD)**                  | 6.537 K                                       | 253.886 K                           |

This represents approximately a 39× reduction in pretraining cost relative to `QWEN2.5‑0.5B-instruct`.

## Benchmarks

| **Model** | **KMMLU (5-shot, acc)** | **HAE-RAE (5-shot, acc)** | **CLiCK (5-shot, acc)** | **KoBEST (5-shot, acc)** |
| --- | --- | --- | --- | --- | 
| HyperCLOVAX-SEED-Text-Base-0.5B | 0.4181 | 0.6370 | 0.5373 | 0.6963
| HyperCLOVAX-SEED-Text-Instruct-0.5B | 0.3815 | 0.5619 | 0.4446 | 0.6299 | 
| QWEN2.5-0.5B-instruct        | 0.2968 | 0.3428 | 0.3805 | 0.5025 | 

## HuggingFace Usage Example

### Python Code
For better inference results with `HyperCLOVAX-SEED-Text-Instruct-0.5B`, we recommend setting `repetition_penalty` to `1.2`.
```python
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

chat = [
    {"role": "tool_list", "content": ""},
    {"role": "system", "content": "- AI 언어모델의 이름은 \"CLOVA X\" 이며 네이버에서 만들었다.\n- 오늘은 2025년 04월 24일(목)이다."},
    {"role": "user", "content": "슈뢰딩거 방정식과 양자역학의 관계를 최대한 자세히 알려줘."},
]

inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = inputs.to("cuda")
output_ids = model.generate(
    **inputs,
    max_length=1024,
    stop_strings=["<|endofturn|>", "<|stop|>"],
    repetition_penalty=1.2,
    tokenizer=tokenizer
    )
print(tokenizer.batch_decode(output_ids)[0])
```

### Result
```bash
[
  '<|im_start|>tool_list\n<|im_end|>\n' \
  '<|im_start|>system\n- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 오늘은 2025년 04월 24일(목)이다.<|im_end|>\n' \
  '<|im_start|>user\n슈뢰딩거 방정식과 양자역학의 관계를 최대한 자세히 알려줘.<|im_end|>\n' \
  '<|im_start|>assistant\n양자역학은 슈뢰딩거 방정식을 통해 물질과 에너지, 공간 등의 현상을 설명합니다.\n\n**1. 슈뢰딩거 방정식**\n\n슈뢰딩거는 파동함수를 이용하여 입자의 위치와 운동량을 계산할 수 있다고 주장했습니다. 이를 위해 다음과 같은 식으로 표현됩니다:\n\n$$\\frac{\\partial \\psi}{\\partial t} = iH \\nabla^2 \\psi + V(x)\\psi $$\n\n여기서 $\\psi$는 파동함수이고 $i$는 허수 단위입니다. 여기서 $t$는 시간, $x$는 공간 좌표이며, $H$는 해밀턴 상수로 시스템의 에너지를 나타냅니다. 또한 $V(x)$는 외부 힘이나 장벽에 의해 영향을 받는 부분을 나타내는 함수로, 일반적으로 전위장을 사용합니다.\n\n**2. 양자역학과 슈뢰딩거 방정식의 관계**\n\n양자역학에서는 슈뢰딩거 방정식이 매우 중요한 역할을 합니다. 이는 모든 물리적 시스템이 불확정성 원리에 따라 행동을 하며, 이러한 시스템들은 확률적으로 상태를 가질 수밖에 없기 때문입니다. 따라서 슈뢰딩거 방정식은 양자역학을 수학적으로 모델링하는 핵심적인 도구 중 하나입니다.\n\n예를 들어, 원자핵 내의 전자들의 상태는 슈뢰딩거 방정식에 의해 결정되며, 이는 물리학적 법칙을 따르는 것으로 보입니다. 또한, 광전 효과에서도 슈뢰딩거 방정식은 빛이 물질 내에서 어떻게 흡수되고 반사되는지를 예측하는데 사용됩니다.\n\n**3. 응용 분야**\n\n슈뢰딩거 방정식은 다양한 분야에서 활용되고 있습니다. 예를 들면, 반도체 기술에서의 트랜지스터 설계, 핵물리학에서의 방사성 붕괴 연구 등이 있으며, 이는 모두 슈뢰딩거 방정식을 기반으로 한 이론적 기반 위에서 이루어집니다.\n\n또한, 현대 과학 기술의 발전에도 큰 기여를 하고 있는데, 특히 인공지능(AI), 컴퓨터 시뮬레이션 등에서 복잡한 문제를 해결하고 새로운 지식을 창출하기 위한 기초가 되고 있습니다.\n\n결론적으로, 슈뢰딩거 방정식은 양자역학의 기본 개념들을 이해하고 해석하며, 그 결과로서 많은 혁신적이고 실용적인 기술을 가능하게 했습니다. 이는 양자역학의 중요성을 보여주는 대표적인 예시라고 할 수 있습니다.<|im_end|>' \
  '<|endofturn|>'
]
```