# Current Branch is dev/llama!
To return to the main branch, [click here!](https://github.com/JakeFRCSE/BiCodeRAG)

To view weekly research progress report (which is in korean), [click here!](https://crystal-air-942.notion.site/1a041c6bef1680e68685f7890655201b)

# How to use LlamaBiCodeLM
1. Add a path for modeling_llama.py
```python
import sys
sys.path.append("YOUR_DIRECTORY_FOR_MODELING_LLAMA.PY")
```
2. Import model and tokenizer.
```python
from modeling_llama import LlamaBiCodeLM
from transformers import AutoTokenizer
```
3. Load the model and tokenizer.
```python
model_name = "meta-llama/llama-3.2-1B"
llama_bi_code_lm = LlamaBiCodeLM.from_pretrained(model_name)
llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
# Add a padding token
llama_tokenizer.pad_token_id = llama_bi_code_lm.pad_token_id
llama_tokenizer.pad_token = llama_tokenizer.added_tokens_decoder[llama_tokenizer.pad_token_id].content
```
4. Use GPU.
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
5. Load data.
```python
data_0 = {
    "passage": "세계 2차 세계대전은 1939년 9월 1일 독일의 폴란드 침공으로 시작되었습니다. 이 전쟁의 원인은 다양하며, 제1차 세계대전 후의 베르사유 조약으로 인한 독일의 불만, 경제 대공황으로 인한 세계적인 경제 위기, 그리고 나치 독일의 영토 확장 정책 등이 주요 요인으로 작용했습니다. 또한, 일본의 아시아 지역 확장 전략과 이탈리아의 지중해 패권 추구도 전쟁의 발발에 기여했습니다.",
    "question": "세계 2차 세계대전의 주요 원인은 무엇이었나요?",
    "answer": "세계 2차 세계대전의 주요 원인은 제1차 세계대전 후의 베르사유 조약으로 인한 독일의 불만, 경제 대공황으로 인한 세계적인 경제 위기, 나치 독일의 영토 확장 정책, 일본의 아시아 지역 확장 전략, 그리고 이탈리아의 지중해 패권 추구 등 여러 요인이 복합적으로 작용했습니다."
}
data_1 = {
    "passage":"애플은 2024년 3월, 새로운 M3 칩이 탑재된 MacBook Air를 출시했다. M3 칩은 이전 세대보다 향상된 성능과 전력 효율성을 제공한다." + " " + "MacBook Air M3는 15인치 모델도 제공되며, 최대 24GB RAM과 최대 2TB SSD 옵션을 지원한다." + " " + "M3 칩은 3나노미터 공정으로 제작되었으며, 이전 M2 칩 대비 멀티코어 성능이 20% 향상되었다.",
    "question":"MacBook Air M3의 특징은 무엇인가요?",
    "answer":"MacBook Air M3는 2024년 3월 출시되었으며, 3나노미터 공정으로 제작된 M3 칩을 탑재하여 이전 세대보다 20% 향상된 멀티코어 성능을 제공합니다. 또한, 15인치 모델이 추가되었고, 최대 24GB RAM과 2TB SSD 옵션을 지원합니다."
}
encoder_input_0 = data_0["question"] + " " + data_0["passage"]
decoder_input_0 = data_0["question"]
answer_0 = data_0["answer"]
encoder_input_1 = data_1["question"] + " " + data_1["passage"]
decoder_input_1 = data_1["question"]
answer_1 = data_1["answer"]
encoder_input_tokenized = llama_tokenizer([encoder_input_0, encoder_input_1], padding=True, return_tensors="pt").to(device)
decoder_input_tokenized = llama_tokenizer([decoder_input_0, decoder_input_1], padding=True, padding_side = "left", return_tensors="pt").to(device)
```
6. Encode and Generate.
```python
# Encoding process.
llama_bi_code_lm_output = llama_bi_code_lm.encode(encoder_input_tokenized.input_ids, encoder_input_tokenized.attention_mask)
llama_bi_code_lm.set_cross_inputs(llama_bi_code_lm_output[0], encoder_input_tokenized.attention_mask)
# Generating process.
llama_bi_code_lm_gen = llama_bi_code_lm.generate(decoder_input_tokenized.input_ids, attention_mask=decoder_input_tokenized.attention_mask, max_length=50, pad_token_id = llama_bi_code_lm.pad_token_id)
```
7. Decode.
```python
llama_tokenizer.batch_decode(llama_bi_code_lm_gen, skip_special_tokens=True)
```

# Contributions of Current Branch (dev/llama)
1. LlamaCrossAttentionLayer
2. LlamaBiCodeModel
3. LlamaBiCodeLM
