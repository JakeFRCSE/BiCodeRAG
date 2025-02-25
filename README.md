# Current Branch is dev/llama!
To return to the main branch, [click here!](https://github.com/JakeFRCSE/CrossDecoder)

To view weekly research progress report (which is in korean), [click here!](https://crystal-air-942.notion.site/1a041c6bef1680e68685f7890655201b)

# How to use
## 1. Import model and tokenizer.
```python
import sys
sys.path.append("the_directory_that_contains_modeling_llama.py")
from modeling_llama import LlamaCrossDecoderLM
from transformers import AutoTokenizer
```
## 2. Initialize the model and the tokenizer.
```python
# Make sure you have an access to this model on HuggingFace
llama_cross_decoder_lm = LlamaCrossDecoderLM.from_pretrained("meta-llama/llama-3.2-1B")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-3.2-1B")
#The next line will be deleted after customizing LlamaConfig
llama_tokenizer.pad_token_id = 128002
```
## 3. Prepare input data.
```python
data = {
    "passage": "세계 2차 세계대전은 1939년 9월 1일 독일의 폴란드 침공으로 시작되었습니다. 이 전쟁의 원인은 다양하며, 제1차 세계대전 후의 베르사유 조약으로 인한 독일의 불만, 경제 대공황으로 인한 세계적인 경제 위기, 그리고 나치 독일의 영토 확장 정책 등이 주요 요인으로 작용했습니다. 또한, 일본의 아시아 지역 확장 전략과 이탈리아의 지중해 패권 추구도 전쟁의 발발에 기여했습니다.",
    "question": "세계 2차 세계대전의 주요 원인은 무엇이었나요?",
    "answer": "세계 2차 세계대전의 주요 원인은 제1차 세계대전 후의 베르사유 조약으로 인한 독일의 불만, 경제 대공황으로 인한 세계적인 경제 위기, 나치 독일의 영토 확장 정책, 일본의 아시아 지역 확장 전략, 그리고 이탈리아의 지중해 패권 추구 등 여러 요인이 복합적으로 작용했습니다."
}
encoder_input = data["question"] + " " + data["passage"]
decoder_input = data["question"]
answer = data["answer"]
```
## 4. Tokenize the input data.
```python
encoder_input_tokenized = llama_tokenizer(encoder_input, padding=True, return_tensors="pt")
decoder_input_tokenized = llama_tokenizer(decoder_input, padding=False, return_tensors="pt")
```
## 5. Set encoder_outputs.
```python
encoder_output = llama_cross_decoder_lm.encode(encoder_input_tokenized.input_ids, encoder_input_tokenized.attention_mask)
llama_cross_decoder_lm.set_encoder_outputs(encoder_output[0], encoder_input_tokenized.attention_mask)
```
## 6. Generate with the model.
```
generated_output = llama_cross_decoder_lm.generate(decoder_input_tokenized.input_ids, attention_mask = decoder_input_tokenized.attention_mask, pad_token_id=llama_tokenizer.pad_token_id)
result = llama_tokenizer.batch_decode(generated_output)
```

# Contributions of Current Branch (dev/llama)
1. LlamaCrossAttention
2. LlamaCrossDecoderLayer
3. LlamaEncodeModel
4. LlamaCrossModel
5. LlamaCrossDecoderLM
