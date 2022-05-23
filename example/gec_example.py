import torch

''' GEC models
1. GecTor
2. LM-Critic
3. https://huggingface.co/deep-learning-analytics/GrammarCorrector
4. https://huggingface.co/vennify/t5-base-grammar-correction
5. https://huggingface.co/Unbabel/gec-t5_small
'''

# https://huggingface.co/vennify/t5-base-grammar-correction
# The T5-base model has been trained on JFLEG dataset.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")

sentence = "I like to swimming"
tokenized_sentence = tokenizer('gec: ' + sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
corrected_sentence = tokenizer.batch_decode(
    model.generate(
        input_ids = tokenized_sentence.input_ids,
        attention_mask = tokenized_sentence.attention_mask, 
        max_length=128,
        num_beams=5,
        early_stopping=True,
        num_return_sequences=4,
    ),
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=True
)
print(corrected_sentence) # -> I like swimming.



# https://huggingface.co/Unbabel/gec-t5_small
# The T5-base model has been trained on CLANG-8 dataset.
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("Unbabel/gec-t5_small")
tokenizer = T5Tokenizer.from_pretrained('t5-small')

sentence = "I like to swimming"
tokenized_sentence = tokenizer('gec: ' + sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
corrected_sentence = tokenizer.batch_decode(
    model.generate(
        input_ids = tokenized_sentence.input_ids,
        attention_mask = tokenized_sentence.attention_mask, 
        max_length=128,
        num_beams=5,
        early_stopping=True,
        num_return_sequences=4,
    ),
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=True
)
print(corrected_sentence) # -> I like swimming.


# https://huggingface.co/deep-learning-analytics/GrammarCorrector
# The T5-base model has been trained on C4_200M dataset.
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
model_name = 'deep-learning-analytics/GrammarCorrector'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)


sentence = "I like to swimming"
tokenized_sentence = tokenizer(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
corrected_sentence = tokenizer.batch_decode(
    model.generate(
        input_ids = tokenized_sentence.input_ids,
        attention_mask = tokenized_sentence.attention_mask, 
        max_length=128,
        num_beams=5,
        early_stopping=True,
        num_return_sequences=4,
    ),
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=True
)
print(corrected_sentence) # -> I like swimming.

def correct_grammar(input_text,num_return_sequences):
  num_beams = num_return_sequences
  batch = tokenizer([input_text],truncation=True,padding='max_length',max_length=64, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=64,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

text = 'I like to swimming'
print(correct_grammar(text, num_return_sequences=4))
