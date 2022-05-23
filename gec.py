import sys, os
sys.path.append(os.getcwd() + "/gector-large")
sys.path.append(os.getcwd() + "/LM-Critic")

from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict

from config import T5Config


class GECT5:
    def __init__(self, model_name='t5-base', device = 'cuda:0'):
        # model names
        # vennify/t5-base-grammar-correction
        # Unbabel/gec-t5_small
        # deep-learning-analytics/GrammarCorrector

        self.model_name = model_name
        self.device = device

        self.model = None
        self.tokenizer = None
        self.config = None
    
    def init(self, config : Dict):
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.config = config

    def preprocess_sent(self, sentence, is_batch=False):
        # vennify/t5-base-grammar-correction
        # Unbabel/gec-t5_small
        # deep-learning-analytics/GrammarCorrector
        if self.model_name == 'vennify/t5-base-grammar-correction'
            prefix = ""
        elif self.model_name == 'Unbabel/gec-t5_small'
            prefix = "gec: "
        
        elif self.model_name == 'vennify/t5-base-grammar-correction'
            prefix = "grammar: "
        else:
            prefix = ""

        if is_batch:
            assert type(sentence) == list
            return list(map(lambda x: prefix + x), sentence)
        else:
            return prefix + sentence

    def correction(self, sentence, is_batch=False):
        
        sentence = self.preprocess_sent(sentence, is_batch)

        tokenized_sentence = self.tokenizer(sentence, **self.config['tokenizer'])
        generated_sentence = self.model.generate(
                            input_ids = tokenized_sentence.input_ids,
                            attention_mask = tokenized_sentence.attention_mask,
                            **self.config['model_generate']
                            )
        corrected_sentence = self.tokenizer.batch_decode(
            generated_sentence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return corrected_sentence


class GECToR:
    pass