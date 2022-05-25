import sys, os
sys.path.append(os.getcwd() + "/gector-large")
sys.path.append(os.getcwd() + "/LM-Critic")

from typing import Dict

from config import T5Config, GECToRConfig
from transformers import T5ForConditionalGeneration, T5Tokenizer

from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel



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
        if self.model_name == 'vennify/t5-base-grammar-correction':
            prefix = ""
        elif self.model_name == 'Unbabel/gec-t5_small':
            prefix = "gec: "
        
        elif self.model_name == 'vennify/t5-base-grammar-correction':
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
    def __init__(self, model_path, input_file, output_file, batch_size, to_normalize=False):
        self.model_path = model_path
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size

        self.config = GECToRConfig

        self.normalize = to_normalize



    def predict_for_file(self, input_file, output_file, model, batch_size=32, to_normalize=False):
        test_data = read_lines(input_file)
        predictions = []
        cnt_corrections = 0
        batch = []
        for sent in test_data:
            batch.append(sent.split())
            if len(batch) == batch_size:
                preds, cnt = model.handle_batch(batch)
                predictions.extend(preds)
                cnt_corrections += cnt
                batch = []
        if batch:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt

        result_lines = [" ".join(x) for x in predictions]
        if to_normalize:
            result_lines = [normalize(line) for line in result_lines]

        with open(output_file, 'w') as f:
            f.write("\n".join(result_lines) + '\n')
        return cnt_corrections


    def predict(self):
        # get all paths
        model = GecBERTModel(model_paths=self.model_paths, **self.config)

        cnt_corrections = self.predict_for_file(self.input_file, self.output_file, model,
                                        batch_size=self.batch_size, 
                                        to_normalize=self.normalize)

        # evaluate with m2 or ERRANT
        print(f"Produced overall corrections: {cnt_corrections}")