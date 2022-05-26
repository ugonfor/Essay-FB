import sys, os
sys.path.append(os.getcwd() + "/gector")
sys.path.append(os.getcwd() + "/LM-Critic")

from typing import Dict

from config import T5Config, GECToRConfig
from transformers import T5ForConditionalGeneration, T5Tokenizer

from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel

import argparse


class GECT5:
    def __init__(self, input_file, output_file, batch_size, model_name='t5-base', device = 'cuda:0'):
        # model names
        # vennify/t5-base-grammar-correction
        # Unbabel/gec-t5_small
        # deep-learning-analytics/GrammarCorrector

        self.model_name = model_name
        self.device = device

        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size

        self.config = T5Config

        self.model = None
        self.tokenizer = None
    
    def init(self):
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

    def preprocess_sent(self, sentence, is_batch=False):
        # vennify/t5-base-grammar-correction
        # Unbabel/gec-t5_small
        # deep-learning-analytics/GrammarCorrector
        if self.model_name == 'deep-learning-analytics/GrammarCorrector':
            prefix = ""
        elif self.model_name == 'Unbabel/gec-t5_small':
            prefix = "gec: "
        
        elif self.model_name == 'vennify/t5-base-grammar-correction':
            prefix = "grammar: "
        else:
            prefix = ""

        if is_batch:
            assert type(sentence) == list
            return list(map(lambda x: prefix + x, sentence))
        else:
            return prefix + sentence

    def correction(self, sentence, is_batch=False):
        
        pre_sentence = self.preprocess_sent(sentence, is_batch)

        tokenized_sentence = self.tokenizer(pre_sentence, **self.config['tokenizer']).to(self.device)
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

        corrected_sentence.tolist()
        cnt = list(map(lambda x,y: 0 if x==y else 1, sentence, corrected_sentence))
        return corrected_sentence, sum(cnt)
    
    def predict_for_file(self, input_file, output_file, batch_size=32):
        test_data = read_lines(input_file)
        predictions = []
        cnt_corrections = 0
        batch = []
        for sent in test_data:
            batch.append(sent)
            if len(batch) == batch_size:
                preds, cnt = self.correction(batch, is_batch=True)
                predictions.extend(preds)
                cnt_corrections += cnt
                batch = []
        if batch:
            preds, cnt = self.correction(batch, is_batch=True)
            predictions.extend(preds)
            cnt_corrections += cnt

        result_lines = [x for x in predictions]

        with open(output_file, 'w') as f:
            f.write("\n".join(result_lines) + '\n')
        return cnt_corrections

    def predict(self):
        # get all paths
        self.init()

        cnt_corrections = self.predict_for_file(self.input_file, self.output_file, batch_size=self.batch_size)

        # evaluate with m2 or ERRANT
        print(f"Produced overall corrections: {cnt_corrections} (lines)")




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
        model = GecBERTModel(model_paths=self.model_path, **self.config)

        cnt_corrections = self.predict_for_file(self.input_file, self.output_file, model,
                                        batch_size=self.batch_size, 
                                        to_normalize=self.normalize)

        # evaluate with m2 or ERRANT
        print(f"Produced overall corrections: {cnt_corrections} (words)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file',
                        help='for both',
                        required=True)
    parser.add_argument('--output_file',
                        help='for both',
                        required=True)
    parser.add_argument('--batch_size',
                        help='for both',
                        required=True)

    parser.add_argument('--model_path',
                        help='for GECToR', nargs='+',
                        required=True)
    parser.add_argument('--model_name',
                        help='for GECT5',
                        default='t5-base')

    args = parser.parse_args()

    gect5  = GECT5(model_name=args.model_name, input_file=args.input_file, output_file=args.output_file+'.T5', batch_size=args.batch_size)
    gector = GECToR(model_path=args.model_path, input_file=args.input_file, output_file=args.output_file+'.GECToR', batch_size=args.batch_size)

    gect5.predict()
    gector.predict()