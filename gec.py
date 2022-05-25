import sys, os
sys.path.append(os.getcwd() + "/gector-large")
sys.path.append(os.getcwd() + "/LM-Critic")

from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict

from config import T5Config, GECToRConfig

import time
import datetime
from utils.helpers import read_lines
from gector.gec_model import GecBERTModel
import torch
from difflib import SequenceMatcher


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
    def __init__(self, model_path, input_file, output_file, batch_size, save_logs):
        self.model_path = model_path
        self.config = GECToRConfig
        self.input_file = input_file
        self.output_file = output_file
        
        self.batch_size = batch_size
        self.save_logs = save_logs

    def predict(self, model_paths):
        # get all paths
        # if args.count_thread != -1:
        #     torch.set_num_threads = str(args.count_thread)
        #     os.environ["OMP_NUM_THREADS"] = str(args.count_thread)
        #     os.environ["MKL_NUM_THREADS"] = str(args.count_thread)
        
        if self.config['cuda_device_index'] != -1:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config['cuda_device_index'])
            os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        
        model = GecBERTModel(model_paths=model_paths, **self.config['model'])

        cnt_corrections = self.predict_for_file(self.input_file, self.output_file, model,
                                        batch_size=self.batch_size, save_logs=self.save_logs)
        # evaluate with m2 or ERRANT
        print(f"Produced overall corrections: {cnt_corrections}")



    def generate_text_for_log(self, processed_lines, total_lines, corrected_lines, prediction_duration, cnt_corrections):
        return "Processed lines: "+str(processed_lines)+"/"+str(total_lines)+" = "+ str(round(100*processed_lines/total_lines, 2))+"%\n"+ "Corrected lines: "+ str(corrected_lines)+"/"+str(processed_lines)+" = "+ str(round(100*corrected_lines/processed_lines, 2))+"%\n"+ "Prediction duration: "+ str(prediction_duration)+"\n"+ "Total corrections: "+str(cnt_corrections)


    def check_corrected_line(self, source_tokens, target_tokens):
        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        raw_diffs = list(matcher.get_opcodes())
        if len(raw_diffs) == 1:
            if raw_diffs[0][0] == 'equal':
                return 0
        return 1    
        
    def get_corrected_lines_for_batch(self, source_batch, target_batch):
        corrected = []
        for source, target in zip(source_batch, target_batch):
            corrected.append(self.check_corrected_line(source, target))
        return corrected
                                
    def predict_for_file(self, input_file, output_file, model, batch_size=32, save_logs=0):
        test_data = read_lines(input_file)
    #     predictions = []
        cnt_corrections = 0
        batch = []
        with open(output_file, 'w') as f:
            f.write("")
        
        if save_logs:
            with open(output_file+".log", 'w') as f:
                f.write("")

            with open(output_file+".check_correction", 'w') as f:
                f.write("")
        
        predicting_start_time = time.time()
        
        total_lines = len(test_data)
        processed_lines = 0
        corrected_lines = 0
        
        for sent in test_data:
            batch.append(sent.split())
            if len(batch) == batch_size:
                preds, cnt = model.handle_batch(batch)
                
                processed_lines += batch_size
                
                pred_sents = [" ".join(x) for x in preds]
                
                with open(output_file, 'a') as f:
                    f.write("\n".join(pred_sents) + '\n')
                    
                cnt_corrections += cnt
                
                if save_logs:
                    checked_lines = self.get_corrected_lines_for_batch(batch, preds)
                    corrected_lines += sum(checked_lines)
                    checked_lines = [str(s) for s in checked_lines]
                    with open(output_file+".check_correction", 'a') as f:
                        f.write("\n".join(checked_lines) + '\n')
                
                    predicting_elapsed_time = time.time() - predicting_start_time
                    prediction_duration = datetime.timedelta(seconds=predicting_elapsed_time)

                    with open(output_file+".log", 'w') as f:
                        f.write(self.generate_text_for_log(processed_lines, total_lines, corrected_lines, prediction_duration, cnt_corrections))


                batch = []
        if batch:
            preds, cnt = model.handle_batch(batch)
            processed_lines += len(batch)
            pred_sents = [" ".join(x) for x in preds]   
            
            with open(output_file, 'a') as f:
                f.write("\n".join(pred_sents) + '\n')
            
            cnt_corrections += cnt
            
            checked_lines = self.get_corrected_lines_for_batch(batch, preds)    
            corrected_lines += sum(checked_lines)
            checked_lines = [str(s) for s in checked_lines]

            if save_logs:
            
                with open(output_file+".check_correction", 'a') as f:
                        f.write("\n".join(checked_lines) + '\n')


                predicting_elapsed_time = time.time() - predicting_start_time
                prediction_duration = datetime.timedelta(seconds=predicting_elapsed_time)

                with open(output_file+".log", 'w') as f:
                    f.write(self.generate_text_for_log(processed_lines, total_lines, corrected_lines, prediction_duration, cnt_corrections))
        
        predicting_elapsed_time = time.time() - predicting_start_time
        prediction_duration = datetime.timedelta(seconds=predicting_elapsed_time)
        
        print(prediction_duration)
        
        return cnt_corrections


