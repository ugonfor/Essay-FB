

T5Config = dict()
T5Config['tokenizer'] =         {
                                "max_length":128, 
                                "truncation":True, 
                                "padding":'max_length', 
                                "return_tensors":'pt'
                                }

T5Config['model_generate'] =    {
                                "max_length":128,
                                "num_beams":5,
                                "early_stopping":True,
                                "num_return_sequences":1 # return sequences
                                }


GECToRConfig = dict()
GECToRConfig = {
                            "vocab_path":'gector/data/output_vocabulary',
                            #"model_paths":args.model_path,
                            "max_len":50, 
                            "min_len":3,
                            "iterations":5,
                            "min_error_probability":0.0,
                            "lowercase_tokens":0,
                            "model_name":'roberta',
                            "special_tokens_fix":1,
                            "log":False,
                            "confidence":0,
                            "del_confidence":0,
                            "is_ensemble":0,
                            "weigths":None
                        }
