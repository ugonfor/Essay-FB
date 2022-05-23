

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
                                "num_return_sequences":4 # return sequences
                                }


GECToRConfig = dict()
