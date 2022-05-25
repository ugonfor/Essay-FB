

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
GECToRConfig['model'] = {
                            "vocab_path": 'gector-large/data/output_vocabulary',
                            "max_len": args.max_len, min_len=args.min_len,
                            "iterations": args.iteration_count,
                            "min_error_probability": args.min_error_probability,
                            "min_probability": args.min_error_probability,
                            "lowercase_tokens": args.lowercase_tokens,
                            "model_name": args.transformer_model,
                            "special_tokens_fix": args.special_tokens_fix,
                            "log": False,
                            "confidence": args.additional_confidence,
                            "is_ensemble": args.is_ensemble,
                            "weigths": args.weights,
                            "use_cpu": bool(args.use_cpu)
                        }

GECToRConfig['cuda_device_index'] = -1 # cuda index