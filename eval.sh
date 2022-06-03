ROOT=$PWD

# download dataset
cd $ROOT/eval/bea19/
bash download.sh

cd $ROOT/eval/jfleg/
bash download.sh

cd $ROOT
pip install -r requirements.txt

wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th
wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gectorv2.th


# vennify/t5-base-grammar-correction
## bea19
python gec.py --input_file ./eval/bea19/input.bea19 --output_file ./eval/bea19/output.bea19 --batch_size 32 --model t5 --t5_name vennify/t5-base-grammar-correction
## jfleg
python gec.py --input_file ./eval/jfleg/test/test.src --output_file ./eval/jfleg/test/test.src.output --batch_size 32 --model t5 --t5_name vennify/t5-base-grammar-correction

cd $ROOT
python eval/eval_gleu/gleu.py -r ./eval/bea19/gold.bea19 -s ./eval/bea19/input.bea19 --hyp ./eval/bea19/output.bea19
# 0.454695
python eval/eval_gleu/gleu.py -r ./eval/jfleg/test/test.ref[0-3] -s ./eval/jfleg/test/test.src --hyp ./eval/jfleg/test/test.src.output
# Running GLEU...
# ./eval/jfleg/test/test.src.output
# [['0.482342', '0.006745', '(0.469,0.496)']]



# Unbabel/gec-t5_small
## bea19
python gec.py --input_file ./eval/bea19/input.bea19 --output_file ./eval/bea19/output.bea19 --batch_size 32 --model t5 --t5_name Unbabel/gec-t5_small
## jfleg
python gec.py --input_file ./eval/jfleg/test/test.src --output_file ./eval/jfleg/test/test.src.output --batch_size 32 --model t5 --t5_name Unbabel/gec-t5_small

cd $ROOT
python eval/eval_gleu/gleu.py -r ./eval/bea19/gold.bea19 -s ./eval/bea19/input.bea19 --hyp ./eval/bea19/output.bea19
# 0.484990
python eval/eval_gleu/gleu.py -r ./eval/jfleg/test/test.ref[0-3] -s ./eval/jfleg/test/test.src --hyp ./eval/jfleg/test/test.src.output
#Running GLEU...
#./eval/jfleg/test/test.src.output
#[['0.431214', '0.005777', '(0.420,0.443)']]



# deep-learning-analytics/GrammarCorrector
## bea19
python gec.py --input_file ./eval/bea19/input.bea19 --output_file ./eval/bea19/output.bea19 --batch_size 32 --model t5 --t5_name deep-learning-analytics/GrammarCorrector
## jfleg
python gec.py --input_file ./eval/jfleg/test/test.src --output_file ./eval/jfleg/test/test.src.output --batch_size 32 --model t5 --t5_name deep-learning-analytics/GrammarCorrector

cd $ROOT
python eval/eval_gleu/gleu.py -r ./eval/bea19/gold.bea19 -s ./eval/bea19/input.bea19 --hyp ./eval/bea19/output.bea19
#0.467454
python eval/eval_gleu/gleu.py -r ./eval/jfleg/test/test.ref[0-3] -s ./eval/jfleg/test/test.src --hyp ./eval/jfleg/test/test.src.output
#Running GLEU...
#./eval/jfleg/test/test.src.output
#[['0.391524', '0.006299', '(0.379,0.404)']]


# GECToR XLNet
## bea19
python gec.py --input_file ./eval/bea19/input.bea19 --output_file ./eval/bea19/output.bea19 --batch_size 32 --model gector --gector_path xlnet_0_gectorv2.th
## jfleg
python gec.py --input_file ./eval/jfleg/test/test.src --output_file ./eval/jfleg/test/test.src.output --batch_size 32 --model gector --gector_path xlnet_0_gectorv2.th

cd $ROOT
python eval/eval_gleu/gleu.py -r ./eval/bea19/gold.bea19 -s ./eval/bea19/input.bea19 --hyp ./eval/bea19/output.bea19
#0.457790
python eval/eval_gleu/gleu.py -r ./eval/jfleg/test/test.ref[0-3] -s ./eval/jfleg/test/test.src --hyp ./eval/jfleg/test/test.src.output
#Running GLEU...
#./eval/jfleg/test/test.src.output.gector.xln
#[['0.329170', '0.006566', '(0.316,0.342)']]



# GECToR RoBERTa
## bea19
python gec.py --input_file ./eval/bea19/input.bea19 --output_file ./eval/bea19/output.bea19 --batch_size 32 --model gector --gector_path roberta_1_gectorv2.th
## jfleg
python gec.py --input_file ./eval/jfleg/test/test.src --output_file ./eval/jfleg/test/test.src.output --batch_size 32 --model gector --gector_path roberta_1_gectorv2.th

cd $ROOT
python eval/eval_gleu/gleu.py -r ./eval/bea19/gold.bea19 -s ./eval/bea19/input.bea19 --hyp ./eval/bea19/output.bea19
#0.693566
python eval/eval_gleu/gleu.py -r ./eval/jfleg/test/test.ref[0-3] -s ./eval/jfleg/test/test.src --hyp ./eval/jfleg/test/test.src.output
#Running GLEU...
#./eval/jfleg/test/test.src.output.gector.xln
#[['0.598976', '0.007676', '(0.584,0.614)']]
