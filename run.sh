ROOT=$PWD

# package
pip install -r requirements.txt

wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th

# test data
echo "this is incrorect sentences.
This is incrrect sentences." > input

python gec.py --input_file ./input --output_file ./output --batch_size 32 --model_path ./roberta_1_gectorv2.th --model_name t5-base


# bea 2019 data
cd ${ROOT}/data
bash ./download.sh

# scoring
cd ${ROOT}
python gec.py --input_file ./data/input.bea2019 --output_file ./data/output.bea2019.1 --batch_size 32 --model_path ./roberta_1_gectorv2.th --model_name deep-learning-analytics/GrammarCorrector
python gec.py --input_file ./data/input.bea2019 --output_file ./data/output.bea2019.2 --batch_size 32 --model_path ./roberta_1_gectorv2.th --model_name deep-learning-analytics/GrammarCorrector
python gec.py --input_file ./data/input.bea2019 --output_file ./data/output.bea2019.3 --batch_size 32 --model_path ./roberta_1_gectorv2.th --model_name deep-learning-analytics/GrammarCorrector
python gec.py --input_file ./data/input.bea2019 --output_file ./data/output.bea2019.4 --batch_size 32 --model_path ./roberta_1_gectorv2.th --model_name deep-learning-analytics/GrammarCorrector

# errant / spacy 
#pip install errant
python -m spacy download en

# annotation
errant_parallel -orig ./data/input.bea2019  -cor ./data/output.bea2019.GECToR -out ./data/bea2019.hyp

# scoring
errant_compare -hyp ./data/bea2019.hyp -ref ./data/fce/m2/fce.test.gold.bea19.m2