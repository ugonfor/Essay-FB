
# package
pip install -r requirements.txt

wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th

# test data
echo "this is incrorect sentences." > input

python gec.py --input_file ./input --output_file ./output --batch_size 32 --model_path ./roberta_1_gectorv2.th --model_name t5-base