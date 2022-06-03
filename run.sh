ROOT=$PWD
# custom next two data path!
TARGET=./input.txt # input path
OUTPUT=./output.txt # output path

cd $ROOT
pip install -r requirements.txt
wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th


# GECToR RoBERTa
python gec.py --input_file $TARGET --output_file $OUTPUT --batch_size 32 --model gector --gector_path roberta_1_gectorv2.th
