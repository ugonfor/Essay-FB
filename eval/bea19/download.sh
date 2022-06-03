wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz
tar xvf fce_v2.1.bea19.tar.gz

python preprocess.py

python m2-correction.py ./fce/m2/fce.test.gold.bea19.m2 > ./gold.bea19

rm ./fce_v2.1.bea19.tar.gz
rm -rf ./fce