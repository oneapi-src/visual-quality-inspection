#! /bin/bash
echo $USER_CONSENT | python -m dataset_librarian.dataset -n mvtec-ad --download --preprocess -d /workspace/data
mkdir -p /workspace/data/{train/{good,bad},test/{good,bad}} 
cd /workspace/data/pill/train/good/ 
cp $(ls | head -n 210) /workspace/data/train/good/ 
cp $(ls | tail -n 65) /workspace/data/test/good/ 
cd /workspace/data/pill/test/combined 
cp $(ls | head -n 17) /workspace/data/train/bad/ 
cp $(ls | tail -n 5) /workspace/data/test/bad/
cd /workspace/src
python clone_dataset.py -d  /workspace/data/

