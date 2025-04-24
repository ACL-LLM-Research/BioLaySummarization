#g4dn
#Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)

git clone https://github.com/linf545UCB/BioLaySumm2025
cd BioLaySumm2025

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init
source ~/.bashrc


conda create --prefix /opt/envs/biolaysum_eval python=3.9
conda activate /opt/envs/biolaysum_eval 

#AWS default tmp have limited disk space, create a new one in the right volume
mkdir -p /opt/tmp
export TMPDIR=/opt/tmp
pip install --no-cache-dir textstat
pip install --no-cache-dir numpy==1.24.1
pip install --no-cache-dir pandas==1.5.3
pip install --no-cache-dir protobuf==6.30.2
pip install --no-cache-dir lens-metric
pip install --no-cache-dir rouge-score
pip install --no-cache-dir bert-score
pip install --no-cache-dir spacy==3.7.5
pip install --no-cache-dir radgraph
pip install --no-cache-dir f1chexbert
pip install --no-cache-dir evaluate
pip install --no-cache-dir sentence-transformers==4.0.2

python -m spacy download en

git clone https://github.com/yuh-zha/AlignScore.git
git clone https://github.com/tingofurro/summac.git


python evaluation_final.py --prediction_file  BioLaySumm2025-eLife_result.json  --groundtruth_file BioLaySumm2025-eLife_result.json --task_name lay_summ
