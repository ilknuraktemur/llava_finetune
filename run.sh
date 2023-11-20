
git clone https://github.com/haotian-liu/LLaVA.git
python3 -m venv project3
source project3/bin/activate
apt update
apt-get install unzip
cd LLaVA
ip install --upgrade pip  # enable PEP 660 support
pip install -e
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
git pull
pip install -e .
pip install deepspeed
cd /workspace/ilknur
mkdir sample_data
cd sample_data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jPM02wNK59Dj3RmqEqu03fPPC33XCpte' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jPM02wNK59Dj3RmqEqu03fPPC33XCpte" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip -d data
wget -O data.json https://drive.google.com/uc?id=1Cgl6magIQZHUB05aWq1ZzFxc0nmgp-bv
cd /workspace/ilknur
python change.py
##run this on LLaVa path
# Hedef dosyada belirli bir satırı değiştir
sed -i '8s~.*~\t--data_path /workspace/ilknur/sample_data/data2.json \\~' "/workspace/ilknur/LLaVA/scripts/v1_5/finetune_task_lora.sh"
sed -i '9s~.*~\t--image_folder /workspace/ilknur/sample_data/data \\~' "/workspace/ilknur/LLaVA/scripts/v1_5/finetune_task_lora.sh"
