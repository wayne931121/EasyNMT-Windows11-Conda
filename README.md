First I use python 3.13 and failed, i see debug log, and second i use pytohn 3.12 and failed too, so I was angry there is two package need correct python version , third I use python 3.9. I think that all deep learn project need tell what python they use, or will be error build package. but error not only these, the easynmt script can not correct download model from network, so i need to use linux(colab) to download, them move to my computer... and set more modify some script config and I success run.

Steps:
```
conda create --prefix C:\Users\TEST\fxt
conda activate C:\Users\TEST\fxt
conda install python=3.9

pip install notebook
pip install https://files.pythonhosted.org/packages/64/c5/ae6008631f67085c7189d1407abea468c80000657778af4d4039de0d893b/tokenizers-0.10.3-cp39-cp39-win_amd64.whl
pip install transformers==4.4
pip install numpy nltk
pip install sentencepiece
pip install protobuf
pip install tqdm
pip install https://files.pythonhosted.org/packages/4a/12/b37a1af2a5a09d9234877bc6e1403fae68adee43afc027fc6da7f576e15a/fasttext_wheel-0.9.2-cp39-cp39-win_amd64.whl

#pls see guide:https://pytorch.org/get-started/locally/
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidiacd C:\Users\TEST\fxt
python
    import torch
    torch.cuda.is_available()
	exit()

mkdir content
cd content

git clone https://github.com/UKPLab/EasyNMT.git
cd C:\Users\TEST\fxt\content\EasyNMT
modify setup.py
python setup.py install

pip install setuptools==80
pip install packaging==21.3

python
import nltk
nltk.download("punkt_tab")

from easynmt import EasyNMT
model = EasyNMT(r"C:\Users\TEST\fxt\content\EasyNMT\get")
print(model.translate("我好棒", target_lang="en",source_lang="zh"))
```

In above step, the setup.py:<br>
https://github.com/wayne931121/EasyNMT-Windows11-Conda/blob/main/setup_afterModify.py#L25


My Device Info:
- Windows 11
- Conda

Model:

https://github.com/wayne931121/EasyNMT-Windows11-Conda/releases/tag/model

In windows you can not download model by eazynmt script ...idk api wtf error I nerver see.. so download from colab

In windows is too complex, if you uncomfortable, only use in linux.

LICENSE:
- all code: Apache License Version 2.0
- this readme.md: Attribution 4.0 International, Copyright (c) 2025 wayne931121.

# Example (en to zh model)
https://github.com/wayne931121/EasyNMT-Windows11-Conda/blob/main/fxt.bat<br>
https://github.com/wayne931121/EasyNMT-Windows11-Conda/blob/main/test.py
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/97a3c387-1a41-490b-ae4f-d8d618e6d3a5" />


# Forked from:

https://github.com/UKPLab/EasyNMT (official)

# Reference

https://github.com/UKPLab/EasyNMT/issues/44#issuecomment-908285722

