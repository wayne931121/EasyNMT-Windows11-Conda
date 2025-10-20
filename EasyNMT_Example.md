#EasyNMT - Example (Opus-MT Model)
This notebook shows the usage of [EasyNMT](https://github.com/UKPLab/EasyNMT) for machine translation.

Here, we use the [Opus-MT model](https://github.com/Helsinki-NLP/Opus-MT). The Helsiniki-NLP group provides 1200+ pre-trained models for various language directions (e.g. en-de, es-fr, ru-fr). Each model has a size of about 300 MB.

We make the usage of the models easy: The suitable model needed for your translation is loaded automatically and kept in memory for future use.

# Colab with GPU
When running this notebook in colab, ensure that you run it with a GPU as hardware accelerator. To enable this:
- Navigate to Edit → Notebook Settings
- select GPU from the Hardware Accelerator drop-down

With `!nvidia-smi` we can check which GPU was assigned to us in Colab.


```
!nvidia-smi
```

    Mon Oct 20 13:18:03 2025       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
    | N/A   43C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+
    

# Installation
You can install EasyNMT by using pip. EasyNMT is using Pytorch. If you have a GPU available on your local machine, have a look at [PyTorch Get Started](https://pytorch.org/get-started/locally/) how to install PyTorch with CUDA support.


```
!pip install -U easynmt
```

    Collecting easynmt
      Downloading EasyNMT-2.0.2.tar.gz (23 kB)
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from easynmt) (4.67.1)
    Requirement already satisfied: transformers<5,>=4.4 in /usr/local/lib/python3.12/dist-packages (from easynmt) (4.57.1)
    Requirement already satisfied: torch>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from easynmt) (2.8.0+cu126)
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from easynmt) (2.0.2)
    Requirement already satisfied: nltk in /usr/local/lib/python3.12/dist-packages (from easynmt) (3.9.1)
    Requirement already satisfied: sentencepiece in /usr/local/lib/python3.12/dist-packages (from easynmt) (0.2.1)
    Collecting fasttext (from easynmt)
      Downloading fasttext-0.9.3.tar.gz (73 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m73.4/73.4 kB[0m [31m3.1 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    Requirement already satisfied: protobuf in /usr/local/lib/python3.12/dist-packages (from easynmt) (5.29.5)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (3.20.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (4.15.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (1.13.3)
    Requirement already satisfied: networkx in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (3.1.6)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (2025.3.0)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (12.6.77)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (12.6.77)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (12.6.80)
    Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (9.10.2.21)
    Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (12.6.4.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (11.3.0.4)
    Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (10.3.7.77)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (11.7.1.2)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (12.5.4.2)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (0.7.1)
    Requirement already satisfied: nvidia-nccl-cu12==2.27.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (2.27.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (12.6.77)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (12.6.85)
    Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (1.11.1.6)
    Requirement already satisfied: triton==3.4.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.6.0->easynmt) (3.4.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.34.0 in /usr/local/lib/python3.12/dist-packages (from transformers<5,>=4.4->easynmt) (0.35.3)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from transformers<5,>=4.4->easynmt) (25.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from transformers<5,>=4.4->easynmt) (6.0.3)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers<5,>=4.4->easynmt) (2024.11.6)
    Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from transformers<5,>=4.4->easynmt) (2.32.4)
    Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers<5,>=4.4->easynmt) (0.22.1)
    Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers<5,>=4.4->easynmt) (0.6.2)
    Collecting pybind11>=2.2 (from fasttext->easynmt)
      Using cached pybind11-3.0.1-py3-none-any.whl.metadata (10.0 kB)
    Requirement already satisfied: click in /usr/local/lib/python3.12/dist-packages (from nltk->easynmt) (8.3.0)
    Requirement already satisfied: joblib in /usr/local/lib/python3.12/dist-packages (from nltk->easynmt) (1.5.2)
    Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<1.0,>=0.34.0->transformers<5,>=4.4->easynmt) (1.1.10)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.6.0->easynmt) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.6.0->easynmt) (3.0.3)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->transformers<5,>=4.4->easynmt) (3.4.4)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->transformers<5,>=4.4->easynmt) (3.11)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->transformers<5,>=4.4->easynmt) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->transformers<5,>=4.4->easynmt) (2025.10.5)
    Using cached pybind11-3.0.1-py3-none-any.whl (293 kB)
    Building wheels for collected packages: easynmt, fasttext
      Building wheel for easynmt (setup.py) ... [?25l[?25hdone
      Created wheel for easynmt: filename=EasyNMT-2.0.2-py3-none-any.whl size=19898 sha256=06fbec9cb11232004bb498496ff9799d535045f95b405b5dc6d538168525c768
      Stored in directory: /root/.cache/pip/wheels/1c/5c/5d/d698bb79f4c9ddc0b910bb71d1ddb9048fb3bc7b0ed7ce40ea
      Building wheel for fasttext (pyproject.toml) ... [?25l[?25hdone
      Created wheel for fasttext: filename=fasttext-0.9.3-cp312-cp312-linux_x86_64.whl size=4498207 sha256=4120d300466d44f5e07c033ace6fcbdd05bd7750f3ee2d1bf2bc84599901aff7
      Stored in directory: /root/.cache/pip/wheels/20/27/95/a7baf1b435f1cbde017cabdf1e9688526d2b0e929255a359c6
    Successfully built easynmt fasttext
    Installing collected packages: pybind11, fasttext, easynmt
    Successfully installed easynmt-2.0.2 fasttext-0.9.3 pybind11-3.0.1
    

# Create EasyNMT instance

Creating an EasyNMT instance and loading a model is easy. You pass the model name you want to use and all needed files are downloaded and cached locally.


```
from easynmt import EasyNMT
model = EasyNMT('opus-mt',cache_folder="./")
```

    11.9kB [00:00, 14.4MB/s]                   
    

# Sentence Translation
When you have individual sentences to translate, you can call the method `translate_sentences`.


```
translations = model.translate("我是誰", target_lang='en',source_lang="zh",)
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    tokenizer_config.json:   0%|          | 0.00/44.0 [00:00<?, ?B/s]



    source.spm:   0%|          | 0.00/805k [00:00<?, ?B/s]



    target.spm:   0%|          | 0.00/807k [00:00<?, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    config.json: 0.00B [00:00, ?B/s]


    /usr/local/lib/python3.12/dist-packages/transformers/models/marian/tokenization_marian.py:175: UserWarning: Recommended: pip install sacremoses.
      warnings.warn("Recommended: pip install sacremoses.")
    


    pytorch_model.bin:   0%|          | 0.00/312M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/293 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/312M [00:00<?, ?B/s]



```
print(translations)
```

    Who am I?
    


```
!pip install wfind
```

    Collecting wfind
      Downloading wfind-0.0.1.tar.gz (2.2 kB)
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Building wheels for collected packages: wfind
      Building wheel for wfind (setup.py) ... [?25l[?25hdone
      Created wheel for wfind: filename=wfind-0.0.1-py3-none-any.whl size=2614 sha256=23c8da4fc4a9840e0f3cd378ca80af7d75f920ae6ffdd441833058ef72f2ccf5
      Stored in directory: /root/.cache/pip/wheels/01/e3/64/3aef1e4f082b67805518c1d3558726586b3e87a88914d06400
    Successfully built wfind
    Installing collected packages: wfind
    Successfully installed wfind-0.0.1
    


```
!python -m find "/" "tokenizer_config.json"
!python -m find "/" "source.spm"
```

    /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/tokenizer_config.json
    /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/source.spm
    


```
!python -m find "/" "target.spm"
!python -m find "/" "vocab.json"
!python -m find "/" "config.json"
```

    /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/target.spm
    /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/vocab.json
    /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/config.json
    /root/.julia/packages/Plots/8ZnR3/docs/user_gallery/config.json
    /root/.julia/packages/Plots/8ZnR3/docs/user_gallery/misc/config.json
    /root/.julia/packages/Plots/8ZnR3/docs/gallery/gaston/config.json
    /root/.julia/packages/Plots/8ZnR3/docs/gallery/pythonplot/config.json
    /root/.julia/packages/Plots/8ZnR3/docs/gallery/inspectdr/config.json
    /root/.julia/packages/Plots/8ZnR3/docs/gallery/unicodeplots/config.json
    /root/.julia/packages/Plots/8ZnR3/docs/gallery/pgfplotsx/config.json
    /root/.julia/packages/Plots/8ZnR3/docs/gallery/gr/config.json
    /root/.julia/packages/Plots/8ZnR3/docs/gallery/plotlyjs/config.json
    /root/.julia/packages/TiffImages/hyrVM/docs/demos/config.json
    /tools/google-cloud-sdk/lib/googlecloudsdk/core/config.json
    


```
!python -m find "/" "pytorch_model.bin"
!python -m find "/" "generation_config.json"
!python -m find "/" "model.safetensors"
```

    /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/pytorch_model.bin
    /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/generation_config.json
    /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/.no_exist/cf109095479db38d6df799875e34039d4938aaa6/model.safetensors
    /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/badebd2bdd4cdfde141a969df82a0f2c4e3b1dfe/model.safetensors
    


```
!cp /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/badebd2bdd4cdfde141a969df82a0f2c4e3b1dfe/model.safetensors /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6
```


```
!python -m find "/" "easynmt.json"
```

    /content/opus-mt/easynmt.json
    


```
!dir /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6
```

    config.json		model.safetensors  source.spm  tokenizer_config.json
    generation_config.json	pytorch_model.bin  target.spm  vocab.json
    


```
!dir /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/badebd2bdd4cdfde141a969df82a0f2c4e3b1dfe
```

    model.safetensors
    


```
!cp /content/opus-mt/easynmt.json /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6
```


```
from easynmt import EasyNMT
model = EasyNMT('/root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6')
```


```
translations = model.translate("我是誰", target_lang='en',source_lang="zh",)
```

    /usr/local/lib/python3.12/dist-packages/transformers/models/marian/tokenization_marian.py:175: UserWarning: Recommended: pip install sacremoses.
      warnings.warn("Recommended: pip install sacremoses.")
    


```
print(translations)
```

    Who am I?
    


```
!zip -r get.zip /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6
```

      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/ (stored 0%)
      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/target.spm (deflated 49%)
      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/config.json (deflated 61%)
      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/pytorch_model.bin (deflated 7%)
      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/tokenizer_config.json (deflated 16%)
      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/generation_config.json (deflated 43%)
      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/source.spm (deflated 46%)
      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/vocab.json (deflated 66%)
      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/model.safetensors (deflated 7%)
      adding: root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6/easynmt.json (deflated 73%)
    


```
from google.colab import files
files.download("get.zip")
```


    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>


# USE COLAB BECAUSE WINDOWS: requests.exceptions.MissingSchema: Invalid URL


# **END**

# Document Translation
You can also pass longer documents (or list of documents) to the `translate()` method.

As Transformer models can only translate inputs up to 512 (or 1024) word pieces, we first perform sentence splitting. Then, each sentence is translated individually.


```
import tqdm
document = """Berlin is the capital and largest city of Germany by both area and population.
Its 3,769,495 inhabitants as of 31 December 2019 make it the most-populous city of the European Union, according to population within city limits.
The city is also one of Germany's 16 federal states. It is surrounded by the state of Brandenburg, and contiguous with Potsdam, Brandenburg's capital.
The two cities are at the center of the Berlin-Brandenburg capital region, which is, with about six million inhabitants and an area of more than 30,000 km2, Germany's third-largest metropolitan region after the Rhine-Ruhr and Rhine-Main regions.
Berlin straddles the banks of the River Spree, which flows into the River Havel (a tributary of the River Elbe) in the western borough of Spandau.
Among the city's main topographical features are the many lakes in the western and southeastern boroughs formed by the Spree, Havel, and Dahme rivers (the largest of which is Lake Müggelsee).
Due to its location in the European Plain, Berlin is influenced by a temperate seasonal climate.
About one-third of the city's area is composed of forests, parks, gardens, rivers, canals and lakes.
The city lies in the Central German dialect area, the Berlin dialect being a variant of the Lusatian-New Marchian dialects.

First documented in the 13th century and at the crossing of two important historic trade routes, Berlin became the capital of the Margraviate of Brandenburg (1417–1701), the Kingdom of Prussia (1701–1918), the German Empire (1871–1918), the Weimar Republic (1919–1933), and the Third Reich (1933–1945).
Berlin in the 1920s was the third-largest municipality in the world.
After World War II and its subsequent occupation by the victorious countries, the city was divided; West Berlin became a de facto West German exclave, surrounded by the Berlin Wall (1961–1989) and East German territory.
East Berlin was declared capital of East Germany, while Bonn became the West German capital.
Following German reunification in 1990, Berlin once again became the capital of all of Germany.

Berlin is a world city of culture, politics, media and science.
Its economy is based on high-tech firms and the service sector, encompassing a diverse range of creative industries, research facilities, media corporations and convention venues.
Berlin serves as a continental hub for air and rail traffic and has a highly complex public transportation network.
The metropolis is a popular tourist destination.
Significant industries also include IT, pharmaceuticals, biomedical engineering, clean tech, biotechnology, construction and electronics."""


print("Output:")
print(model.translate(document, target_lang='de'))
```

    Output:
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=768489.0, style=ProgressStyle(descripti…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=796845.0, style=ProgressStyle(descripti…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1273232.0, style=ProgressStyle(descript…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=42.0, style=ProgressStyle(description_w…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1335.0, style=ProgressStyle(description…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=297928209.0, style=ProgressStyle(descri…


    
    Berlin ist die Hauptstadt und größte Stadt Deutschlands sowohl in der Region als auch in der Bevölkerung.
    Die 3.769,495 Einwohner machen sie zum 31. Dezember 2019 zur bevölkerungsreichsten Stadt der Europäischen Union, nach der Bevölkerung innerhalb der Stadtgrenzen.
    Die Stadt gehört auch zu den 16 Bundesländern Deutschlands. Sie ist von Brandenburg umgeben und mit Potsdam, der Hauptstadt Brandenburgs, verbunden. 
    Die beiden Städte befinden sich im Zentrum der Hauptstadtregion Berlin-Brandenburg, mit rund sechs Millionen Einwohnern und einer Fläche von mehr als 30.000 km2, Deutschlands drittgrößter Metropolregion nach den Regionen Rhein-Ruhr und Rhein-Main. 
    Berlin erstreckt sich über das Ufer der Spree, die in den Havel (ein Nebenfluss der Elbe) im westlichen Bezirk Spandau mündet. 
    Zu den wichtigsten topographischen Merkmalen der Stadt gehören die zahlreichen Seen in den westlichen und südöstlichen Stadtteilen, die von den Flüssen Spree, Havel und Dahme gebildet wurden (der größte davon ist der Müggelsee). 
    Aufgrund seiner Lage in der Europäischen Ebene wird Berlin von einem gemäßigten saisonalen Klima beeinflusst. 
    Etwa ein Drittel des Stadtgebiets besteht aus Wäldern, Parks, Gärten, Flüssen, Kanälen und Seen.
    Die Stadt liegt im mitteldeutschen Dialektgebiet, der Berliner Dialekt ist eine Variante der Lusatian-New Marchian Dialekte.
    
    Jahrhundert und an der Überquerung zweier wichtiger historischer Handelswege erstmals dokumentiert, wurde Berlin zur Hauptstadt der Markgrafschaft Brandenburg (1417–1701), des Königreichs Preußen (1701–1918), des Deutschen Reiches (1871–1918), der Weimarer Republik (1919–1933) und des Dritten Reiches (1933–1945).
    Berlin war in den 1920er Jahren die drittgrößte Gemeinde der Welt.
    Nach dem Zweiten Weltkrieg und seiner anschließenden Besetzung durch die siegreichen Länder wurde die Stadt geteilt; West-Berlin wurde de facto eine westdeutsche Exklave, umgeben von der Berliner Mauer (1961–1989) und dem ostdeutschen Territorium. 
    Ost-Berlin wurde zur Hauptstadt Ostdeutschlands erklärt, während Bonn zur westdeutschen Hauptstadt wurde. 
    Nach der deutschen Wiedervereinigung 1990 wurde Berlin wieder zur Hauptstadt ganz Deutschlands.
    
    Berlin ist eine Weltstadt der Kultur, Politik, Medien und Wissenschaft.
    Seine Wirtschaft basiert auf High-Tech-Firmen und dem Dienstleistungssektor und umfasst eine Vielzahl von Kreativindustrien, Forschungseinrichtungen, Medienunternehmen und Kongressstätten. 
    Berlin dient als kontinentale Drehscheibe für den Luft- und Schienenverkehr und verfügt über ein hochkomplexes öffentliches Verkehrsnetz. 
    Die Metropole ist ein beliebtes Touristenziel.
    Zu den bedeutenden Industriezweigen gehören auch IT, Pharmazeutika, biomedizinische Technik, Clean Tech, Biotechnologie, Bauwesen und Elektronik.
    

# Language Detection
EasyNMT allows easy detection of the language of text. For this, we call the method `model.language_detection(text)`.

For language detection, we use [fastText](https://fasttext.cc/blog/2017/10/02/blog-post.html), which is able to recognize more than 170 languages.



```
sentences = ["This is an English sentence." ,"Dies ist ein deutscher Satz.", "это русское предложение.", "这是一个中文句子。"]

for sent in sentences:
  print(sent)
  print("=> detected language:", model.language_detection(sent), "\n")
```

    This is an English sentence.
    => detected language: en 
    
    Dies ist ein deutscher Satz.
    => detected language: de 
    
    это русское предложение.
    => detected language: ru 
    
    这是一个中文句子。
    => detected language: zh 
    
    

# Beam-Search
You can pass the beam-size as parameter to the `translate()` method. A larger beam size produces higher quality translations, but requires longer for the translation. By default, beam-size is set to 5.


```
import time
model = EasyNMT('opus-mt')

sentence = "Berlin ist die Hauptstadt von Deutschland und sowohl von den Einwohner als auch von der Fläche die größte Stadt in Deutschland, während Hamburg die zweit größte Stadt ist."

#Loading and warm-up of the model
model.translate(sentence, target_lang='en', beam_size=1)

print("\nBeam-Size 1")
start_time = time.time()
print(model.translate(sentence, target_lang='en', beam_size=1))
print("Translated in {:.2f} sec".format(time.time()-start_time))

print("\nBeam-Size 10")
start_time = time.time()
print(model.translate(sentence, target_lang='en', beam_size=10))
print("Translated in {:.2f} sec".format(time.time()-start_time))

```

    
    Beam-Size 1
    Berlin is the capital of Germany and the largest city in Germany, both of its inhabitants and of its area, while Hamburg is the second largest city.
    Translated in 0.18 sec
    
    Beam-Size 10
    Berlin is the capital of Germany and of both the inhabitants and the area the largest city in Germany, while Hamburg is the second largest city.
    Translated in 0.44 sec
    

# Available Models



```
available_models = ['opus-mt', 'mbart50_m2m', 'm2m_100_418M']
#Note: EasyNMT also provides the m2m_100_1.2B. But sadly it requires too much RAM to be loaded with the Colab free version here
#If you start an empty instance in colab and load the 'm2m_100_1.2B' model, it should work.

for model_name in available_models:
  print("\n\nLoad model:", model_name)
  model = EasyNMT(model_name)

  sentences = ['In dieser Liste definieren wir mehrere Sätze.',
              'Jeder dieser Sätze wird dann in die Zielsprache übersetzt.',
              'Puede especificar en esta lista la oración en varios idiomas.',
              'El sistema detectará automáticamente el idioma y utilizará el modelo correcto.']
  translations = model.translate(sentences, target_lang='en')

  print("Translations:")
  for sent, trans in zip(sentences, translations):
    print(sent)
    print("=>", trans, "\n")
  del model

```

    
    
    Load model: opus-mt
    Translations:
    In dieser Liste definieren wir mehrere Sätze.
    => In this list we define several sentences. 
    
    Jeder dieser Sätze wird dann in die Zielsprache übersetzt.
    => Each of these sentences is then translated into the target language. 
    
    Puede especificar en esta lista la oración en varios idiomas.
    => You can specify the sentence in several languages in this list. 
    
    El sistema detectará automáticamente el idioma y utilizará el modelo correcto.
    => The system will automatically detect the language and use the correct model. 
    
    
    
    Load model: mbart50_m2m
    

    100%|██████████| 24.9k/24.9k [00:00<00:00, 242kB/s]
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1429.0, style=ProgressStyle(description…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=2444714899.0, style=ProgressStyle(descr…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=5069051.0, style=ProgressStyle(descript…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=649.0, style=ProgressStyle(description_…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=529.0, style=ProgressStyle(description_…


    
    Translations:
    In dieser Liste definieren wir mehrere Sätze.
    => In this list we define several sentences. 
    
    Jeder dieser Sätze wird dann in die Zielsprache übersetzt.
    => Each of these sentences is then translated into the target language. 
    
    Puede especificar en esta lista la oración en varios idiomas.
    => You can specify in this list the speech in several languages. 
    
    El sistema detectará automáticamente el idioma y utilizará el modelo correcto.
    => The system will automatically detect the language and use the correct model. 
    
    
    
    Load model: m2m_100_418M
    

    100%|██████████| 89.9k/89.9k [00:00<00:00, 425kB/s]
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=908.0, style=ProgressStyle(description_…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1935796948.0, style=ProgressStyle(descr…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=3708092.0, style=ProgressStyle(descript…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=2423393.0, style=ProgressStyle(descript…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=272.0, style=ProgressStyle(description_…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1140.0, style=ProgressStyle(description…


    
    Translations:
    In dieser Liste definieren wir mehrere Sätze.
    => In this list we define several sentences. 
    
    Jeder dieser Sätze wird dann in die Zielsprache übersetzt.
    => Each of these sentences is then translated into the target language. 
    
    Puede especificar en esta lista la oración en varios idiomas.
    => You can specify in this list the prayer in several languages. 
    
    El sistema detectará automáticamente el idioma y utilizará el modelo correcto.
    => The system will automatically detect the language and use the correct model. 
    
    

# Translation Directions & Languages
To get all available translation directions for a model, you can simply call the following property. An entry like 'af-en' means that you can translate from *af* (Afrikaans) to *en* (English).


```
model = EasyNMT('opus-mt')
print("Language directions:")
print(sorted(list(model.lang_pairs)))
```

    Language directions:
    ['aav-en', 'aed-es', 'af-de', 'af-en', 'af-eo', 'af-es', 'af-fi', 'af-fr', 'af-nl', 'af-ru', 'af-sv', 'alv-en', 'am-sv', 'ar-de', 'ar-el', 'ar-en', 'ar-eo', 'ar-es', 'ar-fr', 'ar-he', 'ar-it', 'ar-pl', 'ar-ru', 'ar-tr', 'art-en', 'ase-de', 'ase-en', 'ase-es', 'ase-fr', 'ase-sv', 'az-en', 'az-es', 'az-tr', 'bat-en', 'bcl-de', 'bcl-en', 'bcl-es', 'bcl-fi', 'bcl-fr', 'bcl-sv', 'be-es', 'bem-en', 'bem-es', 'bem-fi', 'bem-fr', 'bem-sv', 'ber-en', 'ber-es', 'ber-fr', 'bg-de', 'bg-en', 'bg-eo', 'bg-es', 'bg-fi', 'bg-fr', 'bg-it', 'bg-ru', 'bg-sv', 'bg-tr', 'bg-uk', 'bi-en', 'bi-es', 'bi-fr', 'bi-sv', 'bn-en', 'bnt-en', 'bzs-en', 'bzs-es', 'bzs-fi', 'bzs-fr', 'bzs-sv', 'ca-de', 'ca-en', 'ca-es', 'ca-fr', 'ca-it', 'ca-nl', 'ca-pt', 'ca-uk', 'cau-en', 'ccs-en', 'ceb-en', 'ceb-es', 'ceb-fi', 'ceb-fr', 'ceb-sv', 'cel-en', 'chk-en', 'chk-es', 'chk-fr', 'chk-sv', 'cpf-en', 'crs-de', 'crs-en', 'crs-es', 'crs-fi', 'crs-fr', 'crs-sv', 'cs-de', 'cs-en', 'cs-eo', 'cs-fi', 'cs-fr', 'cs-sv', 'cs-uk', 'csg-es', 'csn-es', 'cus-en', 'cy-en', 'da-de', 'da-en', 'da-eo', 'da-es', 'da-fi', 'da-fr', 'da-no', 'da-ru', 'de-af', 'de-ar', 'de-ase', 'de-bcl', 'de-bg', 'de-bi', 'de-bzs', 'de-ca', 'de-crs', 'de-cs', 'de-da', 'de-ee', 'de-efi', 'de-el', 'de-en', 'de-eo', 'de-es', 'de-et', 'de-eu', 'de-fi', 'de-fj', 'de-fr', 'de-gaa', 'de-gil', 'de-guw', 'de-ha', 'de-he', 'de-hil', 'de-ho', 'de-hr', 'de-ht', 'de-hu', 'de-ig', 'de-ilo', 'de-is', 'de-iso', 'de-it', 'de-kg', 'de-ln', 'de-loz', 'de-lt', 'de-lua', 'de-ms', 'de-mt', 'de-niu', 'de-nl', 'de-no', 'de-nso', 'de-ny', 'de-pag', 'de-pap', 'de-pis', 'de-pl', 'de-pon', 'de-tl', 'de-uk', 'de-vi', 'dra-en', 'ee-de', 'ee-en', 'ee-es', 'ee-fi', 'ee-fr', 'ee-sv', 'efi-de', 'efi-en', 'efi-fi', 'efi-fr', 'efi-sv', 'el-ar', 'el-eo', 'el-fi', 'el-fr', 'el-sv', 'en-aav', 'en-af', 'en-alv', 'en-ar', 'en-az', 'en-bat', 'en-bcl', 'en-bem', 'en-ber', 'en-bg', 'en-bi', 'en-bnt', 'en-bzs', 'en-ca', 'en-ceb', 'en-cel', 'en-chk', 'en-cpf', 'en-crs', 'en-cs', 'en-cus', 'en-cy', 'en-da', 'en-de', 'en-dra', 'en-ee', 'en-efi', 'en-el', 'en-eo', 'en-es', 'en-et', 'en-eu', 'en-euq', 'en-fi', 'en-fj', 'en-fr', 'en-ga', 'en-gaa', 'en-gil', 'en-gl', 'en-grk', 'en-guw', 'en-gv', 'en-ha', 'en-he', 'en-hi', 'en-hil', 'en-ho', 'en-ht', 'en-hu', 'en-hy', 'en-id', 'en-ig', 'en-ilo', 'en-is', 'en-iso', 'en-it', 'en-jap', 'en-kg', 'en-kj', 'en-kqn', 'en-kwn', 'en-kwy', 'en-lg', 'en-ln', 'en-loz', 'en-lu', 'en-lua', 'en-lue', 'en-lun', 'en-luo', 'en-lus', 'en-map', 'en-mfe', 'en-mg', 'en-mh', 'en-mk', 'en-mkh', 'en-ml', 'en-mos', 'en-mr', 'en-mt', 'en-mul', 'en-ng', 'en-nic', 'en-niu', 'en-nl', 'en-nso', 'en-ny', 'en-nyk', 'en-om', 'en-pag', 'en-pap', 'en-phi', 'en-pis', 'en-pon', 'en-poz', 'en-pqe', 'en-pqw', 'en-rn', 'en-rnd', 'en-ro', 'en-roa', 'en-ru', 'en-run', 'en-rw', 'en-sal', 'en-sg', 'en-sit', 'en-sk', 'en-sm', 'en-sn', 'en-sq', 'en-ss', 'en-st', 'en-sv', 'en-sw', 'en-swc', 'en-tdt', 'en-ti', 'en-tiv', 'en-tl', 'en-tll', 'en-tn', 'en-to', 'en-toi', 'en-tpi', 'en-trk', 'en-ts', 'en-tut', 'en-tvl', 'en-tw', 'en-ty', 'en-uk', 'en-umb', 'en-ur', 'en-vi', 'en-xh', 'en-zh', 'eo-af', 'eo-bg', 'eo-cs', 'eo-da', 'eo-de', 'eo-el', 'eo-en', 'eo-es', 'eo-fi', 'eo-fr', 'eo-he', 'eo-hu', 'eo-it', 'eo-nl', 'eo-pl', 'eo-pt', 'eo-ro', 'eo-ru', 'eo-sh', 'eo-sv', 'es-aed', 'es-af', 'es-ar', 'es-ase', 'es-bcl', 'es-ber', 'es-bg', 'es-bi', 'es-bzs', 'es-ca', 'es-ceb', 'es-crs', 'es-cs', 'es-csg', 'es-csn', 'es-da', 'es-de', 'es-ee', 'es-efi', 'es-el', 'es-en', 'es-eo', 'es-et', 'es-eu', 'es-fi', 'es-fj', 'es-fr', 'es-gaa', 'es-gil', 'es-gl', 'es-guw', 'es-ha', 'es-he', 'es-hil', 'es-ho', 'es-hr', 'es-ht', 'es-id', 'es-ig', 'es-ilo', 'es-is', 'es-iso', 'es-it', 'es-kg', 'es-ln', 'es-loz', 'es-lt', 'es-lua', 'es-lus', 'es-mfs', 'es-mk', 'es-mt', 'es-niu', 'es-nl', 'es-no', 'es-nso', 'es-ny', 'es-pag', 'es-pap', 'es-pis', 'es-pl', 'es-pon', 'es-prl', 'es-rn', 'es-ro', 'es-ru', 'es-rw', 'es-sg', 'es-sl', 'es-sm', 'es-sn', 'es-srn', 'es-st', 'es-swc', 'es-tl', 'es-tll', 'es-tn', 'es-to', 'es-tpi', 'es-tvl', 'es-tw', 'es-ty', 'es-tzo', 'es-uk', 'es-ve', 'es-vi', 'es-war', 'es-wls', 'es-xh', 'es-yo', 'es-yua', 'es-zai', 'et-de', 'et-en', 'et-es', 'et-fi', 'et-fr', 'et-ru', 'et-sv', 'eu-de', 'eu-en', 'eu-es', 'eu-ru', 'euq-en', 'fi-af', 'fi-bcl', 'fi-bem', 'fi-bg', 'fi-bzs', 'fi-ceb', 'fi-crs', 'fi-cs', 'fi-de', 'fi-ee', 'fi-efi', 'fi-el', 'fi-en', 'fi-eo', 'fi-es', 'fi-et', 'fi-fj', 'fi-fr', 'fi-fse', 'fi-gaa', 'fi-gil', 'fi-guw', 'fi-ha', 'fi-he', 'fi-hil', 'fi-ho', 'fi-hr', 'fi-ht', 'fi-hu', 'fi-id', 'fi-ig', 'fi-ilo', 'fi-is', 'fi-iso', 'fi-it', 'fi-kg', 'fi-kqn', 'fi-lg', 'fi-ln', 'fi-lu', 'fi-lua', 'fi-lue', 'fi-lus', 'fi-lv', 'fi-mfe', 'fi-mg', 'fi-mh', 'fi-mk', 'fi-mos', 'fi-mt', 'fi-niu', 'fi-nl', 'fi-no', 'fi-nso', 'fi-ny', 'fi-pag', 'fi-pap', 'fi-pis', 'fi-pon', 'fi-ro', 'fi-ru', 'fi-run', 'fi-rw', 'fi-sg', 'fi-sk', 'fi-sl', 'fi-sm', 'fi-sn', 'fi-sq', 'fi-srn', 'fi-st', 'fi-sv', 'fi-sw', 'fi-swc', 'fi-tiv', 'fi-tll', 'fi-tn', 'fi-to', 'fi-toi', 'fi-tpi', 'fi-tr', 'fi-ts', 'fi-tvl', 'fi-tw', 'fi-ty', 'fi-uk', 'fi-ve', 'fi-war', 'fi-wls', 'fi-xh', 'fi-yap', 'fi-yo', 'fi-zne', 'fj-en', 'fj-fr', 'fr-af', 'fr-ar', 'fr-ase', 'fr-bcl', 'fr-bem', 'fr-ber', 'fr-bg', 'fr-bi', 'fr-bzs', 'fr-ca', 'fr-ceb', 'fr-crs', 'fr-de', 'fr-ee', 'fr-efi', 'fr-el', 'fr-en', 'fr-eo', 'fr-es', 'fr-fj', 'fr-gaa', 'fr-gil', 'fr-guw', 'fr-ha', 'fr-he', 'fr-hil', 'fr-ho', 'fr-hr', 'fr-ht', 'fr-hu', 'fr-id', 'fr-ig', 'fr-ilo', 'fr-iso', 'fr-kg', 'fr-kqn', 'fr-kwy', 'fr-lg', 'fr-ln', 'fr-loz', 'fr-lu', 'fr-lua', 'fr-lue', 'fr-lus', 'fr-mfe', 'fr-mh', 'fr-mos', 'fr-ms', 'fr-mt', 'fr-niu', 'fr-no', 'fr-nso', 'fr-ny', 'fr-pag', 'fr-pap', 'fr-pis', 'fr-pl', 'fr-pon', 'fr-rnd', 'fr-ro', 'fr-ru', 'fr-run', 'fr-rw', 'fr-sg', 'fr-sk', 'fr-sl', 'fr-sm', 'fr-sn', 'fr-srn', 'fr-st', 'fr-sv', 'fr-swc', 'fr-tiv', 'fr-tl', 'fr-tll', 'fr-tn', 'fr-to', 'fr-tpi', 'fr-ts', 'fr-tum', 'fr-tvl', 'fr-tw', 'fr-ty', 'fr-uk', 'fr-ve', 'fr-vi', 'fr-war', 'fr-wls', 'fr-xh', 'fr-yap', 'fr-yo', 'fr-zne', 'fse-fi', 'ga-en', 'gaa-de', 'gaa-en', 'gaa-es', 'gaa-fi', 'gaa-fr', 'gaa-sv', 'gil-en', 'gil-es', 'gil-fi', 'gil-fr', 'gil-sv', 'gl-en', 'gl-es', 'gl-pt', 'grk-en', 'guw-de', 'guw-en', 'guw-es', 'guw-fi', 'guw-fr', 'guw-sv', 'gv-en', 'ha-en', 'ha-es', 'ha-fi', 'ha-fr', 'ha-sv', 'he-ar', 'he-de', 'he-eo', 'he-es', 'he-fi', 'he-fr', 'he-it', 'he-ru', 'he-sv', 'he-uk', 'hi-en', 'hi-ur', 'hil-de', 'hil-en', 'hil-fi', 'ho-en', 'hr-es', 'hr-fi', 'hr-fr', 'hr-sv', 'ht-en', 'ht-es', 'ht-fi', 'ht-fr', 'ht-sv', 'hu-de', 'hu-en', 'hu-eo', 'hu-fi', 'hu-fr', 'hu-sv', 'hu-uk', 'hy-en', 'hy-ru', 'id-en', 'id-es', 'id-fi', 'id-fr', 'id-sv', 'ig-de', 'ig-en', 'ig-es', 'ig-fi', 'ig-fr', 'ig-sv', 'ilo-de', 'ilo-en', 'ilo-es', 'ilo-fi', 'ilo-sv', 'is-de', 'is-en', 'is-eo', 'is-es', 'is-fi', 'is-fr', 'is-it', 'is-sv', 'iso-en', 'iso-es', 'iso-fi', 'iso-fr', 'iso-sv', 'it-ar', 'it-bg', 'it-ca', 'it-de', 'it-en', 'it-eo', 'it-es', 'it-fr', 'it-is', 'it-lt', 'it-ms', 'it-sv', 'it-uk', 'it-vi', 'ja-ar', 'ja-bg', 'ja-da', 'ja-de', 'ja-en', 'ja-es', 'ja-fi', 'ja-fr', 'ja-he', 'ja-hu', 'ja-it', 'ja-ms', 'ja-nl', 'ja-pl', 'ja-pt', 'ja-ru', 'ja-sh', 'ja-sv', 'ja-tr', 'ja-vi', 'jap-en', 'ka-en', 'ka-ru', 'kab-en', 'kg-en', 'kg-es', 'kg-fr', 'kg-sv', 'kj-en', 'kl-en', 'ko-de', 'ko-en', 'ko-es', 'ko-fi', 'ko-fr', 'ko-hu', 'ko-ru', 'ko-sv', 'kqn-en', 'kqn-es', 'kqn-fr', 'kqn-sv', 'kwn-en', 'kwy-en', 'kwy-fr', 'kwy-sv', 'lg-en', 'lg-es', 'lg-fi', 'lg-fr', 'lg-sv', 'ln-de', 'ln-en', 'ln-es', 'ln-fr', 'loz-de', 'loz-en', 'loz-es', 'loz-fi', 'loz-fr', 'loz-sv', 'lt-de', 'lt-eo', 'lt-es', 'lt-fr', 'lt-it', 'lt-pl', 'lt-ru', 'lt-sv', 'lt-tr', 'lu-en', 'lu-es', 'lu-fi', 'lu-fr', 'lu-sv', 'lua-en', 'lua-es', 'lua-fi', 'lua-fr', 'lua-sv', 'lue-en', 'lue-es', 'lue-fi', 'lue-fr', 'lue-sv', 'lun-en', 'luo-en', 'lus-en', 'lus-es', 'lus-fi', 'lus-fr', 'lus-sv', 'lv-en', 'lv-es', 'lv-fi', 'lv-fr', 'lv-ru', 'lv-sv', 'mfe-en', 'mfe-es', 'mfs-es', 'mg-en', 'mg-es', 'mh-en', 'mh-es', 'mh-fi', 'mk-en', 'mk-es', 'mk-fi', 'mk-fr', 'mkh-en', 'ml-en', 'mos-en', 'mr-en', 'ms-de', 'ms-fr', 'ms-it', 'mt-en', 'mt-es', 'mt-fi', 'mt-fr', 'mt-sv', 'mul-en', 'ng-en', 'nic-en', 'niu-de', 'niu-en', 'niu-es', 'niu-fi', 'niu-fr', 'niu-sv', 'nl-af', 'nl-ca', 'nl-en', 'nl-eo', 'nl-es', 'nl-fi', 'nl-fr', 'nl-no', 'nl-sv', 'nl-uk', 'no-da', 'no-de', 'no-es', 'no-fi', 'no-fr', 'no-nl', 'no-pl', 'no-ru', 'no-sv', 'no-uk', 'nso-de', 'nso-en', 'nso-es', 'nso-fi', 'nso-fr', 'nso-sv', 'ny-de', 'ny-en', 'ny-es', 'nyk-en', 'om-en', 'pa-en', 'pag-de', 'pag-en', 'pag-es', 'pag-fi', 'pag-sv', 'pap-de', 'pap-en', 'pap-es', 'pap-fi', 'pap-fr', 'phi-en', 'pis-en', 'pis-es', 'pis-fi', 'pis-fr', 'pis-sv', 'pl-ar', 'pl-de', 'pl-en', 'pl-eo', 'pl-es', 'pl-fr', 'pl-lt', 'pl-no', 'pl-sv', 'pl-uk', 'pon-en', 'pon-es', 'pon-fi', 'pon-fr', 'pon-sv', 'pqe-en', 'prl-es', 'pt-ca', 'pt-eo', 'pt-gl', 'pt-tl', 'pt-uk', 'rn-de', 'rn-en', 'rn-es', 'rn-fr', 'rn-ru', 'rnd-en', 'rnd-fr', 'rnd-sv', 'ro-eo', 'ro-fi', 'ro-fr', 'ro-sv', 'roa-en', 'ru-af', 'ru-ar', 'ru-bg', 'ru-da', 'ru-en', 'ru-eo', 'ru-es', 'ru-et', 'ru-eu', 'ru-fi', 'ru-fr', 'ru-he', 'ru-hy', 'ru-lt', 'ru-lv', 'ru-no', 'ru-sl', 'ru-sv', 'ru-uk', 'ru-vi', 'run-en', 'run-es', 'run-sv', 'rw-en', 'rw-es', 'rw-fr', 'rw-sv', 'sal-en', 'sg-en', 'sg-es', 'sg-fi', 'sg-fr', 'sg-sv', 'sh-eo', 'sh-uk', 'sk-en', 'sk-es', 'sk-fi', 'sk-fr', 'sk-sv', 'sl-es', 'sl-fi', 'sl-fr', 'sl-ru', 'sl-sv', 'sl-uk', 'sm-en', 'sm-es', 'sm-fr', 'sn-en', 'sn-es', 'sn-fr', 'sn-sv', 'sq-en', 'sq-es', 'sq-sv', 'srn-en', 'srn-es', 'srn-fr', 'srn-sv', 'ss-en', 'ssp-es', 'st-en', 'st-es', 'st-fi', 'st-fr', 'st-sv', 'sv-af', 'sv-ase', 'sv-bcl', 'sv-bem', 'sv-bg', 'sv-bi', 'sv-bzs', 'sv-ceb', 'sv-chk', 'sv-crs', 'sv-cs', 'sv-ee', 'sv-efi', 'sv-el', 'sv-en', 'sv-eo', 'sv-es', 'sv-et', 'sv-fi', 'sv-fj', 'sv-fr', 'sv-gaa', 'sv-gil', 'sv-guw', 'sv-ha', 'sv-he', 'sv-hil', 'sv-ho', 'sv-hr', 'sv-ht', 'sv-hu', 'sv-id', 'sv-ig', 'sv-ilo', 'sv-is', 'sv-iso', 'sv-kg', 'sv-kqn', 'sv-kwy', 'sv-lg', 'sv-ln', 'sv-lu', 'sv-lua', 'sv-lue', 'sv-lus', 'sv-lv', 'sv-mfe', 'sv-mh', 'sv-mos', 'sv-mt', 'sv-niu', 'sv-nl', 'sv-no', 'sv-nso', 'sv-ny', 'sv-pag', 'sv-pap', 'sv-pis', 'sv-pon', 'sv-rnd', 'sv-ro', 'sv-ru', 'sv-run', 'sv-rw', 'sv-sg', 'sv-sk', 'sv-sl', 'sv-sm', 'sv-sn', 'sv-sq', 'sv-srn', 'sv-st', 'sv-swc', 'sv-th', 'sv-tiv', 'sv-tll', 'sv-tn', 'sv-to', 'sv-toi', 'sv-tpi', 'sv-ts', 'sv-tum', 'sv-tvl', 'sv-tw', 'sv-ty', 'sv-uk', 'sv-umb', 'sv-ve', 'sv-war', 'sv-wls', 'sv-xh', 'sv-yap', 'sv-yo', 'sv-zne', 'swc-en', 'swc-es', 'swc-fi', 'swc-fr', 'swc-sv', 'taw-en', 'th-en', 'th-fr', 'ti-en', 'tiv-en', 'tiv-fr', 'tiv-sv', 'tl-de', 'tl-en', 'tl-es', 'tl-pt', 'tll-en', 'tll-es', 'tll-fi', 'tll-fr', 'tll-sv', 'tn-en', 'tn-es', 'tn-fr', 'tn-sv', 'to-en', 'to-es', 'to-fr', 'to-sv', 'toi-en', 'toi-es', 'toi-fi', 'toi-fr', 'toi-sv', 'tpi-en', 'tpi-sv', 'tr-ar', 'tr-az', 'tr-en', 'tr-eo', 'tr-es', 'tr-fr', 'tr-lt', 'tr-sv', 'tr-uk', 'trk-en', 'ts-en', 'ts-es', 'ts-fi', 'ts-fr', 'ts-sv', 'tum-en', 'tum-es', 'tum-fr', 'tum-sv', 'tvl-en', 'tvl-es', 'tvl-fi', 'tvl-fr', 'tvl-sv', 'tw-es', 'tw-fi', 'tw-fr', 'tw-sv', 'ty-es', 'ty-fi', 'ty-fr', 'ty-sv', 'tzo-es', 'uk-bg', 'uk-ca', 'uk-cs', 'uk-de', 'uk-en', 'uk-es', 'uk-fi', 'uk-fr', 'uk-he', 'uk-hu', 'uk-it', 'uk-nl', 'uk-no', 'uk-pl', 'uk-pt', 'uk-ru', 'uk-sh', 'uk-sl', 'uk-sv', 'uk-tr', 'umb-en', 'ur-en', 've-en', 've-es', 'vi-de', 'vi-en', 'vi-eo', 'vi-es', 'vi-fr', 'vi-it', 'vi-ru', 'vsl-es', 'wa-en', 'wal-en', 'war-en', 'war-es', 'war-fi', 'war-fr', 'war-sv', 'wls-en', 'wls-fr', 'wls-sv', 'xh-en', 'xh-es', 'xh-fr', 'xh-sv', 'yap-en', 'yap-fr', 'yap-sv', 'yo-en', 'yo-es', 'yo-fi', 'yo-fr', 'yo-sv', 'zai-es', 'zh-bg', 'zh-de', 'zh-en', 'zh-fi', 'zh-he', 'zh-it', 'zh-ms', 'zh-nl', 'zh-sv', 'zh-uk', 'zh-vi', 'zne-es', 'zne-fi', 'zne-fr', 'zne-sv']
    

To check which languages are supported, you can use the following method:


```
print("All Languages:")
print(model.get_languages())

print("\n\nAll languages with source_lang=en. I.e., we can translate English (en) to these languages.")
print(model.get_languages(source_lang='en'))

print("\n\nAll languages with target_lang=de. I.e., we can translate from these languages to German (de).")
print(model.get_languages(target_lang='de'))
```

    All Languages:
    ['aav', 'aed', 'af', 'alv', 'am', 'ar', 'art', 'ase', 'az', 'bat', 'bcl', 'be', 'bem', 'ber', 'bg', 'bi', 'bn', 'bnt', 'bzs', 'ca', 'cau', 'ccs', 'ceb', 'cel', 'chk', 'cpf', 'crs', 'cs', 'csg', 'csn', 'cus', 'cy', 'da', 'de', 'dra', 'ee', 'efi', 'el', 'en', 'eo', 'es', 'et', 'eu', 'euq', 'fi', 'fj', 'fr', 'fse', 'ga', 'gaa', 'gil', 'gl', 'grk', 'guw', 'gv', 'ha', 'he', 'hi', 'hil', 'ho', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'iso', 'it', 'ja', 'jap', 'ka', 'kab', 'kg', 'kj', 'kl', 'ko', 'kqn', 'kwn', 'kwy', 'lg', 'ln', 'loz', 'lt', 'lu', 'lua', 'lue', 'lun', 'luo', 'lus', 'lv', 'map', 'mfe', 'mfs', 'mg', 'mh', 'mk', 'mkh', 'ml', 'mos', 'mr', 'ms', 'mt', 'mul', 'ng', 'nic', 'niu', 'nl', 'no', 'nso', 'ny', 'nyk', 'om', 'pa', 'pag', 'pap', 'phi', 'pis', 'pl', 'pon', 'poz', 'pqe', 'pqw', 'prl', 'pt', 'rn', 'rnd', 'ro', 'roa', 'ru', 'run', 'rw', 'sal', 'sg', 'sh', 'sit', 'sk', 'sl', 'sm', 'sn', 'sq', 'srn', 'ss', 'ssp', 'st', 'sv', 'sw', 'swc', 'taw', 'tdt', 'th', 'ti', 'tiv', 'tl', 'tll', 'tn', 'to', 'toi', 'tpi', 'tr', 'trk', 'ts', 'tum', 'tut', 'tvl', 'tw', 'ty', 'tzo', 'uk', 'umb', 'ur', 've', 'vi', 'vsl', 'wa', 'wal', 'war', 'wls', 'xh', 'yap', 'yo', 'yua', 'zai', 'zh', 'zne']
    
    
    All languages with source_lang=en. I.e., we can translate English (en) to these languages.
    ['aav', 'af', 'alv', 'ar', 'az', 'bat', 'bcl', 'bem', 'ber', 'bg', 'bi', 'bnt', 'bzs', 'ca', 'ceb', 'cel', 'chk', 'cpf', 'crs', 'cs', 'cus', 'cy', 'da', 'de', 'dra', 'ee', 'efi', 'el', 'eo', 'es', 'et', 'eu', 'euq', 'fi', 'fj', 'fr', 'ga', 'gaa', 'gil', 'gl', 'grk', 'guw', 'gv', 'ha', 'he', 'hi', 'hil', 'ho', 'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'iso', 'it', 'jap', 'kg', 'kj', 'kqn', 'kwn', 'kwy', 'lg', 'ln', 'loz', 'lu', 'lua', 'lue', 'lun', 'luo', 'lus', 'map', 'mfe', 'mg', 'mh', 'mk', 'mkh', 'ml', 'mos', 'mr', 'mt', 'mul', 'ng', 'nic', 'niu', 'nl', 'nso', 'ny', 'nyk', 'om', 'pag', 'pap', 'phi', 'pis', 'pon', 'poz', 'pqe', 'pqw', 'rn', 'rnd', 'ro', 'roa', 'ru', 'run', 'rw', 'sal', 'sg', 'sit', 'sk', 'sm', 'sn', 'sq', 'ss', 'st', 'sv', 'sw', 'swc', 'tdt', 'ti', 'tiv', 'tl', 'tll', 'tn', 'to', 'toi', 'tpi', 'trk', 'ts', 'tut', 'tvl', 'tw', 'ty', 'uk', 'umb', 'ur', 'vi', 'xh', 'zh']
    
    
    All languages with target_lang=de. I.e., we can translate from these languages to German (de).
    ['af', 'ar', 'ase', 'bcl', 'bg', 'ca', 'crs', 'cs', 'da', 'ee', 'efi', 'en', 'eo', 'es', 'et', 'eu', 'fi', 'fr', 'gaa', 'guw', 'he', 'hil', 'hu', 'ig', 'ilo', 'is', 'it', 'ja', 'ko', 'ln', 'loz', 'lt', 'ms', 'niu', 'no', 'nso', 'ny', 'pag', 'pap', 'pl', 'rn', 'tl', 'uk', 'vi', 'zh']
    
