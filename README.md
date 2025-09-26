<h1 align="center"> <a href=>Generative Agents for Multimodal Controversy Detection</a></h2>

## :sparkles: Keypoints
We explore incorporating LLM-based agents into the task of multimodal controversy detection, thereby enhancing the explainability of the process.

<p align="center">
    <img src="figures/pic1.png" alt="pic1" width="400">
</p>

## :memo: Overall Framework
We introduce a novel Agent-based Multimodal Controversy Detection (AgentMCD) framework that employs a three-stage reasoning process to systematically evaluate controversy. Additionally, we propose a multi-agent simulation mechanism designed to model the formation of controversy in the early stages of video dissemination.
<p align="center">
    <img src="figures/pic2.png" alt="pic2" width="900">
</p>

## :mag: Case Study
We present case studies that highlight the good explainability of our framework.
<p align="center">
    <img src="figures/pic3.png" alt="pic3" width="900">
</p>

## :rocket: Getting Started
Dependencies
- Python: 3.9.2
- Pytorch: 1.13.1+cu117

To begin, install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

Next, prepare the Multimodal Controversy Detection (MMCD) dataset by creating a new folder named `dataset`. Within this folder, place the validation and test data files, specifically `metadata_valid.json` and `metadata_test.json`, which can be obtained from the [MMCD repository](https://github.com/skylie-xtj/MM_Controversy_Detection_Released) or alternatively downloaded [here](https://pan.quark.cn/s/886b78c6e67d) (password: UnWZ).
```bash
mkdir dataset
mkdir storage
python generate_role.py
```

Download the [GLM-4-9B-CHAT model](https://huggingface.co/THUDM/glm-4-9b-chat/tree/main) and then execute the model using the following command:
```bash
bash lm_run.sh
```

Once everything is set up, you can initiate the process:
```bash
python rich_comments_main.py
python simulate_comments_train.py
python simulate_comments_test.py
```

Alternatively, the result files can be directly downloaded [here](https://pan.quark.cn/s/886b78c6e67d) (password: UnWZ), with the expected directory name `storage`.

## :busts_in_silhouette: Ethical Statement
As discussed, LLMs may occasionally produce irrelevant or harmful outputs, necessitating caution when interpreting their results. In our approach, LLM-based multi-agent systems are employed solely to enhance the simulation of controversy formation. However, additional research is required for language models intended for practical applications to refine prediction accuracy and bolster the model's authenticity and safety, thereby mitigating potential user risks.

## :book: Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :book:.

```BibTeX
@inproceedings{agent_mcd,
  author       = {Xu, Tianjiao and Gao, Jinfei and Kong, Keyi and Yin, Jianhua and Gan, Tian and Nie, Liqiang},
  title        = {Generative Agents for Multimodal Controversy Detection},
  booktitle    = {International Joint Conferences on Artificial Intelligence},
  page         = {9963--9971},
  year         = {2025},
}
```
