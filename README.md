# [MICCAI'26] Can Agents Distinguish Visually Hard-to-Separate Diseases in a Zero-Shot Setting? A Pilot Study  

by Zihao Zhao, Frederik Hauke, Juliana De Castilhos, Sven Nebelung, and Daniel Truhn<br/>

<!-- <div style="display: flex; justify-content: center; align-items: center;">
    <img src="asset/overview.png" alt="Image" style="width: 60%;">
</div> -->

<div align="center">
  <img src="asset/teaser.jpeg" style="width: 100%">
</div>

> <p align="justify"> 
> <strong>(Left)</strong> Despite highly overlapping visual patterns, some disease pairs can have totally different etiologies and managements, which makes imaging-only differentiation challenging and high-stakes
> <strong>(Right)</strong> The overview of our proposed Contrastive Agent REasoning (CARE). Two disease-specific agents generate opposing evidence from the same input image. A judge agent adjudicates the arguments, flags unsupported evidence, and outputs the final diagnosis in a training-free, zero-shot setting.
</p>



## Prerequisite
For Gemini-based experiments:

```bash
conda env create -f environment.yml
```

For experiments based on open-source MLLMs:

```bash
uv venv vlm --python 3.10
source vlm/bin/activate
uv pip install -r uv_req.txt
```
Put the resized (512×512) version of [mimic-cxr-jpg](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) dataset and raw [derm7pt](https://derm.cs.sfu.ca/Welcome.html) dataset into ./data
## Usage
For Geminis, specify your personal API Key in 
```
client = genai.Client(api_key = "xxxxxxxxxxxxxxx")
```
and run 
```
conda activate openai
python gemini_cxr_care.py
```
For open-source MLLMs, run the following command on slurm cluster
```
sbatch open-source-vlm.sh
```
or run the following command on your local computer as suggested in open-source-vlm.sh
```
python [derm/cxr]_agent_script_[/care].py --model_name medvlm --output_path OUTPUT_PATH
```

## 📎 Citation

If you find this repository useful for your work, please cite our arXiv paper:

```bibtex
@article{zhao2026can,
  title={Can Agents Distinguish Visually Hard-to-Separate Diseases in a Zero-Shot Setting? A Pilot Study},
  author={Zhao, Zihao and Hauke, Frederik and De Castilhos, Juliana and Nebelung, Sven and Truhn, Daniel},
  journal={arXiv preprint arXiv:2602.22959},
  year={2026}
}
```

