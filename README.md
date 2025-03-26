# MHA2HLA


[llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)

[checkpoint](https://huggingface.co/AICrossSim/clm-60m)

[Perplexity ](https://gist.github.com/ChengZhang-98/5eaa628d26dc4edb6fd22c3705b218dc)

[For quick fine-tuning and eval, use wikitext](https://huggingface.co/datasets/Salesforce/wikitext)

[later you may want to use a subset of fineweb to finetune the transformed model and cross validate on wikitext2.](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

# Traing test list:
- With / without frozen qkv_proj 
- recover upscale with / without latent
- inference with original / changed S code (inference的时候啥时候用到了S？好像没用到？)
- try model 60m/200m
  
to be continued...
add-test-workflow
change PR rule

## to start a PR
git checkout -b newbranch

git add .

git commit -m "pr"

git push commit origin

this allow a test before push
# Language Model Training and Evaluation

## 🏋️‍♂️ Training

All training-related code can be found in the `train/` directory.

We trained models on both:
- **Google Colab** with A100 GPUs
- **Imperial College HPC** with V100 GPUs

To start training:
```bash
python fine-tune.py  # Run inside each experiment folder
```

Google Colab Jupyter Notebooks used for training are available in our Google Drive.

---

## 📁 Google Drive – Model Weights & Evaluation

Our trained model checkpoints and evaluation notebooks are available in Google Drive.  
Each experiment includes:
- Model weights
- Perplexity evaluation on WikiText-2
- Text generation testing

### 📦 Checkpoints

#### 🔹 Original `clm-60m` tuning results  
📎 [Link](https://drive.google.com/drive/folders/105usjkc0ZiZjhLqxZFPmo9dBHafRnJH1?usp=sharing)

#### 🔹 `clm-60m` + HLA (with original RoPE)  
- Best validation at **epoch 18**  
- Includes training & test notebooks  
📎 [Link](https://drive.google.com/drive/folders/1UNUGjmVJyrsqFIp6Vn-Sx4KfkrB0WGX9?usp=sharing)

#### 🔹 `clm-60m` + HLA (with latent RoPE)  
- Weights saved in `checkpoint-56252`  
- Includes training & test notebooks  
📎 [Link](https://drive.google.com/drive/folders/1tItskPkZejRLY606_ZEvusaWz2T2SQiw?usp=drive_link)

---

Feel free to open an issue if you have questions about using the checkpoints or running training.

## Achievements:
- Prove:
    - Prove the math of transforming the RoPE to latent space to save the cache space and computaional time. 
- implement:
    - 1. Finetuned the 60m model on wiki dataset: as the baseline
    - 2. Test the perplexity of origianl checkpoint and finetuned checkpoint from origianl model.
    - 3. Implement the MLA code:
        - apply SVD to qkv_proj.weight to get up and down scale matrix
        - apply down-scale matrix to hidden_state to get the C(latent) state.
        - apply SVD to up-scale matrix to get the B matrix, and compute RoPE to C(latent) state.
        - cache key_states_h and queue_states_h
        - apply up-scale matrix to recover to original dimention to compute attenntion.

## TODO
- 1. finetune the original model to see the baseline
- 2. solve the problem of SVD: the computation of SVD is slow
    - set B matrix as learnable parameters
    - use fast SVD 
- 3. finish training lists as Above to see the influence