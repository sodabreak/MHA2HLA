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
## Google drive containing models
https://drive.google.com/drive/folders/1cg_nFA9tmdDLZJ7h3oUyqO-vubNTQHLP?usp=sharing
5 epoch training on wikitext-2-v1 


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