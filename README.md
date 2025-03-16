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