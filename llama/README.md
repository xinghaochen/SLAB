# PRepBN for Llama-350M

## Dependencies

- torch==1.13.1
- tensorboardX
- numpy
- rouge_score
- fire
- openai==0.27.6
- transformers==4.29.1
- datasets==2.17.0
- sentencepiece
- tokenizers==0.13.3
- deepspeed==0.8.3
- accelerate==0.27.2
- scikit-learn

## Evaluate llama-350M

```shell
python evaluation.py --ckpt <checkpoint-path>
```
