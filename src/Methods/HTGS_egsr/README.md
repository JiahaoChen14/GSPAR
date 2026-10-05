# HTGS_egsr
HTGS with Background Model

Install all method dependencies with:

```shell
python ./scripts/install.py -m HTGS_egsr
```

This also installs `transformers` and its model-loading dependency `accelerate`,
which are used to generate sky masks for the foreground/background separation
loss. SegFormer weights are downloaded on the first training run unless they
are already present in the Hugging Face cache.
