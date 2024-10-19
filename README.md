# Decoupling and Aligning: A New Paradigm for Computationally Efficient Prompt Learning

Prompt learning, as a parameter-efficient fine-tuning paradigm, has emerged as a trend in adapting large pre-trained vision-language models (VLMs) to downstream tasks. However, most existing methods, like CoCoOp and KgCoOp, require converting category names from specific tasks into textual descriptions as text prompt inputs, resulting in the computational cost of the text encoder increasing in direct proportion to the number of categories in the downstream task. To address this challenge, we propose a novel Computationally Efficient Prompt Learning (CEPL) method, which showcases remarkable performance improvement while significantly reducing computation cost. Our CEPL involves the following two key points. 1) Boosted computation efficiency. We propose a textual prompt decoupling module (TPD) that decouples category names from text prompts by learning an image-conditioned text prompt, rather than directly embedding the complete category names. 2) Enhanced tuning effectiveness. We introduce a semantic alignment adaptation module (SAA) which fine-tunes original image features by optimizing task-specific and task-agnostic losses, so that image features are not only aligned with semantic-level text but also adaptable to downstream tasks. Extensive experiments demonstrate that our CEPL achieves superior classification performance at extremely low computational overhead. In particular, CEPL reduces GFLOPs by 95% compared to the state-of-the-art KgCoOp, and yields an average accuracy improvement of 7.57% across 16-shot classifications in 11 datasets.

![intro](imgs/CEPL.png)


## How to Install

This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CEPL_Code/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

## How to Run

### Few-Shot Learning

This corresponds to the experiments in Section 3.1, Few-shot classification.

#### First, you need to generate text features:
Here `CEPL_Code/exp/cross_modal_engine/config/default.py` is configured to save the path of text features and the path of the data set. Then run `CEPL_Code/get_text_feature.sh` to generate the text feature and save it. Next, modify the text feature path inside `CEPL_Code/get_linear_head_weight.py`, which is the text feature generated above.

#### Then, start training the model:
You will need `CEPL_Code/bash.sh` . The bash train and evaluate the model on all classes. Both scripts have three input arguments, i.e., `DATASET`, `EPOCH` and `SEED`.
`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files names in `CEPL_Code/configs/datasets/`.

#### Below we provide an example on how to evaluate the model on Stanford_Cars.

```bash
# seed=1
bash scripts/efficient_prompts/xd_train.sh  stanford_cars 50 1 
bash scripts/efficient_prompts/xd_test.sh  stanford_cars 50 1 
```

#### After you finish the bash using the aforementioned commands, you would get:

```
output
|–– CEPL/
|   |–– stanford_cars/
|   |   |–– vit_b16_cepl_16shots/
|   |   |   |–– seed1/
output
|–– evaluation/
|   |–– stanford_cars/
|   |   |–– vit_b16_cepl_16shots/
|   |   |   |–– seed1/
```


# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Acknowledgement
We would like to thank the authors of  [KgCoOp]( https://github.com/htyao89/KgCoOp) and [CoOp](https://github.com/KaiyangZhou/CoOp), based on which this codebase was built.

