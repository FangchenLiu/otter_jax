# 🦦 OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction
by <a href="https://qingh097.github.io/">Huang Huang*</a>, <a href="https://fangchenliu.github.io/">Fangchen Liu*</a>, <a href="https://max-fu.github.io">Letian Fu*</a>, <a href="https://scholar.google.com/citations?user=9bt2Z5QAAAAJ&hl=en">Tingfan Wu</a>, <a href="https://www.mustafamukadam.com/">Mustafa Mukadam</a>, <a href="https://people.eecs.berkeley.edu/~malik/">Jitendra Malik</a>, <a href="https://goldberg.berkeley.edu">Ken Goldberg</a>, <a href="https://people.eecs.berkeley.edu/~pabbeel/">Pieter Abbeel</a> at UC Berkeley and Meta (*equal contribution).

[[Paper](http://arxiv.org/abs/2503.03734)] | [[Project Page](https://ottervla.github.io/)]

This repo contains the official implementation for *Otter: A Vision-Language-Action Model with Text-Aware Feature Extraciton*. We also released a [Pytorch Implementation](https://github.com/Max-Fu/otter).


Further information please contact <a href="https://qingh097.github.io/">Huang Huang</a>, <a href="https://fangchenliu.github.io/">Fangchen Liu</a>, <a href="https://max-fu.github.io">Letian Fu</a>, or post an issue on Github!

## Updates 
- 2025-03-05: Initial code release. 
- WIP: instructions on training, inference.
- WIP: release pretrained models.

## Training

```
python scripts/train.py --config.save_dir=<...>
```

## Contributing
Experimental things and training/eval scripts should go in `experiments/<your_name>`. To make any changes to files outside of your experiments directory, please open a pull request.

To enable code checks and auto-formatting, please install pre-commit hooks:
```
pre-commit install
```

## Environment
```
conda create -n otter_jax python=3.10
conda activate otter_jax
pip install -e .
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.8
conda install -c conda-forge cudnn=8.9
```
For GPU:
```
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]==0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

## License
This project is under the Apache 2.0 license. See [LICENSE](LICENSE.txt) for details.

## Acknowledgement
We thank the authors of [Octo](https://octo-models.github.io/) for providing an easy-to-use codebase for training vision-language-action models.

## Citation 
Please give us a star 🌟 on Github to support us!

Please cite our work if you find our work inspiring or use our code in your work:
```
@article{huang2025otter,
    title={Otter: A Vision-Language-Action Model with Text-Aware Feature Extraciton}, 
    author={Huang Huang and Fangchen Liu and Letian Fu and Tingfan Wu and Mustafa Mukadam and Jitendra Malik and Ken Goldberg and Pieter Abbeel},
    journal={arXiv preprint arXiv:2503.15980},
    year={2025}
}
```
