# Out-of-domain FrameNet Semantic Role Labeling

This code is an implementation of a simple frame identification approach (SimpleFrameId) described in the paper "Out-of-domain FrameNet Semantic Role Labeling".
Please use the following citation:

```
@inproceedings{TUD-CS-2017-0011,
	title = {Out-of-domain FrameNet Semantic Role Labeling},
	author = {Hartmann, Silvana and Kuznetsov, Ilia and Martin, Teresa and Gurevych, Iryna},
	publisher = {Association for Computational Linguistics},
	booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017)},
	pages = {to appear},
	month = apr,
	year = {2017},
	location = {Valencia, Spain},
}
```

> **Abstract:**
Domain dependence of NLP systems is one of the major obstacles to their application in  large-scale  text  analysis,  also  restricting the applicability of FrameNet semantic role labeling (SRL) systems. Yet, current FrameNet SRL systems are still only evaluated on a single in-domain test set.  For the first time, we study the domain dependence of FrameNet SRL on a wide range of benchmark sets. We create a novel test set for FrameNet SRL based on user-generated web text and find that the major bottleneck for  out-of-domain  FrameNet  SRL  is  the frame identification step.  To address this problem, we develop a simple, yet efficient system  based  on  distributed  word  representations. Our system closely approaches the state-of-the-art in-domain while outperforming the best available frame identification system out-of-domain.

Contact persons: Teresa Martin, martin@aiphes.tu-darmstadt.de; Ilia Kuznetsov, kuznetsov@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Project structure
The implementation is a single package. Two most important modules are:

* `main.py` -- the entry point for experiments
* `globals.py` -- global variables used in experiments
* `classifier.py` -- the classifiers
* `representation.py` -- representation builders

The system requires a specific folder structure where the data is stored:
* `ROOT` -- your project root (just a folder somewhere on your disk)
* `ROOT/srl_data` -- source data
* `ROOT/srl_data/corpora` -- input corpora
* `ROOT/srl_data/embeddings` -- external VSMs
* `ROOT/srl_data/lexicons` -- external lexicons
* `ROOT/out` -- here the experiment results are stored

## Requirements

* Python 2.7
* Python dependencies: keras, lightfm, sklearn, numpy, networkx

## Installation

Install the dependencies, adjust the paths in `main.py` and `globals.py` accordingly and run via `python main.py`

### Parameter description

* to define in `globals.py`: filenames for
  * pretrained embeddings e.g, Levy dependency embeddings
  * FrameNet lexicon
  * train data
  * test data
* to define in `main.py`
  * `vsms` -- vector space model to use
  * `lexicons` -- lexicon to use (mind the all_unknown setting!)
  * `multiword_averaging` -- treatment of multiword predicates, false - use head embedding, true - use avg
  * `all_unknown` -- makes the lexicon treat all LU as unknown, corresponds to the no-lex setting
  * `num_components` -- for wsabie classifier: dimension for the learned latent representations
  * `max_sampled` -- for wsabie classifier: maximum number of negative samples used during WARP fitting 'warp'
  * `num_epochs` -- for wsabie classifier: number of epochs to train the model




