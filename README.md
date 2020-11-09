# Naturalness of Hardware Descriptions

This repository contains our data used to measure naturalness of
hardware descriptions and models for assignment completion. Details
can be found in our ESEC/FSE'20 paper "[On the Naturalness of Hardware
Descriptions](HDLP.pdf)".

If you have used our data or code in a research project, please cite
the research paper in any related publication:
```
@inproceedings{LeeETAL20HDLP,
  author =       {Jaeseong Lee and Pengyu Nie and Junyi Jessy Li and Milos Gligoric},
  title =        {On the Naturalness of Hardware Descriptions},
  booktitle =    {Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  pages =        {to appear},
  year =         {2020},
}
```


# Data

Our data is located in `data/`.  Its five sub-directories,
`data/vhdl`, `data/verilog`, `data/systemverilog`,
`data/java-popular`, and `data/java-naturalness` holds the data for
each corpus.  In the remainder of this file, I will use `$lang` to
denote any of the five corpora.  Under `data/$lang`, there is an `ALL`
sub-directory which holds the data relevant for the corpus as a whole.
The remaining sub-directories contain the data relevant for individual
repositories: a sub-directory with name `user_repo` contains the data
relevant to the repository `https://github.com/user/repo`. I will use
`$repo` to denote any repository (excluding `ALL`).

- `data/$lang/repositories.txt`: List of repositories in each corpus.
  On each line of this file, there is a URL to the GitHub repository,
  then a space, and then the SHA (revision) we used in our
  study.  Due to license restrictions and size limitations, we cannot
  directly share all the files we analyzed, but they are accessible
  via the URLs and SHAs we provide.

- `data/$lang/ALL/cksum.txt`: List of non-duplicate parsable files in
  each corpus.  On each line of this file, there is a checksum
  (obtained by using `cksum` command), then a space, then the path of
  that file (after removing the `_downloads/$lang/repos/` prefix).
  
- `data/$lang/$repo/cksum.txt`: List of non-duplicate parsable files
  in each repository.  Same format as `data/$lang/ALL/cksum.txt`.
  
- `data/$lang/$repo/num-lines.txt`: Number of lines of code in each
  repository.

- `data/$lang/ALL/ce/`: The results of naturalness experiments on this
  corpus.  This directory contains `order-$n.json` files where `$n` is
  in {1, ..., 10}. Each `order-$n.json` file contains the cross
  entropies obtained on the 10 folds using n-gram language model.

- `data/$lang/$repo/ce/`: The results of naturalness experiments on
  this corpus.  Same format as `data/$lang/ALL/ce/`.

- `data/vhdl/ALL/assignments.json`: All concurrent assignments in VHDL.


# Assignment Completion Models

Based on our collected data, we build deep learning models for
predicting the right hand side of concurrent assignments in VHDL.  The
code for our models is located at `completion/`.

## Requirements

- Python>=3.7.6

- PyTorch==1.1.0 (see
  https://pytorch.org/get-started/previous-versions/#v110 for
  installation instructions)
  
- Other Python package requirements listed at
  `completion/requirements.txt`. To install them, run the following
  command after installing Python and Pytorch:

```
# cd completion
pip install -r requirements.txt
```

Our code is partially based on the OpenNMT framework, whose license is
at `completion/LICENSE.md`.

## Steps to reproduce

To reproduce our experiments (training and testing the models), run
this command to split the concurrent assignments dataset
(data/vhdl/ALL/assignments.json):

```
# cd data
python -m hdlp.main split_dataset --cross-file --random-seed=27
```

Then, the commands to train and test each model are listed below (all
commands should be executed under the `completion` directory).  In the
comment before each command, we also note down the time for training
each model on our machine (Intel i7-8700 CPU @ 3.20GHz with 6 cores,
64GB RAM, Nvidia Geforce GTX 1080, Ubuntu 18.04). After training each
model, their results can be found at `completion/tests/$model` where
`$model` is the name of the model after removing parentheses.

- S2S

```
# 5 hours
python3 ex_ms2.py --mode train --feat l --save_dir "S2S"
```

- S2S-PA(1)

```
# 6 hours
python3 ex_ms2.py --mode train --feat lpa --save_dir "S2S-PA1" --pa_index 1
```

- S2S-PA(1)+Type

```
# 12 hours
python3 ex_ms2.py --mode train --feat lpa+typeappend --save_dir "S2S-PA1+Type"
```

- S2S-PA(2)+Type

```
# 12 hours
python3 ex_ms2.py --mode train --feat lpa+typeappend --pa_index 2 --save_dir "S2S-PA2+Type"
```

- S2S-PA(3)+Type

```
# 12 hours
python3 ex_ms2.py --mode train --feat lpa+typeappend --pa_index 3 --save_dir "S2S-PA3+Type"
```

- S2S-PA(4)+Type

```
# 12 hours
python3 ex_ms2.py --mode train --feat lpa+typeappend --pa_index 4 --save_dir "S2S-PA4+Type"
```

- S2S-PA(5)+Type

```
# 12 hours
python3 ex_ms2.py --mode train --feat lpa+typeappend --pa_index 5 --save_dir "S2S-PA5+Type"
```

- S2S-PA(Ensemb-1-5)+Type

Require first training S2S-PA(1)+Type, S2S-PA(2)+Type, S2S-PA(3)+Type,
S2S-PA(4)+Type, S2S-PA(5)+Type.

```
# several minutes
python3 ex_ms2.py --mode testval --feat lpa+typeappend --save_dir "S2S-PA1+Type"
python3 ex_ms2.py --mode testval --feat lpa+typeappend --pa_index 2 --save_dir "S2S-PA2+Type"
python3 ex_ms2.py --mode testval --feat lpa+typeappend --pa_index 3 --save_dir "S2S-PA3+Type"
python3 ex_ms2.py --mode testval --feat lpa+typeappend --pa_index 4 --save_dir "S2S-PA4+Type"
python3 ex_ms2.py --mode testval --feat lpa+typeappend --pa_index 5 --save_dir "S2S-PA5+Type"
python3 ex_ms2.py --mode assemble --save_dir "S2S-PAEnsemb-1-5+Type" --which "S2S-PA1+Type" "S2S-PA2+Type" "S2S-PA3+Type" "S2S-PA4+Type" "S2S-PA5+Type"
```

- S2S-PA(1-2)+Type

```
# 25 hours
python3 ex_msap.py --mode train --feat apa+typeappend --num_pa 2 --save_dir "S2S-PA1-2+Type"
```

- S2S-PA(1-3)+Type

```
# 35 hours
python3 ex_msap.py --mode train --feat apa+typeappend --num_pa 3 --save_dir "S2S-PA1-3+Type"
```

- S2S-PA(1-4)+Type

```
# 46 hours
python3 ex_msap.py --mode train --feat apa+typeappend --num_pa 4 --save_dir "S2S-PA1-4+Type"
```

- S2S-PA(1-5)+Type

```
# 56 hours
python3 ex_msap.py --mode train --feat apa+typeappend --num_pa 5 --save_dir "S2S-PA1-5+Type"
```

- S2S-PA(Concat-1-5)+Type

```
# 48 hours
python3 ex_s2s.py --mode train --num_pa=5 --type_append --save_dir "S2S-LHS+PAConcat-1-5+Type"
```

- Rule-based baseline

Require first running S2S-PA(1) that performs necessary data processing.

```
# < 1 minute
python3 ex_baseline.py --ref_modelname "S2S-PA1" --modelname "Baseline"
```

- 10gramLM

Require first running S2S-PA(1-5)+Type that performs necessary data processing.

```
# < 1 minute
python3 ex_ngram.py --order 10 --pa 0 --ref_modelname "S2S-PA1-5+Type" --modelname "10gramLM"
```

- 10gramLM+PA(1)

Require first running S2S-PA(1-5)+Type that performs necessary data processing.

```
# < 1 minute
python3 ex_ngram.py --order 10 --pa 1 --ref_modelname "S2S-PA1-5+Type" --modelname "10gramLM+PA1"
```

- 10gramLM+PA(1-5)

Require first running S2S-PA(1-5)+Type that performs necessary data processing.

```
# 5 minutes
python3 ex_ngram.py --order 10 --pa 5 --ref_modelname "S2S-PA1-5+Type" --modelname "10gramLM+PA1-5"
```

- RNNLM

Require first running S2S-PA(1-5)+Type that performs necessary data processing.

```
# 10 minutes
python3 ex_ngram.py --pa 0 --rnn True --ref_modelname "S2S-PA1-5+Type" --modelname "RNNLM"
```

- RNNLM+PA(1)

Require first running S2S-PA(1-5)+Type that performs necessary data processing.

```
# 20 minutes
python3 ex_ngram.py --pa 1 --rnn True --ref_modelname "S2S-PA1-5+Type" --modelname "RNNLM+PA1"
```

- RNNLM+PA(1-5)

Require first running S2S-PA(1-5)+Type that performs necessary data processing.

```
# 30 minutes
python3 ex_ngram.py --pa 5 --rnn True --ref_modelname "S2S-PA1-5+Type" --modelname "RNNLM+PA1-5"
```


# Other Code

The `code` directory contains some miscellaneous source code used in
our experiments, described as follows.  We found and adapted them from
other open-source repositories, and would like to share them to
facilitate future work.

- `code/grammars`: The ANTLR grammar source code for generating VHDL,
  Verilog, and SystemVerilog parsers. They are adapted from
  (https://github.com/antlr/grammars-v4) and
  (https://github.com/eirikpre/VSCode-SystemVerilog).

- `code/tool`: The n-gram language model used in our naturalness
  experiments, adapted from
  [SLP](https://github.com/SLP-team/SLP-Core).
