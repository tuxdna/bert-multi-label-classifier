#!/bin/bash

# enter virtualenv here, you can replace it with your own setup
source /home/tuxdna/venv3/bin/activate

export ZEROMQ_SOCK_TMP_DIR=/tmp

pushd data/uncased_L-12_H-768_A-12/

bert-serving-start -model_dir ./ -graph_tmp_dir /tmp -fp16 -num_worker 1 -max_seq_len NONE  -show_tokens_to_client
popd
deactivate
