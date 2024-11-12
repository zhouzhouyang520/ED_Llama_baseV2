#!/bin/sh

#source /etc/network_turbo
#source activate py310_chat
# conda activate chatglm_etuning

out=out.log

nohup sh ed.sh 0 > $out 2>&1 &

tail -f $out
