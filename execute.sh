#!/bin/bash
conda activate myenv
python Random.py  -nc=40 -iid=False;
wait
python Random.py  -nc=60 -iid=False;
wait
python Random.py  -nc=80 -iid=False;
wait
python Random.py  -nc=100 -iid=False;
wait
python BQN.py -nc=40 -mlr=0.01 -iid=False;
wait
python BQN.py -nc=40 -mlr=0.001 -iid=False;
wait
python BQN.py -nc=60 -mlr=0.01 -iid=False;
wait
python BQN.py -nc=60 -mlr=0.001 -iid=False;
wait
python BQN.py -nc=80 -mlr=0.01 -iid=False;
wait
python BQN.py -nc=80 -mlr=0.001 -iid=False;
wait
python BQN.py -nc=100 -mlr=0.01 -iid=False;
wait
python BQN.py -nc=100 -mlr=0.001 -iid=False;
wait
