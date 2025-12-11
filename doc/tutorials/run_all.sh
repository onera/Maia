#!/bin/bash

mystnb-to-jupyter 01_basic.myst -o
mystnb-to-jupyter 02_pre.myst -o
mystnb-to-jupyter 04_post.myst -o

python3 exec_par.py 01_basic.ipynb 01_basic.ipynb 3 &
python3 exec_par.py 02_pre.ipynb 02_pre.ipynb 4 &
python3 exec_par.py 04_post.ipynb 04_post.ipynb 6 &

wait

rm *.cgns
rm *.log
