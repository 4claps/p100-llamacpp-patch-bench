#!/bin/bash
cd /data/p100ab
until grep -q POST-DONE queue.log; do sleep 30; done
python3 srvbench.py /data/p100ab/res/srv 3 prod2 quick2 n4p0_2 n3p02_2 b32k_2 nomtp2 kf2 prod4 quick4
echo SRV-QUEUE-DONE >> queue.log
