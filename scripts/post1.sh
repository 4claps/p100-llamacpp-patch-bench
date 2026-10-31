#!/bin/bash
cd /data/p100ab
until grep -q QUEUE-DONE queue.log; do sleep 30; done
./correct2.sh
echo POST-DONE >> queue.log
