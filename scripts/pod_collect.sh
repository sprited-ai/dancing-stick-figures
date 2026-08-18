#!/bin/bash
# poll pod chain; on "eval done" pull results and terminate the pod. detached; log: pod_results/collect.log
cd /Users/jin/dev/dancing-stick-figure; mkdir -p pod_results; P="-p 48433"; H="root@103.196.86.118"; POD=dzvnhylay9w9r3
while ! ssh $P -o ConnectTimeout=20 $H 'grep -q "eval done" /workspace/pod_chain.log' 2>/dev/null; do sleep 300; done
scp -q $P -P 48433 $H:/workspace/pod_chain.log $H:/workspace/eval_pod.log pod_results/ 2>/dev/null
for r in B0_shift1 B1_shift3 B2_shift3_mix b64; do mkdir -p pod_results/$r; scp -q -P 48433 "$H:/workspace/runs/$r/log.txt" "$H:/workspace/runs/$r/args.json" pod_results/$r/; scp -q -P 48433 "$H:/workspace/runs/$r/sample_00*000.gif" pod_results/$r/ 2>/dev/null; scp -q -r -P 48433 "$H:/workspace/runs/$r/eval" pod_results/$r/ 2>/dev/null; done
python3 scripts/runpod.py terminate $POD >> pod_results/collect.log 2>&1; echo "collected+terminated $(date)" >> pod_results/collect.log
