
export JOBNAME=parsl.bindcraft_folding.block-0.1761756388.1145294
set -e
export CORES=$(getconf _NPROCESSORS_ONLN)
[[ "1" == "1" ]] && echo "Found cores : $CORES"
WORKERCOUNT=1
FAILONANY=0
PIDS=""

CMD() {
process_worker_pool.py --debug --max_workers_per_node=4 -a 10.140.56.189,10.201.0.188,127.0.0.1,10.201.0.184 -p 0 -c 4 -m None --poll 10 --port=54060 --cert_dir None --logdir=/lus/eagle/projects/FoundEpidem/msinclair/github/MDAgent/runinfo/002/bindcraft_folding --block_id=0 --hb_period=30  --hb_threshold=120 --drain_period=None --cpu-affinity none  --mpi-launcher=mpiexec --available-accelerators 0 1 2 3
}
for COUNT in $(seq 1 1 $WORKERCOUNT); do
    [[ "1" == "1" ]] && echo "Launching worker: $COUNT"
    CMD $COUNT &
    PIDS="$PIDS $!"
done

ALLFAILED=1
ANYFAILED=0
for PID in $PIDS ; do
    wait $PID
    if [ "$?" != "0" ]; then
        ANYFAILED=1
    else
        ALLFAILED=0
    fi
done

[[ "1" == "1" ]] && echo "All workers done"
if [ "$FAILONANY" == "1" ]; then
    exit $ANYFAILED
else
    exit $ALLFAILED
fi
