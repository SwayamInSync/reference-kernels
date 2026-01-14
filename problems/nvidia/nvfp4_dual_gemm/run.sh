rm -rf __pycache__ 
rm -rf ~/.cache/torch_extensions
mode=$1
eval_file=eval.py

# compute mean for benchmark and leaderboard modes
if [ "$mode" = "benchmark" ] || [ "$mode" = "leaderboard" ]; then
    CUDA_VISIBLE_DEVICES=0 POPCORN_FD=1 CUTE_DSL_ARCH=sm_100a python $eval_file $mode task.yml | tee /tmp/benchmark_output.txt
    
    # Extract mean values and compute geometric mean
    echo ""
    echo "========== SUMMARY =========="
    means=$(grep "\.mean:" /tmp/benchmark_output.txt | awk -F': ' '{print $2}')
    
    # Print individual means
    i=0
    for mean in $means; do
        spec=$(grep "benchmark.$i.spec:" /tmp/benchmark_output.txt | cut -d':' -f2-)
        printf "Test %d:%s -> mean: %.2f ns\n" $i "$spec" $mean
        i=$((i + 1))
    done
    
    # Compute geometric mean
    geomean=$(echo "$means" | awk '
    BEGIN { product = 1; count = 0 }
    { product *= $1; count++ }
    END { 
        if (count > 0) 
            printf "%.2f", exp(log(product) / count)
        else 
            print "N/A"
    }')
    
    echo ""
    echo "Geometric Mean: $geomean ns"
    echo "============================="
else
    CUDA_VISIBLE_DEVICES=0 POPCORN_FD=1 CUTE_DSL_ARCH=sm_100a python $eval_file $mode task.yml
fi