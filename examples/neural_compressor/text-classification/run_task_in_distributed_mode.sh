mpirun -np <NUM_PROCESS> \
        -mca btl_tcp_if_include <NETWORK_INTERFACE> \
        -x OMP_NUM_THREADS=<MAX_NUM_THREADS> \
        --host <HOSTNAME1>,<HOSTNAME2>,<HOSTNAME3>... \
        bash -c """
            source ~/miniconda3/etc/profile.d/conda.sh
            conda activate conda_env
            cd path/to/optimum-intel/examples/neural_compressor/text-classification
            python -u ./run_glue_post_training.py \
                --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
                --task_name sst2 \
                --apply_quantization \
                --quantization_approach static \
                --num_calibration_samples 50 \
                --do_eval \
                --verify_loading \
                --output_dir /tmp/sst2_output
            """