# Llama judge queue

Launched 2026-10-02, supervisor PID 344912 on the experiment EC2 instance.
Remote root: /home/ubuntu/experiments/judge-llama-20261002

Model: RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16
Pinned revision: da23851b92f5ea8023d6c70f790feb8e05269812
Safetensors size: 36.811 GiB. Actual GPU fit remains unverified until scheduled loading.

Three conditions: baseline, saliency, nearest neighbour, each 5,190 flagged test examples (15,570 decisions total). Same frozen weighted probes, validation thresholds, labels, rubrics and explanation records as Qwen/Gemma. Semantic instruction hashes checked before inference. Native Llama chat template; temperature zero, one-token restricted 0/1 generation, seed 20261001. No reasoning generation.

Supervisor downloads and prepares inputs, then waits for successful completion of the four-condition Qwen/Gemma explanation queue. It stops on failures without retry. No currently running model is interrupted. Llama uses batches of 4, max_num_batched_tokens 1024 and GPU memory utilization .92, with eager execution. Model quantization and reduced batching are explicit implementation differences.

Fresh state: QUEUE_STATUS.json, queue.log, per-condition run/status.json, run/FAILED.json and run/COMPLETED. Baseline lives at the Llama root; explanation conditions live under judge-explanations-20261002/llama_saliency and llama_nearest. Aggregate results are generated after each full condition; complete raw records remain on the instance.
