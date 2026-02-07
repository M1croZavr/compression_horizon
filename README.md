# compression_horizon

## TODOs

### Embeddings initialization ablation

### Compare activations diff with different prefix tokens for the same sequence and plot its PCA progressively increasing sequence length

## Linear probing for different NLP Tasks


### Add script for github actions exeriment run.


### Save artifacs. Add telegram for for artifacs exploration.

uv run python ./scripts/reproduction.py --model_checkpoint unsloth/Llama-3.2-1B --limit_dataset_items 10 --max_sequence_length 512 --number_of_mem_tokens 1 --max_optimization_steps_per_sample 5000 --random_seed 1337 --per_device_train_batch_size 10 --learning_rate 0.01 --adam_beta1 0.9 --adam_beta2 0.9 --weight_decay 0.01 --dtype bfloat16

uv run python ./scripts/reproduction.py --model_checkpoint unsloth/Llama-3.2-3B --limit_dataset_items 10 --max_sequence_length 1024 --number_of_mem_tokens 1 --max_optimization_steps_per_sample 5000 --random_seed 1337 --per_device_train_batch_size 10 --learning_rate 0.01 --adam_beta1 0.9 --adam_beta2 0.9 --weight_decay 0.01 --dtype bfloat16

uv run python ./scripts/reproduction.py --model_checkpoint unsloth/Meta-Llama-3.1-8B --limit_dataset_items 10 --max_sequence_length 1568 --number_of_mem_tokens 1 --max_optimization_steps_per_sample 5000 --random_seed 1337 --per_device_train_batch_size 2 --learning_rate 0.01 --adam_beta1 0.9 --adam_beta2 0.9 --weight_decay 0.01 --dtype bfloat16

debug
uv run python ./scripts/reproduction.py --model_checkpoint unsloth/Llama-3.2-1B --limit_dataset_items 2 --max_sequence_length 128 --number_of_mem_tokens 1 --max_optimization_steps_per_sample 5000 --random_seed 1337 --per_device_train_batch_size 2 --learning_rate 0.01 --adam_beta1 0.9 --adam_beta2 0.9 --weight_decay 0.01 --dtype bfloat16

Я посчитал обновленную статистику. Обнови LateX таблицу. *_bos - это такой эксперимент, при котором был проигногирован loss у BOS токена при обучении. *_2leading - это такой эксперимент, при котором у первых двух токенов последовательности loss был увеличен в 3 раза при обучении.
В целом, моя задача была - понять ограничения ситуации, когда сжатие обучилось на 99%, а не на 100%. mismatch@k - это доля примеров на которых после обучения неправильно классифицируется токен индекса k при инференсе с логитами (teacher forcing).  mean_teacher_forcing_convergence - средняя аккураси для всех сжатых примеров. mean_greedy_convergence - среднее аккураси при жадном декодинге, начиная только с компрессионного.
[
    {"model": "Llama-3.2-1B", "sequence length": "512", "mismatch@0": 0.9, "mismatch@1": 1.0, "mismatch@2": 0.0, "mean_teacher_forcing_convergence": "0.9880859375", "mean_greedy_convergence": "0.007293731532990932"},
    {"model": "Llama-3.2-1B_bos", "sequence length": "512", "mismatch@0": 1.0, "mismatch@1": 1.0, "mismatch@2": 0.0, "mean_teacher_forcing_convergence": "0.990234375", "mean_greedy_convergence": "0.0005870841443538666"},
    {"model": "Llama-3.2-1B_2leading", "sequence length": "512", "mismatch@0": 0.0, "mismatch@1": 1.0, "mismatch@2": 0.0, "mean_teacher_forcing_convergence": "0.990234375", "mean_greedy_convergence": "0.0203472007997334"},

    {"model": "Llama-3.2-3B", "sequence length": "1024", "mismatch@0": 1.0, "mismatch@1": 1.0, "mismatch@2": 0.0, "mean_teacher_forcing_convergence": "0.9671875", "mean_greedy_convergence": "0.00048828125"},
    {"model": "Llama-3.2-3B_bos", "sequence length": "1024", "mismatch@0": 1.0, "mismatch@1": 1.0, "mismatch@2": 0.0, "mean_teacher_forcing_convergence": "0.9578125", "mean_greedy_convergence": "0.007914391811937094"},
    {"model": "Llama-3.2-3B_2leading", "sequence length": "1024", "mismatch@0": 0.0, "mismatch@1": 1.0, "mismatch@2": 0.0, "mean_teacher_forcing_convergence": "0.960546875", "mean_greedy_convergence": "0.01557386815547943"},

    {"model": "Llama-3.1-8B", "sequence length": "1568", "mismatch@0": 1.0, "mismatch@1": 1.0, "mismatch@2": 0.0, "mean_teacher_forcing_convergence": "0.9705994963645935", "mean_greedy_convergence": "0.002997448929818347"},
    {"model": "Llama-3.1-8B_bos", "sequence length": "1568", "mismatch@0": 0.9, "mismatch@1": 1.0, "mismatch@2": 0.0, "mean_teacher_forcing_convergence": "0.9880859375", "mean_greedy_convergence": "0.0078125"},
    {"model": "Llama-3.1-8B_2leading", "sequence length": "1568", "mismatch@0": 1.0, "mismatch@1": 1.0, "mismatch@2": 0.0, "mean_teacher_forcing_convergence": "0.9705994963645935", "mean_greedy_convergence": "0.016547608398832382"},
]
Обнови таблицу и текст раздела. То есть опиши, что если на этапе тренировки сжать неидеально, то жадный декодинг почти всегда полностью ломается, так как ошибка происходит в 0 и 1 токене.