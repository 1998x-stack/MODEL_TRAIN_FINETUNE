nohup python main.py \
--method_type t5 \
--model_type channels_zh \
--language zh \
--choose_finetune 0 \
--device_ids 2 \
--default_size small \
--num_epochs 30 \
--batch_size 16 \
--lr 2e-5 \
> logs/channels_zh_new_small.log 2>&1 &