nohup python main.py \
--method_type t5 \
--model_type extract_title_zh \
--language zh \
--choose_finetune 1 \
--device_ids 1 \
--default_size small \
--num_epochs 30 \
--batch_size 16 \
--lr 2e-5 \
> logs/extract_title_zh_finetune.log 2>&1 &