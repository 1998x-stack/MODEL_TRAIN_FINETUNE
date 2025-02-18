nohup python main.py \
--method_type t5 \
--model_type extract_title_en \
--language en \
--choose_finetune 0 \
--device_ids 3 \
--default_size small \
--num_epochs 30 \
--batch_size 24 \
--lr 2e-5 \
> logs/extract_title_en.log 2>&1 &