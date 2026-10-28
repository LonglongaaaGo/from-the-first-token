# Exercise map

Each function listed below is a TODO. Read the matching section before editing.

| Section | File | Assigned functions |
| --- | --- | --- |
| 01.01 | [01_01_text_to_tokens.py](exercises/01_01_text_to_tokens.py) | `ByteTokenizer.decode`, `ByteTokenizer.encode`, `make_example` |
| 02.01 | [02_01_bpe_pair_counts.py](exercises/02_01_bpe_pair_counts.py) | `count_pairs` |
| 02.02 | [02_02_bpe_merge.py](exercises/02_02_bpe_merge.py) | `merge_pair` |
| 02.03 | [02_03_bpe_train.py](exercises/02_03_bpe_train.py) | `train_bpe` |
| 02.04 | [02_04_bpe_select_rule.py](exercises/02_04_bpe_select_rule.py) | `select_merge_pair` |
| 02.05 | [02_05_bpe_encode.py](exercises/02_05_bpe_encode.py) | `encode_bpe` |
| 02.06 | [02_06_bpe_decode.py](exercises/02_06_bpe_decode.py) | `decode_bpe` |
| 02.07 | [02_07_bpe_save_load.py](exercises/02_07_bpe_save_load.py) | `pack_tokenizer`, `unpack_tokenizer` |
| 03.01 | [03_01_embedding_lookup.py](exercises/03_01_embedding_lookup.py) | `lookup_embeddings` |
| 03.02 | [03_02_scalar_gradient_step.py](exercises/03_02_scalar_gradient_step.py) | `gradient_step` |
| 03.03 | [03_03_scalar_training_loop.py](exercises/03_03_scalar_training_loop.py) | `train_scalar` |
| 03.05 | [03_05_scalar_autograd.py](exercises/03_05_scalar_autograd.py) | `autograd_loss_and_gradient` |
| 03.06 | [03_06_tensor_gradient_step.py](exercises/03_06_tensor_gradient_step.py) | `tensor_gradient_step` |
| 03.07 | [03_07_tensor_training_loop.py](exercises/03_07_tensor_training_loop.py) | `train_tensor` |
| 03.08 | [03_08_embedding_gradient_step.py](exercises/03_08_embedding_gradient_step.py) | `embedding_gradient_step` |
| 03.09 | [03_09_logits_and_softmax.py](exercises/03_09_logits_and_softmax.py) | `compute_logits`, `softmax_probabilities` |
| 03.10 | [03_10_cross_entropy_loss.py](exercises/03_10_cross_entropy_loss.py) | `cross_entropy_loss` |
| 03.11 | [03_11_next_token_train_step.py](exercises/03_11_next_token_train_step.py) | `next_token_train_step` |
| 03.12 | [03_12_next_token_training_loop.py](exercises/03_12_next_token_training_loop.py) | `train_next_token` |
| 03.13 | [03_13_predict_next_token.py](exercises/03_13_predict_next_token.py) | `predict_next_token` |
| 03.14 | [03_14_autoregressive_generation.py](exercises/03_14_autoregressive_generation.py) | `generate_token_ids` |
| 03.15 | [03_15_temperature_and_sampling.py](exercises/03_15_temperature_and_sampling.py) | `sample_token_id`, `temperature_probabilities` |
| 03.16 | [03_16_sampled_generation.py](exercises/03_16_sampled_generation.py) | `generate_sampled_ids` |
| 03.17 | [03_17_generation_stopping.py](exercises/03_17_generation_stopping.py) | `generate_until_eos` |
| 03.18 | [03_18_decode_continuation.py](exercises/03_18_decode_continuation.py) | `decode_continuation` |
| 03.19 | [03_19_next_token_batch.py](exercises/03_19_next_token_batch.py) | `make_next_token_batch` |
| 03.20 | [03_20_batched_logits_and_loss.py](exercises/03_20_batched_logits_and_loss.py) | `batch_cross_entropy`, `compute_batch_logits` |
| 03.21 | [03_21_batch_train_step.py](exercises/03_21_batch_train_step.py) | `batch_train_step` |
| 03.22 | [03_22_batch_training_loop.py](exercises/03_22_batch_training_loop.py) | `train_batches` |
| 03.23 | [03_23_evaluate_batches.py](exercises/03_23_evaluate_batches.py) | `evaluate_batches` |
| 03.24 | [03_24_baseline_pipeline.py](exercises/03_24_baseline_pipeline.py) | `run_baseline` |
| 04.01 | [04_01_context_baseline.py](exercises/04_01_context_baseline.py) | `prefix_mean` |
| 04.02 | [04_02_registered_model.py](exercises/04_02_registered_model.py) | `TokenModel.forward` |
| 04.03 | [04_03_qkv_projection.py](exercises/04_03_qkv_projection.py) | `project_qkv` |
| 04.04 | [04_04_scaled_attention.py](exercises/04_04_scaled_attention.py) | `attention` |
| 04.05 | [04_05_causal_mask.py](exercises/04_05_causal_mask.py) | `causal_attention` |
| 04.06 | [04_06_multihead_attention.py](exercises/04_06_multihead_attention.py) | `multihead` |
| 04.07 | [04_07_position_embeddings.py](exercises/04_07_position_embeddings.py) | `add_positions` |
| 04.08 | [04_08_residual_norm_ffn.py](exercises/04_08_residual_norm_ffn.py) | `pre_norm_ffn` |
| 04.09 | [04_09_decoder_model.py](exercises/04_09_decoder_model.py) | `TinyDecoder.forward` |
| 04.10 | [04_10_train_tiny_transformer.py](exercises/04_10_train_tiny_transformer.py) | `fit_tiny` |
| 05.01 | [05_01_rmsnorm.py](exercises/05_01_rmsnorm.py) | `rmsnorm` |
| 05.02 | [05_02_rotary_positions.py](exercises/05_02_rotary_positions.py) | `apply_rope` |
| 05.03 | [05_03_swiglu.py](exercises/05_03_swiglu.py) | `swiglu` |
| 05.04 | [05_04_modern_model.py](exercises/05_04_modern_model.py) | `ModernBlock.forward` |
| 05.05 | [05_05_adamw_initialization.py](exercises/05_05_adamw_initialization.py) | `adamw_update` |
| 05.06 | [05_06_schedule_clipping.py](exercises/05_06_schedule_clipping.py) | `scheduled_clip` |
| 05.07 | [05_07_checkpoint_resume.py](exercises/05_07_checkpoint_resume.py) | `checkpoint_state` |
| 06.01 | [06_01_data_manifest.py](exercises/06_01_data_manifest.py) | `data_manifest` |
| 06.02 | [06_02_document_splits.py](exercises/06_02_document_splits.py) | `split_documents` |
| 06.03 | [06_03_clean_deduplicate.py](exercises/06_03_clean_deduplicate.py) | `clean_deduplicate` |
| 06.04 | [06_04_frozen_tokenizer.py](exercises/06_04_frozen_tokenizer.py) | `encode_documents` |
| 06.05 | [06_05_packing_masks.py](exercises/06_05_packing_masks.py) | `pack_tokens` |
| 06.06 | [06_06_pretrain_run.py](exercises/06_06_pretrain_run.py) | `pretrain` |
| 07.01 | [07_01_language_metrics.py](exercises/07_01_language_metrics.py) | `language_metrics` |
| 07.02 | [07_02_fixed_evaluation.py](exercises/07_02_fixed_evaluation.py) | `evaluate_fixed` |
| 07.03 | [07_03_kv_cache.py](exercises/07_03_kv_cache.py) | `cached_attention` |
| 07.04 | [07_04_inference_interface.py](exercises/07_04_inference_interface.py) | `generate_cached` |
| 08.01 | [08_01_chat_format.py](exercises/08_01_chat_format.py) | `format_chat` |
| 08.02 | [08_02_assistant_sft.py](exercises/08_02_assistant_sft.py) | `assistant_loss` |
| 08.03 | [08_03_direct_preference.py](exercises/08_03_direct_preference.py) | `dpo_loss` |
| 08.04 | [08_04_policy_objective.py](exercises/08_04_policy_objective.py) | `clipped_policy_loss` |
| 09.01 | [09_01_budget_profiling.py](exercises/09_01_budget_profiling.py) | `training_budget` |
| 09.02 | [09_02_gradient_accumulation.py](exercises/09_02_gradient_accumulation.py) | `accumulated_step` |
| 09.03 | [09_03_mixed_precision.py](exercises/09_03_mixed_precision.py) | `mixed_precision_step` |
| 09.04 | [09_04_efficient_attention.py](exercises/09_04_efficient_attention.py) | `efficient_attention` |
| 09.05 | [09_05_distributed_recovery.py](exercises/09_05_distributed_recovery.py) | `distributed_gradient` |
| 10.01 | [10_01_experiment_provenance.py](exercises/10_01_experiment_provenance.py) | `experiment_record` |
| 10.02 | [10_02_scaling_fit.py](exercises/10_02_scaling_fit.py) | `fit_scaling` |
| 10.03 | [10_03_ablations_uncertainty.py](exercises/10_03_ablations_uncertainty.py) | `paired_effect` |
