#!/bin/bash
# Train all models (LSTM, Transformer, HRM) with the same configuration for comparison

DATA_PATH="dataset/long_short_ratio.csv"
TARGET_COL="long_short_ratio"
VQVAE_CHECKPOINT="checkpoints_vq/vqvae_epoch_5.pt"
HIDDEN_SIZE=128
NUM_LAYERS=4
NUM_HEADS=4
BATCH_SIZE=8
SEQ_LEN=16
EPOCHS=40
LR=5e-4
SEED=42

echo "========================================"
echo "Training All Models for Comparison"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Target Column: $TARGET_COL"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Layers: $NUM_LAYERS"
echo "  Epochs: $EPOCHS"
echo ""

# Train LSTM Baseline
echo "========================================"
echo "1. Training LSTM Baseline"
echo "========================================"
python scripts/train_lstm_baseline.py \
    --data_path "$DATA_PATH" \
    --target_col "$TARGET_COL" \
    --vqvae_checkpoint "$VQVAE_CHECKPOINT" \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --checkpoint_dir checkpoints_lstm_baseline \
    --seed $SEED

echo ""
echo "LSTM training complete!"
echo ""

# Train Transformer Baseline
echo "========================================"
echo "2. Training Transformer Baseline"
echo "========================================"
python scripts/train_transformer_baseline.py \
    --data_path "$DATA_PATH" \
    --target_col "$TARGET_COL" \
    --vqvae_checkpoint "$VQVAE_CHECKPOINT" \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --checkpoint_dir checkpoints_transformer \
    --seed $SEED

echo ""
echo "Transformer training complete!"
echo ""

# Train HRM with LoRA
echo "========================================"
echo "3. Training HRM with LoRA"
echo "========================================"
python scripts/train_hrm_lora_simple.py \
    --data_path "$DATA_PATH" \
    --target_col "$TARGET_COL" \
    --vqvae_checkpoint "$VQVAE_CHECKPOINT" \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_dropout 0.1 \
    --epochs $EPOCHS \
    --lr $LR \
    --checkpoint_dir checkpoints_hrm_lora \
    --seed $SEED

echo ""
echo "HRM training complete!"
echo ""

# Compare all models
echo "========================================"
echo "4. Comparing All Models"
echo "========================================"
python scripts/compare_models.py \
    --data_path "$DATA_PATH" \
    --target_col "$TARGET_COL" \
    --vqvae_checkpoint "$VQVAE_CHECKPOINT" \
    --lstm_checkpoint checkpoints_lstm_baseline/best_lstm_baseline.pt \
    --transformer_checkpoint checkpoints_transformer/best_transformer_baseline.pt \
    --hrm_checkpoint checkpoints_hrm_lora/best_lora.pt \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --output_dir . \
    --seed $SEED

echo ""
echo "========================================"
echo "All Done!"
echo "========================================"
echo ""
echo "Check the following files for results:"
echo "  - model_comparison_results.json"
echo "  - model_comparison_results.txt"
echo ""

