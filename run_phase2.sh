#!/bin/bash
# Phase 2: Knowledge Distillation Experiments
# Mitigating Inductive Bias Mismatch in Heterogeneous Knowledge Distillation
#
# Prerequisites: Phase 1 teacher models must be trained to target accuracy:
#   - ResNet-18: >94% validation accuracy
#   - ConvNeXt V2: >95% validation accuracy
#
# This script runs three experiments:
#   EXP-1: DeiT + ResNet-18 (soft KL distillation) - expected: marginal/negative transfer
#   EXP-2: DeiT + ConvNeXt V2 (soft KL distillation) - expected: better alignment
#   EXP-3: DeiT + DINOv2 (CKA structural distillation) - expected: SOTA >89%

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKDIR="/lambda/nfs/lambda-cloud-data/dataset-adaptive-CNN"
RESNET_CHECKPOINT="$WORKDIR/outputs/resnet18_cifar/checkpoints/best_model.pth"
CONVNEXT_CHECKPOINT="$WORKDIR/outputs/convnext_v2_cifar/checkpoints/best_model.pth"

# Logging
LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    Phase 2: Knowledge Distillation Experiments${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

#######################################
# Go/No-Go Gate: Verify Phase 1 Checkpoints
#######################################
echo -e "${YELLOW}[GATE] Verifying Phase 1 teacher checkpoints...${NC}"

GATE_PASS=true

# Check ResNet-18 checkpoint
if [ -f "$RESNET_CHECKPOINT" ]; then
    RESNET_SIZE=$(du -h "$RESNET_CHECKPOINT" | cut -f1)
    echo -e "${GREEN}  [OK] ResNet-18 checkpoint exists ($RESNET_SIZE)${NC}"
else
    echo -e "${RED}  [FAIL] ResNet-18 checkpoint not found: $RESNET_CHECKPOINT${NC}"
    GATE_PASS=false
fi

# Check ConvNeXt V2 checkpoint
if [ -f "$CONVNEXT_CHECKPOINT" ]; then
    CONVNEXT_SIZE=$(du -h "$CONVNEXT_CHECKPOINT" | cut -f1)
    echo -e "${GREEN}  [OK] ConvNeXt V2 checkpoint exists ($CONVNEXT_SIZE)${NC}"
else
    echo -e "${RED}  [FAIL] ConvNeXt V2 checkpoint not found: $CONVNEXT_CHECKPOINT${NC}"
    GATE_PASS=false
fi

# Gate decision
if [ "$GATE_PASS" = false ]; then
    echo ""
    echo -e "${RED}[ABORT] Go/No-Go gate FAILED. Phase 1 checkpoints missing.${NC}"
    echo -e "${RED}        Please ensure Phase 1 training has completed successfully.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}[GATE] Go/No-Go gate PASSED. Proceeding with Phase 2...${NC}"
echo ""

#######################################
# Experiment Execution
#######################################

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local config=$2
    local command=$3
    local gpu=$4
    local log_file=$5

    echo -e "${BLUE}[EXP] Starting $exp_name on GPU $gpu${NC}"
    echo "      Config: $config"
    echo "      Log: $log_file"

    CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$((29500 + gpu)) \
        nohup python "$WORKDIR/main.py" $command "$WORKDIR/configs/$config" \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "      PID: $pid"
    echo "$pid" > "$LOG_DIR/${exp_name}.pid"
    return $pid
}

#######################################
# Strategy Selection
#######################################
echo -e "${YELLOW}Select execution strategy:${NC}"
echo "  1) Sequential: Run all experiments one after another (safest)"
echo "  2) Parallel EXP-1/2, then EXP-3: Run EXP-1 and EXP-2 in parallel, then EXP-3"
echo "  3) All parallel: Run all experiments in parallel (requires sufficient VRAM)"
echo ""
read -p "Enter choice [1-3, default=2]: " STRATEGY
STRATEGY=${STRATEGY:-2}

case $STRATEGY in
    1)
        echo -e "${BLUE}Running experiments sequentially...${NC}"

        # EXP-1
        echo ""
        echo -e "${BLUE}======== EXP-1: DeiT + ResNet-18 ========${NC}"
        CUDA_VISIBLE_DEVICES=0 python "$WORKDIR/main.py" train-distill \
            "$WORKDIR/configs/deit_resnet18_distill_config.yaml" \
            2>&1 | tee "$LOG_DIR/exp1_deit_resnet18.log"

        # EXP-2
        echo ""
        echo -e "${BLUE}======== EXP-2: DeiT + ConvNeXt V2 ========${NC}"
        CUDA_VISIBLE_DEVICES=0 python "$WORKDIR/main.py" train-distill \
            "$WORKDIR/configs/deit_convnext_distill_config.yaml" \
            2>&1 | tee "$LOG_DIR/exp2_deit_convnext.log"

        # EXP-3
        echo ""
        echo -e "${BLUE}======== EXP-3: DeiT + DINOv2 CKA ========${NC}"
        CUDA_VISIBLE_DEVICES=0 python "$WORKDIR/main.py" train-ss-distill \
            "$WORKDIR/configs/deit_ss_distill_cka_cifar_config.yaml" \
            2>&1 | tee "$LOG_DIR/exp3_deit_dinov2_cka.log"
        ;;

    2)
        echo -e "${BLUE}Running EXP-1 and EXP-2 in parallel, then EXP-3...${NC}"
        echo ""

        # EXP-1 on GPU 0
        echo -e "${BLUE}======== EXP-1: DeiT + ResNet-18 (GPU 0) ========${NC}"
        CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 \
            nohup python "$WORKDIR/main.py" train-distill \
            "$WORKDIR/configs/deit_resnet18_distill_config.yaml" \
            > "$LOG_DIR/exp1_deit_resnet18.log" 2>&1 &
        EXP1_PID=$!
        echo "  EXP-1 PID: $EXP1_PID"
        echo "$EXP1_PID" > "$LOG_DIR/exp1.pid"

        # EXP-2 on GPU 1
        echo -e "${BLUE}======== EXP-2: DeiT + ConvNeXt V2 (GPU 1) ========${NC}"
        CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 \
            nohup python "$WORKDIR/main.py" train-distill \
            "$WORKDIR/configs/deit_convnext_distill_config.yaml" \
            > "$LOG_DIR/exp2_deit_convnext.log" 2>&1 &
        EXP2_PID=$!
        echo "  EXP-2 PID: $EXP2_PID"
        echo "$EXP2_PID" > "$LOG_DIR/exp2.pid"

        echo ""
        echo -e "${YELLOW}Waiting for EXP-1 and EXP-2 to complete...${NC}"
        echo "  Monitor with: tail -f $LOG_DIR/exp1_deit_resnet18.log"
        echo "                tail -f $LOG_DIR/exp2_deit_convnext.log"

        wait $EXP1_PID
        EXP1_STATUS=$?
        wait $EXP2_PID
        EXP2_STATUS=$?

        echo ""
        if [ $EXP1_STATUS -eq 0 ]; then
            echo -e "${GREEN}  [OK] EXP-1 completed successfully${NC}"
        else
            echo -e "${RED}  [FAIL] EXP-1 failed with status $EXP1_STATUS${NC}"
        fi

        if [ $EXP2_STATUS -eq 0 ]; then
            echo -e "${GREEN}  [OK] EXP-2 completed successfully${NC}"
        else
            echo -e "${RED}  [FAIL] EXP-2 failed with status $EXP2_STATUS${NC}"
        fi

        # EXP-3 (uses both GPUs for full power)
        echo ""
        echo -e "${BLUE}======== EXP-3: DeiT + DINOv2 CKA ========${NC}"
        python "$WORKDIR/main.py" train-ss-distill \
            "$WORKDIR/configs/deit_ss_distill_cka_cifar_config.yaml" \
            2>&1 | tee "$LOG_DIR/exp3_deit_dinov2_cka.log"
        ;;

    3)
        echo -e "${BLUE}Running all experiments in parallel...${NC}"
        echo -e "${YELLOW}WARNING: This requires ~100GB VRAM total. Ensure sufficient resources.${NC}"
        echo ""

        # EXP-1 on GPU 0
        echo -e "${BLUE}======== EXP-1: DeiT + ResNet-18 (GPU 0) ========${NC}"
        CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 \
            nohup python "$WORKDIR/main.py" train-distill \
            "$WORKDIR/configs/deit_resnet18_distill_config.yaml" \
            > "$LOG_DIR/exp1_deit_resnet18.log" 2>&1 &
        EXP1_PID=$!
        echo "  EXP-1 PID: $EXP1_PID"

        # EXP-2 on GPU 1
        echo -e "${BLUE}======== EXP-2: DeiT + ConvNeXt V2 (GPU 1) ========${NC}"
        CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 \
            nohup python "$WORKDIR/main.py" train-distill \
            "$WORKDIR/configs/deit_convnext_distill_config.yaml" \
            > "$LOG_DIR/exp2_deit_convnext.log" 2>&1 &
        EXP2_PID=$!
        echo "  EXP-2 PID: $EXP2_PID"

        # Wait a moment for GPU allocation
        sleep 5

        # EXP-3 (try to share GPU 0 if EXP-1 fits)
        echo -e "${BLUE}======== EXP-3: DeiT + DINOv2 CKA (GPU 0) ========${NC}"
        CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29502 \
            nohup python "$WORKDIR/main.py" train-ss-distill \
            "$WORKDIR/configs/deit_ss_distill_cka_cifar_config.yaml" \
            > "$LOG_DIR/exp3_deit_dinov2_cka.log" 2>&1 &
        EXP3_PID=$!
        echo "  EXP-3 PID: $EXP3_PID"

        echo ""
        echo -e "${YELLOW}All experiments launched. Monitor with:${NC}"
        echo "  tail -f $LOG_DIR/exp1_deit_resnet18.log"
        echo "  tail -f $LOG_DIR/exp2_deit_convnext.log"
        echo "  tail -f $LOG_DIR/exp3_deit_dinov2_cka.log"
        echo ""
        echo "Waiting for all experiments to complete..."

        wait $EXP1_PID $EXP2_PID $EXP3_PID
        ;;

    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}    Phase 2 Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Checkpoints saved to:"
echo "  EXP-1: $WORKDIR/outputs/deit_resnet18_distill/checkpoints/"
echo "  EXP-2: $WORKDIR/outputs/deit_convnext_distill/checkpoints/"
echo "  EXP-3: $WORKDIR/outputs/deit_dinov2_cka/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Run analytics: python main.py analyze <config> <checkpoint>"
echo "  2. Generate comparison plots"
echo "  3. Document results for portfolio"
