#!/usr/bin/env bash
set -euo pipefail

: "${TEMPORAL_ADDRESS:=127.0.0.1:7233}"
: "${TEMPORAL_NAMESPACE:=clipper}"
: "${TASK_QUEUE:=cpu-tq}"

WF_TYPE="${1:-HelloWorkflow}"
INPUT="${2:-\"ping\"}"
WF_ID="smoke-$(date +%s)"

echo "[smoke] address=$TEMPORAL_ADDRESS namespace=$TEMPORAL_NAMESPACE tq=$TASK_QUEUE type=$WF_TYPE id=$WF_ID input=$INPUT"

temporal workflow start \
  --address "$TEMPORAL_ADDRESS" \
  --namespace "$TEMPORAL_NAMESPACE" \
  --task-queue "$TASK_QUEUE" \
  --type "$WF_TYPE" \
  --workflow-id "$WF_ID" \
  --input "$INPUT" >/dev/null

temporal workflow show \
  --address "$TEMPORAL_ADDRESS" \
  --namespace "$TEMPORAL_NAMESPACE" \
  --workflow-id "$WF_ID"

echo
echo "[smoke] UI: http://192.168.2.36:8080/namespaces/$TEMPORAL_NAMESPACE/workflows/$WF_ID"
