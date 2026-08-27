#!/usr/bin/env bash
set -uo pipefail

# This task's verifier (test_outputs.py) is a dependency-free plain script,
# not pytest -- see task/README.md's "task-review findings" section for why
# (environment_hygiene: no test-only deps baked into environment/Dockerfile).

TESTS_DIR="${TESTS_DIR:-/tests}"
LOG_DIR="${LOG_DIR:-/logs/verifier}"
TEST_LOG="${LOG_DIR}/test_outputs.log"
REWARD_PATH="${LOG_DIR}/reward.txt"

mkdir -p "${LOG_DIR}"

python3 "${TESTS_DIR}/test_outputs.py" 2>&1 | tee "${TEST_LOG}"
status=${PIPESTATUS[0]}

if [ "${status}" -eq 0 ]; then
    echo "1" > "${REWARD_PATH}"
else
    echo "0" > "${REWARD_PATH}"
fi

exit "${status}"
