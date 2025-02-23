#!/bin/bash

# Check if the invariant still holds for the IR program
grep -q "Constant" ../ir/ir_1.txt

if [ $? -eq 0 ]; then
    exit 0  # Success: Invariant holds, so IRReduce can continue reducing
else
    exit 1  # Failure: Stop
fi
