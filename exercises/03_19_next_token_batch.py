# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.19: construct a batch of shifted next-token training windows.

Run after the lesson walkthrough, in the environment described in README.md:
    python exercises/03_19_next_token_batch.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Extend 01.01's single input/target window to B tensor rows. Review ordered
token IDs, Python slices with an excluded end, tensors, loops, and lists.
Each target is the NEXT token after the corresponding input position.
This is deterministic data construction. Start positions are supplied;
there is no random sampling, forward pass, loss, or parameter update here.

EXACT INPUTS AND VALID WINDOW CONTRACT
token_ids: one-dimensional torch.long tensor [N], an already encoded stream.
start_positions: nonempty list[int], one chosen start index per batch row.
context_length: positive Python int T, tokens in EACH input/target row.
B=len(start_positions). Every start satisfies 0 <= start and start+T < N,
so T+1 consecutive source tokens are available. Assume valid inputs; input
validation is not assigned. Preserve start order, including repeats and
overlapping windows. IDs can include learned vocabulary IDs above 255.

RETURN AND STATE
Return (input_ids, target_ids), two fresh torch.long tensors [B,T], on the
source device. Preserve token_ids, start_positions, caller gradient mode,
and the global random stream. Returned tensors must not share storage with
the source or each other. Subsequent calls return independent tensors too.
No padding, empty batches, shuffled starts, document-boundary policy, or
special-token changes are implemented by this exercise.

FORMULAS (zero-based indices)
    input_ids[b,t]  = token_ids[start_positions[b] + t]
    target_ids[b,t] = token_ids[start_positions[b] + t + 1]
    b=0,...,B-1; t=0,...,T-1
Both rows have T items; the source span needed for one pair has T+1 items.
T counts input positions. B counts samples. Neither is vocabulary size.

COMPLETE WORKED EXAMPLE: toy-ascii-stream-v1 (not a training result)
    token_ids = tensor([97,98,99,100,101,102,103])  # abcdefg
    start_positions = [0,3]; context_length = 3
    start 0: source span [97,98,99,100]
        x_row = [97,98,99]; y_row = [98,99,100]
    start 3: source span [100,101,102,103]
        x_row = [100,101,102]; y_row = [101,102,103]
    input_ids  = [[97,98,99],[100,101,102]]  # abc / def
    target_ids = [[98,99,100],[101,102,103]] # bcd / efg
    Return two tensors with shape [2,3]. There are six supervised positions.
At row 0, position 1, input ID 98 ('b') predicts target ID 99 ('c').
The final input 'c' predicts 'd', even though 'd' is outside that input row.

SYNTAX BEFORE IMPLEMENTATION
token_ids[start:end] selects start through end-1. A tensor slice is a view,
so do not modify x_row or y_row in place. [] creates an empty Python list;
append(row) adds one tensor to it. for start in start_positions: visits
the supplied starts in order; indent its body eight spaces.
torch.stack(rows,dim=0) adds a new batch dimension: B tensors [T] -> [B,T].
torch.cat(rows,dim=0) would concatenate them into [B*T], losing row shape.
Stack after the loop, then return the two tensors in input/target order.
The list is nonempty and all rows are equal length under the valid contract.

TRAINING INTERPRETATION
For a future causal contextual model, position t uses only its row's prefix
0..t to predict target_ids[b,t]. Targets stay outside the forward input.
The full training row stores future input positions for their own losses;
a causal model must block their influence on earlier predictions. This
window builder does not implement that mask or add attention to the earlier
lookup-plus-head baseline. During free generation, unknown future IDs still
require sequential prediction. The supplied final check is only a small
fixed-parameter loss/backward bridge using completed helpers, not training.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def make_next_token_batch(
    token_ids: torch.Tensor,
    start_positions: list[int],
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fresh [B,T] input IDs and one-position-shifted target IDs."""
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, create fresh lists input_rows and target_rows.
    # Each will collect B tensors, each with shape [context_length].
    #
    # STEP 2: At four spaces, loop over start_positions with for start in ...:
    # The example visits 0 then 3. Steps 3-4 use eight spaces inside the loop.
    #
    # STEP 3: At eight spaces, take x_row from token_ids[start:start+context_length].
    # Then take y_row starting at start+1 and ending at start+context_length+1.
    # Slice ends are excluded. First x_row=[97,98,99], y_row=[98,99,100].
    # Both must have T items; do not modify the slice views in place.
    #
    # STEP 4: At eight spaces, append x_row to input_rows and y_row to
    # target_rows, preserving their supplied order. After two iterations,
    # each list contains two [3] tensors. append adds one row at a time.
    #
    # STEP 5: Back at four spaces AFTER the loop, use torch.stack(input_rows,
    # dim=0) to obtain input_ids. Shape [2,3], not a flat six-ID sequence.
    # Stack allocates fresh output storage and preserves dtype/device.
    #
    # STEP 6: At four spaces, stack target_rows on dim=0 as target_ids.
    # Example [[98,99,100],[101,102,103]], shape [2,3].
    #
    # STEP 7: At four spaces after both stack operations, return input_ids,
    # target_ids in that order. Do not return inside the for loop.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement make_next_token_batch for this exercise")
    # ===== END LEARNER CODE =====

# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _load_module(filename, module_name):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


make_example = _load_module("01_01_text_to_tokens.py", "completed_batch_single_window").make_example
compute_logits = _load_module("03_09_logits_and_softmax.py", "completed_batch_projection").compute_logits
cross_entropy_loss = _load_module("03_10_cross_entropy_loss.py", "completed_batch_loss").cross_entropy_loss


def _check(token_ids, starts, length, expected=None):
    original_ids, original_starts = token_ids.clone(), starts.copy()
    global_state, mode = torch.get_rng_state().clone(), torch.is_grad_enabled()
    result = make_next_token_batch(token_ids, starts, length)
    assert isinstance(result, tuple) and len(result) == 2
    input_ids, target_ids = result
    for tensor in result:
        assert isinstance(tensor, torch.Tensor) and tensor.shape == (len(starts), length)
        assert tensor.dtype == token_ids.dtype and tensor.device == token_ids.device
        assert tensor.data_ptr() != token_ids.data_ptr() and tensor.grad_fn is None
    assert input_ids.data_ptr() != target_ids.data_ptr()
    assert torch.equal(token_ids, original_ids) and starts == original_starts
    assert torch.equal(torch.get_rng_state(), global_state) and torch.is_grad_enabled() == mode
    if expected is None:
        pairs = [make_example(original_ids.tolist(), start, length) for start in starts]
        expected = ([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    assert input_ids.tolist() == expected[0], (input_ids.tolist(), expected[0])
    assert target_ids.tolist() == expected[1], (target_ids.tolist(), expected[1])
    return result


def run_checks():
    stream = torch.tensor([97, 98, 99, 100, 101, 102, 103])
    _check(stream, [0, 3], 3, ([[97, 98, 99], [100, 101, 102]],
                            [[98, 99, 100], [101, 102, 103]]))
    print("PASS: complete worked batch, shifted final targets, and two [2,3] tensors")

    _check(torch.tensor([7, 8]), [0], 1, ([[7]], [[8]]))
    _check(stream, [0], 6)
    _check(stream, [3], 3)
    _check(stream, [0, 5], 1)
    print("PASS: one sample, one position, maximal span, and the last valid start")

    _check(stream, [2, 0, 2, 1], 3)
    _check(torch.tensor([256, 900, 257, 900, 258, 901]), [1, 0, 2], 3)
    print("PASS: supplied order, repeated/overlapping starts, and vocabulary IDs above 255")

    strided = torch.tensor([97, 999, 98, 999, 99, 999, 100, 999, 101, 999])[::2]
    _check(strided, [0, 1], 3)
    with torch.no_grad():
        _check(stream, [0, 3], 3)
    meta_source = torch.empty(7, dtype=torch.long, device="meta")
    meta_input, meta_target = make_next_token_batch(meta_source, [0, 3], 3)
    assert meta_input.shape == meta_target.shape == (2, 3)
    assert meta_input.dtype == meta_target.dtype == torch.long
    assert meta_input.device == meta_target.device == meta_source.device
    print("PASS: strided source, dtype/device propagation, caller mode, and unchanged random stream")

    original = stream.clone()
    first_x, first_y = _check(stream, [0], 3)
    second_x, second_y = _check(stream, [0], 3)
    first_x[0, 1] = 500
    assert first_y.tolist() == [[98, 99, 100]]
    first_y[0, 0] = 501
    assert torch.equal(stream, original)
    assert second_x.tolist() == [[97, 98, 99]] and second_y.tolist() == [[98, 99, 100]]
    print("PASS: fresh outputs independent of the source, each other, and later calls")

    varied = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8])
    for length in (1, 2, 5):
        _check(varied, [varied.numel() - length - 1, 0, 1, 0], length)
    print("PASS: varied batch/window sizes agree with completed single-window references")

    ids = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    batch_x, batch_y = _check(ids, [0, 3], 3)
    table = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    hidden = table[batch_x]
    assert hidden.shape == (2, 3, 2)
    scores = compute_logits(hidden.reshape(-1, 2), weights, bias)
    loss = cross_entropy_loss(scores, batch_y.reshape(-1))
    reference = torch.nn.functional.cross_entropy(scores, batch_y.reshape(-1))
    assert scores.shape == (6, 3) and torch.allclose(loss, reference)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (table, weights, bias))
    print("PASS: six aligned supervised positions feed completed projection/loss with connected gradients")
    print("Lesson 03.19 checks passed.")


if __name__ == "__main__":
    run_checks()
