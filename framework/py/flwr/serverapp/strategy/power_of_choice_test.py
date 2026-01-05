
"""POWER-OF-CHOICE strategy tests."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from flwr.common import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict

from .power_of_choice import PowerOfChoice
from .strategy_utils_test import create_mock_reply


class _FakeGrid:  # minimal Grid stub used in tests
    def __init__(self, node_ids: list[int], losses: dict[int, float]) -> None:
        self._node_ids = node_ids
        self._losses = losses
        self.train_dst_node_ids: list[list[int]] = []
        self.evaluate_dst_node_ids: list[list[int]] = []

    def get_node_ids(self) -> Iterable[int]:
        return self._node_ids

    def send_and_receive(
        self, messages: Iterable[Message], *, timeout: float | None = None
    ) -> Iterable[Message]:
        del timeout
        msgs = list(messages)
        if not msgs:
            return []

        msg_type = msgs[0].metadata.message_type

        if msg_type == MessageType.EVALUATE:
            self.evaluate_dst_node_ids.append([m.metadata.dst_node_id for m in msgs])
            replies: list[Message] = []
            for m in msgs:
                node_id = m.metadata.dst_node_id
                metrics = MetricRecord(
                    {"num-examples": 1.0, "loss": float(self._losses[node_id])}
                )
                # Exactly one MetricRecord in the RecordDict, consistent with strategy assumptions
                replies.append(Message(content=RecordDict({"metrics": metrics}), reply_to=m))
            return replies

        if msg_type == MessageType.TRAIN:
            self.train_dst_node_ids.append([m.metadata.dst_node_id for m in msgs])
            replies: list[Message] = []
            for m in msgs:
                node_id = m.metadata.dst_node_id
                arrays = ArrayRecord([np.array([float(node_id)], dtype=np.float32)])
                replies.append(create_mock_reply(arrays, num_examples=1.0))
            return replies

        raise ValueError(f"Unexpected message type: {msg_type}")


def _extract_dst_ids(messages: Iterable[Message]) -> set[int]:
    return {m.metadata.dst_node_id for m in messages}


def test_power_of_choice_configure_train_selects_highest_loss_clients() -> None:
    """Select the highest-loss clients when all nodes are in the candidate set."""
    losses = {1: 0.1, 2: 0.9, 3: 0.2, 4: 0.8, 5: 0.3}
    grid = _FakeGrid([1, 2, 3, 4, 5], losses)

    # Total nodes=5, fraction_train=0.4 => m=2; candidate_set_size=5 => all nodes are candidates
    strategy = PowerOfChoice(
        fraction_train=0.4,
        fraction_evaluate=0.0,
        min_train_nodes=2,
        candidate_set_size=5,
        loss_key="loss",
        seed=123,
    )

    arrays = ArrayRecord([np.array([0.0], dtype=np.float32)])
    cfg = ConfigRecord({"lr": 0.1})

    train_msgs = list(strategy.configure_train(server_round=1, arrays=arrays, config=cfg, grid=grid))

    # Candidate probing should have occurred against all nodes
    assert len(grid.evaluate_dst_node_ids) == 1
    assert set(grid.evaluate_dst_node_ids[0]) == {1, 2, 3, 4, 5}

    # Top-2 losses are nodes 2 (0.9) and 4 (0.8)
    assert _extract_dst_ids(train_msgs) == {2, 4}

    # Ensure we configured TRAIN messages
    assert all(m.metadata.message_type == MessageType.TRAIN for m in train_msgs)


def test_power_of_choice_raises_if_candidate_set_smaller_than_sample_size() -> None:
    """candidate_set_size must be >= m once m is known."""
    losses = {1: 1.0, 2: 1.0, 3: 1.0}
    grid = _FakeGrid([1, 2, 3], losses)

    strategy = PowerOfChoice(
        fraction_train=1.0,   # would want to select m=3 (all nodes)
        fraction_evaluate=0.0,
        min_train_nodes=2,
        candidate_set_size=2,  # invalid: d < m
        seed=123,
    )

    arrays = ArrayRecord([np.array([0.0], dtype=np.float32)])
    cfg = ConfigRecord()

    with pytest.raises(ValueError):
        _ = list(strategy.configure_train(server_round=1, arrays=arrays, config=cfg, grid=grid))


def test_power_of_choice_skips_candidate_loss_probe_when_d_equals_m() -> None:
    """When d == m, probing losses must be skipped."""
    losses = {1: 0.1, 2: 0.9, 3: 0.2, 4: 0.8, 5: 0.3}
    grid = _FakeGrid([1, 2, 3, 4, 5], losses)

    # fraction_train=1.0 => m=5; candidate_set_size=0 => d=m
    strategy = PowerOfChoice(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=2,
        candidate_set_size=0,
        seed=123,
    )

    arrays = ArrayRecord([np.array([0.0], dtype=np.float32)])
    cfg = ConfigRecord()

    train_msgs = list(strategy.configure_train(server_round=1, arrays=arrays, config=cfg, grid=grid))

    # Loss probing must be skipped entirely
    assert grid.evaluate_dst_node_ids == []

    # All nodes should be selected to train
    assert _extract_dst_ids(train_msgs) == {1, 2, 3, 4, 5}

    assert all(m.metadata.message_type == MessageType.TRAIN for m in train_msgs)


def test_power_of_choice_end_to_end_start_still_works() -> None:
    """Sanity-check that Strategy.start orchestrates rounds using configure_train."""
    losses = {1: 0.1, 2: 0.9, 3: 0.2, 4: 0.8, 5: 0.3}
    grid = _FakeGrid([1, 2, 3, 4, 5], losses)

    strategy = PowerOfChoice(
        fraction_train=0.4,
        fraction_evaluate=0.0,
        min_train_nodes=2,
        candidate_set_size=5,
        loss_key="loss",
        seed=123,
    )

    initial = ArrayRecord([np.array([0.0], dtype=np.float32)])
    res = strategy.start(grid=grid, initial_arrays=initial, num_rounds=1, timeout=1.0)

    # configure_train should have led to a single TRAIN round to nodes {2,4}
    assert len(grid.train_dst_node_ids) == 1
    assert set(grid.train_dst_node_ids[0]) == {2, 4}

    # Aggregation sanity (same expectation as your original tests):
    # average of arrays [2] and [4] -> [3]
    assert res.arrays is not None
    np.testing.assert_allclose(
        res.arrays.to_numpy_ndarrays()[0],
        np.array([3.0], dtype=np.float32),
    )