
"""POWER-OF-CHOICE (pow-d) selection strategy.

Paper: https://arxiv.org/pdf/2010.01243
"""


import math
import random
import time
from collections.abc import Callable, Iterable
from logging import INFO, WARNING

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid

from .fedavg import FedAvg


class PowerOfChoice(FedAvg):
    """POWER-OF-CHOICE (pow-d) client selection strategy.

    This strategy implements the three-step POWER-OF-CHOICE client selection scheme
    proposed by Cho et al., 2020:

    1. Sample a candidate set `A` of size `d` (without replacement).
    2. Estimate local losses `F_k(w)` for all clients in `A` at the current global model.
    3. Select the `m` clients in `A` with the highest loss to participate in training.

    Notes
    -----
    - The paper samples candidates in proportion to `p_k` (client data fraction). In a
      general Flower deployment, the server typically does not know `p_k` ahead of time.
      This implementation estimates `p_k` from the most recently observed value of
      `weighted_by_key` (default: `"num-examples"`) in evaluation replies. If no estimate
      is available yet, uniform sampling is used.
    - Local loss estimation is performed via an additional federated evaluation step
      on the candidate set in every round (message type: `MessageType.EVALUATE`).

    Parameters
    ----------
    candidate_set_size : int (default: 0)
        The candidate set size `d` in the paper. If set to 0, `d` defaults to `m`,
        which reduces POWER-OF-CHOICE to random sampling without replacement.
        Must satisfy `d >= m` once `m` is determined.
    loss_key : str (default: "loss")
        Metric key used to read the local loss value from candidate evaluation replies.
    candidate_eval_timeout : Optional[float] (default: 3600.0)
        Timeout (in seconds) used for querying candidate clients for their local loss.
        If `None`, the server will wait until all candidate replies are received.
    seed : Optional[int] (default: None)
        Seed for the internal RNG used for weighted candidate sampling and tie-breaking.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        *,
        candidate_set_size: int = 0,
        loss_key: str = "loss",
        candidate_eval_timeout: float | None = 3600.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        if candidate_set_size < 0:
            raise ValueError("`candidate_set_size` must be >= 0.")
        self.candidate_set_size = candidate_set_size
        self.loss_key = loss_key
        self.candidate_eval_timeout = candidate_eval_timeout
        self._rng = random.Random(seed)
        self._node_weight_estimates: dict[int, float] = {}

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> POWER-OF-CHOICE settings:")
        log(INFO, "\t│\t├── Candidate set size (d): %s", self.candidate_set_size)
        log(INFO, "\t│\t├── Loss key: '%s'", self.loss_key)
        log(INFO, "\t│\t└── Candidate eval timeout (s): %s", self.candidate_eval_timeout)
        super().summary()

    def _await_min_nodes(self, grid: Grid, min_nodes: int) -> list[int]:
        """Wait until at least `min_nodes` are connected and return all node IDs."""
        while len(all_nodes := list(grid.get_node_ids())) < min_nodes:
            log(
                INFO,
                "Waiting for nodes to connect: %d connected (minimum required: %d).",
                len(all_nodes),
                min_nodes,
            )
            time.sleep(1)
        return all_nodes

    def _weighted_sample_without_replacement(
        self, node_ids: list[int], sample_size: int
    ) -> list[int]:
        """Sample `sample_size` node IDs without replacement, weighted by estimates.

        Uses the Efraimidis-Spirakis method for weighted sampling without replacement.
        """
        if sample_size > len(node_ids):
            raise ValueError("`sample_size` must be <= number of available nodes.")
        if sample_size == 0:
            return []

        keys: list[tuple[float, int]] = []
        for node_id in node_ids:
            weight = self._node_weight_estimates.get(node_id, 1.0)
            weight = max(float(weight), 1e-12)
            u = self._rng.random()
            keys.append((-math.log(max(u, 1e-12)) / weight, node_id))

        keys.sort(key=lambda x: x[0])
        return [node_id for _, node_id in keys[:sample_size]]

    def _extract_single_metricrecord(self, record: RecordDict) -> MetricRecord | None:
        """Return the single MetricRecord in `record`, if present."""
        if len(record.metric_records) != 1:
            return None
        return next(iter(record.metric_records.values()))

    def _select_by_highest_loss(
        self,
        candidate_ids: list[int],
        candidate_losses: dict[int, float],
        sample_size: int,
    ) -> list[int]:
        """Select `sample_size` clients from candidates by descending loss."""
        candidates_with_loss = [nid for nid in candidate_ids if nid in candidate_losses]
        self._rng.shuffle(candidates_with_loss)  # random tie-breaking
        candidates_with_loss.sort(key=lambda nid: candidate_losses[nid], reverse=True)

        selected = candidates_with_loss[:sample_size]
        if len(selected) < sample_size:
            selected_set = set(selected)
            remaining = [nid for nid in candidate_ids if nid not in selected_set]
            self._rng.shuffle(remaining)
            selected.extend(remaining[: sample_size - len(selected)])
        return selected

    def _evaluate_candidate_losses(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
        candidate_ids: list[int],
        timeout: float | None,
    ) -> dict[int, float]:
        """Query candidate clients for their local loss at the current global model."""
        # Copy config to avoid unintended mutations across steps
        candidate_config = ConfigRecord(dict(config))
        candidate_config["server-round"] = server_round

        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: candidate_config}
        )
        messages = self._construct_messages(record, candidate_ids, MessageType.EVALUATE)
        replies = grid.send_and_receive(messages=messages, timeout=timeout)

        losses: dict[int, float] = {}
        for msg in replies:
            if msg.has_error():
                continue

            metricrecord = self._extract_single_metricrecord(msg.content)
            if metricrecord is None:
                continue

            # Update `p_k` estimate from latest `weighted_by_key`
            if (
                self.weighted_by_key in metricrecord
                and not isinstance(metricrecord[self.weighted_by_key], list)
            ):
                try:
                    self._node_weight_estimates[msg.metadata.src_node_id] = float(
                        metricrecord[self.weighted_by_key]
                    )
                except (TypeError, ValueError):
                    pass

            # Extract loss used for selection
            if (
                self.loss_key in metricrecord
                and not isinstance(metricrecord[self.loss_key], list)
            ):
                try:
                    losses[msg.metadata.src_node_id] = float(metricrecord[self.loss_key])
                except (TypeError, ValueError):
                    pass

        return losses

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []

        # current_total = len(list(grid.get_node_ids()))
        # m = max(int(current_total * self.fraction_train), self.min_train_nodes)
        # TODO check if we can avoid double counting nodes here. Should we just wait for min_available_nodes?
        all_nodes = self._await_min_nodes(grid, self.min_available_nodes)
        num_total = len(all_nodes)
        m = max(int(num_total * self.fraction_train), self.min_train_nodes)
        m = min(m, num_total)
        if m > num_total:
            m = num_total

        d = self.candidate_set_size or m
        if d < m:
            raise ValueError(
                "`candidate_set_size` must be >= the number of selected training "
                "nodes in a round."
            )
        if d > num_total:
            d = num_total

        candidate_ids = self._weighted_sample_without_replacement(all_nodes, d)
        log(
            INFO,
            "configure_train: Candidate set size %s, selecting %s clients (out of %s)",
            len(candidate_ids),
            m,
            num_total,
        )

        # When d == m, selecting the "top m losses" among m candidates always yields the
        # entire candidate set, so probing losses would add cost without changing the
        # selected set.
        if d == m:
            selected_node_ids = candidate_ids
        else:
            candidate_losses = self._evaluate_candidate_losses(
                server_round,
                arrays,
                config,
                grid,
                candidate_ids,
                self.candidate_eval_timeout,
            )
            selected_node_ids = self._select_by_highest_loss(
                candidate_ids, candidate_losses, m
            )
        if len(selected_node_ids) < m:
            log(
                WARNING,
                "POWER-OF-CHOICE selected %s clients, expected %s.",
                len(selected_node_ids),
                m,
            )

        # Always inject current server round
        train_config = ConfigRecord(dict(config))
        train_config["server-round"] = server_round

        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: train_config})
        return self._construct_messages(record, selected_node_ids, MessageType.TRAIN)
