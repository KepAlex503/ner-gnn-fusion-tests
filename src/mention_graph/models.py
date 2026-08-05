from __future__ import annotations

import torch
from torch import nn


class LocalClassifier(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        class_count: int,
        hidden_dimension: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dimension > 0:
            self.network = nn.Sequential(
                nn.Linear(input_dimension, hidden_dimension),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dimension, class_count),
            )
        else:
            self.network = nn.Linear(input_dimension, class_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def mean_neighbor_aggregation(
    hidden: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    aggregation = torch.zeros_like(hidden)
    if edge_index.numel() == 0:
        return aggregation
    sources, targets = edge_index
    aggregation.index_add_(0, targets, hidden[sources])
    degrees = torch.zeros(
        hidden.shape[0],
        dtype=hidden.dtype,
        device=hidden.device,
    )
    degrees.index_add_(
        0,
        targets,
        torch.ones(targets.shape[0], dtype=hidden.dtype, device=hidden.device),
    )
    return aggregation / degrees.clamp_min(1.0).unsqueeze(1)


def mean_neighbor_aggregation_with_presence(
    hidden: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    aggregation = torch.zeros_like(hidden)
    degrees = torch.zeros(
        hidden.shape[0],
        dtype=hidden.dtype,
        device=hidden.device,
    )
    if edge_index.numel() == 0:
        return aggregation, degrees
    sources, targets = edge_index
    aggregation.index_add_(0, targets, hidden[sources])
    degrees.index_add_(
        0,
        targets,
        torch.ones(targets.shape[0], dtype=hidden.dtype, device=hidden.device),
    )
    aggregation = aggregation / degrees.clamp_min(1.0).unsqueeze(1)
    return aggregation, (degrees > 0).to(hidden.dtype)


class GraphSAGELayer(nn.Module):
    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.neighbor_projection = nn.Linear(
            hidden_dimension,
            hidden_dimension,
            bias=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        neighbors = mean_neighbor_aggregation(hidden, edge_index)
        return self.self_projection(hidden) + self.neighbor_projection(neighbors)


class  GraphSAGEClassifier(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        class_count: int,
        dropout: float,
        layer_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dimension, hidden_dimension)
        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dimension), GraphSAGELayer(hidden_dimension)]
        )
        self.normalizations = nn.ModuleList(
            [
                nn.LayerNorm(hidden_dimension)
                if layer_normalization
                else nn.Identity()
                for _ in self.layers
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dimension, class_count)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.relu(self.input_projection(features))
        hidden = self.dropout(hidden)
        for layer, normalization in zip(
            self.layers,
            self.normalizations,
            strict=True,
        ):
            hidden = layer(hidden, edge_index)
            hidden = normalization(torch.relu(hidden))
            hidden = self.dropout(hidden)
        return self.output_projection(hidden)


class RelationalGatedLayer(nn.Module):
    def __init__(
        self,
        hidden_dimension: int,
        relation_names: list[str],
        *,
        layer_normalization: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.relation_names = tuple(relation_names)
        self.self_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.neighbor_projections = nn.ModuleDict(
            {
                relation: nn.Linear(
                    hidden_dimension,
                    hidden_dimension,
                    bias=False,
                )
                for relation in self.relation_names
            }
        )
        self.gates = nn.ModuleDict(
            {
                relation: nn.Linear(2 * hidden_dimension, 1)
                for relation in self.relation_names
            }
        )
        for gate in self.gates.values():
            nn.init.constant_(gate.bias, -0.5)
        self.normalization = (
            nn.LayerNorm(hidden_dimension)
            if layer_normalization
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        edge_indices: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        update = self.self_projection(hidden)
        relation_sum = torch.zeros_like(hidden)
        active_relations = torch.zeros(
            (hidden.shape[0], 1),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        for relation in self.relation_names:
            neighbors, present = mean_neighbor_aggregation_with_presence(
                hidden,
                edge_indices[relation],
            )
            gate = torch.sigmoid(
                self.gates[relation](torch.cat([hidden, neighbors], dim=1))
            )
            relation_sum = relation_sum + (
                gate
                * self.neighbor_projections[relation](neighbors)
                * present.unsqueeze(1)
            )
            active_relations = active_relations + present.unsqueeze(1)
        update = update + relation_sum / active_relations.clamp_min(1.0)
        return self.normalization(
            hidden + self.dropout(torch.relu(update))
        )


class RelationalGatedClassifier(nn.Module):
    """Relation-aware residual correction of fixed local probabilities."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        class_count: int,
        relation_names: list[str],
        *,
        layers: int,
        dropout: float,
        layer_normalization: bool,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("a relational classifier needs at least one layer")
        self.relation_names = tuple(relation_names)
        self.input_projection = nn.Linear(input_dimension, hidden_dimension)
        self.layers = nn.ModuleList(
            [
                RelationalGatedLayer(
                    hidden_dimension,
                    relation_names,
                    layer_normalization=layer_normalization,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.input_dropout = nn.Dropout(dropout)
        self.correction_projection = nn.Linear(hidden_dimension, class_count)
        self.correction_gate = nn.Linear(hidden_dimension, 1)
        nn.init.constant_(self.correction_gate.bias, -1.0)
        self.local_logit_scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        features: torch.Tensor,
        edge_indices: dict[str, torch.Tensor],
        local_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_dropout(
            torch.relu(self.input_projection(features))
        )
        for layer in self.layers:
            hidden = layer(hidden, edge_indices)
        correction = self.correction_projection(hidden)
        correction_gate = torch.sigmoid(self.correction_gate(hidden))
        local_logits = torch.log(local_probabilities.clamp_min(1e-7))
        return (
            self.local_logit_scale * local_logits
            + correction_gate * correction
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
