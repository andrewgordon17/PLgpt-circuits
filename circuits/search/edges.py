from collections import defaultdict
from dataclasses import dataclass

import torch

from circuits import Circuit, Edge, EdgeGroup, Node
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator
from circuits.search.divergence import (
    compute_downstream_magnitudes,
    patch_feature_magnitudes,
)
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


@dataclass(frozen=True)
class EdgeSearchResult:
    """
    Result of an edge search.
    """

    # Maps an edge to a normalized MSE increase
    edge_importance: dict[Edge, float]

    # Maps an edge group to a normalized MSE increase
    token_importance: dict[EdgeGroup, float]


class EdgeSearch:
    """
    Analyze edge importance in a circuit by ablating each edge between two adjacent layers.
    """

    def __init__(
        self,
        model: SparsifiedGPT,
        model_profile: ModelProfile,
        upstream_ablator: ResampleAblator,
        num_samples: int,
    ):
        """
        :param model: The sparsified model to use for circuit analysis.
        :param model_profile: The model profile containing cache feature metrics.
        :param upstream_ablator: Ablator to use for patching upstream feature magnitudes.
        :param num_samples: The number of samples to use for ablation.
        """
        self.model = model
        self.model_profile = model_profile
        self.upstream_ablator = upstream_ablator
        self.num_samples = num_samples
        print(f"Using {num_samples} samples for edge analysis.")

    def search(
        self,
        tokens: list[int],
        upstream_nodes: frozenset[Node],
        downstream_nodes: frozenset[Node],
        target_token_idx: int,
    ) -> EdgeSearchResult:
        """
        Map each edge in a sparsified model to a normalized MSE increase that results from its ablation.
        """
        assert len(downstream_nodes) > 0
        downstream_idx = next(iter(downstream_nodes)).layer_idx
        upstream_idx = downstream_idx - 1
        print(f"\nAnalyzing edge importance between layers {upstream_idx} and {downstream_idx}...")

        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)

        # Get feature magnitudes
        with torch.no_grad():
            model_output: SparsifiedGPTOutput = self.model(input)

        upstream_magnitudes = model_output.feature_magnitudes[upstream_idx].squeeze(0)  # Shape: (T, F)

        # Find all edges that could exist between upstream and downstream nodes
        all_edges = set()
        for upstream in sorted(upstream_nodes):
            for downstream in sorted(downstream_nodes):
                if upstream.token_idx <= downstream.token_idx:
                    all_edges.add(Edge(upstream, downstream))
        all_edges = frozenset(all_edges)

        # Get average downstream feature magnitudes
        # NOTE: We get better results from averaging sampled magnitudes than we do from using the original
        # downstream feature magnitudes from the model output.
        downstream_means = self.estimate_downstream_node_means(
            downstream_nodes,
            all_edges,
            upstream_magnitudes,
            target_token_idx,
        )

        # Set baseline MSE to use for comparisons
        baseline_mses = self.estimate_downstream_node_mses(
            downstream_nodes,
            all_edges,
            upstream_magnitudes,
            downstream_means,
            target_token_idx,
        )

        # Compute edge importance
        edge_importance = self.compute_edge_importance(
            all_edges,
            downstream_nodes,
            baseline_mses,
            upstream_magnitudes,
            downstream_means,
            target_token_idx,
        )

        # Compute token importance
        token_importance = self.compute_token_importance(
            all_edges,
            downstream_nodes,
            baseline_mses,
            upstream_magnitudes,
            downstream_means,
            target_token_idx,
        )

        return EdgeSearchResult(edge_importance=edge_importance, token_importance=token_importance)

    def get_placeholders(
        self,
        upstream_nodes: frozenset[Node],
        downstream_nodes: frozenset[Node],
    ) -> EdgeSearchResult:
        """
        Get a placeholder result for edge search.
        """
        # Find all edges that could exist between upstream and downstream nodes
        edges = set()
        for upstream in sorted(upstream_nodes):
            for downstream in sorted(downstream_nodes):
                if upstream.token_idx <= downstream.token_idx:
                    edges.add(Edge(upstream, downstream))

        # Create all edge groups
        edge_groups: set[EdgeGroup] = set()
        upstream_blocks = {(node.layer_idx, node.token_idx) for node in upstream_nodes}
        for upstream_layer_idx, upstream_token_idx in upstream_blocks:
            downstream_token_idxs = {n.token_idx for n in downstream_nodes if n.token_idx >= upstream_token_idx}
            for downstream_token_idx in downstream_token_idxs:
                edge_groups.add(EdgeGroup(upstream_layer_idx, upstream_token_idx, downstream_token_idx))

        return EdgeSearchResult(
            edge_importance={edge: 1.0 for edge in edges},
            token_importance={edge_group: 0.0 for edge_group in edge_groups},
        )

    def compute_edge_importance(
        self,
        all_edges: frozenset[Edge],
        downstream_nodes: frozenset[Node],
        baseline_mses: dict[Node, float],
        upstream_magnitudes: torch.Tensor,
        downstream_means: dict[Node, float],
        target_token_idx: int,
    ) -> dict[Edge, float]:
        """
        Compute the importance of edges between upstream and downstream nodes.

        :param all_edges: Set of all possible edges between layers
        :param downstream_nodes: Set of downstream nodes
        :param baseline_mses: Dictionary mapping downstream nodes to their baseline mean-squared errors
        :param upstream_magnitudes: The upstream feature magnitudes (shape: T, F)
        :param downstream_means: The downstream feature magnitudes to use for calculating the mean-squared error
        :param target_token_idx: The target token index

        :return: Dictionary mapping edges to their importance scores
        """
        # Map edges to ablation effects
        ablation_mses = self.estimate_edge_ablation_effects(
            downstream_nodes,
            all_edges,
            upstream_magnitudes,
            downstream_means,
            target_token_idx,
        )

        # Calculate MSE increase from baseline
        edge_mse_increase = {}
        for edge, mse in sorted(ablation_mses.items(), key=lambda x: x[0]):
            baseline_mse = baseline_mses[edge.downstream]
            edge_mse_increase[edge] = mse - baseline_mse

        # Calculate MSE increase stats per downstream node
        min_mse_increases: dict[Node, float] = {}
        max_mse_increases: dict[Node, float] = {}
        for downstream_node in downstream_nodes:
            upstream_edges = {edge for edge in edge_mse_increase.keys() if edge.downstream == downstream_node}
            mse_increases = [edge_mse_increase[edge] for edge in upstream_edges]
            min_mse_increase = min(mse_increases, default=0)
            min_mse_increases[downstream_node] = min_mse_increase
            max_mse_increase = max(mse_increases, default=0)
            max_mse_increases[downstream_node] = max_mse_increase

        # Print MSE increase stats
        for downstream_node in sorted(downstream_nodes):
            print(
                f"Upstream from {downstream_node} - "
                f"Baseline: {baseline_mses[downstream_node]:.4f} - "
                f"Min MSE increase: {min_mse_increases[downstream_node]:.4f} - "
                f"Max MSE increase: {max_mse_increases[downstream_node]:.4f}"
            )

        # Normalize MSE increase by max MSE increase
        edge_importance = {}
        for edge, mse_increase in edge_mse_increase.items():
            mse_increase = max(mse_increase, 0)  # Avoid negative values
            max_mse_increase = max(max_mse_increases[edge.downstream], 1e-6)  # Avoid negative values
            edge_importance[edge] = mse_increase / max_mse_increase

        return edge_importance

    def compute_token_importance(
        self,
        all_edges: frozenset[Edge],
        downstream_nodes: frozenset[Node],
        baseline_mses: dict[Node, float],
        upstream_magnitudes: torch.Tensor,
        downstream_means: dict[Node, float],
        target_token_idx: int,
    ) -> dict[EdgeGroup, float]:
        """
        Compute the importance of upstream tokens for downstream tokens.

        :param all_edges: Set of all possible edges between layers
        :param downstream_nodes: Set of downstream nodes
        :param baseline_mses: Dictionary mapping downstream nodes to their baseline mean-squared errors
        :param upstream_magnitudes: The upstream feature magnitudes (shape: T, F)
        :param downstream_means: The downstream feature magnitudes to use for calculating the mean-squared error
        :param target_token_idx: The target token index

        :return: Dictionary mapping downstream token indicies to upstream token indices and their importance scores
        """
        # Downstream token indices
        upstream_layer_idx = next(iter(downstream_nodes)).layer_idx - 1
        downstream_token_idxs = tuple(sorted({node.token_idx for node in downstream_nodes}))

        # Set baseline MSE to use for comparisons
        token_baseline_mses: dict[int, float] = {}
        for downstream_token_idx in downstream_token_idxs:
            baseline_node_mses = [mse for node, mse in baseline_mses.items() if node.token_idx == downstream_token_idx]
            token_baseline_mses[downstream_token_idx] = sum(baseline_node_mses) / len(baseline_node_mses)

        # For each downstream token index, map upstream token indices to an MSE
        token_ablation_mses = self.estimate_token_ablation_effects(
            downstream_nodes,
            all_edges,
            upstream_magnitudes,
            downstream_means,
            target_token_idx,
        )

        # Calculate token MSE increase from baseline
        token_mse_increases = defaultdict(dict)
        for downstream_token_idx, upstream_token_mses in sorted(token_ablation_mses.items(), key=lambda x: x[0]):
            for upstream_token_idx, token_mse in sorted(upstream_token_mses.items(), key=lambda x: x[0]):
                baseline_mse = token_baseline_mses[downstream_token_idx]
                token_mse_increases[downstream_token_idx][upstream_token_idx] = token_mse - baseline_mse

        # Calculate MSE increase stats per downstream node
        min_mse_increases: dict[int, float] = {}
        max_mse_increases: dict[int, float] = {}
        for downstream_token_idx in downstream_token_idxs:
            min_mse_increase = min(token_mse_increases[downstream_token_idx].values(), default=0)
            min_mse_increases[downstream_token_idx] = min_mse_increase
            max_mse_increase = max(token_mse_increases[downstream_token_idx].values(), default=0)
            max_mse_increases[downstream_token_idx] = max_mse_increase

        # Print MSE increase stats
        for downstream_token_idx in downstream_token_idxs:
            print(
                f"Upstream from token {downstream_token_idx} - "
                f"Baseline: {token_baseline_mses[downstream_token_idx]:.4f} - "
                f"Min MSE increase: {min_mse_increases[downstream_token_idx]:.4f} - "
                f"Max MSE increase: {max_mse_increases[downstream_token_idx]:.4f}"
            )

        # Normalize MSE increase by max MSE increase
        token_importance = {}
        for downstream_token_idx, token_mses in token_mse_increases.items():
            for upstream_token_idx, mse_increase in token_mses.items():
                edge_group = EdgeGroup(upstream_layer_idx, upstream_token_idx, downstream_token_idx)
                mse_increase = max(mse_increase, 0)  # Avoid negative values
                max_mse_increase = max(max_mse_increases[downstream_token_idx], 1e-6)  # Avoid negative value
                token_importance[edge_group] = mse_increase / max_mse_increase

        return token_importance

    def estimate_token_ablation_effects(
        self,
        downstream_nodes: frozenset[Node],
        all_edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        downstream_means: dict[Node, float],  # Shape: (T, F)
        target_token_idx: int,
    ) -> dict[int, dict[int, float]]:
        """
        Estimate the downstream feature mean-squared error that results from ablating each token in a circuit.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param downstream_means: The downstream feature magnitudes to use for calculating the mean-squared error.
        :param target_token_idx: The target token index.
        """
        token_ablation_mses = defaultdict(lambda: defaultdict(list))
        for token_idx in sorted({edge.upstream.token_idx for edge in all_edges}):
            # Exclude edges that are connected to the target token
            patched_edges = frozenset({edge for edge in all_edges if edge.upstream.token_idx != token_idx})
            estimated_mses = self.estimate_downstream_node_mses(
                downstream_nodes,
                patched_edges,
                upstream_magnitudes,
                downstream_means,
                target_token_idx,
            )
            # Look for downstream nodes that have an edge to the target token
            for downstream_node in {edge.downstream for edge in all_edges if edge.upstream.token_idx == token_idx}:
                # Set the mean-squared error from the downstream node
                mse = estimated_mses[downstream_node]
                token_ablation_mses[downstream_node.token_idx][token_idx].append(mse)

        # Average the MSEs for each downstream node grouped by token index
        average_token_ablation_mses = defaultdict(dict)
        for downstream_token_idx, upstream_token_mses in token_ablation_mses.items():
            for upstream_token_idx, mses in upstream_token_mses.items():
                average_token_ablation_mses[downstream_token_idx][upstream_token_idx] = sum(mses) / len(mses)

        return average_token_ablation_mses

    def estimate_edge_ablation_effects(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        downstream_means: dict[Node, float],  # Shape: (T, F)
        target_token_idx: int,
    ) -> dict[Edge, float]:
        """
        Estimate the downstream feature mean-squared error that results from ablating each edge in a circuit.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param downstream_means: The downstream feature magnitudes to use for calculating the mean-squared error.
        :param target_token_idx: The target token index.
        """
        # Maps edge to downstream mean-squared error
        edge_to_mse: dict[Edge, float] = {}

        # Create a set of circuit variants with one edge removed
        edge_to_circuit_variant: dict[Edge, Circuit] = {}
        for edge in edges:
            circuit_variant = Circuit(downstream_nodes, edges=frozenset(edges - {edge}))
            edge_to_circuit_variant[edge] = circuit_variant

        # Compute downstream feature magnitude errors that results from ablating each edge
        for edge, circuit_variant in edge_to_circuit_variant.items():
            downstream_errors = self.estimate_downstream_node_mses(
                downstream_nodes,
                circuit_variant.edges,
                upstream_magnitudes,
                downstream_means,
                target_token_idx,
            )
            edge_to_mse[edge] = downstream_errors[edge.downstream]
        return edge_to_mse

    def estimate_downstream_node_mses(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        downstream_means: dict[Node, float],
        target_token_idx: int,
    ) -> dict[Node, float]:
        """
        Use downstream feature magnitudes derived from upstream feature magnitudes and edges to produce a mean-squared
        error per downstream node.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param downstream_means: The downstream feature magnitudes to use for calculating the mean-squared error.
        :param target_token_idx: The target token index.

        :return: The mean-squared error per downstream node.
        """
        # Get feature magnitude samples
        sampled_feature_magnitudes = self.sample_downstream_feature_magnitudes(
            downstream_nodes,
            edges,
            upstream_magnitudes,
            target_token_idx,
            num_samples=self.num_samples,
        )  # Shape: (num_samples, T, F)

        # Caculate normalization coefficients for downstream features, which scale magnitudes to [0, 1]
        norm_coefficients = torch.ones(len(downstream_nodes))
        downstream_layer_idx = next(iter(downstream_nodes)).layer_idx
        layer_profile = self.model_profile[downstream_layer_idx]
        for i, node in enumerate(sampled_feature_magnitudes.keys()):
            feature_profile = layer_profile[int(node.feature_idx)]
            norm_coefficients[i] = 1.0 / feature_profile.max

        # Calculate mean-squared error from original downstream feature magnitudes
        downstream_mses = {}
        for node, magnitudes in sampled_feature_magnitudes.items():
            original_magnitude = downstream_means[node]
            normalized_mse = torch.mean((norm_coefficients[i] * (magnitudes - original_magnitude)) ** 2)
            downstream_mses[node] = normalized_mse.item()
        return downstream_mses

    def estimate_downstream_node_means(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
    ) -> dict[Node, float]:
        """
        Use downstream feature magnitudes derived from upstream feature magnitudes and edges to produce a mean per
        downstream node.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param target_token_idx: The target token index.

        :return: The mean per downstream node.
        """
        sampled_feature_magnitudes = self.sample_downstream_feature_magnitudes(
            downstream_nodes,
            edges,
            upstream_magnitudes,
            target_token_idx,
            # Use a larger number of samples to get a more accurate mean
            num_samples=min(self.num_samples * 4, self.upstream_ablator.k_nearest or int(1e10)),
        )
        # Calculate mean from sampled feature magnitudes
        downstream_means = {}
        for node, magnitudes in sampled_feature_magnitudes.items():
            downstream_means[node] = torch.mean(magnitudes).item()
        return downstream_means

    def sample_downstream_feature_magnitudes(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
        num_samples: int,
    ) -> dict[Node, torch.Tensor]:
        """
        Sample downstream feature magnitudes derived from upstream feature magnitudes and edges.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param downstream_means: The downstream feature magnitudes to use for calculating the mean-squared error.
        :param target_token_idx: The target token index.
        :param num_samples: The number of samples to produce per feature.
        """
        # Map downstream nodes to upstream dependencies
        node_to_dependencies: dict[Node, frozenset[Node]] = {}
        for node in downstream_nodes:
            node_to_dependencies[node] = frozenset([edge.upstream for edge in edges if edge.downstream == node])
        dependencies_to_nodes: dict[frozenset[Node], set[Node]] = defaultdict(set)
        for node, dependencies in node_to_dependencies.items():
            dependencies_to_nodes[dependencies].add(node)

        # Patch upstream feature magnitudes for each set of dependencies
        circuit_variants = [Circuit(nodes=dependencies) for dependencies in dependencies_to_nodes.keys()]
        upstream_layer_idx = next(iter(downstream_nodes)).layer_idx - 1
        patched_upstream_magnitudes = patch_feature_magnitudes(  # Shape: (num_samples, T, F)
            self.upstream_ablator,
            upstream_layer_idx,
            target_token_idx,
            circuit_variants,
            upstream_magnitudes.cpu().numpy(),
            num_samples=num_samples,
        )

        # Compute downstream feature magnitudes for each set of dependencies
        sampled_downstream_magnitudes = {}
        for circuit_variant, patched_magnitudes in patched_upstream_magnitudes.items():
            downstream_magnitudes = compute_downstream_magnitudes(  # Shape: (num_samples, T, F)
                self.model,
                upstream_layer_idx,
                torch.tensor(patched_magnitudes, device=upstream_magnitudes.device),
            )
            sampled_downstream_magnitudes[circuit_variant] = downstream_magnitudes

        # Map each downstream node to a set of sampled feature magnitudes
        sampled_magnitudes: dict[Node, torch.Tensor] = {}
        for circuit_variant, magnitudes in sampled_downstream_magnitudes.items():
            for node in dependencies_to_nodes[circuit_variant.nodes]:
                sampled_magnitudes[node] = magnitudes[:, node.token_idx, node.feature_idx]

        return sampled_magnitudes

    @property
    def num_layers(self) -> int:
        """
        Get the number of SAE layers in the model.
        """
        return self.model.gpt.config.n_layer + 1  # Add 1 for the embedding layer
