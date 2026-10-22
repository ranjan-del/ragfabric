import { GraphEdge, Subgraph } from '@ragfabric/sdk';

/** Plain words for the reasons the API gives when the graph walked nothing. */
export const GRAPH_REASONS: Record<string, string> = {
  no_graph_coverage: 'No knowledge graph covers these documents yet.',
  no_entity_matched: 'No entity in the question matched the knowledge graph.',
  no_walkable_edges: 'The matched entities have no relationships to follow.',
};

/** The sentence for an empty walk, or null when the walk found edges. */
export function emptyGraphReason(graph: Subgraph | null | undefined): string | null {
  const reason = graph?.empty_reason;
  if (!reason) {
    return null;
  }
  return GRAPH_REASONS[reason] ?? 'The knowledge graph had nothing to answer from.';
}

export interface WalkedEdge {
  /** 1-based position in `Subgraph.edges`: the k of the answer's `[E k]` marker. */
  k: number;
  edge: GraphEdge;
  /** The entity the walk left from, and the one it arrived at. */
  from: string;
  to: string;
  /** The relation as walked (the inverse name when the walk ran against the stored direction). */
  relation: string;
}

/** Read one edge in its walked direction, the same reading the graph prompt gives the model. */
export function walkedEdge(graph: Subgraph, index: number): WalkedEdge | null {
  const edge = graph.edges[index];
  if (edge === undefined) {
    return null;
  }
  const name = (id: number) => graph.nodes.find((n) => n.id === id)?.name ?? `#${id}`;
  const [first, second] = edge.reversed ? [edge.target_id, edge.source_id] : [edge.source_id, edge.target_id];
  return { k: index + 1, edge, from: name(first), to: name(second), relation: edge.walked_as };
}

/** Every edge of the walk, in the order the answer's markers number them. */
export function walkedEdges(graph: Subgraph | null | undefined): WalkedEdge[] {
  if (!graph) {
    return [];
  }
  return graph.edges.map((_, i) => walkedEdge(graph, i)).filter((e): e is WalkedEdge => e !== null);
}

/** "REPORTS_TO" reads as "reports to". */
export function relationLabel(relation: string): string {
  return relation.toLowerCase().replace(/_/g, ' ');
}
