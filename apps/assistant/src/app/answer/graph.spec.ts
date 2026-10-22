// Reading a walked sub graph the way the graph prompt does.

import { Subgraph } from '@ragfabric/sdk';

import { emptyGraphReason, relationLabel, walkedEdge, walkedEdges } from './graph';

const NODES = [
  { id: 1, name: 'Anjali Rao', entity_type: 'person', depth: 0 },
  { id: 2, name: 'Dev Malhotra', entity_type: 'person', depth: 1 },
];

function graph(partial: Partial<Subgraph>): Subgraph {
  return { nodes: NODES, edges: [], truncated: false, ...partial };
}

describe('walkedEdge', () => {
  it('test_a_forward_edge_reads_source_then_target', () => {
    const g = graph({
      edges: [{ id: 5, source_id: 1, target_id: 2, relation_type: 'MANAGES', walked_as: 'MANAGES', reversed: false, source_chunk_ids: [] }],
    });
    const e = walkedEdge(g, 0)!;
    expect([e.k, e.from, e.relation, e.to]).toEqual([1, 'Anjali Rao', 'MANAGES', 'Dev Malhotra']);
  });

  it('test_a_reversed_edge_reads_target_first_through_the_inverse_name', () => {
    const g = graph({
      edges: [{ id: 5, source_id: 1, target_id: 2, relation_type: 'MANAGES', walked_as: 'MANAGED_BY', reversed: true, source_chunk_ids: [] }],
    });
    const e = walkedEdge(g, 0)!;
    expect([e.from, e.relation, e.to]).toEqual(['Dev Malhotra', 'MANAGED_BY', 'Anjali Rao']);
  });

  it('test_an_unknown_node_is_named_by_its_id_and_a_missing_edge_is_null', () => {
    const g = graph({
      nodes: [],
      edges: [{ id: 5, source_id: 1, target_id: 2, relation_type: 'MANAGES', walked_as: 'MANAGES', reversed: false, source_chunk_ids: [] }],
    });
    expect(walkedEdge(g, 0)!.from).toBe('#1');
    expect(walkedEdge(g, 3)).toBeNull();
    expect(walkedEdges(null)).toEqual([]);
  });
});

describe('emptyGraphReason', () => {
  it('test_each_api_reason_has_plain_words', () => {
    expect(emptyGraphReason(graph({ empty_reason: 'no_graph_coverage' }))).toContain('No knowledge graph covers');
    expect(emptyGraphReason(graph({ empty_reason: 'no_entity_matched' }))).toContain('No entity in the question');
    expect(emptyGraphReason(graph({ empty_reason: 'no_walkable_edges' }))).toContain('no relationships to follow');
  });

  it('test_an_unknown_reason_is_still_said_plainly_and_no_reason_is_null', () => {
    expect(emptyGraphReason(graph({ empty_reason: 'something_new' }))).toContain('nothing to answer from');
    expect(emptyGraphReason(graph({}))).toBeNull();
    expect(emptyGraphReason(null)).toBeNull();
  });

  it('test_relation_names_read_as_words', () => {
    expect(relationLabel('REPORTS_TO')).toBe('reports to');
  });
});
