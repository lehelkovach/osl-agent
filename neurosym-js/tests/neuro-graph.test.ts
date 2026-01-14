/**
 * Tests for NeuroGraph module
 */

import { NeuroGraph, createGraph, parseNeuroJSON } from '../src/neuro-graph';
import { NeuroJSON } from '../src/types';

describe('NeuroGraph', () => {
  const sampleDoc: NeuroJSON = {
    version: '1.0',
    name: 'test-graph',
    variables: {
      A: { type: 'bool', prior: 0.5 },
      B: { type: 'bool', prior: 0.3 },
      C: { type: 'bool', prior: 0.1 }
    },
    rules: [
      {
        id: 'rule1',
        type: 'IMPLICATION',
        inputs: ['A'],
        output: 'B',
        op: 'IDENTITY',
        weight: 0.9
      },
      {
        id: 'rule2',
        type: 'CONJUNCTION',
        inputs: ['A', 'B'],
        output: 'C',
        op: 'AND',
        weight: 0.8
      }
    ],
    constraints: [
      {
        id: 'attack1',
        type: 'ATTACK',
        source: 'C',
        target: 'A',
        weight: 0.7
      }
    ]
  };

  describe('Constructor and Loading', () => {
    it('should create an empty graph', () => {
      const graph = new NeuroGraph();
      expect(graph.variableCount).toBe(0);
      expect(graph.ruleCount).toBe(0);
      expect(graph.constraintCount).toBe(0);
    });

    it('should load a NeuroJSON document', () => {
      const graph = new NeuroGraph(sampleDoc);
      expect(graph.variableCount).toBe(3);
      expect(graph.ruleCount).toBe(2);
      expect(graph.constraintCount).toBe(1);
    });

    it('should validate on load and return errors for invalid docs', () => {
      const graph = new NeuroGraph();
      const result = graph.load({} as any);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });
  });

  describe('Variable Management', () => {
    let graph: NeuroGraph;

    beforeEach(() => {
      graph = new NeuroGraph(sampleDoc);
    });

    it('should get variable definitions', () => {
      const varA = graph.getVariable('A');
      expect(varA).toBeDefined();
      expect(varA?.type).toBe('bool');
      expect(varA?.prior).toBe(0.5);
    });

    it('should return undefined for non-existent variables', () => {
      expect(graph.getVariable('nonexistent')).toBeUndefined();
    });

    it('should list all variable names', () => {
      const names = graph.getVariableNames();
      expect(names).toContain('A');
      expect(names).toContain('B');
      expect(names).toContain('C');
      expect(names.length).toBe(3);
    });

    it('should check variable existence', () => {
      expect(graph.hasVariable('A')).toBe(true);
      expect(graph.hasVariable('X')).toBe(false);
    });

    it('should add new variables', () => {
      graph.addVariable('D', { type: 'bool', prior: 0.4 });
      expect(graph.hasVariable('D')).toBe(true);
      expect(graph.getValue('D')).toBeCloseTo(0.4);
    });
  });

  describe('State Management', () => {
    let graph: NeuroGraph;

    beforeEach(() => {
      graph = new NeuroGraph(sampleDoc);
    });

    it('should get variable state', () => {
      const state = graph.getState('A');
      expect(state).toBeDefined();
      expect(state?.value).toBe(0.5);
      expect(state?.locked).toBe(false);
    });

    it('should get current value', () => {
      expect(graph.getValue('A')).toBe(0.5);
      expect(graph.getValue('nonexistent')).toBeUndefined();
    });

    it('should set value for unlocked variables', () => {
      expect(graph.setValue('A', 0.8)).toBe(true);
      expect(graph.getValue('A')).toBe(0.8);
    });

    it('should clamp values to [0, 1]', () => {
      graph.setValue('A', 1.5);
      expect(graph.getValue('A')).toBe(1);
      graph.setValue('A', -0.5);
      expect(graph.getValue('A')).toBe(0);
    });

    it('should not set value for non-existent variables', () => {
      expect(graph.setValue('X', 0.5)).toBe(false);
    });

    it('should get all values', () => {
      const values = graph.getAllValues();
      expect(values['A']).toBe(0.5);
      expect(values['B']).toBe(0.3);
      expect(values['C']).toBe(0.1);
    });

    it('should set multiple values at once', () => {
      graph.setValues({ A: 0.9, B: 0.8 });
      expect(graph.getValue('A')).toBe(0.9);
      expect(graph.getValue('B')).toBe(0.8);
    });

    it('should reset to priors', () => {
      graph.setValue('A', 0.9);
      graph.setValue('B', 0.8);
      graph.resetToPriors();
      expect(graph.getValue('A')).toBe(0.5);
      expect(graph.getValue('B')).toBe(0.3);
    });
  });

  describe('Locking (Evidence)', () => {
    let graph: NeuroGraph;

    beforeEach(() => {
      graph = new NeuroGraph(sampleDoc);
    });

    it('should lock a variable to a value', () => {
      expect(graph.lockVariable('A', 1.0)).toBe(true);
      expect(graph.getValue('A')).toBe(1.0);
      expect(graph.isLocked('A')).toBe(true);
    });

    it('should not allow setValue on locked variables', () => {
      graph.lockVariable('A', 1.0);
      expect(graph.setValue('A', 0.5)).toBe(false);
      expect(graph.getValue('A')).toBe(1.0);
    });

    it('should unlock a variable', () => {
      graph.lockVariable('A', 1.0);
      graph.unlockVariable('A');
      expect(graph.isLocked('A')).toBe(false);
      expect(graph.setValue('A', 0.5)).toBe(true);
    });

    it('should unlock all variables', () => {
      graph.lockVariable('A', 1.0);
      graph.lockVariable('B', 0.0);
      graph.unlockAll();
      expect(graph.isLocked('A')).toBe(false);
      expect(graph.isLocked('B')).toBe(false);
    });

    it('should not reset locked variables', () => {
      graph.lockVariable('A', 1.0);
      graph.resetToPriors();
      expect(graph.getValue('A')).toBe(1.0);
    });
  });

  describe('Gradient Management', () => {
    let graph: NeuroGraph;

    beforeEach(() => {
      graph = new NeuroGraph(sampleDoc);
    });

    it('should get and set gradients', () => {
      expect(graph.getGradient('A')).toBe(0);
      graph.setGradient('A', 0.5);
      expect(graph.getGradient('A')).toBe(0.5);
    });

    it('should accumulate gradients', () => {
      graph.setGradient('A', 0.3);
      graph.accumulateGradient('A', 0.2);
      expect(graph.getGradient('A')).toBeCloseTo(0.5);
    });

    it('should clear all gradients', () => {
      graph.setGradient('A', 0.5);
      graph.setGradient('B', 0.3);
      graph.clearGradients();
      expect(graph.getGradient('A')).toBe(0);
      expect(graph.getGradient('B')).toBe(0);
    });
  });

  describe('Rule Management', () => {
    let graph: NeuroGraph;

    beforeEach(() => {
      graph = new NeuroGraph(sampleDoc);
    });

    it('should get rule by ID', () => {
      const rule = graph.getRule('rule1');
      expect(rule).toBeDefined();
      expect(rule?.type).toBe('IMPLICATION');
      expect(rule?.inputs).toEqual(['A']);
    });

    it('should get all rules', () => {
      const rules = graph.getRules();
      expect(rules.length).toBe(2);
    });

    it('should find rules by input variable', () => {
      const rulesWithA = graph.getRulesWithInput('A');
      expect(rulesWithA.length).toBe(2);
      expect(rulesWithA.map(r => r.id)).toContain('rule1');
      expect(rulesWithA.map(r => r.id)).toContain('rule2');
    });

    it('should find rules by output variable', () => {
      const rulesToB = graph.getRulesWithOutput('B');
      expect(rulesToB.length).toBe(1);
      expect(rulesToB[0]?.id).toBe('rule1');
    });

    it('should update rule weight', () => {
      expect(graph.setRuleWeight('rule1', 0.5)).toBe(true);
      expect(graph.getRule('rule1')?.weight).toBe(0.5);
    });

    it('should add new rules', () => {
      graph.addRule({
        id: 'rule3',
        type: 'DISJUNCTION',
        inputs: ['B'],
        output: 'A',
        op: 'OR',
        weight: 0.6
      });
      expect(graph.ruleCount).toBe(3);
      expect(graph.getRule('rule3')).toBeDefined();
    });
  });

  describe('Constraint Management', () => {
    let graph: NeuroGraph;

    beforeEach(() => {
      graph = new NeuroGraph(sampleDoc);
    });

    it('should get constraint by ID', () => {
      const constraint = graph.getConstraint('attack1');
      expect(constraint).toBeDefined();
      expect(constraint?.type).toBe('ATTACK');
    });

    it('should get all constraints', () => {
      const constraints = graph.getConstraints();
      expect(constraints.length).toBe(1);
    });

    it('should find constraints by source', () => {
      const fromC = graph.getConstraintsFromSource('C');
      expect(fromC.length).toBe(1);
      expect(fromC[0]?.id).toBe('attack1');
    });

    it('should find constraints by target', () => {
      const toA = graph.getConstraintsToTarget('A');
      expect(toA.length).toBe(1);
      expect(toA[0]?.id).toBe('attack1');
    });

    it('should add new constraints', () => {
      graph.addConstraint({
        id: 'support1',
        type: 'SUPPORT',
        source: 'A',
        target: 'B',
        weight: 0.5
      });
      expect(graph.constraintCount).toBe(2);
    });
  });

  describe('Graph Analysis', () => {
    let graph: NeuroGraph;

    beforeEach(() => {
      graph = new NeuroGraph(sampleDoc);
    });

    it('should compute topological order', () => {
      const order = graph.getTopologicalOrder();
      expect(order.length).toBe(3);
      // A should come before B (A -> B)
      expect(order.indexOf('A')).toBeLessThan(order.indexOf('B'));
      // B should come before C (A,B -> C)
      expect(order.indexOf('B')).toBeLessThan(order.indexOf('C'));
    });

    it('should identify source variables', () => {
      const sources = graph.getSourceVariables();
      expect(sources).toContain('A');
      expect(sources).not.toContain('B');
      expect(sources).not.toContain('C');
    });

    it('should identify sink variables', () => {
      // C has no outgoing rules (it's used as attack source but not in rule inputs)
      const sinks = graph.getSinkVariables();
      expect(sinks).toContain('C');
    });
  });

  describe('Export and Clone', () => {
    it('should export to NeuroJSON', () => {
      const graph = new NeuroGraph(sampleDoc);
      const exported = graph.export();
      
      expect(exported.version).toBe('1.0');
      expect(exported.name).toBe('test-graph');
      expect(Object.keys(exported.variables).length).toBe(3);
      expect(exported.rules.length).toBe(2);
      expect(exported.constraints.length).toBe(1);
    });

    it('should create independent clone', () => {
      const graph = new NeuroGraph(sampleDoc);
      const clone = graph.clone();
      
      graph.setValue('A', 0.9);
      expect(clone.getValue('A')).toBe(0.5); // Unchanged
    });
  });

  describe('Factory Functions', () => {
    it('createGraph should create empty graph', () => {
      const graph = createGraph('my-graph');
      expect(graph.variableCount).toBe(0);
    });

    it('parseNeuroJSON should parse JSON string', () => {
      const json = JSON.stringify(sampleDoc);
      const graph = parseNeuroJSON(json);
      expect(graph.variableCount).toBe(3);
    });
  });

  describe('Clear', () => {
    it('should clear all data', () => {
      const graph = new NeuroGraph(sampleDoc);
      graph.clear();
      expect(graph.variableCount).toBe(0);
      expect(graph.ruleCount).toBe(0);
      expect(graph.constraintCount).toBe(0);
    });
  });
});
