/**
 * Tests for Inference Engine module
 */

import { InferenceEngine, createEngine, infer, query } from '../src/inference';
import { NeuroGraph } from '../src/neuro-graph';
import { NeuroJSON, TrainingExample } from '../src/types';

describe('Inference Engine', () => {
  // Simple causal chain: rain -> wet_ground -> slippery
  const causalChainDoc: NeuroJSON = {
    version: '1.0',
    name: 'causal-chain',
    variables: {
      rain: { type: 'bool', prior: 0.3 },
      wet_ground: { type: 'bool', prior: 0.1 },
      slippery: { type: 'bool', prior: 0.05 }
    },
    rules: [
      {
        id: 'rain_wets_ground',
        type: 'IMPLICATION',
        inputs: ['rain'],
        output: 'wet_ground',
        op: 'IDENTITY',
        weight: 0.9
      },
      {
        id: 'wet_makes_slippery',
        type: 'IMPLICATION',
        inputs: ['wet_ground'],
        output: 'slippery',
        op: 'IDENTITY',
        weight: 0.8
      }
    ],
    constraints: []
  };

  // Argumentation example: pro and con arguments
  const argumentationDoc: NeuroJSON = {
    version: '1.0',
    name: 'argumentation',
    variables: {
      pro_argument: { type: 'bool', prior: 0.7 },
      con_argument: { type: 'bool', prior: 0.6 },
      conclusion: { type: 'bool', prior: 0.5 }
    },
    rules: [
      {
        id: 'pro_supports',
        type: 'IMPLICATION',
        inputs: ['pro_argument'],
        output: 'conclusion',
        op: 'IDENTITY',
        weight: 0.9
      }
    ],
    constraints: [
      {
        id: 'con_attacks',
        type: 'ATTACK',
        source: 'con_argument',
        target: 'conclusion',
        weight: 0.8
      }
    ]
  };

  describe('Configuration', () => {
    it('should use default configuration', () => {
      const engine = new InferenceEngine();
      const config = engine.getConfig();
      expect(config.maxIterations).toBe(100);
      expect(config.convergenceThreshold).toBe(0.001);
    });

    it('should accept custom configuration', () => {
      const engine = new InferenceEngine({ maxIterations: 50 });
      expect(engine.getConfig().maxIterations).toBe(50);
    });

    it('should update configuration', () => {
      const engine = new InferenceEngine();
      engine.setConfig({ learningRate: 0.05 });
      expect(engine.getConfig().learningRate).toBe(0.05);
    });
  });

  describe('Forward Pass', () => {
    it('should propagate values through rules', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      // Lock rain to true
      graph.lockVariable('rain', 1.0);
      
      // Single forward pass
      engine.forwardPass(graph);
      
      // wet_ground should be increased (rain * weight = 0.9)
      expect(graph.getValue('wet_ground')).toBeGreaterThan(0.1);
    });

    it('should not modify locked variables', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      graph.lockVariable('wet_ground', 0.5);
      engine.forwardPass(graph);
      
      expect(graph.getValue('wet_ground')).toBe(0.5);
    });
  });

  describe('Rule Evaluation', () => {
    it('should evaluate IMPLICATION rules', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      const rule = graph.getRule('rain_wets_ground')!;
      
      graph.setValue('rain', 1.0);
      const result = engine.evaluateRule(graph, rule);
      
      expect(result).toBeCloseTo(0.9); // 1.0 * 0.9
    });

    it('should evaluate CONJUNCTION rules', () => {
      const doc: NeuroJSON = {
        version: '1.0',
        variables: {
          A: { type: 'bool', prior: 0.8 },
          B: { type: 'bool', prior: 0.9 },
          C: { type: 'bool', prior: 0.1 }
        },
        rules: [{
          id: 'conj',
          type: 'CONJUNCTION',
          inputs: ['A', 'B'],
          output: 'C',
          op: 'AND',
          weight: 1.0
        }],
        constraints: []
      };
      
      const graph = new NeuroGraph(doc);
      const engine = new InferenceEngine();
      const rule = graph.getRule('conj')!;
      
      const result = engine.evaluateRule(graph, rule);
      // Lukasiewicz AND: max(0, 0.8 + 0.9 - 1) = 0.7
      expect(result).toBeCloseTo(0.7);
    });

    it('should evaluate DISJUNCTION rules', () => {
      const doc: NeuroJSON = {
        version: '1.0',
        variables: {
          A: { type: 'bool', prior: 0.3 },
          B: { type: 'bool', prior: 0.4 },
          C: { type: 'bool', prior: 0.1 }
        },
        rules: [{
          id: 'disj',
          type: 'DISJUNCTION',
          inputs: ['A', 'B'],
          output: 'C',
          op: 'OR',
          weight: 1.0
        }],
        constraints: []
      };
      
      const graph = new NeuroGraph(doc);
      const engine = new InferenceEngine();
      const rule = graph.getRule('disj')!;
      
      const result = engine.evaluateRule(graph, rule);
      // Lukasiewicz OR: min(1, 0.3 + 0.4) = 0.7
      expect(result).toBeCloseTo(0.7);
    });
  });

  describe('Constraint Application', () => {
    it('should apply ATTACK constraints', () => {
      const graph = new NeuroGraph(argumentationDoc);
      const engine = new InferenceEngine();
      
      // Set initial values
      graph.setValue('con_argument', 1.0);
      graph.setValue('conclusion', 0.8);
      
      engine.applyConstraints(graph);
      
      // conclusion should be reduced by attack
      expect(graph.getValue('conclusion')).toBeLessThan(0.8);
    });

    it('should apply SUPPORT constraints', () => {
      const doc: NeuroJSON = {
        version: '1.0',
        variables: {
          supporter: { type: 'bool', prior: 0.8 },
          target: { type: 'bool', prior: 0.3 }
        },
        rules: [],
        constraints: [{
          id: 'support1',
          type: 'SUPPORT',
          source: 'supporter',
          target: 'target',
          weight: 0.5
        }]
      };
      
      const graph = new NeuroGraph(doc);
      const engine = new InferenceEngine();
      
      engine.applyConstraints(graph);
      
      // target should be increased
      expect(graph.getValue('target')).toBeGreaterThan(0.3);
    });
  });

  describe('Full Inference', () => {
    it('should converge to stable state', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      const result = engine.infer(graph);
      
      expect(result.converged).toBe(true);
      expect(result.iterations).toBeLessThan(100);
    });

    it('should record history when requested', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      const result = engine.infer(graph, true);
      
      expect(result.history).toBeDefined();
      expect(result.history!.length).toBeGreaterThan(0);
    });

    it('should propagate through chain', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      graph.lockVariable('rain', 1.0);
      const result = engine.infer(graph);
      
      // With rain = 1, wet_ground should be high, slippery should follow
      expect(result.states['wet_ground']?.value).toBeGreaterThan(0.5);
      expect(result.states['slippery']?.value).toBeGreaterThan(0.3);
    });
  });

  describe('Inference with Evidence', () => {
    it('should set evidence and run inference', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      const result = engine.inferWithEvidence(graph, { rain: 1.0 });
      
      expect(graph.getValue('rain')).toBe(1.0);
      expect(graph.isLocked('rain')).toBe(true);
      expect(result.states['wet_ground']?.value).toBeGreaterThan(0.5);
    });

    it('should reset to priors before applying evidence', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      // First set rain high
      engine.inferWithEvidence(graph, { rain: 1.0 });
      
      // Then set rain low - should reset other variables
      const result = engine.inferWithEvidence(graph, { rain: 0.0 });
      
      expect(result.states['rain']?.value).toBe(0);
    });
  });

  describe('Query', () => {
    it('should query single variable given evidence', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      const wetProb = engine.query(graph, 'wet_ground', { rain: 1.0 });
      
      expect(wetProb).toBeGreaterThan(0.5);
    });

    it('should return 0.5 for unknown variable', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      const prob = engine.query(graph, 'unknown', {});
      
      expect(prob).toBe(0.5);
    });
  });

  describe('Explain', () => {
    it('should return values of unobserved variables', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const engine = new InferenceEngine();
      
      const explanation = engine.explain(graph, { rain: 1.0 });
      
      expect(explanation).not.toHaveProperty('rain');
      expect(explanation).toHaveProperty('wet_ground');
      expect(explanation).toHaveProperty('slippery');
    });
  });

  describe('Training', () => {
    it('should reduce loss over epochs', () => {
      // Simple learnable graph
      const doc: NeuroJSON = {
        version: '1.0',
        variables: {
          input: { type: 'bool', prior: 0.5 },
          output: { type: 'bool', prior: 0.5 }
        },
        rules: [{
          id: 'learn_rule',
          type: 'IMPLICATION',
          inputs: ['input'],
          output: 'output',
          op: 'IDENTITY',
          weight: 0.5,
          learnable: true
        }],
        constraints: []
      };
      
      const graph = new NeuroGraph(doc);
      const engine = new InferenceEngine({ learningRate: 0.2 });
      
      const examples: TrainingExample[] = [
        { inputs: { input: 1.0 }, outputs: { output: 0.9 } },
        { inputs: { input: 0.0 }, outputs: { output: 0.1 } }
      ];
      
      const result = engine.train(graph, examples, 50);
      
      // Loss should decrease
      expect(result.lossHistory[result.lossHistory.length - 1]).toBeLessThan(
        result.lossHistory[0]!
      );
    });

    it('should compute loss correctly', () => {
      const doc: NeuroJSON = {
        version: '1.0',
        variables: {
          A: { type: 'bool', prior: 0.5 },
          B: { type: 'bool', prior: 0.5 }
        },
        rules: [{
          id: 'r1',
          type: 'IMPLICATION',
          inputs: ['A'],
          output: 'B',
          op: 'IDENTITY',
          weight: 1.0
        }],
        constraints: []
      };
      
      const graph = new NeuroGraph(doc);
      const engine = new InferenceEngine();
      
      // With A=1 and weight=1, B should become ~1
      // If we expect B=0, loss should be high
      const loss = engine.computeLoss(graph, {
        inputs: { A: 1.0 },
        outputs: { B: 0.0 }
      });
      
      expect(loss).toBeGreaterThan(0);
    });

    it('should early stop when loss is very low', () => {
      const doc: NeuroJSON = {
        version: '1.0',
        variables: {
          A: { type: 'bool', prior: 0.5 },
          B: { type: 'bool', prior: 0.5 }
        },
        rules: [{
          id: 'r1',
          type: 'IMPLICATION',
          inputs: ['A'],
          output: 'B',
          op: 'IDENTITY',
          weight: 0.9,
          learnable: true
        }],
        constraints: []
      };
      
      const graph = new NeuroGraph(doc);
      const engine = new InferenceEngine();
      
      // Training example that already matches
      const examples: TrainingExample[] = [
        { inputs: { A: 1.0 }, outputs: { B: 0.9 } }
      ];
      
      const result = engine.train(graph, examples, 100);
      
      // Should stop early due to low loss
      expect(result.epochs).toBeLessThan(100);
    });
  });

  describe('Factory Functions', () => {
    it('createEngine should create engine with config', () => {
      const engine = createEngine({ maxIterations: 50 });
      expect(engine.getConfig().maxIterations).toBe(50);
    });

    it('infer should run inference on graph', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const result = infer(graph);
      expect(result.converged).toBe(true);
    });

    it('query should query variable', () => {
      const graph = new NeuroGraph(causalChainDoc);
      const prob = query(graph, 'wet_ground', { rain: 1.0 });
      expect(prob).toBeGreaterThan(0.5);
    });
  });

  describe('Edge Cases', () => {
    it('should handle graph with no rules', () => {
      const doc: NeuroJSON = {
        version: '1.0',
        variables: {
          A: { type: 'bool', prior: 0.5 }
        },
        rules: [],
        constraints: []
      };
      
      const graph = new NeuroGraph(doc);
      const engine = new InferenceEngine();
      const result = engine.infer(graph);
      
      expect(result.converged).toBe(true);
      expect(result.states['A']?.value).toBe(0.5);
    });

    it('should handle cyclic dependencies gracefully', () => {
      const doc: NeuroJSON = {
        version: '1.0',
        variables: {
          A: { type: 'bool', prior: 0.5 },
          B: { type: 'bool', prior: 0.5 }
        },
        rules: [
          {
            id: 'a_to_b',
            type: 'IMPLICATION',
            inputs: ['A'],
            output: 'B',
            op: 'IDENTITY',
            weight: 0.8
          },
          {
            id: 'b_to_a',
            type: 'IMPLICATION',
            inputs: ['B'],
            output: 'A',
            op: 'IDENTITY',
            weight: 0.8
          }
        ],
        constraints: []
      };
      
      const graph = new NeuroGraph(doc);
      const engine = new InferenceEngine({ maxIterations: 50 });
      const result = engine.infer(graph);
      
      // Should either converge or hit max iterations
      expect(result.iterations).toBeGreaterThan(0);
    });
  });
});
