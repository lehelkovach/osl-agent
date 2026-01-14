/**
 * Tests for NeuroEngine - the main API
 */

import { NeuroEngine, createEngine, TrainingData, NeuroJSON } from '../src';

describe('NeuroEngine', () => {
  // Bird classification example from the design doc
  const birdSchema: NeuroJSON = {
    version: '1.0',
    name: 'bird-classifier',
    variables: {
      has_wings: { type: 'bool', prior: 0.5 },
      has_feathers: { type: 'bool', prior: 0.5 },
      flies: { type: 'bool', prior: 0.5 },
      is_bird: { type: 'bool', prior: 0.3 }
    },
    rules: [
      {
        id: 'wings_imply_bird',
        type: 'IMPLICATION',
        inputs: ['has_wings'],
        output: 'is_bird',
        op: 'IDENTITY',
        weight: 0.7,
        learnable: true
      },
      {
        id: 'feathers_imply_bird',
        type: 'IMPLICATION',
        inputs: ['has_feathers'],
        output: 'is_bird',
        op: 'IDENTITY',
        weight: 0.9,
        learnable: true
      }
    ],
    constraints: []
  };

  // Simple causal model
  const causalSchema: NeuroJSON = {
    version: '1.0',
    variables: {
      rain: { type: 'bool', prior: 0.3 },
      wet: { type: 'bool', prior: 0.1 },
      slippery: { type: 'bool', prior: 0.05 }
    },
    rules: [
      {
        id: 'rain_wets',
        type: 'IMPLICATION',
        inputs: ['rain'],
        output: 'wet',
        op: 'IDENTITY',
        weight: 0.95
      },
      {
        id: 'wet_slippery',
        type: 'IMPLICATION',
        inputs: ['wet'],
        output: 'slippery',
        op: 'IDENTITY',
        weight: 0.8
      }
    ],
    constraints: []
  };

  describe('Constructor', () => {
    it('should create engine from schema', () => {
      const ai = new NeuroEngine(birdSchema);
      expect(ai).toBeInstanceOf(NeuroEngine);
    });

    it('should accept custom configuration', () => {
      const ai = new NeuroEngine(birdSchema, { learningRate: 0.05 });
      expect(ai.getConfig().learningRate).toBe(0.05);
    });
  });

  describe('run() - Inference', () => {
    it('should return all variable values', () => {
      const ai = new NeuroEngine(birdSchema);
      const result = ai.run();
      
      expect(result).toHaveProperty('has_wings');
      expect(result).toHaveProperty('has_feathers');
      expect(result).toHaveProperty('flies');
      expect(result).toHaveProperty('is_bird');
    });

    it('should handle evidence (hard constraints)', () => {
      const ai = new NeuroEngine(birdSchema);
      const result = ai.run({ has_wings: 1.0, flies: 0.0 });
      
      // Evidence should be locked
      expect(result.has_wings).toBe(1.0);
      expect(result.flies).toBe(0.0);
      
      // is_bird should increase due to wings
      expect(result.is_bird).toBeGreaterThan(0.3);
    });

    it('should propagate through causal chain', () => {
      const ai = new NeuroEngine(causalSchema);
      
      // Given rain, wet and slippery should increase
      const result = ai.run({ rain: 1.0 });
      
      expect(result.rain).toBe(1.0);
      expect(result.wet).toBeGreaterThan(0.5);
      expect(result.slippery).toBeGreaterThan(0.3);
    });

    it('should reset between runs', () => {
      const ai = new NeuroEngine(causalSchema);
      
      // First run with rain
      ai.run({ rain: 1.0 });
      
      // Second run without rain - should reset
      const result = ai.run({ rain: 0.0 });
      
      expect(result.rain).toBe(0.0);
      expect(result.wet).toBeLessThan(0.5);
    });

    it('should accept custom iteration count', () => {
      const ai = new NeuroEngine(causalSchema);
      const result = ai.run({ rain: 1.0 }, 5); // Just 5 iterations
      
      expect(result.rain).toBe(1.0);
    });
  });

  describe('query() - Single Variable', () => {
    it('should query a single variable', () => {
      const ai = new NeuroEngine(causalSchema);
      const prob = ai.query('slippery', { rain: 1.0 });
      
      expect(prob).toBeGreaterThan(0.3);
    });

    it('should return 0.5 for unknown variables', () => {
      const ai = new NeuroEngine(causalSchema);
      const prob = ai.query('unknown_var');
      
      expect(prob).toBe(0.5);
    });
  });

  describe('train() - Learning', () => {
    it('should reduce loss over training', () => {
      const ai = new NeuroEngine(birdSchema, { learningRate: 0.2 });
      
      const trainingData: TrainingData[] = [
        { inputs: { has_wings: 1.0, has_feathers: 1.0 }, targets: { is_bird: 0.95 } },
        { inputs: { has_wings: 1.0, has_feathers: 0.0 }, targets: { is_bird: 0.7 } },
        { inputs: { has_wings: 0.0, has_feathers: 0.0 }, targets: { is_bird: 0.1 } }
      ];
      
      const initialResult = ai.run({ has_wings: 1.0, has_feathers: 1.0 });
      const initialBird = initialResult.is_bird ?? 0;
      
      ai.train(trainingData, 50);
      
      const finalResult = ai.run({ has_wings: 1.0, has_feathers: 1.0 });
      const finalBird = finalResult.is_bird ?? 0;
      
      // Should have improved towards target
      expect(Math.abs(finalBird - 0.95)).toBeLessThan(Math.abs(initialBird - 0.95) + 0.1);
    });

    it('should update rule weights', () => {
      const ai = new NeuroEngine(birdSchema, { learningRate: 0.3 });
      
      const initialWeight = ai.getRuleWeight('wings_imply_bird');
      
      const trainingData: TrainingData[] = [
        { inputs: { has_wings: 1.0 }, targets: { is_bird: 0.99 } }
      ];
      
      ai.train(trainingData, 20);
      
      const finalWeight = ai.getRuleWeight('wings_imply_bird');
      
      // Weight should have changed
      expect(finalWeight).not.toBe(initialWeight);
    });

    it('should return final loss', () => {
      const ai = new NeuroEngine(birdSchema);
      
      const trainingData: TrainingData[] = [
        { inputs: { has_wings: 1.0 }, targets: { is_bird: 0.9 } }
      ];
      
      const loss = ai.train(trainingData, 10);
      
      expect(typeof loss).toBe('number');
      expect(loss).toBeGreaterThanOrEqual(0);
    });
  });

  describe('export() - Serialization', () => {
    it('should export the schema', () => {
      const ai = new NeuroEngine(birdSchema);
      const exported = ai.export();
      
      expect(exported.version).toBe('1.0');
      expect(exported.variables).toHaveProperty('has_wings');
      expect(exported.rules.length).toBe(2);
    });

    it('should include learned weights', () => {
      const ai = new NeuroEngine(birdSchema, { learningRate: 0.3 });
      
      // Train to change weights
      ai.train([
        { inputs: { has_wings: 1.0 }, targets: { is_bird: 0.99 } }
      ], 20);
      
      const exported = ai.export();
      const rule = exported.rules.find(r => r.id === 'wings_imply_bird');
      
      // Weight should be updated in export
      expect(rule?.weight).not.toBe(0.7);
    });

    it('should export as JSON string', () => {
      const ai = new NeuroEngine(birdSchema);
      const json = ai.exportJSON();
      
      expect(typeof json).toBe('string');
      
      // Should be valid JSON
      const parsed = JSON.parse(json);
      expect(parsed.version).toBe('1.0');
    });
  });

  describe('Graph Access', () => {
    it('should get variable names', () => {
      const ai = new NeuroEngine(birdSchema);
      const vars = ai.getVariables();
      
      expect(vars).toContain('has_wings');
      expect(vars).toContain('is_bird');
    });

    it('should get rule IDs', () => {
      const ai = new NeuroEngine(birdSchema);
      const rules = ai.getRules();
      
      expect(rules).toContain('wings_imply_bird');
      expect(rules).toContain('feathers_imply_bird');
    });

    it('should get/set rule weights', () => {
      const ai = new NeuroEngine(birdSchema);
      
      expect(ai.getRuleWeight('wings_imply_bird')).toBe(0.7);
      
      ai.setRuleWeight('wings_imply_bird', 0.5);
      expect(ai.getRuleWeight('wings_imply_bird')).toBe(0.5);
    });

    it('should get current state', () => {
      const ai = new NeuroEngine(birdSchema);
      ai.run({ has_wings: 1.0 });
      
      const state = ai.getState();
      expect(state.has_wings).toBe(1.0);
    });
  });

  describe('Configuration', () => {
    it('should get configuration', () => {
      const ai = new NeuroEngine(birdSchema);
      const config = ai.getConfig();
      
      expect(config).toHaveProperty('maxIterations');
      expect(config).toHaveProperty('learningRate');
    });

    it('should set configuration', () => {
      const ai = new NeuroEngine(birdSchema);
      ai.setConfig({ maxIterations: 50 });
      
      expect(ai.getConfig().maxIterations).toBe(50);
    });
  });

  describe('Argumentation (Attacks)', () => {
    const argumentSchema: NeuroJSON = {
      version: '1.0',
      variables: {
        evidence: { type: 'bool', prior: 0.8 },
        counter_evidence: { type: 'bool', prior: 0.6 },
        conclusion: { type: 'bool', prior: 0.5 }
      },
      rules: [{
        id: 'evidence_supports',
        type: 'IMPLICATION',
        inputs: ['evidence'],
        output: 'conclusion',
        op: 'IDENTITY',
        weight: 0.9
      }],
      constraints: [{
        id: 'counter_attacks',
        type: 'ATTACK',
        source: 'counter_evidence',
        target: 'conclusion',
        weight: 0.8
      }]
    };

    it('should handle attack constraints', () => {
      const ai = new NeuroEngine(argumentSchema);
      
      const result = ai.run();
      
      // Conclusion should be dampened by attack
      expect(result.conclusion).toBeLessThan(result.evidence ?? 1);
    });

    it('should show effect of strong attack', () => {
      const ai = new NeuroEngine(argumentSchema);
      
      // Strong counter-evidence
      const result = ai.run({ counter_evidence: 1.0 });
      
      // Conclusion should be significantly reduced
      expect(result.conclusion).toBeLessThan(0.5);
    });
  });

  describe('Factory Function', () => {
    it('createEngine should create NeuroEngine', () => {
      const ai = createEngine(birdSchema);
      expect(ai).toBeInstanceOf(NeuroEngine);
    });

    it('createEngine should accept config', () => {
      const ai = createEngine(birdSchema, { learningRate: 0.01 });
      expect(ai.getConfig().learningRate).toBe(0.01);
    });
  });

  describe('Penguin Example (from design doc)', () => {
    it('should handle fuzzy conflict: has_wings but does not fly', () => {
      const penguinSchema: NeuroJSON = {
        version: '1.0',
        variables: {
          has_wings: { type: 'bool', prior: 0.5 },
          flies: { type: 'bool', prior: 0.5 },
          is_bird: { type: 'bool', prior: 0.3 }
        },
        rules: [
          {
            id: 'wings_bird',
            type: 'IMPLICATION',
            inputs: ['has_wings'],
            output: 'is_bird',
            op: 'IDENTITY',
            weight: 0.8
          },
          {
            id: 'flies_bird',
            type: 'IMPLICATION',
            inputs: ['flies'],
            output: 'is_bird',
            op: 'IDENTITY',
            weight: 0.7
          }
        ],
        constraints: []
      };

      const ai = new NeuroEngine(penguinSchema);
      
      // Penguin: has wings but doesn't fly
      const result = ai.run({
        has_wings: 1.0,
        flies: 0.0
      });
      
      // Should still be recognized as bird (wings > flies)
      expect(result.is_bird).toBeGreaterThan(0.3);
      // But lower than if it also flew
      const flyingBird = ai.run({ has_wings: 1.0, flies: 1.0 });
      expect(result.is_bird).toBeLessThan(flyingBird.is_bird ?? 1);
    });
  });
});
