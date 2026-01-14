/**
 * Tests for Types and Validation module
 */

import {
  validateNeuroJSON,
  createVariable,
  createDefaultConfig,
  NeuroJSON
} from '../src/types';

describe('Types and Validation', () => {
  describe('validateNeuroJSON', () => {
    const validDoc: NeuroJSON = {
      version: '1.0',
      variables: {
        A: { type: 'bool', prior: 0.5 },
        B: { type: 'continuous', prior: 0.3 }
      },
      rules: [
        {
          id: 'rule1',
          type: 'IMPLICATION',
          inputs: ['A'],
          output: 'B',
          op: 'IDENTITY',
          weight: 0.9
        }
      ],
      constraints: [
        {
          id: 'attack1',
          type: 'ATTACK',
          source: 'B',
          target: 'A',
          weight: 0.5
        }
      ]
    };

    it('should validate a correct document', () => {
      const result = validateNeuroJSON(validDoc);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should reject non-object input', () => {
      expect(validateNeuroJSON(null).valid).toBe(false);
      expect(validateNeuroJSON(undefined).valid).toBe(false);
      expect(validateNeuroJSON('string').valid).toBe(false);
      expect(validateNeuroJSON(123).valid).toBe(false);
    });

    it('should require version field', () => {
      const doc = { ...validDoc };
      delete (doc as any).version;
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'version')).toBe(true);
    });

    it('should require variables to be an object', () => {
      const doc = { ...validDoc, variables: null };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'variables')).toBe(true);
    });

    it('should validate variable types', () => {
      const doc = {
        ...validDoc,
        variables: {
          A: { type: 'invalid', prior: 0.5 }
        }
      };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'variables.A.type')).toBe(true);
    });

    it('should validate variable priors are in range', () => {
      const doc = {
        ...validDoc,
        variables: {
          A: { type: 'bool', prior: 1.5 }
        }
      };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'variables.A.prior')).toBe(true);
    });

    it('should validate prior is not negative', () => {
      const doc = {
        ...validDoc,
        variables: {
          A: { type: 'bool', prior: -0.1 }
        }
      };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
    });

    it('should require rules to be an array', () => {
      const doc = { ...validDoc, rules: {} };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'rules')).toBe(true);
    });

    it('should validate rule structure', () => {
      const doc = {
        ...validDoc,
        rules: [
          { id: 123 } // invalid: id should be string
        ]
      };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'rules[0].id')).toBe(true);
    });

    it('should validate rule types', () => {
      const doc = {
        ...validDoc,
        rules: [{
          id: 'r1',
          type: 'INVALID_TYPE',
          inputs: ['A'],
          output: 'B',
          op: 'IDENTITY',
          weight: 0.9
        }]
      };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'rules[0].type')).toBe(true);
    });

    it('should validate rule inputs is an array', () => {
      const doc = {
        ...validDoc,
        rules: [{
          id: 'r1',
          type: 'IMPLICATION',
          inputs: 'A', // should be array
          output: 'B',
          op: 'IDENTITY',
          weight: 0.9
        }]
      };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'rules[0].inputs')).toBe(true);
    });

    it('should validate rule weight is in range', () => {
      const doc = {
        ...validDoc,
        rules: [{
          id: 'r1',
          type: 'IMPLICATION',
          inputs: ['A'],
          output: 'B',
          op: 'IDENTITY',
          weight: 2.0 // out of range
        }]
      };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'rules[0].weight')).toBe(true);
    });

    it('should require constraints to be an array', () => {
      const doc = { ...validDoc, constraints: {} };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'constraints')).toBe(true);
    });

    it('should validate constraint types', () => {
      const doc = {
        ...validDoc,
        constraints: [{
          id: 'c1',
          type: 'INVALID',
          source: 'A',
          target: 'B',
          weight: 0.5
        }]
      };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === 'constraints[0].type')).toBe(true);
    });

    it('should allow valid constraint types', () => {
      for (const type of ['ATTACK', 'SUPPORT', 'MUTEX']) {
        const doc = {
          ...validDoc,
          constraints: [{
            id: 'c1',
            type,
            source: 'A',
            target: 'B',
            weight: 0.5
          }]
        };
        const result = validateNeuroJSON(doc);
        expect(result.errors.filter(e => e.path === 'constraints[0].type')).toHaveLength(0);
      }
    });

    it('should allow valid rule types', () => {
      for (const type of ['IMPLICATION', 'EQUIVALENCE', 'CONJUNCTION', 'DISJUNCTION']) {
        const doc = {
          ...validDoc,
          rules: [{
            id: 'r1',
            type,
            inputs: ['A'],
            output: 'B',
            op: 'IDENTITY',
            weight: 0.9
          }]
        };
        const result = validateNeuroJSON(doc);
        expect(result.errors.filter(e => e.path === 'rules[0].type')).toHaveLength(0);
      }
    });

    it('should allow empty rules and constraints', () => {
      const doc = {
        version: '1.0',
        variables: { A: { type: 'bool', prior: 0.5 } },
        rules: [],
        constraints: []
      };
      const result = validateNeuroJSON(doc);
      expect(result.valid).toBe(true);
    });
  });

  describe('createVariable', () => {
    it('should create variable with defaults', () => {
      const variable = createVariable();
      expect(variable.type).toBe('bool');
      expect(variable.prior).toBe(0.5);
    });

    it('should create variable with custom values', () => {
      const variable = createVariable(0.7, 'continuous');
      expect(variable.type).toBe('continuous');
      expect(variable.prior).toBe(0.7);
    });
  });

  describe('createDefaultConfig', () => {
    it('should create config with sensible defaults', () => {
      const config = createDefaultConfig();
      expect(config.maxIterations).toBe(100);
      expect(config.convergenceThreshold).toBe(0.001);
      expect(config.learningRate).toBe(0.1);
      expect(config.dampingFactor).toBe(0.5);
    });
  });
});
