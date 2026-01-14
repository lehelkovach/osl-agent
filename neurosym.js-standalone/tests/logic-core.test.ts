/**
 * Tests for Logic Core module
 */

import {
  clamp,
  isValidTruthValue,
  not,
  and,
  or,
  implies,
  equivalent,
  weightedAverage,
  weightedAnd,
  weightedOr,
  inhibit,
  support,
  mutexNormalize,
  applyOperation,
  andGradient,
  orGradient,
  impliesGradient,
  inhibitGradient
} from '../src/logic-core';

describe('Logic Core', () => {
  describe('Utility Functions', () => {
    describe('clamp', () => {
      it('should return value unchanged if in range', () => {
        expect(clamp(0.5)).toBe(0.5);
        expect(clamp(0)).toBe(0);
        expect(clamp(1)).toBe(1);
      });

      it('should clamp values below 0 to 0', () => {
        expect(clamp(-0.5)).toBe(0);
        expect(clamp(-100)).toBe(0);
      });

      it('should clamp values above 1 to 1', () => {
        expect(clamp(1.5)).toBe(1);
        expect(clamp(100)).toBe(1);
      });
    });

    describe('isValidTruthValue', () => {
      it('should return true for valid truth values', () => {
        expect(isValidTruthValue(0)).toBe(true);
        expect(isValidTruthValue(0.5)).toBe(true);
        expect(isValidTruthValue(1)).toBe(true);
      });

      it('should return false for invalid values', () => {
        expect(isValidTruthValue(-0.1)).toBe(false);
        expect(isValidTruthValue(1.1)).toBe(false);
        expect(isValidTruthValue(NaN)).toBe(false);
      });
    });
  });

  describe('Lukasiewicz Logic Operations', () => {
    describe('not (negation)', () => {
      it('should negate truth values', () => {
        expect(not(0)).toBe(1);
        expect(not(1)).toBe(0);
        expect(not(0.3)).toBeCloseTo(0.7);
        expect(not(0.7)).toBeCloseTo(0.3);
      });

      it('should be idempotent (double negation)', () => {
        expect(not(not(0.4))).toBeCloseTo(0.4);
      });
    });

    describe('and (conjunction)', () => {
      it('should handle empty input', () => {
        expect(and()).toBe(1); // Identity element
      });

      it('should handle single input', () => {
        expect(and(0.7)).toBeCloseTo(0.7);
      });

      it('should compute Lukasiewicz AND for two values', () => {
        // max(0, a + b - 1)
        expect(and(1, 1)).toBe(1);
        expect(and(0, 0)).toBe(0);
        expect(and(0, 1)).toBe(0);
        expect(and(1, 0)).toBe(0);
        expect(and(0.8, 0.7)).toBeCloseTo(0.5); // max(0, 1.5 - 1) = 0.5
        expect(and(0.4, 0.4)).toBe(0); // max(0, 0.8 - 1) = 0
      });

      it('should handle multiple inputs', () => {
        // max(0, sum - (n-1))
        expect(and(1, 1, 1)).toBe(1);
        expect(and(0.5, 0.5, 0.5)).toBe(0); // max(0, 1.5 - 2) = 0
        expect(and(0.9, 0.9, 0.9)).toBeCloseTo(0.7); // max(0, 2.7 - 2) = 0.7
      });

      it('should be commutative', () => {
        expect(and(0.3, 0.7)).toBeCloseTo(and(0.7, 0.3));
      });
    });

    describe('or (disjunction)', () => {
      it('should handle empty input', () => {
        expect(or()).toBe(0); // Identity element
      });

      it('should handle single input', () => {
        expect(or(0.7)).toBeCloseTo(0.7);
      });

      it('should compute Lukasiewicz OR for two values', () => {
        // min(1, a + b)
        expect(or(0, 0)).toBe(0);
        expect(or(1, 1)).toBe(1);
        expect(or(0, 1)).toBe(1);
        expect(or(0.3, 0.4)).toBeCloseTo(0.7);
        expect(or(0.6, 0.7)).toBe(1); // min(1, 1.3) = 1
      });

      it('should be commutative', () => {
        expect(or(0.3, 0.7)).toBeCloseTo(or(0.7, 0.3));
      });
    });

    describe('implies (implication)', () => {
      it('should compute Lukasiewicz implication', () => {
        // min(1, 1 - a + b)
        expect(implies(0, 0)).toBe(1); // False implies anything
        expect(implies(0, 1)).toBe(1);
        expect(implies(1, 1)).toBe(1);
        expect(implies(1, 0)).toBe(0); // True implies false = false
        expect(implies(0.8, 0.5)).toBeCloseTo(0.7); // min(1, 0.2 + 0.5)
      });

      it('should satisfy: implies(a, 1) = 1', () => {
        expect(implies(0.3, 1)).toBe(1);
        expect(implies(0.9, 1)).toBe(1);
      });

      it('should satisfy: implies(1, b) = b', () => {
        expect(implies(1, 0.5)).toBeCloseTo(0.5);
        expect(implies(1, 0.8)).toBeCloseTo(0.8);
      });
    });

    describe('equivalent (equivalence)', () => {
      it('should compute equivalence', () => {
        // 1 - |a - b|
        expect(equivalent(0.5, 0.5)).toBe(1);
        expect(equivalent(0, 1)).toBe(0);
        expect(equivalent(1, 0)).toBe(0);
        expect(equivalent(0.3, 0.7)).toBeCloseTo(0.6);
      });

      it('should be symmetric', () => {
        expect(equivalent(0.3, 0.8)).toBeCloseTo(equivalent(0.8, 0.3));
      });
    });
  });

  describe('Weighted Operations', () => {
    describe('weightedAverage', () => {
      it('should handle empty input', () => {
        expect(weightedAverage([])).toBe(0.5);
      });

      it('should compute weighted average', () => {
        expect(weightedAverage([[0.8, 1], [0.4, 1]])).toBeCloseTo(0.6);
        expect(weightedAverage([[0.8, 2], [0.4, 1]])).toBeCloseTo(0.667, 2);
      });

      it('should handle zero total weight', () => {
        expect(weightedAverage([[0.5, 0], [0.7, 0]])).toBe(0.5);
      });
    });

    describe('weightedAnd', () => {
      it('should handle empty input', () => {
        expect(weightedAnd([])).toBe(1);
      });

      it('should soften inputs based on weight', () => {
        // Weight 1 keeps original, weight 0 treats as 1
        expect(weightedAnd([[0.5, 1]])).toBeCloseTo(0.5);
        expect(weightedAnd([[0.5, 0]])).toBe(1);
        expect(weightedAnd([[0.5, 0.5]])).toBeCloseTo(0.75); // 0.5*0.5 + 0.5 = 0.75
      });
    });

    describe('weightedOr', () => {
      it('should handle empty input', () => {
        expect(weightedOr([])).toBe(0);
      });

      it('should soften inputs based on weight', () => {
        expect(weightedOr([[0.5, 1]])).toBeCloseTo(0.5);
        expect(weightedOr([[0.5, 0]])).toBe(0);
      });
    });
  });

  describe('Argumentation Logic', () => {
    describe('inhibit (attack)', () => {
      it('should reduce target based on attacker strength', () => {
        // target * (1 - attacker * weight)
        expect(inhibit(1, 1, 1)).toBe(0); // Full attack
        expect(inhibit(1, 0, 1)).toBe(1); // No attack
        expect(inhibit(0.8, 0.5, 1)).toBeCloseTo(0.4); // 0.8 * 0.5
        expect(inhibit(0.8, 1, 0.5)).toBeCloseTo(0.4); // Half-strength attack
      });

      it('should not affect target when attacker is 0', () => {
        expect(inhibit(0.7, 0, 1)).toBeCloseTo(0.7);
      });
    });

    describe('support (reinforcement)', () => {
      it('should increase target based on supporter', () => {
        // target + (1 - target) * supporter * weight
        expect(support(0, 1, 1)).toBe(1); // Full support
        expect(support(0.5, 1, 1)).toBe(1); // Full support from 0.5
        expect(support(0.5, 0, 1)).toBeCloseTo(0.5); // No support
        expect(support(0.5, 0.5, 1)).toBeCloseTo(0.75); // 0.5 + 0.5*0.5
      });

      it('should not exceed 1', () => {
        expect(support(0.9, 1, 1)).toBe(1);
      });
    });

    describe('mutexNormalize', () => {
      it('should handle empty array', () => {
        expect(mutexNormalize([])).toEqual([]);
      });

      it('should not change values that sum to <= 1', () => {
        const result = mutexNormalize([0.3, 0.4, 0.2]);
        expect(result[0]).toBeCloseTo(0.3);
        expect(result[1]).toBeCloseTo(0.4);
        expect(result[2]).toBeCloseTo(0.2);
      });

      it('should normalize values that sum to > 1', () => {
        const result = mutexNormalize([0.6, 0.6]);
        expect(result[0]! + result[1]!).toBeCloseTo(1);
        expect(result[0]).toBeCloseTo(0.5);
        expect(result[1]).toBeCloseTo(0.5);
      });
    });
  });

  describe('Operation Dispatcher', () => {
    describe('applyOperation', () => {
      it('should handle IDENTITY', () => {
        expect(applyOperation('IDENTITY', [0.7])).toBeCloseTo(0.7);
        expect(applyOperation('IDENTITY', [])).toBe(0.5);
      });

      it('should handle AND', () => {
        expect(applyOperation('AND', [0.8, 0.9])).toBeCloseTo(and(0.8, 0.9));
      });

      it('should handle OR', () => {
        expect(applyOperation('OR', [0.3, 0.4])).toBeCloseTo(or(0.3, 0.4));
      });

      it('should handle NOT', () => {
        expect(applyOperation('NOT', [0.3])).toBeCloseTo(not(0.3));
      });

      it('should handle WEIGHTED with weights', () => {
        expect(applyOperation('WEIGHTED', [0.8, 0.4], [1, 1])).toBeCloseTo(0.6);
      });

      it('should throw for unknown operation', () => {
        expect(() => applyOperation('UNKNOWN' as any, [])).toThrow();
      });
    });
  });

  describe('Gradient Functions', () => {
    describe('andGradient', () => {
      it('should propagate gradient when result > 0', () => {
        const grads = andGradient([0.9, 0.9], 1.0);
        expect(grads[0]).toBe(1.0);
        expect(grads[1]).toBe(1.0);
      });

      it('should block gradient when result = 0', () => {
        const grads = andGradient([0.3, 0.3], 1.0);
        expect(grads[0]).toBe(0);
        expect(grads[1]).toBe(0);
      });
    });

    describe('orGradient', () => {
      it('should propagate gradient when result < 1', () => {
        const grads = orGradient([0.3, 0.4], 1.0);
        expect(grads[0]).toBe(1.0);
        expect(grads[1]).toBe(1.0);
      });

      it('should block gradient when result = 1', () => {
        const grads = orGradient([0.8, 0.8], 1.0);
        expect(grads[0]).toBe(0);
        expect(grads[1]).toBe(0);
      });
    });

    describe('impliesGradient', () => {
      it('should return correct gradients when not saturated', () => {
        const [dA, dB] = impliesGradient(0.8, 0.5, 1.0);
        expect(dA).toBe(-1.0);
        expect(dB).toBe(1.0);
      });

      it('should return zero gradients when saturated at 1', () => {
        const [dA, dB] = impliesGradient(0.3, 0.9, 1.0);
        expect(dA).toBe(0);
        expect(dB).toBe(0);
      });
    });

    describe('inhibitGradient', () => {
      it('should compute correct gradients', () => {
        const [dTarget, dAttacker] = inhibitGradient(0.8, 0.5, 1.0, 1.0);
        expect(dTarget).toBeCloseTo(0.5); // (1 - 0.5*1) * 1
        expect(dAttacker).toBeCloseTo(-0.8); // -0.8 * 1 * 1
      });
    });
  });
});
