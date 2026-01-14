# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-14

### Added
- Initial release of NeuroSym.js
- **Logic Core** module with Lukasiewicz T-Norms
  - `and()`, `or()`, `not()` - Basic fuzzy operations
  - `implies()`, `equivalent()` - Logical connectives
  - `inhibit()`, `support()` - Argumentation operations
  - `mutexNormalize()` - Mutual exclusion constraint
  - Gradient functions for learning
- **NeuroGraph** module for state management
  - Variable management with priors
  - Rule and constraint tracking
  - Evidence injection (locking)
  - Topological ordering for inference
- **NeuroEngine** main API class
  - `run(evidence, iterations)` - Inference
  - `train(data, epochs)` - Weight learning
  - `export()` - Schema serialization
  - `query(variable, evidence)` - Single variable query
- **InferenceEngine** for advanced use
  - Forward chaining
  - Belief propagation
  - Constraint application
- **NeuroJSON** protocol specification
  - Variables with type and prior
  - Rules: IMPLICATION, CONJUNCTION, DISJUNCTION, EQUIVALENCE
  - Constraints: ATTACK, SUPPORT, MUTEX
- Full TypeScript type definitions
- 163 unit tests with 80%+ coverage
- Comprehensive documentation

### Technical Details
- Zero runtime dependencies
- ES Module and CommonJS builds
- Browser and Node.js compatible
