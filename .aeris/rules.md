# AERIS-X Roadmap Control Rules

1. Never implement a future milestone before the current milestone.
2. The current milestone is always the lowest-numbered incomplete milestone.
3. Historical out-of-order work may remain, but must not change the current milestone.
4. Never modify completed milestone functionality unnecessarily.
5. Inspect the repository before implementation.
6. Run tests and milestone-specific validation before completion.
7. Diagnose and repair validation failures, then rerun validation.
8. Never fabricate datasets, metrics, experiment results, or artifacts.
9. Never use unavailable datasets merely to satisfy a specification.
10. Do not claim completion without repository evidence.
11. Never randomly split overlapping temporal windows for ML evaluation.
12. Never fit preprocessing or scalers using validation or test data.
13. Preserve all previously completed experiment trees.
14. Do not silently skip milestones.
15. Do not jump to a future milestone because its specification is attached to a conversation.
