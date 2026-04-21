# Phase 1 Annotation Guidelines

## Label Set
- `realistic_hope`
- `unrealistic_hope`
- `generalized_hope`
- `no_hope`

## Decision Rules
1. If hope is specific and achievable with realistic context -> `realistic_hope`
2. If hope is wishful/exaggerated with weak grounding -> `unrealistic_hope`
3. If hope is broad/community/religious without specific action -> `generalized_hope`
4. If text is neutral/negative/sarcastic/no clear hope -> `no_hope`

## Ambiguity Handling
- Mark uncertain cases for joint review.
- Resolve by majority agreement after guideline discussion.
- Keep a changelog of revised labels.

## Quality Process
- Review 200–300 ambiguous samples.
- Calculate agreement snapshot before and after discussion.
- Freeze guideline version before final model training.

## Data Format Constraint
- CSV columns must be: `text`, `label`, `source`.
