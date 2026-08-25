# Motion prompt source map

Working note for a provenance-aware prompt bank. This is a source audit, not
legal advice. Do not import or redistribute source annotations until the exact
license and attribution route has been checked.

## Release rule

Each released prompt record should contain:

- `intent_id`: our normalized motion concept;
- `prompt_text`: released wording;
- `semantic_attributes`: direction, speed, count, body side, style, object;
- `source_dataset` and `source_reference`;
- `source_text`: exact upstream text only when redistribution is permitted;
- `transformation`: exact, normalized, paraphrased, or independently written;
- `license_status`: approved, attribution-required, research-only, or excluded;
- `rig_feasibility`: whether a single 27-joint unpropped figure can express it.

Exact captions and motions are distinct assets. A public action name can inspire
an independently written prompt without making the upstream motion part of this
dataset, but systematic reuse of a protected taxonomy still needs review.

## Candidate sources

| Source | Useful coverage | Annotation form | Initial decision |
|---|---|---|---|
| HumanML3D | locomotion, daily actions, gestures, transitions | multiple natural-language descriptions per motion | High-value candidate. Audit the annotation and underlying AMASS terms before importing exact text. Prefer normalized intents plus independently written prompts. |
| KIT Motion-Language | everyday whole-body actions and compositional descriptions | free-form English descriptions | High-value candidate. Confirm dataset terms and attribution before retaining exact descriptions. |
| BABEL | dense atomic actions, transitions, overlapping actions | sequence and frame-aligned language labels | Strong ontology source. Underlying AMASS terms apply; do not redistribute motions. Audit label redistribution separately. |
| AIST++ | ten dance genres and varied choreographies | genre/choreography identifiers, not rich captions | Approved candidate for dance taxonomy; annotations are CC BY 4.0. Attribute Google/AIST++ and write our own prompt sentences. |
| FLAG3D | exercise and fitness instructions | 60 activity categories with language instructions | Strong fitness source. Hold exact text until the official dataset license is confirmed. |
| BEAT / BEAT2 | conversational and expressive gestures | speech, text, emotion, body gesture | Useful for gesture concepts, but much motion depends on speech context and hands/face absent from the stick rig. Import only clearly legible whole-body intents after license audit. |
| GRAB | pick, lift, pass, drink, use-object actions | object-action identifiers and motion intent | Useful ontology, but many prompts require visible props and the dataset is non-commercial research only. Keep as an optional prop-enabled future track. |
| Kinetics-700 | broad human action vocabulary | one action-class phrase per video | Use only as a coverage checklist until annotation terms are clear. Filter multi-person, camera-dependent, object-dependent, and non-rig-legible classes. |
| FineGym | fine-grained gymnastics | hierarchical event/action labels | Research-only/CC BY-NC annotation route. Do not merge into a permissive release without a compatible license decision. |
| NTU RGB+D 120 | daily actions, health events, interactions | 120 fixed action labels | Exclude from released prompt sourcing: official terms prohibit derivation or generation of a new dataset without permission. |
| BONES-SEED | broad mocap taxonomy and up to six descriptions per motion | proprietary hierarchy and natural-language variants | Exclude. The current license prohibits using the dataset or results to create or augment a competing/synthetic motion dataset. |

## Prompt families to preserve

For every normalized intent, distinguish surface-form variation from motion
attributes:

- subject form: `a person`, `someone`, `he`, `she`;
- syntax: `does`, `is doing`, `performs`;
- lexical paraphrase: `runs`, `sprints`, `moves forward at a run`;
- count/aspect: once, repeatedly, continuously;
- speed/energy: slow, brisk, rapid, energetic;
- direction: forward, backward, left, right, clockwise;
- laterality: left hand, right hand, both hands;
- style/emotion: proudly, cautiously, tiredly;
- composition: action A then action B.

Subject and syntax variants normally share the same motion target. Count,
speed, direction, laterality, style, and composition are controlled attributes
and must not be collapsed into simple paraphrases.

## Discovery protocol

1. Normalize source labels into an internal intent ontology.
2. Remove actions that cannot be read from a single unpropped 27-joint rig.
3. Write several surface-form variants and controlled-attribute variants.
4. Generate every variant with matched ARDY seeds.
5. Score rig validity, within-intent consistency, between-intent separation,
   and attribute response.
6. Manually inspect borderline clusters and record rejection reasons.
7. Release accepted prompts, failures, seeds, model revision, and raw rigs so
   the selection process is reproducible.

## First pilot

Start with 50 intents balanced across locomotion, exercise, gesture, dance, and
transition. Use four surface paraphrases, two controlled variants, and three
matched seeds per intent. This yields 900 ARDY generations and is large enough
to test the discovery method without committing to a full expansion.
