---
name: brainstorming
description: "Turns vague ideas into validated designs/specs through collaborative questions, options, and incremental design sections. Use before creative work such as creating features, building components, adding functionality, or changing behavior."
---

# Brainstorming Ideas Into Designs

## Purpose

Help turn ideas into fully formed designs and specs through natural collaborative dialogue.

This skill is mandatory before creative work (new features, new components, new functionality, or behavior changes).

## Workflow

### 1) Understand the idea (one question at a time)

- Start by checking the current project context (relevant files, docs, and recent changes).
- Ask exactly one question per message to refine the idea.
- Prefer multiple-choice questions when possible; use open-ended questions only when necessary.
- Focus on:
  - **purpose**: what problem this solves and for whom
  - **constraints**: performance, UX, compatibility, timeline, dependencies
  - **success criteria**: what “done” looks like and how to verify it

### 2) Explore approaches

- Propose 2–3 approaches with trade-offs.
- Lead with the recommended option and explain why.
- Apply YAGNI ruthlessly: remove non-essential features and defer nice-to-haves.

### 3) Present the design (incremental validation)

- Once the goal is clear, present the design in small sections (200–300 words each).
- After each section, ask whether it looks right so far before continuing.
- Cover:
  - architecture and boundaries
  - components/modules and responsibilities
  - data flow and interfaces
  - error handling and observability
  - testing strategy

## After the design

### Documentation

- Write the validated design to `docs/plans/YYYY-MM-DD-<topic>-design.md`.
- Keep it concise and clear.
- Commit the design document to git.

### Implementation (if continuing)

- Ask: “Ready to set up for implementation?”
- Then create a detailed implementation plan and proceed with changes.
