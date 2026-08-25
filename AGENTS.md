# Collaboration format

For every user-facing response in this repository:

1. Begin with one simple Unicode emotion that reflects the assistant's current stance, such as `🙂`, `🤔`, `😮`, or `✅`. Do not use multi-line ASCII art.
2. Do **not** include a progress display in every response. Show one only when the user explicitly asks for status, a material milestone or state change has occurred, or a long-running work interval needs an update.
3. Omit the display for direct answers, acknowledgements, clarifications, and small edits. Never repeat an unchanged status block merely because another message was sent.
4. When a progress display is useful, put it in a fenced `python-repl` block rather than a Markdown table. Keep it monospaced and aligned, with fixed-width task labels, a ten-character progress field, and a final percentage or status column.
5. Report only active, relevant work. Percentages must be supported by concrete evidence; otherwise use a status such as `RUNNING`, `PENDING`, or `VERIFYING`.

Example:

🙂

```python-repl
TASK                     PROGRESS       STATUS
Video training           [█████-----]   50%
Rebuild verification     [----------]   RUNNING
Colab T4 verification    [----------]   RUNNING
```

Keep any status display compact so that it supports the substantive answer instead of replacing it.
