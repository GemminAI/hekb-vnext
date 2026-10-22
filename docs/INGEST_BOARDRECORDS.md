# Ingesting BoardRecords: value-less cells (board-42-v2)

`scripts/ingest_boardrecords.py` stores a session's BoardRecord cells as an int64 trajectory (one
column per cell). In `board-42-v2` (nvs-kernel) the P15 cell has no value field: its wire value is
`null` and its state is `NOT_MEASURED` (or `ERROR`). `board-42-v1` sent P15 as the number 0.

| Case | Behaviour |
|---|---|
| `board-42-v1`, 38 columns (P15 = 0) | unchanged: 38 columns, same `space_id`, `object_id`s and index as before |
| a value-less cell, default | **IngestError** naming the cell(s) and `--drop-valueless-columns`; nothing is stored |
| a value-less cell, `--drop-valueless-columns` | the column is left out (37 columns); see "What is recorded" |
| no value-less cell in a `board-42-v2` session | ingested as 38 columns; the option changes nothing |
| a cell that is `null` in only some records, a non-integer value, or sessions with different value-less columns | IngestError (never a blob-encoder error) |

## What is recorded when a column is dropped

- the **space fingerprint**: the optional space key `excluded_columns` (`"P15=valueless"`, comma-separated
  and sorted for several) is part of `space_id`; without it a space hashes exactly as before. A 37-column
  trajectory therefore never shares a `space_id` with a 38-column one (`board_format` already differs too);
- the **trajectory metadata**: `ext.hekbv3.ports.excluded_columns` = `[{"port_id": "P15", "reason": "valueless"}]`,
  and `ports.dim` / `ports.port_ids` list only the 37 stored columns;
- `--raw-logs-root` (when given) still carries every port's raw reading, P15 included, in the causal log.

P15 is not part of the experience index's feature (it reads ten named ports: P01 P03 P09 P13 P18 P19 P25 P28
P31 P35), so dropping it changes no index key, distance or neighbour.

## Not decided here

What a missing value means to HEKB in general (a NaN column, a mask, a separate "absent" channel) is a
future, separate decision. This ingest only refuses by default, or leaves the column out and says so.
