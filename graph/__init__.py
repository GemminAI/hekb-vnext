"""HEKBv3 graph layer (GRAPH_SEARCH v0.1, ``docs/design/GRAPH_SEARCH_v0_1.md``).

Adds nodes and edges *beside* the existing content-addressed store: nothing here changes an
``object_id``, a blob, or a field of ``objects/*.json``. The graph is the source of truth for
relations; the catalog is a derived index that can be dropped and rebuilt from it.
"""
