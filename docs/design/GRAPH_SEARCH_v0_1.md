# HEKBv3 追補 C（草案 v0.1）— グラフ構造と検索の層

v0（`GRAPH_SEARCH_v0.md`、ad4fc81）は履歴として残す。本書は v0 の §0–§5 の記述を引き継ぎ、**D-69 の確認を反映した改訂点**と、**Q6・Q7 の追加**、**Q1〜Q7 のどこまでを v0.1 で実装するか**を定める。区分: [FACT] 確かめた、[提案] 設計案、[未確認]。

> **D-70 による改訂（本書 §2・§3 Q6–Q8・付録 A）**: ノードの種類を D-69 の `kind`（observation／claim／evaluation）から、Observation・Evidence（HEXT）・Interpretation・Selection（HANKO）・Outcome の 5 種類に改めた。Q2 の縦の一本の実装は変えない（§2.4）。

## 1. D-69 の反映（v0 §6）

| # | D-69 の結果 | v0.1 の扱い |
|---|---|---|
| D-GR-1 | (B) 確認 | `content_id` と辺 `same_content_as` を足す。`object_id` は変えない。`content_id = SHA-256(canonical{schema, space_id, records − trajectory_hash − annotations})`（`graph/content.py`）。実行の識別（`experiment_id`・`run_id`）は `object_id` 側に残る |
| D-GR-2 | (B) 確認 | `space` は変えない。`policy_set` ノード（`window_policy_{name,sha256}`、`time`・`wiring`・`anchor` の指紋）を作り、辺 `measured_under` で Trajectory に結ぶ。窓の方針は `space` に無いので取り込み時に付け、`n == min(token+1, 8)` を全トークンで再計算して `verification` を決める |
| D-GR-3 | (B) 確認 | `source_text` を blob（`hekbv3.source_text/1`: 本文・トークン ID・バイト範囲・文字範囲）とする。MSD0001 v1.1 の `meta.json`＋`tokens.npz` から作り、Locked は ID で除外 |
| D-GR-4 | (B) 確認 | 追記のみ・ハッシュ連鎖の辺のログ（`graph/edges.jsonl`）。ノードも同様（`graph/nodes.jsonl`）。既存の `objects/`・`meta/`・`lineages/`・`blobs/` の欄は書き換えない |
| D-GR-5 | (A) 確認 | 類似は厳密（numpy）。v0.1 では未実装（Q3） |
| D-GR-6 | (A) 確認 | 全文は SQLite FTS5。v0.1 では未実装（Q5） |
| **D-GR-7** | **変更**: 主張の層は作らないが、**全ノードに `kind` と `verification` を最初から持たせる** | §2 |
| D-GR-8 | (A) 確認 | 来歴の追記は Q4 で。v0.1 では未実装 |
| D-GR-9 | (A) 確認 | `set` ノード。v0.1 では未実装（Q1 の縦の一本で作る） |

## 2. ノードの種類（D-70。D-69 の `kind` を置き換える）[提案]

### 2.1 5 種類

| 種類 | 何か | `verification` | 書き換え |
|---|---|---|---|
| **Observation** | 測った・記録した事実（Trajectory・因果ログ・原文・計器の値・環境・実行） | **持たない** | 書き換えない（D-18）。誤りは §2.2 |
| **Evidence**（HEXT） | 観測を根拠として残した文書・要約（`hext_doc`、カーネルの `hext/1.0` の要約を含む） | 持つ | 値の変更は新しいノード＋`supersedes` |
| **Interpretation** | 観測・根拠からの解釈・主張・評価（旧 claim／evaluation）。**採用されなかったものも消さない** | 持つ | 同上 |
| **Selection**（HANKO） | 人または規則が、Interpretation または Action を**選んだ出来事**。Observation は変更しない | 持たない（出来事そのもの） | 追記のみ |
| **Outcome** | 介入・選択の後に観測された結果（分岐が消えた、別の束に移った、等） | 持つ（結果の確かさ） | 追記のみ |

`verification` の値（Evidence・Interpretation・Outcome）: `unverified`／`machine_verified`（何を確かめたかを `attrs.check` に書く）／`human_verified`（Selection への辺を要する）／`contested`／`retracted`（ログからは消さない）。

### 2.2 計器の誤り（D-18 のグラフでの実装）

元の Observation は書き換えない。訂正・再計測は**新しい Observation**として足し、辺 `corrects`（誤りの訂正）または `supersedes`（より新しい計測条件・計器での置き換え）で元の Observation に結ぶ。検索は既定で「訂正されていない最新」を返し、元のノードは辿れる（EXP-C18-RECAL001 の P18 の注釈・`window_policy_sha256` の後付けも、この型で足す）。

### 2.3 Selection

属性: 誰が（人・規則・主体の種類）、いつ、どの条件（`space_id`・方針の指紋・参照した Evidence）と**目的**（Goal Graph の目的・`time_frame`）で選んだか。対象: Interpretation または Action への辺 `selects`。選ばれなかった Interpretation への辺 `considered`（残す）。**Skill への昇格も Selection** として記録する（Q7）。Selection を持たない昇格は書込みで拒否する。

### 2.4 Q2 の実装との関係 [FACT]

実装済みの縦の一本（`graph/`）は、D-69 の `kind`・`verification` を全ノードに持つ（`observation` × `machine_verified`／`unverified`）。**実装は変えない。** 対応: 実装の `node_type` のうち `trajectory`・`causal_log`・`source_text`・`policy_set` は Observation、`hext_kernel` は Evidence に当たる。Observation ノードの `verification` は D-70 では「観測の確かさ」ではなく**取り込み時の検査の結果**（`attrs.check`）であり、将来 `ingest_check` に読み替える（欄の追加のみ。既存の値は変えない）。`hext_kernel` の `verification: unverified` は Evidence の `verification` としてそのまま有効。辺の語彙に `corrects`・`supersedes`・`selects`・`considered`・`outcome_of` を足すのは**後の実装**（本書は設計）。

## 3. 追加の問い合わせ

| # | 問い合わせ | 入力 | 出力 | 合格の条件 |
|---|---|---|---|---|
| **Q6 文脈の組み立て（RAG）** | ある Trajectory／状況に対し、過去の Trajectory・結果を**比較できる条件**で取り出し、プロンプトの文脈として渡す | 対象の `object_id`、`space_id`、方針の指紋、欲しい種類・`verification` の下限 | 同じ `space_id`＋同じ `policy_set` の候補。各項目に種類・`verification`・根拠の辺。**支持する例と、反対・失敗の例の両方**を返す | (1) `space_id` または方針の指紋が違うものは混ぜない（混ぜるときは `cross_space` で明示）。(2) ノードの種類（5 種類）と `verification` を区別して返す。(3) 支持だけでなく反対・失敗の例を含める。(4) **`unverified` の Interpretation・Evidence は根拠として渡さない**（`EVIDENCE_VERIFICATIONS` に入らないものは「参考・未確認」の欄に分け、根拠の欄に入れない）。Observation は `verification` を持たず、事実として渡せる |
| **Q7 Skill へのフィードバック** | 過去の手順・結果を、適用できた条件と失敗した条件を**来歴つき**で取り出し、Skill の改訂の材料にする | 手順（Skill）の識別、期間・条件 | 手順の各回の結果、適用できた条件（空間・方針・入力の種類）、失敗した条件、各項目の来歴（Q4）。Skill への昇格の候補 | (1) 結果と条件は同じ `space_id`／方針の指紋の中で対にして返す。(2) **Skill への昇格は人の承認（HANKO）を要し、Selection として記録する**（§2.3）。(3) 記録は**グラフのノード・辺**（Selection: 誰が・いつ・どの条件と目的で、対象 `node_id`、`selects`／`considered`）。採用されなかった Interpretation も残す。Selection の無い昇格は書込みで拒否する |

| **Q8 情報不足の循環** | 分岐・逸脱の原因として「情報不足」を疑い、足りない情報を足して出し直した結果を残す | 分岐・逸脱の Observation（トークンの位置） | 連鎖（下記）。各段が別ノードで、辿れる | 連鎖が 1 回の問い合わせで返る。**「情報不足が原因だった」は、介入の Outcome でしか確かめられない**（Interpretation の段では仮説のまま。`verification` は Outcome が付くまで `unverified`） |

Q8 の連鎖 [提案]:

1. **Observation**: 分岐・逸脱（トークンの位置、`has_causal_log` の範囲）。
2. **Observation の欄**: その時点で使えた入力・HEXT（Evidence）・目的（Goal Graph、T06 の `time_frame` 等）。**使えなかったもの**は「記録なし」と明示する。
3. **Interpretation**「不足情報の候補」: 候補の内容、根拠の辺、**推定した主体（LLM／人）**を属性に持つ。複数候補を並べ、選ばれなかったものも残す。
4. **Selection**「ユーザーへの確認」: 質問の内容、誰が・いつ・何の条件と目的で選んだか。
5. **Observation**: 不足情報を足して再実行した Trajectory（`content_id` は別。足した情報を入力の指紋に持つ）。
6. **Outcome**: 分岐が消えたか、別の束に移ったか（EXP-HEKB-BUNDLE001 の語）。

## 4. Q1〜Q8 と v0.1 の範囲

| # | v0.1 で実装する | v0.1 で実装しない |
|---|---|---|
| Q1 | カタログ（`space_id`・`policy_set`・`session`・`content_id` で引く。グラフから再構築可）。**縦の一本ではない**（索引のみ）。条件の集合の指紋は未実装 | ラベル・分割・タグでの絞り込み、`set` ノード、集合の指紋 |
| **Q2** | **縦の一本を実装**: P18 の値 → 因果ログ（38 ポート＋`cycle_raw`）→ 原文の文字範囲・トークン ID → `hext/1.0`。48 件で受け入れ | HEXT 文書（`hext_doc`）の保存、blob 全体を読まないトークン単位の取り出し（現状は blob を読む。オフセット表は未実装） |
| Q3 | なし | 類似の索引・API・D-68 の語の記述子 |
| Q4 | `policy_set`（窓・時間・配線・アンカー）まで | モデル `revision`・環境・Pod・実験・コミットのノード |
| Q5 | なし | `hext_doc`、全文・意味の索引 |
| Q6 | 語彙（`kind`・`verification`・`EVIDENCE_VERIFICATIONS`）と、同じ `space_id`／`policy_set` の候補の取り出しの下地（カタログ） | 文脈の組み立て、反対・失敗の例の取り出し（結果・評価のノードが無い） |
| Q7 | Selection の語彙と拒否の規則の定義のみ | 手順・結果のノード、Selection のノード・辺の書込み、昇格 |
| Q8 | なし（設計と付録 A の実験案のみ） | Interpretation・Selection・Outcome の書込み、再実行の連携 |

## 5. 実装（`feature/graph-v0`）[FACT]

- `graph/`: `model`（語彙・検査）、`chainlog`（追記のみ・ハッシュ連鎖）、`store`（ノード・辺）、`content`（`content_id`）、`source_text`、`ingest`、`catalog`、`trace`（Q2）、`api`（`GET /graph/q2`、読み出しのみ）。
- `tools/graph_select.py`（選び方を先に記録）、`tools/graph_ingest.py`（新しい data dir へ取り込み）、`tools/graph_accept.py`（受け入れ）。
- 辺: `has_causal_log`（トークン範囲）、`has_source_text`（トークン範囲・文字範囲）、`token_hext`（因果ログのトークン → `cycle_raw.hext`、`hext/1.0`）、`measured_under`、`same_content_as`（`a < b` で 1 本）。
- 既存の欄・`object_id`・blob は変えない。既存コードの変更は `store_service.py` への `include_router` の 1 行のみ（追加）。

## 6. 未確認のこと

- `hext/1.0` はカーネルの要約（`cycle_raw.hext`）で、HEXT Standard 1.0.0 の文書ではない。`token_hext` の先は `hext_kernel` ノード（`verification: unverified`: 独立の記録が無い）。
- 原文の文字範囲は、バイト単位の BPE を UTF-8 として復号したもの。`generated_text` はトークナイザーで整形されており、空白を除いて一致を確かめる（`machine_verified` の条件）。多バイト文字の途中で切れるトークンの文字範囲は空になり、文字は完成させたトークンに帰属させる。
- 窓の方針（`window-8-v1`）は元のデータに保存されていない。`n == min(token+1, 8)` の再計算は容量 8 の確認であり、方針の同一性の証明ではない。
- 辺のログの規模: 48 件でノード 8,636・辺 8,635 行（`hext_kernel` が 1 トークン 1 件）。1,296 件では約 27 倍（約 23 万行ずつ）と見積もるが、実測していない。

## 付録 A. Q8 を確かめる実験の設計案（実装しない）[提案]

型は ONSET002 の分岐点の再サンプリングと同じ（分岐点を固定し、条件を 1 つだけ変えて出し直し、分岐の出方を比べる）。**変えるのは乱数ではなく、与える情報**である。ONSET002 のデータ・事前登録には触れない。

### A.1 生成なしの探索（既存の MSD0001、Mac）

- 問い: recall/hard と contradictory とで、分岐・逸脱の**起き方**が違うか。生成は行わず、非 Locked の既存の因果ログ（INFO001）と TRACE001 の分岐の位置だけを使う。
- 比べるもの: 分岐の位置（プロンプト末尾からのトークン数）、分岐前後の P02・P03・P18 の変化、その時点で入力に含まれていた情報（recall は想起すべき事実の有無、contradictory は矛盾する指示の対）。
- 限界: 因果の向きは言えない。「不足していた」の判断は記述にとどまり、**Outcome の無い Interpretation**である。

### A.2 介入（RunPod の段階）

- 設計: 同じセッションの分岐点（トークン位置 t*）までを固定し、(a) 元のまま、(b) 不足情報の候補を 1 件ずつ足した入力、(c) 無関係な情報を同じ長さだけ足した対照、の 3 条件で出し直す（乱数の種は揃える）。
- 比べるもの: 分岐が消えたか、別の束に移ったか（BUNDLE001 の語）。(c) でも変わるなら、効果は情報の中身ではなく入力長の影響と読む。
- 事前に登録するもの: 候補の作り方（LLM／人の別を記録）、成否の定義、対象の件数、停止の条件。結果は Outcome ノードとして記録し、`verification` は (c) との比較の後に付ける。
- 前提: GPU の Pod が要る（現在の Pod は終了済み）。本書では実行しない。

### A.3 未確認のこと

- 「不足情報の候補」を LLM が出す場合の、候補の質の評価方法。
- T06（`time_frame`）・Goal Graph のどのノードを「目的の欄」の正本とするか。
- Outcome の「別の束に移った」の定義は BUNDLE001 の探索の語であり、判定の基準ではない。
