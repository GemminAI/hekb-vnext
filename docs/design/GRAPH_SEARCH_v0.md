# HEKBv3 追補 C（草案）— グラフ構造と検索の層

Doc ID: `HEKBV3-ADDENDUM-C-GRAPH-SEARCH-v0`　状態: **草案（提案）。実装しない。既存のコード・データ・スキーマは変えない。**
作成: 2026-10-14（HEKB-GRAPH001）。前提: HEKBv3 改修設計書 v0.1・追補 A・追補 B、HEKBv3 × HEXT 連携設計書 v0.1、
Decisions D-44・D-45・D-47・D-62・D-64〜D-68。

ラベル: **[FACT]** 今回コード・データで確かめた / **[提案]** 設計の案 / **[Tomo の判断待ち]** §6。

## 0. 目的と方針

今日の作業で実際に必要になった 5 つの問い合わせ（§2）を**受け入れの試験**として先に置き、今の hekb-vnext に何が足りないかを
示す。設計の骨格は、**グラフを正本**とし、識別子・条件の索引、グラフの走査、全文の索引、類似の索引を**グラフから作り直せる
派生物**として持つこと。

## 1. 今の状態 [FACT]（hekb-vnext `feature/space-policy-fingerprints` の `6d5fd0b`、INFO001 の data dir）

### 1.1 ノード

| ノード | 置き場所 | 識別子（決め方） | 主な内容 |
|---|---|---|---|
| 経験（Experience） | `objects/<object_id>.json` | `object_id = SHA-256(canonical(payload))`（`store_service.py:491`、`vendor/hekb_vnext/core/object.py:49-51`）。payload の予約キーは 7 つ（`:238-257`、`extra="forbid"`） | `schema`・`raw_input`・`resolved_meaning`・`action`・`title`・`tags`・`ext`。軌跡は `ext.hekbv3`: `kind="trajectory"`、`origin`、`space`（空間の指紋）、`ports`・`trajectory`・`causal_log` の blob 参照、`labels`、`outcome` |
| 系譜（Lineage） | `lineages/<lineage_id>.json` | 最初の経験の `object_id` | 経験の `object_id` の並び |
| 射（Morphism） | `relations/<morphism_id>.json` | `branch()` が作る | `dom`・`cod`・`label` |
| 副情報（Sidecar） | `meta/<object_id>.json` | `object_id` | `recall_count`・`archived`・`provenance`（ハッシュの外） |
| Blob | `blobs/<aa>/<bb>/<sha256>` | 内容の SHA-256（`PUT /v3/blobs/{sha}`、不変） | `ports`（盤面の量子化値）、`trajectory`（64 次元の位置、任意）、`causal_log`（トークンごとの 38 ポートの生値と `cycle_raw`） |
| 空間（Space） | 経験の中に埋め込み（独立のノードではない） | `space_id = BLAKE2b-256(9 キー)`（`index/space_id.py:18-28`） | `model_ref`・`layer`・`projection_sha256`・`w_sha256`・`anchor_policy_sha256`・`wiring_policy_sha256`・`time_policy_sha256`・`board_format`・`quant_step_log2` |
| 索引 | `index/<space_id>/experience/{keys,ids,meta}`（`index/experience_index.py:101`）、`hekb_vnext_index.db`（`semantic_addresses`、`index/semantic_index.py`） | space ごとに分割 | 10 ポート × 8 トークンの窓の特徴（80 次元）。旧系統の `semantic_addresses`（`/trajectories/recalibrate`）は別の旧い軌跡ストア用 |

因果ログの中（1 トークン = 1 レコード）: `token_index`・`seq`・`trajectory_hash`・`ports{P01..P38}`（`capability`・`status`・`result`・
`input_hash`・`output_hash`・`event_id`・`parent_event_id`・`drive_digest`・`annotations`）・`cycle_raw`（`belief`・`geometry`・`risk`・
`control`・`hext` ほか）。

### 1.2 エッジ

| エッジ | 何がどのキーで結ばれているか | 備考 |
|---|---|---|
| 親 → 子（系譜） | `parent_id` ＋ `relations/` の `dom`→`cod`（`label`＝`op`）。`GET /graph/topology`（`store_service.py:679`）の `lineage_parent` | 1 親 1 子の木 |
| 経験 → blob | `ext.hekbv3.{ports,trajectory,causal_log}.blob_sha256` | 内容アドレス。`causal_log` は経験の `object_id` を決める |
| 経験 → 空間 | `ext.hekbv3.space.space_id`（埋め込み。再計算で検証、`:1109-1150`） | 空間は値の組で、ノードではない |
| ポートの連鎖（トークン内） | `parent_event_id` → `event_id`（38 ポートの鎖）。鎖の根は直前サイクルの `final_hext_object_id` または `session:<id>:no-prior-cycle` | 因果ログの中だけ |
| 経験 → 経験（利用者作成） | `POST /morphisms`（`:873`）、`/experience/{id}/closure`（`:823`）、`/graph/geodesic`（`:912`） | 意味の閉包。INFO001 の data dir では 0 件（`relations/` が空） |
| 経験 → セッション（文字列） | `title`・`raw_input = "edge-hekb-slice session=<id>"`（`nvs_runtime/integrations/hekb_vnext.py:245`） | 型のあるエッジではなく**文字列** |

### 1.3 確かめた 4 点

| 点 | 結果 | 根拠 |
|---|---|---|
| (a) テキスト・トークン ID | **HEKB に保存されていない。** 因果ログの `cycle_raw.observations[].text` は空文字（Edge が `text: ""` で送る、`nvs_edge/edge.py:310`）で、`digest` のみ。`raw_input` は文字列の札。原文・トークン ID は MSD0001 の `sessions/<id>/meta.json`（`generated_text`・`messages`・`prompt_id`）と `tokens.npz`（HEKB の外）にあり、`session_id` の**文字列の一致**でしか辿れない。連携設計書 §3.2・§3.4 の `ext.hekbv3.text.hext_blob_sha256` は**未実装**（`store_service.py`・`index/` に `text`・`hext_blob_sha256` が無い） | `store_service.py:238-257, 1109-1236`、`nvs_edge/edge.py:310`、MSD0001 `meta.json` |
| (b) HEXT の中身 | **カーネルの `hext/1.0` の要約だけ。** `cycle_raw.hext` = `{basin, object_id, risk, risk_state, schema_version: "hext/1.0", tags, tier, timestamp_ns}` で、因果ログ blob の中にのみ存在する。hekb-vnext の Python（vendor・tests を除く）に HEXT の語が無い。HEXT Standard 1.0.0 の文書（`kind: "hext_doc"`）も、HEXT v2 も、保存されていない（`hext_doc` が無い）。「HEXT」の 3 つの意味の衝突は Lineage-HEKB #10 | `grep` 結果 0 件、INFO001 の因果ログ |
| (c) `trajectory_hash` と `experiment_id` | **入っている。同じ内容が別の `object_id` になる。** `trajectory_hash = hash{experiment_id, run_id, chain_root, 38 ポートの event_id}`（`nvs_kernel/api/routers/experiment.py:319-326`）→ 各トークンのレコードに入り（`nvs_edge/causal_log.py:51-57`）→ `causal_log` blob の sha が payload に入り → `object_id`。実測（同じセッション p11-jailbreak-1737 を、`experiment_id` だけ変えて別の新しい hekb-vnext で 2 回）: **`ports` blob と `space_id` は一致、`causal_log` blob と `object_id` は不一致**。**意図の記録**: `trajectory_hash` が `experiment_id`・`run_id` を含むことは事実として記録（HANDOFF 10-03 v4 L41、v0.6 L35）、`TrajectoryIdentityResponse` の docstring（`schemas.py:436-442`）は「どの run が作ったかを辿る」ためと述べる。「同じ内容が別の `object_id` になる」ことを意図したとする記録は**見つからない（意図の記録なし）** | 上の行、`/tmp` の 2 回の実測（`EXPIDTEST-A-`・`-B-`） |
| (d) `window_policy_sha256` | **`space` に無い。** `REQUIRED_KEYS` は 9 キーで窓の方針を含まず、`SpaceIn` は `extra="forbid"`（`index/space_id.py:18-28`、`store_service.py:1058-1068`）。カーネルは応答に出し、Edge は毎回照合するが、保存されない（EXP-C18-RECAL001 `INSTRUMENT_CALIBRATION.md`） | 同上 |

補足 [FACT]: 同じ 1,296 セッションを 2 つの data dir（INFO001 と EXP-C18-RECAL001）に測ると、`space_id` は 1,296/1,296 で一致、
`ports` blob は 1,296/1,296 で一致、`causal_log` blob と `object_id` は 0/1,296 で一致（実験 ID の違いと P18 の `annotations` の追加が
重なる）。

## 2. 受け入れの試験（問い合わせ）— 先に書く

各試験は「入力 → 期待する出力 → 合格の条件」。今日の作業で手作業だったものを、1 回の問い合わせにする。

| # | 問い合わせ（今日の実例） | 入力 | 期待する出力 | 合格の条件 |
|---|---|---|---|---|
| **Q1 識別子・条件** | 条件の揃った Trajectory の集合を取り出す（EXP-C18-RECAL001 の fit／dev、INFO001 の 1,296） | `space_id`（または空間の全フィールド）、方針の指紋（`wiring`・`time`・`window`・`anchor`）、ラベル・分割・タグの条件 | `object_id` の一覧（ソート済み）、`space_id`・方針の指紋・件数・集合の指紋（`sha256`） | 1 回の問い合わせで返り、全データの走査を要しない。同じ条件で同じ指紋。`window_policy_sha256` が違う Trajectory は混ざらない |
| **Q2 グラフ** | あるトークンの P18 の値 → そのトークンの因果ログ → 原文 → 根拠の HEXT（PORT38-TRACE001 が手作業でやった） | `object_id` と `token_index`（または P18 の値の条件） | そのトークンの 38 ポートの生値と `cycle_raw`、原文（プロンプト・生成文・そのトークンの位置）、根拠の HEXT 文書または要約 | 1 回の問い合わせで、blob 全体（約 13 MB）を読まずに当該トークンの記録が返り、原文と HEXT へのエッジがたどれる |
| **Q3 類似** | 同じ space で、ある Trajectory と似た変動・逸脱・分岐を示す過去の Trajectory を探す（D-68 の語。BUNDLE001 の問い） | `object_id`、特徴の種類（窓の 10 ポート／V の系列／位置の経路）、基準の集合（Q1 の集合）、`k` | 近い順の `object_id`、距離、記述子（V の位置づけ、急変の位置、束・分岐の情報）。D-68 の語で「変動／逸脱／分岐」を区別して返す | 同じ `space_id` の中だけで検索する。空間をまたぐときは明示の引数が要る。基準の集合が指定できる |
| **Q4 来歴** | ある Trajectory を生成したモデル・環境・実験・Pod まで遡る（RUNPOD-INVENTORY001 の手作業） | `object_id` | モデル（`revision`）、`torch`・`transformers`・`dtype`、GPU の機種、Pod ID、生成設定（T・top_p・種）、実験 ID・コミット、根拠の種類（記録・推定） | 各項目が記録または「記録なし」として返り、推定には根拠（Pod ID 一致／日報／日時）が付く |
| **Q5 意味・全文** | 同じ主張や状況を含む過去の HEXT を探す | 文字列または HEXT 文書、`space` の条件（任意） | HEXT 文書と、それが根拠づける Trajectory（範囲）の一覧、一致の箇所 | 語の一致（全文）と意味の近さ（類似）の両方で返る。1 つの HEXT が複数の Trajectory を根拠づける関係が保たれる |

## 3. 今の状態で足りないもの（試験ごと）

| # | 今できること | 足りないもの | 足すもの [提案] |
|---|---|---|---|
| Q1 | `POST /v3/index/{space_id}/build`（`:1249`）で space ごとの窓の索引。`space_id` の再計算 | 条件（方針の指紋・ラベル・分割・実験）で引く**カタログの索引**。ビルドは全 `objects/` の走査（`experience_index.py:69`）。`window_policy_sha256` が `space` に無い（§1.3-d）。集合（`kind: "set"`）が無い（連携設計書 §3.5） | カタログ（§4.2 I1）と、方針の集合を持つ空間（§6 D-GR-2）。`set` ノード（メンバー・規則・指紋・`space_id`） |
| Q2 | `GET /v3/blobs/{sha}` で因果ログ全体（JSON）を取り、手で当該トークンを探す | トークン単位の取り出し、**原文のノード**（§1.3-a）、HEXT のノード（§1.3-b）、トークン範囲 → HEXT の根拠の辺 | `source_text` ノード（プロンプト・生成文・トークン ID・トークンのオフセット）、`hext_doc` ノード、辺 `derived_from`・`evidence_of`、トークンの索引（blob 内のオフセット表）、走査の API（§4.3） |
| Q3 | `POST /v3/compose`（`:1318`）: 10 ポート × 8 窓の特徴 80 次元の Softmax・カーネル合成。カテゴリーの事後確率と値の合成を返す | 「似た Trajectory の一覧」を返す API、特徴の種類の登録（V の系列・位置の経路・始まり／終わりの群）、基準の集合の指定、D-68 の語の記述子、束・分岐の記述子 | 特徴の仕様（名前・版・パラメータの指紋）を持つ類似の索引（§4.5）、`similar` の API、記述子の計算（EXP-HEKB-BUNDLE001 の定義） |
| Q4 | `space.model_ref`（文字列 `meta-llama/Llama-3.1-8B-Instruct`）のみ。`revision`・`torch`・`transformers`・`dtype` は MSD0001 の `meta.json`（HEKB の外）。GPU・Pod は**どこにも保存されていない**（RUNPOD-INVENTORY001: Pod ID の直接の痕跡は 70 台中 1 台、他は日報・日時の推定） | 環境・実行・データセットのノードと辺。実験 ID・コミットの記録 | `environment`（モデルの `revision`・版・精度・GPU・Pod）、`run`（`experiment_id`・`run_id`・nvs-kernel／nvs-runtime のコミット・方針の指紋）、`dataset`（MSD0001 の INDEX の sha256）ノードと辺 `generated_by`・`run_of`。旧データは台帳から**根拠つきで**追記 |
| Q5 | `/experience/{id}/closure`（意味の閉包）は利用者が作った射の上のみ。全文の索引なし | HEXT 文書の保存、主張の層（追補 B）、全文・意味の索引 | `hext_doc`（連携設計書 §3.1）、`claim`（追補 B §2.3）、全文の索引（§4.4 I3）、意味の索引（§4.5 の特徴の 1 つ） |

## 4. 設計の草案 [提案]

### 4.1 正本はグラフ

- **ノード**: 経験（trajectory）、blob（`ports`・`trajectory`・`causal_log`・`source_text`・`hext`）、`source_text`、`hext_doc`、
  `claim`、`set`、`environment`、`run`、`dataset`。ノードは内容アドレスで不変（既存の `object_id`・blob の sha をそのまま使う）。
- **エッジ**: 型つき・方向つき・**追記のみ**の別の記録（辺のログ、ハッシュの鎖）。ノードの payload に埋め込まない（既存の
  `object_id` を変えない）。属性（トークン範囲・確度・出典）を持てる。多対多を許す: 1 つの HEXT → 複数の Trajectory（範囲つき）、
  1 つの Trajectory → 複数の HEXT、1 つの `source_text` → 複数の `claim`。
- **辺の型**（最小）: `parent`（既存）、`measured_in`（経験 → 空間）、`has_blob`、`derived_from`（経験 → `source_text`）、
  `evidence_of`（`hext_doc`・`claim` → 経験、トークン範囲つき）、`member_of`（経験 → `set`）、`generated_by`（経験 → `run` → `environment`）、
  `same_content_as`（経験 ↔ 経験、`ports` blob の一致）。
- **時間は三つを区別**（追補 B）: 観測の時間（順序づけに使う単調な時計。Edge の論理時刻）、主語の時間（区間の属性。非単調）、
  歴史の時間（区間の属性）。OS の壁時計（`created_at`）は意味の時間ではなく、台帳の事実として持ち、ハッシュにも検索の鍵にも
  使わない（HEKB14 の原則）。

### 4.2 派生物（グラフから作り直せる。壊れたら再構築）

| # | 索引 | 鍵と内容 | 再構築の入力 |
|---|---|---|---|
| I1 | カタログ（識別子・条件） | `object_id` → `space_id`・方針の指紋・`session_id`・`experiment_id`・`run_id`・ラベル・分割・タグ・件数・`created_at`（台帳）。SQLite | `objects/` と辺のログ |
| I2 | 走査（グラフ） | 隣接表（辺の型別）。トークンの索引（`causal_log` blob の中のレコードのオフセット） | 辺のログと blob のヘッダ |
| I3 | 全文 | `source_text`・`hext_doc` の本文（SQLite FTS5。日本語は trigram）。ノードと範囲への逆引き | `source_text`・`hext_doc` の blob |
| I4 | 類似（`space_id` ごと） | 特徴の仕様（名前・版・パラメータの指紋 `feature_spec_sha256`）ごとの特徴ベクトル。まず厳密（numpy）、規模に応じて近似 | 因果ログ・`ports`・`trajectory` の blob と特徴の仕様 |
| I5 | 集合 | 条件（宣言的・幾何的・混合）の結果の凍結 | I1・I4 とノードの規則 |

- 各索引の `meta` に入力の指紋（`index_sha256`）、仕様の版、構築の日時を持ち、入力が変われば古い索引を使えない。
- **類似の索引は `space_id` ごとに分ける**。空間をまたぐ検索は、`cross_space=true` と空間の一覧（と理由）が明示されたときだけ許し、
  結果に各項目の `space_id` を付ける（H5「`space_id` が違うものは合成しない」を守る）。

### 4.3 問い合わせの API [提案。追加のみ]

| 問い合わせ | 概要 |
|---|---|
| `GET /v4/catalog?space_id=&policy.*=&label=&split=&tag=` | I1。`object_id` の一覧と指紋（Q1） |
| `GET /v4/trace?object_id=&token=&fields=ports.P18,cycle_raw,text,hext` | I2。1 回の走査でトークンの記録・原文・HEXT へ（Q2） |
| `POST /v4/similar` `{object_id, feature_spec, baseline_set, k, cross_space?}` | I4（Q3）。D-68 の語の記述子つき |
| `GET /v4/provenance?object_id=` | `generated_by` の連鎖（Q4）。各項目に基づく（記録／推定） |
| `POST /v4/search` `{text|hext, mode: fulltext|semantic, space?}` | I3・I4（Q5） |

### 4.4 識別の層を分ける

`object_id`（実行の識別を含む）はそのまま。**内容の識別**（`ports` blob の sha ＝ 値だけ）を `content_id` として辺 `same_content_as`
で明示し、異なる実験が同じ内容を測ったことを検索で扱える（§1.3-c）。

### 4.5 実装の順（段階 [提案]）

1. 辺のログと I1（カタログ）。既存の経験から全再構築できること。
2. `source_text`・`environment`・`run` ノードの取り込み（旧データの追記は「推定」の根拠つき）。I2（トークンの索引）。
3. I3（全文）。`hext_doc` の取り込み（連携設計書 §3.1）。
4. I4（類似）：まず窓の 10 ポート（既存の `compose` と同じ特徴）→ V の系列・位置の経路。`set` ノード。
5. 主張の層（追補 B）と Q5 の意味検索。

## 5. 各問い合わせの受け入れの検査 [提案]

| 試験 | 検査 |
|---|---|
| Q1 | INFO001 の 1,296 件で、`space_id` ＋方針の指紋の問い合わせの結果が、全走査の結果と一致。`window_policy_sha256` の異なる 2 件が混ざらない |
| Q2 | TRACE001 の 37 件のうち 5 件を、手作業の結果と一致させる（P18 の値・因果ログ・原文・HEXT） |
| Q3 | BUNDLE001 の束の群の同じ始まりの群から、近い順の一覧が取れ、`space_id` が違う Trajectory は返らない（`cross_space` を付けない限り） |
| Q4 | MSD0001 の 2,000 セッションのうち非 Locked 1,296 について、`revision`・`torch`・`transformers` が返り、GPU・Pod は「記録なし」か推定（根拠つき）で返る |
| Q5 | 同じ文を含む 2 件の HEXT から、両方が引ける。1 つの HEXT が 2 つの Trajectory を根拠づける辺が保たれる |

## 6. Tomo の判断待ち（選択肢と推奨）

| # | 決める点 | 選択肢 | 推奨 |
|---|---|---|---|
| D-GR-1 | 同じ内容を別の実験で測ったとき `object_id` を分けるか（§1.3-c。意図の記録なし） | (A) 今のまま（実行を区別）。(B) 今のまま＋内容の識別 `content_id` と辺 `same_content_as` を足す。(C) `trajectory_hash` から `experiment_id` を外し、内容だけで同一にする（カーネル・基準線の変更を伴う） | **(B)**。既存の ID・基準線を変えず、検索で内容の一致を扱える |
| D-GR-2 | 窓の方針（と他の方針）を `space` に入れるか（§1.3-d、D-65 と同型） | (A) `window_policy_sha256` を `space` に足す（全 `space_id` が変わる。旧データは明示の記号で取り込む）。(B) `space` は変えず、方針の集合を別ノード（`policy_set`）として辺で結ぶ。(C) 入れない | **(B) を第一**（`space_id` を安定させ、後から (A) に移れる）。旧データの扱いを (A) と同じ明示の記号に |
| D-GR-3 | 原文の保存の形（§1.3-a、連携設計書 §3.4 の未確認） | (A) HEXT 文書として保存（原文を入れる要素が未確認）。(B) まず `source_text`（プロンプト・生成文・トークン ID・オフセット）を blob とし、HEXT は根拠として別に足す。 | **(B)**。Q2 を先に満たし、HEXT の要素の未確認に依存しない |
| D-GR-4 | 辺の置き場所 | (A) payload に埋め込む（`object_id` が変わる）。(B) 別の追記のみの辺のログ。(C) `relations/` を拡張 | **(B)**。既存の経験を変えない |
| D-GR-5 | 類似の索引の技術 | (A) 厳密（numpy）のみ。(B) 近似（HNSW 等）。(C) 規模で切替 | **(A) から始め**、10^5 件を超える見通しで (C) |
| D-GR-6 | 全文の索引 | (A) SQLite FTS5（trigram）。(B) 別の検索エンジン | **(A)**。依存を増やさず、グラフから作り直せる |
| D-GR-7 | 主張（claim）の層の範囲（追補 B §6） | (A) この草案に含めず、Q5 は HEXT の本文の全文・意味に限る。(B) 含める | **(A)** |
| D-GR-8 | 旧データの来歴の追記 | (A) RUNPOD の台帳から「推定」の根拠つきで `run`・`environment` に追記。(B) 新しいデータだけ | **(A)**（根拠の列を必須に） |
| D-GR-9 | 基準の集合（`set`）をどこに置くか、D-68 の語の API 出力への反映 | (A) `set` ノード（連携設計書 §3.5）。(B) 索引のみ | **(A)** |

## 7. 未確認のこと

- HEXT Standard 1.0.0 のどの要素に原文を入れるか（連携設計書 §1.3 の未確認）。
- トークン単位の取り出しの方式（blob の分割／オフセット表）と、既存の 13 MB の JSON blob の読み出しとの互換。
- 辺のログの規模（Trajectory 1 件あたりの辺の数）と、SQLite の上限。今回は見積もっていない。
- `nvs-kernel` のコミットと環境（GPU・Pod）を、生成の時点でどこに記録するか（EXP-HEKB 側の取り込みの設計）。
- Q3 の記述子（V の位置づけ、急変、束・分岐）の定義と閾値は、EXP-HEKB-BUNDLE001 の探索の結果であり、判定に使う基準ではない。
- 旧系統（`/trajectories/recalibrate`、`semantic_addresses`）と v3 の索引の関係。今回は v3 のみを対象にした。
