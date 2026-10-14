# GRAPH_SEARCH_v0 追補 — 検証された関係（Verification）と未解決の逆引き

- 版: 0.1（提案）。対象: `GRAPH_SEARCH_v0.md`（`ad4fc81`）。前提: D-69（グラフが正本）・D-70（ノードの種類。Observation は書き換えない）・D-71・D-72（HEXT v2）。
- 位置づけ: **新しい Graph を作らない。** 既存のグラフ層（`graph/`、追記のみ・ハッシュ連鎖のノードと辺のログ）に、ノードの型 `verification`・`raw_log`・`hext_document` と辺 `verification_input` を**足す**。既存のノード・辺・欄は変えない。
- 区分: [FACT] 今回確かめた／[提案]／[未確認]。
- 既存との関係 [FACT]: `GRAPH_SEARCH_v0.md` には Verification 相当のノード・辺は無い（検索した）。`GRAPH_SEARCH_v0_1.md`（`feature/graph-v0`）はノードに**欄**としての `verification`（`unverified`／`machine_verified`…）を持たせる。本追補はそれを置き換えず、**関係の検証を別のレコード**として足す（欄は、そのノード単体の検査の状態。Verification は 2 つの記録の間の関係の検査）。

## 1. 認識上のステータス（3 つ）[提案]

ステータスは「データの種類」ではなく、**認識上の位置**である。

| ステータス | 意味 | 例（本件） |
|---|---|---|
| `observed` | 計器・記録がそう記録した。書き換えない（D-70） | 生ログ、保存された因果ログ、軌跡の経験、因果ログ内の `hext/1.0` の参照 |
| `derived` | 機械が観測から**導出**した（原資料そのものではない） | 生ログと MSD0001 の原文から生成した HEXT v2 の `source` 文書、索引、要約 |
| `verified_relation` | 2 つの記録の間の**関係**を、規則と入力を明示して検査した結果 | 本追補の Verification |

- **導出された記録が検証されても、`observed` には変わらない。** 検証の結果は別の `verified_relation` の記録になり、元の記録は `derived` のまま。
- `verified_relation` は「関係の確認」であって、記録の真偽・意味の同一・因果関係を保証しない。
- 既定（欄が無いノード）: `trajectory`・`causal_log`・`source_text`・`policy_set`・`raw_log`・`hext_kernel` は `observed`、`hext_document` は `derived`、`verification` は `verified_relation`。

## 2. Verification（検証された関係）[提案。実装: `graph/verification.py`]

| 欄 | 内容 |
|---|---|
| `verification_id` | 規則・入力・主語・結果の正準化の SHA-256（`ver:…`）。時刻を持たない（同じ入力・規則なら同じ ID） |
| `epistemic_status` | 常に `verified_relation` |
| `rule` | `{id, version, sha256}`。規則の本文のハッシュ |
| `a`・`b` | 関係づける 2 つの記録のノード ID（`b` は無くてもよい） |
| `inputs[]` | `{role, sha256}`。**生ログ・HEXT 文書・因果ログ**など、検査に使った入力の SHA-256（必須） |
| `status` | `VERIFIED`／`FAILED`／`NOT_EVALUATED`／`UNRESOLVED` |
| `verified_means` | 何を確認したか（識別子の一致など）。`VERIFIED` では必須 |
| `detail`・`reason` | 件数・位置・理由。`FAILED`・`NOT_EVALUATED`・`UNRESOLVED` では理由が必須 |

- **保証しないこと**: 意味が同じこと、因果関係、真実性。`VERIFIED` は「`verified_means` の条件が、`inputs` に対して成り立った」だけを意味する。
- **D-70 との整合**: 検証は既存のノードを変更しない（主語のノードが無ければ書込みを拒否し、検証が Observation を作ることもない）。訂正は `corrects`／`supersedes` の辺（本追補では足さない）。**採用されなかった Interpretation も消さない**（ログは追記のみで、削除の操作が無い）。
- ステータスの意味: `VERIFIED` 条件が成立／`FAILED` 規則は動いたが条件が不成立／`NOT_EVALUATED` 規則を動かせない（必要な入力が無い。理由つき）。

## 3. UNRESOLVED の定義 [提案]

**参照先が存在することを確認できない**こと。`FAILED`（参照先は見つかったが条件が不成立）でも、「参照先は存在しない」という主張でもない。**探索の範囲（`search_scope`）を記録に含める**。

- 本件 [FACT]: 因果ログ内の `hext/1.0` の参照（48 セッション・13,077 件）は、カーネルの wire object（BLAKE2b でアドレスされる意味状態のスナップショット。HEXT Standard の文書ではない）の ID で、**範囲内（保存された経験の `object_id`・系譜・blob、生成した HEXT v2 文書の `object.id`・`state_hash`・blob）に実在する文書を確認できない** → すべて `UNRESOLVED`。
- **識別子の一致は別の関係**として `VERIFIED` に記録する（`R-VER-ID-HEXT10-01`: `cycle_raw.hext.object_id` ＝ P01 の連鎖の根 ＝ `cycle_provenance` の `hext_object_id`）。つまり「参照は一貫しているが、指す先の文書は未確認」。
- **逆引き**: `UNRESOLVED` の記録から、「この ID を参照している記録（セッション・トークン）」の一覧を、グラフから作り直せる索引として出す（`unresolved_reverse_index`）。

## 4. 新しく生成した HEXT の位置づけ [提案。依頼文の末尾が切れていたため、前段の原則から書いた]

- HEXT-V2-GEN001 の 48 件の `source` 文書は、保存済みのファイルから機械が生成したもので、**`derived`**。原資料（生ログ・MSD0001 の原文）が `observed`。
- 文書と軌跡・因果ログ・生ログの関係は、Verification として別に記録する（`R-VER-DOC-TRAJ-01`、`R-VER-DOC-CL-01`、`R-VER-RAW-01`）。文書の `state_hash` や HEKB の `object_id` は別物で、置き換えない。
- 文書に `causal_log_ref` が無い（仕様に単位が無い）ため、`R-VER-DOC-CL-01` は `NOT_EVALUATED`（理由つき）。

## 5. 規則（ID・版）

| ID | 版 | 内容 |
|---|---|---|
| `R-VER-RAW-01` | 1 | 因果ログ ← 生ログ。ファイルの SHA-256、各レコードの行の SHA-256 と行番号が `provenance.raw_ref` と一致 |
| `R-VER-DOC-TRAJ-01` | 1 | HEXT 文書 ↔ 軌跡。`observation_refs` のセッション・範囲と `space_id` が一致し、文書がレベル 2 で検証できる |
| `R-VER-DOC-CL-01` | 1 | HEXT 文書 ↔ 因果ログ（`causal_log_ref`）。無ければ `NOT_EVALUATED` |
| `R-VER-ID-HEXT10-01` | 1 | `hext/1.0` 参照の識別子の一致 |
| `R-VER-REF-RESOLVE-01` | 1 | `hext/1.0` の ID が、範囲内の保存された文書の識別子に解決できる |

## 6. プロトタイプの結果（48 件。読み取り専用）[FACT]

`tools/graph_align001.py`。出力は `experiment-evidence/GRAPH-ALIGN001/`（グラフは新しい data dir `graph_data/`）。元の data dir は前後のダイジェストが一致（変更なし）。

| 規則 | 状態 | 件数 |
|---|---|---|
| `R-VER-RAW-01` | VERIFIED | 48 |
| `R-VER-DOC-TRAJ-01` | VERIFIED | 48 |
| `R-VER-DOC-CL-01` | NOT_EVALUATED | 48 |
| `R-VER-ID-HEXT10-01` | VERIFIED | 13,077 |
| `R-VER-REF-RESOLVE-01` | UNRESOLVED | 13,077 |

Verification 記録 26,298 件、グラフ（本体）: ノード 39,519・辺 52,596。HEXT の射影は別のログ（文書 48・`observation_ref` の辺 48・`unresolved` 0）。同じ入力で 2 回作って、ノード・辺のログの連鎖の先頭が一致した。

## 7. 未確認

- `hext/1.0` の参照が、ディスク外（VM の稼働中の 8300、他のマシン）にある文書を指す可能性。探索範囲は本書の `search_scope` のとおりで、「存在しない」とは言っていない。
- Verification を HEKB の API（`PUT/GET`）に載せる形式。本追補は読み取り専用のプロトタイプまで。
- `GRAPH_SEARCH_v0_1.md` のノードの `verification` 欄との統合（欄を Verification の集約として導出するか）。

## 8. HEXT v2 JSON → グラフの射影 [提案。実装: `graph/hext_projection.py`]

**対象**: HEXT 文書（D-70 の Evidence）と、その `claims`・`relations`・`observation_refs`・`provenance` だけ。**Trajectory・因果ログ・Interpretation・Selection・Outcome を HEXT から作らない。** 射影は導出された（`derived`）層で、HEXT の JSON から**いつでも作り直せる**。Verification など HEXT 由来でない記録は別のログ（本体のグラフ）にあり、射影の作り直しの影響を受けない。

### 8.1 写像表

| HEXT のフィールド | グラフ | ノード ID・属性の導き方 |
|---|---|---|
| 文書全体 | ノード `hext_document`（`d70_kind=Evidence`、`epistemic_status`＝導出なら `derived`） | ID ＝ `"hextdoc:" + final_seal.state_hash`。属性: `hext_object_id`（`object.id`）、`state_hash`、`hekb_object_id`（HEKB の `object_id`、あれば）、`blob_sha256`（文書のバイト列の SHA-256）、`validated_level`、`hext_kind`、`object_type`、`lifecycle`、`sealed_at` |
| `object.id` | 属性 `hext_object_id` | ノード ID には使わない（版が変わっても `state_hash` で区別する） |
| `final_seal.state_hash` | 属性 `state_hash`、ノード ID の元 | `hextdoc:` ＋ その値 |
| HEKB の `object_id` | 属性 `hekb_object_id` | **HEXT の `id`・`state_hash` とは別の属性**。ノード ID の元にしない |
| `time.claims[]` | ノード `hext_claim` ＋ 辺 `has_claim`（文書 → claim） | ID ＝ `"hextclaim:" + state_hash + "#" + time_id`。属性: `time_id`・`frame`・`expression`・`status`・`value`・`event_ref`・`source`（`{evidence_id, start, end}`）・`anchor_ref` |
| `time.relations[]` | 辺 `time_relation`（claim → claim） | 属性: `rel_id`・`relation`・`offset`・`status`。端点が文書自身または `derived_from` の（射影済みの）元文書の claim でなければ、辺を作らず `unresolved`（`relation_endpoint`） |
| `provenance.derived_from[]` | 辺 `derived_from`（文書 → 元の文書） | 元の `state_hash` の文書が射影の集合にあるときだけ。無ければ辺を作らず `unresolved`（`derived_from`） |
| `provenance.contributors`・`agency` | 文書ノードの属性 | ノードにしない |
| `time.observed_at`・`recorded_at` | 文書ノードの属性 | そのまま（`null` は `null`）。現在時刻で埋めない |
| `measurement.observation_refs[]` | 辺 `observation_ref`（文書 → 軌跡）。属性: `session_id`・`seq_from`・`seq_to`・`ref` | `urn:gmm:nvs:session:<session_id>#seq=a-b` の `session_id` が、グラフ内の軌跡に解決できるときだけ。**解決できなければ辺を作らず、UNRESOLVED の記録**（規則 `R-VER-OBSREF-RESOLVE-01`）にする |
| `measurement.space_id`・`calibration`・`causal_log_ref` | 文書ノードの属性 | 辺にしない（`causal_log_ref` は未定義の単位のため属性のみ） |
| `evidence[]` | 文書ノードの属性 `evidence_ids` | 本文はノードに複製しない（blob で参照） |

### 8.2 辺を作ったことと、意味を検証したことは別
- 射影が作る辺は「識別子が解決した」ことだけを表す（例: `observation_ref` は、セッション ID が一致した軌跡を指す）。
- その関係に意味があるか（範囲・`space_id` の一致など）は、**別の Verification 記録**（`R-VER-DOC-TRAJ-01` など）が持つ。辺があることは `VERIFIED` ではない。
- 作れなかった辺は、射影の `unresolved`（構造の副産物）と、本体のグラフの Verification（`UNRESOLVED`、理由つき）の両方に残る。

### 8.3 決定性と作り直し
- 射影は、文書のバイト列と 2 つの参照（セッション → 軌跡、元文書の有無）だけの関数。入力の順序に依存しない（`blob_sha256` で整列）。時刻・乱数を使わない。
- 射影は専用のログ（`graph/hext_projection/{nodes,edges,unresolved}.jsonl`、ハッシュ連鎖）で、作り直しは新しいログを書いて入れ替える。本体のログ（Verification を含む）は変更しない。
- 試験（`tests/test_hext_projection.py`）: 新しいプロセス 2 回（`PYTHONHASHSEED=0`）で連鎖の先頭が一致／射影を消して入力順を変えて作り直しても一致／本体の Verification が残る／HEXT 由来の型が `hext_document`・`hext_claim` だけ。
- 実データ（HEXT-V2-GEN001 の 48 件）: 文書ノード 48、`observation_ref` の辺 48（すべて解決）、`unresolved` 0、claims 0（生成した文書は時間の主張を持たない）。claims・relations・`derived_from` の写像は、**テストデータ**（`GemminAI/hext` の P0 の例文書）でのみ確かめた。

### 8.4 `blob_sha256` の対象と、同一識別子の入力の扱い（固定）

**`blob_sha256` が何のバイト列か**
- 対象は、**封印済みの文書の正準化バイト列（JCS）そのもの**。表示用に整形した JSON でも、再シリアライズした JSON でもない。
- ファイル規約としての末尾の改行 1 個だけは、取り除いてから（封印のバイト列として）ハッシュを取る。それ以外で正準形でない入力（整形済み・キー順違い・空白あり）は**拒否**する（射影は、文書を「合わせるために」作り直さない）。
- 射影は `final_seal.state_hash` を再計算して照合し、一致しない文書を拒否する（ノード ID は `state_hash` から導くため）。署名の検証は射影の範囲外。

**同一の識別が複数の入力にあるときの扱い**

| 状況 | 扱い | 理由 |
|---|---|---|
| バイト列が同一（同じ `blob_sha256`）で、付随のメタデータ（`hekb_object_id`・検証水準・認識状態）も同一 | **重複として統合**する（冪等）。`manifest.duplicates_merged` に件数を残す | 同じものを 2 度渡しただけ |
| バイト列が同一で、メタデータが違う | **衝突として拒否**（`ProjectionConflict`）。射影は置き換えない（前の射影が残る） | どちらが正しいか決められない |
| `state_hash` が同一で、バイト列が違う（例: `signature` だけが違う） | **衝突として拒否** | `state_hash` は署名を除くため、同じ内容の別の署名の版。どれを採るかを選ばない |
| `object.id` が同一で、`state_hash` が違う | **別の版として両方を保持**（別のノード）。`manifest.same_object_id_versions` に一覧を残す。`supersedes` の辺は**作らない**（文書が述べていないため） | 改訂は別の文書にする（v0.3 §2.2） |
| `hekb_object_id` が同じで HEXT 文書が違う | 属性として持つだけで、判定しない | HEKB の `object_id` は別の識別子 |

- 拒否は、入力の集合全体に対する再構築の失敗として扱い、部分的な射影を書かない。
