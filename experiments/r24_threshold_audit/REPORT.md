# R24：§5.2 門檻的盤點——哪些還測得到、哪些測的是不存在的機制

起因：準備解除 R2（黃金集）前先確認要標什麼。發現
`FRAME_CLASS_ACCURACY` 測的 `frame_class` 欄位，在 v0.3 用 D16
移除 CV 分類後**全部是 `slide`**（實測 2519/2519）。

架構改了三版（v0.2→v0.3→v0.4），必須逐項確認門檻是否還對得上實作，
否則會標註一個沒人讀的欄位。

---

## 結論（先講）

**10 項門檻中，只有 3 項真的在驗收。**

| 狀態 | 數量 | 門檻 |
|---|---|---|
| ✅ 真的在跑 | 3 | 合成 boundary F1、逐條動畫合併、溯源通過率／unverified 比例 |
| ⏸ 因黃金集而 skip | 2 | 真實 boundary F1、對齊誤差 |
| ❌ **測的機制已不存在** | 1 | speaker/slide 分類 accuracy |
| ❌ **從來沒有驗收測試** | 2 | 術語校正 precision、術語校正 recall |

「從來沒有驗收測試」的兩項最值得注意：它們**只出現在
`test_unit_metrics.py` 的常數值斷言裡**（`assert T.X == 0.90`），
也就是只驗證「這個常數等於 0.90」，沒有任何測試拿真實輸出去比它。

---

## 逐項

### ✅ 1. `BOUNDARY_F1_SYNTHETIC = 0.95`

合成影片 A1–A9。`test_slide_detection_meets_synthetic_f1_threshold`
實際跑 S1b 並比對。**有效。**

### ⏸ 2. `BOUNDARY_F1_REAL = 0.75`

`test_boundary_f1_on_real_videos` 存在且會跑真實 S1b，
但 `_golden_annotations()` 為空 → skip。**機制仍在，缺標註。**

### ❌ 3. `FRAME_CLASS_ACCURACY = 0.95` —— 測的機制已不存在

測 `candidates.frames[].frame_class`。實測 `zIglvjoU9vo` 的 2519 幀
**全部是 `slide`**——v0.3 的 D16 把 CV 的 speaker/slide 分類整個移除，
`FrameLabel.frame_class` 的 docstring 自己也寫了：

> v0.3 移除了 speaker/slide 二分類——CV 只負責找靜止區段，分類交給 VLM。
> `frame_class` 保留為 `slide`（意為「候選幀」）

**這項門檻現在即使標註了也測不到東西。**

分類這件事並沒有消失，只是搬到 **S4a 的 `is_slide`**（§4.7a）。
而我**已經有那份標註**——`experiments/r14_image_binding/labels_is_slide.json`，
49 張人工逐張判讀。門檻應該改綁到那裡。

### ✅ 4. `PROGRESSIVE_MERGE_ACCURACY = 1.00`

`test_progressive_animation_merges_into_one_slide`（合成 A2）。**有效。**

### ❌ 5. `TERM_CORRECTION_PRECISION = 0.90` —— 沒有驗收測試

`test_term_correction_precision` 是**整個函式 body 只有一行 skip**：

```python
pytest.skip("需要黃金集標註逐字稿中的術語錯誤；標註完成後啟用")
```

它連 `_golden_annotations()` 都沒讀——**不是「有標註就會跑」，是根本還沒寫**。

諷刺的是這一項**其實有大量實測資料**：R13 量過首跑 9 筆的 precision
（0.56→1.00）、R20 量過詞庫事後校正（100%）。那些量測都在
`experiments/` 裡，**沒有一項接進驗收**。

### ❌ 6. `TERM_CORRECTION_RECALL = None`

值是 `None`，語意是「必須量測並寫入報告，但不 assert」。
**實際上從來沒有任何地方量測它並寫入報告。**
R13 §「未量」明白記過「recall 從未量測」，至今仍是。

### ⏸ 7. `ALIGNMENT_MEDIAN_ERROR_SEC = 5.0`

同 #5，`test_alignment_boundary_error_within_threshold` 也是只有一行 skip。
但這一項還有**更根本的問題**：v0.3 移除了語意吸附（R10），
S3 現在只做粗切，`boundary_shift_sec` 恆為 0。
**沒有「吸附」可言，這個門檻量的是什麼需要重新定義。**

### ✅ 8–9. `PROVENANCE_PASS_RATE = 0.95` / `MAX_UNVERIFIED_RATIO = 0.05`

S6 實際執行並在超標時擋下寫入（實跑本地配置時 0.838 被擋）。**有效。**

### ✅ 10. `BOUNDARY_TOLERANCE_SEC = 2.0`

不是門檻是參數，被 #1／#2 使用。

---

## 這次盤點暴露的模式

**三項門檻（#3、#5、#6）的共同點是：它們在 `test_unit_metrics.py`
裡有「常數值斷言」。**

```python
assert T.TERM_CORRECTION_PRECISION == 0.90
```

那條斷言會綠燈，看起來像「門檻有在管」，但它只證明了**常數沒被改過**——
與「有沒有拿真實輸出去比」是兩回事。

`test_thresholds_are_not_configurable` 與 `ACCEPTANCE_THRESHOLDS` 也一樣：
它們防的是「有人偷偷調門檻」，防不了「門檻根本沒接上」。

**§5.5 #7 防的是調門檻，沒有規定防「門檻懸空」。**

---

## 建議的處置（A2 執行）

| 門檻 | 處置 |
|---|---|
| `FRAME_CLASS_ACCURACY` | **改綁到 S4a 的 `is_slide`**。標註已存在（49 張），改名為 `SLIDE_CLASSIFICATION_ACCURACY` |
| `TERM_CORRECTION_PRECISION` | 寫真的驗收測試。R13/R20 的對照組可直接用 |
| `TERM_CORRECTION_RECALL` | 寫量測並輸出報告（不 assert）。目前連量都沒量 |
| `ALIGNMENT_MEDIAN_ERROR_SEC` | **需要決定**：v0.3 移除吸附後這一項量什麼？暫記為待決 |
| 其餘 | 不動 |

**另外建議加一條機械護欄**：檢查每個 `ACCEPTANCE_THRESHOLDS` 的成員
都至少被一個**非常數斷言**使用——否則「門檻懸空」還會再發生。
