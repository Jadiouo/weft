# 前人工作勘查：分段粒度怎麼控、怎麼驗

2026-08-09。起因是 R37 量到分段切太碎（57 刀 vs 17 個真實邊界，3.4 倍），
而使用者的指示是「這個問題一定有很多人做過，把各家好的地方擷取下來」。

**這份文件只記別人做過什麼，以及它對不對得上我們的處境。
沒有實測的地方明寫「未驗證」。**

---

## 0. 三個直接可用的結論

| | |
|---|---|
| **我們的驗收指標選錯了** | boundary F1 對「切太碎」結構上不敏感；**WindowDiff 就是為此設計的** |
| **α 是既有的旋鈕，但它有記載的代價** | `cutoff = µ + α·σ`；調嚴會提高 precision、壓低 recall，**淨效果可能更差** |
| **文獻對「語意切塊到底有沒有用」沒有共識** | 有 benchmark 說它輸給固定長度切塊，也有研究說它 87% vs 13% |

---

## 1. 驗收指標：boundary F1 是這個領域公認不該當主指標的

Pevzner & Hearst (2002) 開宗明義批評的就是我們現在用的東西。

### 1.1 Pk（Beeferman et al. 1997）

滑動一個寬度 `k` 的視窗（`k` = 參考分段平均段長的一半），
看視窗**兩端**是否落在同一段——參考與預測不一致就記一分。
分數越低越好，完美為 0。

Pevzner & Hearst 指出四個缺陷，其中**第四個正中我們的處境**：

> 「預測出遠多於參考的邊界」的系統，拿到的懲罰與沒有那麼誇張的
> 系統差不多。

也就是說 **Pk 對切太碎也是遲鈍的**，因為它不懲罰落在參考邊界
`k` 距離內的假陽性。

### 1.2 WindowDiff（Pevzner & Hearst 2002）

改成比較視窗內的**邊界數量**：

```
WindowDiff = (1/(N−k)) · Σ_i  [ |b_ref(i, i+k) − b_hyp(i, i+k)| > 0 ]
```

`b(i, i+k)` 是位置 i 到 i+k 之間的邊界數。加權版本則直接累加
`|b_ref − b_hyp|` 而不是封頂在 1。

**這正是我們需要的**：切 3.4 倍的刀會讓幾乎每個視窗的數量都對不上，
逐窗被罰。而 boundary F1 只要那些多出來的刀「剛好落在容忍窗裡」
就照樣給分——R37 量到的 ±20s 覆蓋 95% 就是這個機制。

nltk 有實作（`nltk.metrics.segmentation.windowdiff`，`k` 未指定時
預設為參考平均段長的一半）。**本機環境沒裝 nltk**，要自己寫或加依賴。

### 1.3 更後面的東西

- **Segmentation Similarity (S) / Boundary Similarity (B)**，Fournier & Inkpen，
  基於 boundary edit distance——把「near-miss」建模成一次**移動**編輯，
  而不是一次刪除加一次插入。理論上比 WindowDiff 乾淨。
  **未評估**：沒查它有沒有堪用的 Python 實作。
- **"When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic
  Segmentation"**（arXiv 2512.17083，2025-12）——標題就是我們踩到的坑。
  它提出對「預測數量遠多於參考」施加**漸進式懲罰**而非當成一般假陽性。
  **未細讀**：抓下來的 PDF 摘要偏空泛，公式沒讀到。

### 1.4 對我們的意思

Q4 當時決定「密度當主要、±10s F1 當輔助」。**密度的方向是對的，
但它是我自己想的粗糙版本，而 WindowDiff 是同一個直覺的成熟形式**——
它同時處理了近失與過度分割，而純密度比值（57/17）完全不看**位置**。

## 2. 粒度怎麼控：α 是既有旋鈕，但有代價

TextTiling 的門檻在文獻裡被參數化成：

```
cutoff(α) = µ + α·σ        µ、σ 是深度分數的平均與標準差
```

**我們現在的 `mean − std/2` 就是 α = −0.5**（Hearst 1997 的原始設定）。
α 越大門檻越嚴、邊界越少。所以 Q2(a) 的「調嚴深度門檻」在文獻裡
是標準做法，不是自創。

**但有一則明確的警告**（Song et al., Interspeech 2016 一系的工作）：

> TextTiling 會放置多餘的邊界（假陽性），而設定較保守的深度門檻
> 提高了 precision、犧牲了 recall，**淨效果反而更差**。

這對我們的意義：**單獨轉 α 很可能不會贏**。它會把 57 刀降到 20 刀，
但降掉的不保證是錯的那 40 刀。**這是假設，我們的素材上未驗證**——
但它是文獻上記載過的失敗，值得先當成預期而不是意外。

### 2.1 其他路線（都未評估）

- **Embedding-Enhanced TextTiling**（Song et al. 2016）——用詞向量取代
  詞頻，緩解「同義不同詞」。我們 R30 試過 sbert，±10s 下在講經上贏
  ngram、在 STEM 上輸，互有勝負。
- **TopicTiling**——用 LDA 主題編號取代詞彙。中文短逐字稿上 LDA 能不能
  訓得起來未知。
- **LLM embeddings 版 TextTiling**（`saeedabc/llm-text-tiling`）——
  同一個骨架換更強的表示。與 §2.1 第一項同類。
- **Utterance-Pair Coherence Scoring**（arXiv 2106.06719）——訓一個
  「這兩句話接得上嗎」的判別器，取代無監督的相似度。**需要訓練資料。**

## 3. 粒度目標該定在哪：文獻互相矛盾

Q1 真正的理由不是通過閘門，是**chunk 太碎會傷害在 Obsidian 裡的查詢**。
那密度目標就不該只由黃金集決定。查了 RAG 切塊的實證研究：

| 來源 | 說法 |
|---|---|
| 一般建議 | 事實型問答 **64–128 token**；需要脈絡的 **512–1024 token** |
| Vecta benchmark（2026-02，50 篇論文 × 7 種策略）| 遞迴 512-token 切塊 **69%** 居首；**語意切塊 54%**，因為它產出平均 **43 token** 的碎片 |
| NAACL 2025 Findings | 語意切塊的計算成本**不划算**，固定 200 字切塊打平或更好 |
| MDPI Bioengineering（2025-11，臨床決策）| **對齊話題邊界**的自適應切塊 **87%** vs 固定長度 **13%**（p = 0.001）|

**沒有共識。** 最後兩列的方向完全相反。

> 這批數字**不能直接搬**：它們的素材是論文與臨床文件（書面、結構清楚），
> 我們的是中文口語逐字稿。而且前三列量的是英文。
> **列在這裡是為了知道別人量到什麼，不是拿來當我們的目標。**

### 3.1 我們現在的位置

實測 166 個 chunk：

| | 中位 | 平均 | P10 | P90 |
|---|---|---|---|---|
| 字數 | **68** | 84 | 41 | 141 |
| 秒數 | **42** | 44 | 21 | 69 |

19.3% 的 chunk **不到 50 字**，最短 20 字。

也就是說：**一個 chunk 是 40 秒口語壓成的 68 個字**。
中文 68 字約 45–70 token（依 tokenizer），**正落在 Vecta 那份
benchmark 描述的「43 token 碎片」失敗區間**。

這比「刀數 3.4 倍」更能說明問題：不只邊界太多，
**產出的東西本身就太小**。

### 3.2 一個結構性的觀察

chunk 的長度 = 段落長度 × 每段的 block 數 × S4c 的壓縮率。
中位 chunk 涵蓋 42 秒 ≈ 整個段落，所以目前**一段大約產出一個 chunk**。

**把段落拉長會直接讓 chunk 變大**——S4c 拿到更多材料可寫。
這是修分段的第二個好處，而且比閘門那 8 個 block 重要得多。

## 4. 沒查的東西

- **中文**的話題分段有沒有專門的工作（本次只搜到英文為主的文獻）。
- 講課／演講這個 genre 專屬的分段研究（我們的素材有投影片切換
  這個額外訊號，雖然 R30 量過它單獨用不夠好）。
- Boundary Similarity (B) 的可用實作。
- "When F1 Fails" 那篇的實際公式。

---

## 引用

- Pevzner & Hearst (2002), *A Critique and Improvement of an Evaluation
  Metric for Text Segmentation* —
  https://people.ischool.berkeley.edu/~hearst/papers/pevzner-01.pdf
- *Recent Trends in Linear Text Segmentation: a Survey* (arXiv 2411.16613)
- *When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic
  Segmentation* (arXiv 2512.17083)
- Song et al., *Dialogue Session Segmentation by Embedding-Enhanced
  TextTiling*, Interspeech 2016 —
  https://www.isca-archive.org/interspeech_2016/song16b_interspeech.pdf
- *Improving Unsupervised Dialogue Topic Segmentation with Utterance-Pair
  Coherence Scoring* (arXiv 2106.06719)
- AssemblyAI, *Text Segmentation — Approaches, Datasets, and Evaluation
  Metrics* —
  https://www.assemblyai.com/blog/text-segmentation-approaches-datasets-and-evaluation-metrics
- nltk `metrics.segmentation` — https://www.nltk.org/_modules/nltk/metrics/segmentation.html
- *Rethinking Chunk Size For Long-Document Retrieval: A Multi-Dataset
  Analysis* (arXiv 2505.21700)
- *Mix-of-Granularity: Optimize the Chunking Granularity for RAG*
  (arXiv 2406.00456)

---

## 5. 補查（同日）：中文的次詞表示——R30 的推論被證實了

R30 選字元 n-gram 的理由寫的是推論：「中文沒有詞邊界，而 R18 量過 ASR
的錯誤幾乎全是等長同音替換，bigram 只是變稀疏不是歸零」。
當時**沒有查過中文分段的既有做法**。

查到了，而且結論一致：

> Chinese broadcast news 的 story segmentation 上，直接量**詞**層次的
> 詞彙連貫性不可行——語音辨識錯誤會打斷詞的連貫；**次詞**層次可以，
> 因為中文的次詞單位承載相當的語義且對辨識錯誤穩健。
>
> TDT2 Mandarin 語料：**字元 bigram 相對 F-measure 比詞為單位高 8.84%**，
> 音節 bigram 高 7.11%。次詞 bigram 在所有尺度中最好。

**這是我們少數「推論後來被文獻證實」的案例**，值得記下來——
R30 的理由不只是合理，是對的。

### 5.1 一個「取各家好處」的候選：Multi-Scale TextTiling

同一批工作提出把**詞的 specificity** 與**次詞的 robustness** 在
相似度計算上整合起來（multi-scale）。

**未評估。** 記在這裡是因為使用者的指示是「把各人好的地方擷取下來」，
而這是目前看到最像那回事的一個——它不是換掉字元 n-gram，是加一層。
要做的前提是先把粒度控制解決（現在切 3.4 倍的刀，換表示法也是白搭）。

### 引用（補）

- Multi-Scale TextTiling for Automatic Story Segmentation in Chinese
  Broadcast News — https://link.springer.com/chapter/10.1007/978-3-540-68636-1_33
- Unsupervised Measure of Chinese Lexical Semantic Similarity Using
  Correlated Graph Model for News Story Segmentation
