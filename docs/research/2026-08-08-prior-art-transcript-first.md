# 前人工作勘查：逐字稿主幹 + 投影片輔助

2026-08-08。觸發問題（使用者）：

> 以逐字稿為主，沒有就用 Whisper 取得；投影片變成修正逐字稿的方法。
> 這個方法有比較好嗎？我不會是第一個遇到這種問題的人。

**結論：這是學界的主流路線，不是獨創也不是走偏。** 四個具體結果：

1. 「逐字稿主幹 + 語意分段」是有名字的成熟方法（TextTiling + Sentence-BERT），
   **無監督、純本地、有中文實證**
2. **有沒有投影片，分段用的是同一套方法**——這正是本專案想要的通用性
3. **R20「解碼層沒用」的結論站不住**——見 §4，這是本文最重要的修正
4. Video-RAC 已經做掉本專案想做的約 80%，但**沒做本專案的溯源部分**

> **修訂 2026-08-08（同日）**：本文第一版把 DocWhisper 寫成「encoder-level
> fusion，成本高不建議追」。**那是錯的**——DocWhisper 就是用 prompt，
> 與 R20 測的是同一個插入點。§4 已改寫，結論從「不建議追」翻轉為
> 「這條路很便宜，值得重測」。

---

## 1. 一句話對照

| 本專案的設計 | 學界／業界對應 | 判定 |
|---|---|---|
| 投影片切換驅動分段（現行 S3） | visual content-based segmentation | **已知有缺陷**，見 §2 |
| 提案 §6：逐字稿主幹 + 語意分段 | TextTiling + Sentence-BERT | **成熟標準方法**，非首創 |
| 無投影片素材怎麼辦 | dialogue / meeting / podcast topic segmentation | **同一套方法**，見 §3.3 |
| S4b 事後校正（D19/D25） | ASR post-processing error correction | 方向正確，已完成 |
| R20：`initial_prompt` 沒用 | **與 DocWhisper 矛盾** | **見 §4，需重測** |
| 多模態 chunk 供 RAG 檢索 | **Video-RAC** | 直接對應，開源可裝 |
| §5.4 溯源／防幻覺閘門 | **未找到直接對應** | 見 §8 |

---

## 2. 視覺驅動分段的已知缺陷

[Automated Video Segmentation for Lecture Videos (HKU)][hku] 與
[Structuring low-quality videotaped lectures (Pattern Recognition)][pr]：

> 「Slide transition detection based on visual cues is **greatly affected by
> the lecturer's motion**.」

**這正是本專案量到的 35% 假邊界**（49 個候選邊界中 17 個是「鏡頭切回講者」，
見 `docs/proposals/v0.4-transcript-first.md` §3）。不是本專案素材特有的問題，
是視覺驅動分段的**結構性缺陷**，文獻早有記載。

文獻同時指出逐字稿路線自己的難處：

> 「Topic boundaries within lecture transcripts tend to be **more subtle and
> fuzzy** because of unprofessional and spontaneous speech.」

**兩條路各有各的糊。** 換到逐字稿主幹不是換到一條乾淨的路，是換到一條
**缺陷型態不同**的路——而那個型態（語意漸變）比「鏡頭切回講者」
更接近真實的內容邊界。

---

## 3. 逐字稿語意分段

### 3.1 標準方法：TextTiling + Sentence-BERT

提案 §6 方向 2 寫「需要語意相似度，那正是 R10 移除的東西，
會把 sentence-transformers 加回來」——這件事有名字、有原始論文、有實作：

> 原始 TextTiling 用**詞頻**算相似度；改良版改用 **BERT embedding**，
> 把主題轉換偵測為連續段落之間**語意連貫度的下降**。
> ——[Solbiati et al., Unsupervised Topic Segmentation of Meetings with
> BERT Embeddings (2021)][solbiati]

三個對本專案重要的性質：

1. **無監督**——不需標註資料，不用擴充黃金集
2. **純本地**——sentence-transformers 可離線跑，零額度（符合「不再用雲端 API」的決定）
3. **與現行訊號可疊加**——投影片切換降級為候選邊界，語意連貫度下降作為採納條件，
   正是提案 §6 的方向 2

**§6 方向 2 的風險評估應下修**：它原本被記為「需要新量測、要加回一個依賴」，
實際上是一個有標準實作的既有無監督方法。

開源實作：[SliceCast][slicecast]（RNN，podcast）、
[unsupervised-topic-segmentation-roberta][roberta-seg]（RoBERTa，podcast 逐字稿）。

### 3.2 中文適用性（本專案素材是中文，必須單獨確認）

TextTiling 原為英文設計，中文沒有詞邊界，不能假設直接適用。查到直接對應的工作：

> **Multi-Scale TextTiling for Automatic Story Segmentation in
> Chinese Broadcast News**——對中文 ASR 逐字稿做 story segmentation，
> 用 character／syllable **n-gram** 序列上的 LSA。
> 關鍵發現：**subword n-grams 對語音辨識錯誤（尤其 OOV 詞）具有韌性。**
> ——[researchgate][cn-texttiling]

**這一條對本專案特別重要**：本專案的 ASR 錯誤型態已由 R18 量出——
幾乎全是**等長同音替換**（`七經開竅` vs `七精開竅`）。
subword n-gram 對這類錯誤天然有韌性，因為錯字周邊的 n-gram 大多仍然匹配。

也與 R18 的既有發現一致：繁簡差異「bigram 只是變稀疏不是歸零」。
**本專案已經在用這個表示法，只是還沒用在分段上。**

### 3.3 沒有投影片的素材：同一套方法

使用者要收的素材有兩支是純口播／訪談（`Best Partners TV`、`大问题 Dialectic`），
完全沒有投影片。查證結果：**這不是特例，是一個成熟的子領域**——
dialogue / meeting / podcast topic segmentation。

| 工作 | 對象 | 方法 |
|---|---|---|
| [Solbiati et al. 2021][solbiati] | 會議錄音 | TextTiling + SBERT，無監督 |
| [Unsupervised Dialogue Topic Segmentation][dialog-seg] | 對話 | topic-aware utterance representation |
| [CobSeg][cobseg] | 對話 | coherence boundary modeling |
| [TreeSeg][treeseg] | **長逐字稿** | **階層式**分段 |
| [SliceCast][slicecast] | podcast | RNN |

> **這是本次勘查對架構決策最有力的支持**：一旦主幹改成逐字稿，
> 「有沒有投影片」對**分段**就不再是架構問題——同一套語意分段方法兩邊都適用。
> 投影片退化成純粹的加分項（校正來源 + 圖表描述），而不是管線能不能跑的前提。
>
> 這正是使用者提案想達成的通用性，而文獻支持它做得到。

### 3.4 長片可能需要階層式分段

使用者要跑的機器人學課程是 **26 支、30–90 分鐘**；講經影片也是長片。

[TreeSeg][treeseg] 專門處理長逐字稿的**階層式**主題分段（章 → 節 → 段），
而非單層切分。對「進 Obsidian 建立知識網絡」可能特別合適——
Obsidian 的結構本身就是階層加網狀，單層 chunk 會丟掉層級關係。

**未評估，僅記錄為候選。**

---

## 4. R20 的結論站不住（本文最重要的修正）

R20 量了兩個插入點，結論寫「**解碼沒用**、事後校正可用」。

**文獻裡有一個做同一件事、結論相反的工作。**

[SlideAVSR][slideavsr] 提出的 **DocWhisper**：

> 「DocWhisper provides texts extracted from slides to Whisper **as prompts**.
> It captures screenshots at the **midpoint of each utterance**, feeds them
> into the OCR module, and uses the recognized text as prompts.
> The prompts are presented as **word sequences**, such as
> "word 1, word 2, ..., word n".」
>
> 結果：相較 fine-tuned Whisper，**TestA 最大改善 14.3%、TestB 11%**。
> 且「**as the maximum word count of prompts increased, the performance
> improved**」。

**DocWhisper 用的就是 prompt，與 R20 測的是同一個插入點。**
但 R20 量到的是術語 4/19 → 5/19、CER 12.3% → 12.0%（幾乎沒動）。

### 三處實作差異，都可能解釋這個落差

| | R20 | DocWhisper |
|---|---|---|
| **粒度** | 整支影片的 `slide_text` 建**一個全域 prompt** | **逐 utterance 各自截圖 OCR**，只餵當下那張投影片 |
| **格式** | **經文原文段落**（「一月為胞，精血凝也。三月陽神為三魂…」） | **詞序列**（`word 1, word 2, ...`） |
| **長度** | 固定 200 字 | 越長越好（實測遞增改善） |

粒度差異最可能是主因：全域 prompt 把整支影片所有投影片的詞混在一起，
對任一句話而言，絕大多數都是雜訊；DocWhisper 只給當下這一句對應的那張。

### 對本專案的意義

**成本很低，而且基礎設施已經有了**：本專案有 `slide_text`、有 S3 對齊
（算得出「這一句對應哪張投影片」）、有 S1c 去重。
把 R20 的全域 prompt 改成逐段局部 prompt，是**設定層的改動，不是架構改動**。

**R20 的結論必須改寫**：從「解碼層沒用」改成
「**全域 200 字經文 `initial_prompt` 這個做法沒用**」。

前者是把一次實驗的結論推廣到整個層級——**與 §1.3 把單支影片推廣成系列通則、
與 R26 把調校集的 1.000 當成判準成立，是同一種錯誤的第三次出現。**

相關工作另見 [LCB-net][lcb]（long-context biasing）、
[ED-CEC][edcec]（事後錯誤偵測與情境校正）、
[OCR-Enhanced Multimodal ASR][ocr-asr]。

---

## 5. 投影片幫不幫得上忙：是**條件性**的

[Do Slides Help? Multi-modal Context for Automatic Transcription of
Conference Talks (EMNLP 2025 main)][doslides] 直接研究這個問題。

**有幫助的條件**：投影片含相關術語與領域詞彙、投影片與語音**時間對齊良好**、
投影片文字清晰且有一定密度。

**會變差的條件**：

> 「**Misaligned or off-topic slides can degrade performance.**」
> 「Diminishing returns when slides contain minimal textual content.」

**這是對本專案的直接警告。** 本專案已知有 35% 假邊界——那正是 misaligned。
**在對齊修好之前擴大投影片對逐字稿的影響力，文獻預測會讓結果變差。**

**因此「先修分段對齊，再談投影片校正」的順序有文獻依據，不只是工程偏好。**
這一條也直接約束 §4 的重測：**逐段局部 prompt 的前提是分段是對的。**

---

## 6. 公開資料集

本專案黃金集為手工標註（`tests/golden/`，目前 4 支）。以下語料含
同步投影片 + 轉錄：

| 資料集 | 規模 | 內容 | 出處 |
|---|---|---|---|
| **SlideSpeech** | 1,705 支、1,000+ 小時（473 小時高品質轉錄） | 語音、逐字稿、**OCR 結果、keyword bias list**、分段 | [ICASSP 2024][slidespeech] / [OpenSLR 144][openslr] / [下載腳本][ss-dl] |
| **SlideAVSR** | 論文解說影片 | AVSR，DocWhisper 的評測集 | [ACL ALVR 2024][slideavsr] |
| **M3AV** | 367 小時 | 投影片文字與語音標註，評測 contextual ASR | 見 §4 出處線索 |
| **EduViQA** | 20 支 × 50 QA | 雙語，Video-RAC 的評測集，HuggingFace | [Video-RAC][videorac] |

> ⚠️ **語種：SlideSpeech 官網未明說。** 官網（CC BY-SA 4.0）只列規模與檔案，
> 沒有語言欄位。多處二手描述指向**來自 YouTube 的英文會議影片**。
> **傾向英文，但未證實。**
>
> 本專案素材是中文（含簡繁混雜）。**若語種為英文，這些語料只能用於驗證
> 「方法在乾淨資料上會不會動」，不能用於決定本專案的門檻或參數**——
> 拿英文語料的結論套中文素材，是同一種推廣錯誤的又一次。
>
> 中文的對應資料在 §3.2（Chinese Broadcast News）。

---

## 7. 最重要的發現：Video-RAC

[Video-RAC: Retrieval Adaptive Chunking for Lecture Video RAG][videorac]
是本次勘查中**與本專案目標最接近的既有工作**。

**做法**：CLIP embeddings + SSIM 偵測投影片切換切出 chunk；
每 chunk 取三張代表幀（最大熵、首、末）；
逐字稿**以語意邊界遞迴切分**，再時間對映回視覺 chunk。

**成效**：較 naive 切片在 RAGAS 指標上 **+12–15%**；
多模態（圖+文）配 GPT-4o 達 Answer Relevance 0.87 /
Context Relevance 0.82 / Faithfulness 0.91。

**可取得性**：`pip install VideoRAC`，GitHub 開源，資料集在 HuggingFace。

### 與本專案的異同

| | Video-RAC | weft |
|---|---|---|
| 投影片切換偵測 | CLIP + SSIM | ink Jaccard + 灰階距離（純 CV，更輕） |
| 代表幀 | 最大熵 + 首 + 末 | 段內 ink 量最大、排除兩端（D15） |
| 逐字稿語意切分 | **有** | **尚未實作**（提案 §6） |
| **防幻覺／溯源** | **無** | **§5.4 溯源閘門 + §5.3 十條不變量** |
| **素材勘查** | **無** | **S-1**（逐支 profile + 中止判準） |
| 額度／續跑 | 無 | §6 Quota Ledger |
| 無投影片素材 | 未處理 | 目標支援 |

**本專案沒有被取代。** Video-RAC 解的是「怎麼切才好檢索」，
本專案多解了「**怎麼確定產出的東西不是編的**」——那是要進個人知識庫、
長期複利、會被引用的內容才需要的保證。

### 一個本專案目前缺的東西

Video-RAC 用 **RAGAS**（Answer Relevance / Context Relevance / Faithfulness）
當評測出口。

**本專案 §5.2 的十項門檻全部在量「有沒有做錯」，沒有一項在量
「做出來的東西好不好用」。** 這可能是「不知道怎麼收尾」的深層原因：
**沒有任何一個指標會告訴你「夠好了」**，所以只剩下永遠差一點的 0.95。

建議把「檢索得出來嗎、答得準嗎」納入驗收，作為與現有機械門檻**互補**的出口。
**未評估 RAGAS 對中文與本專案素材的適用性，僅記錄為方向。**

---

## 8. 未解與待查

| # | 項目 | 狀態 |
|---|---|---|
| 1 | SlideSpeech / M3AV 語種分布 | **部分解**：官網未列語言，二手描述指向英文。需下載後確認 |
| 2 | DocWhisper 原始出處與數字 | **已解**：出自 [SlideAVSR][slideavsr]，且**是 prompt 不是 encoder fusion**（§4） |
| 3 | 溯源獨立性的前人工作 | **未解**，見下 |
| 4 | 無投影片素材的分段 | **已解**：成熟子領域，同一套方法（§3.3） |
| 5 | TreeSeg 階層分段是否適用 | **未評估**（§3.4） |
| 6 | RAGAS 對中文素材的適用性 | **未評估**（§7） |

### 關於 #3：查了，沒找到直接對應

本專案的問題是：**S4b 用投影片文字校正逐字稿，而 §5.4 又拿逐字稿與投影片
互相比對做溯源**——校正把來源 A 的資訊寫進來源 B，兩個來源不再獨立
（實作證據：`src/weft/validation/provenance.py:325`
`transcript = seg.transcript_corrected or seg.transcript_raw`）。

查到的 hallucination detection 文獻（[綜述][hallu-survey]、
[Decomposed Entailment][decomp]、[span-level][span]）都假設**來源是外部固定的
文件**，不處理「管線內部資訊回流污染驗證基準」這種循環。

最接近的既有概念是軟體測試的 **test oracle problem**（拿被測系統自己的輸出
當預期值）與 ML 的 **data leakage**，但都沒有針對這個具體形態的工作。

**暫記為本專案可能的獨有部分。也可能只是關鍵字沒下對——不宣稱原創。**

---

## 出處

[hku]: https://pweb.fbe.hku.hk/~mchau/papers/AutomatedVideoSegmentation.pdf
[pr]: https://www.sciencedirect.com/science/article/abs/pii/S0031320308001209
[ncaa]: https://link.springer.com/article/10.1007/s00521-025-11740-2
[solbiati]: https://arxiv.org/pdf/2106.12978
[cn-texttiling]: https://www.researchgate.net/publication/221055663_Multi-Scale_TextTiling_for_Automatic_Story_Segmentation_in_Chinese_Broadcast_News
[treeseg]: https://arxiv.org/pdf/2407.12028
[cobseg]: https://arxiv.org/pdf/2605.30668
[dialog-seg]: https://arxiv.org/pdf/2305.02747
[slicecast]: https://github.com/bmmidei/SliceCast
[roberta-seg]: https://github.com/schimmerd/unsupervised-topic-segmentation-roberta
[mmvts]: https://arxiv.org/pdf/2312.00220
[doslides]: https://aclanthology.org/2025.emnlp-main.814.pdf
[slidespeech]: https://arxiv.org/pdf/2309.05396
[openslr]: https://www.openslr.org/144/
[ss-dl]: https://github.com/Mashiro009/slidespeech_dl
[slideavsr]: https://aclanthology.org/2024.alvr-1.11.pdf
[lcb]: https://arxiv.org/pdf/2401.06390
[edcec]: https://arxiv.org/pdf/2310.05129
[ocr-asr]: https://www.arxiv.org/pdf/2601.18393
[videorac]: https://prismaticlab.github.io/Video-RAC/
[hallu-survey]: https://arxiv.org/html/2510.06265v1
[decomp]: https://arxiv.org/html/2608.05823
[span]: https://arxiv.org/html/2607.00895

**分段**
- Automated Video Segmentation for Lecture Videos (HKU) — [PDF][hku]
- Structuring low-quality videotaped lectures — [Pattern Recognition][pr]
- Multimodal segmentation and labeling of lecture recordings (2025) — [NCAA][ncaa]
- Unsupervised Topic Segmentation of Meetings with BERT Embeddings — [Solbiati et al. 2021][solbiati]
- Multi-Scale TextTiling for Chinese Broadcast News — [researchgate][cn-texttiling]
- TreeSeg: Hierarchical Topic Segmentation of Large Transcripts — [arXiv 2407.12028][treeseg]
- CobSeg: Coherence Boundary Modeling for Dialogue Topic Segmentation — [arXiv 2605.30668][cobseg]
- Unsupervised Dialogue Topic Segmentation — [arXiv 2305.02747][dialog-seg]
- Multi-Modal Video Topic Segmentation — [arXiv 2312.00220][mmvts]
- SliceCast — [GitHub][slicecast] ／ unsupervised-topic-segmentation-roberta — [GitHub][roberta-seg]

**投影片 × ASR**
- Do Slides Help? — [EMNLP 2025 main][doslides]
- SlideAVSR（DocWhisper 出處）— [ACL ALVR 2024][slideavsr]
- SlideSpeech — [ICASSP 2024][slidespeech] ／ [OpenSLR 144][openslr] ／ [下載][ss-dl]
- LCB-net — [arXiv 2401.06390][lcb]
- ED-CEC — [arXiv 2310.05129][edcec]
- OCR-Enhanced Multimodal ASR — [arXiv 2601.18393][ocr-asr]

**檢索與幻覺**
- Video-RAC — [PrismaticLab][videorac]
- Hallucination in LLMs: A Comprehensive Survey — [arXiv 2510.06265][hallu-survey]
- Decomposed Entailment for Factuality Checking — [arXiv 2608.05823][decomp]
- Span-Level Hallucination Detection — [arXiv 2607.00895][span]
