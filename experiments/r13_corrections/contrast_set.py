"""R13 校準的對照組：術語校正「授權 vs 未授權」。

**方法**比照 D1 與 R12——建對照組、量分離度，**分離倍數低於 2 倍即判該規則
不可用**，不靠調參數硬拉。

**授權的定義取自 prompt 自己寫的規範**（`understand.py` SYSTEM_PROMPT §3）：

    只改有把握的：正確寫法出現在這張投影片上，且與錯字同音或近音
    寧可漏改，不可亂改

真實錨點：`zIglvjoU9vo` 首跑輸出的全部 9 筆校正（標 `real=True`），未經修改。
其餘為手寫，刻意取同一領域（佛道講經逐字稿的 ASR 錯誤），
使兩組的詞彙分布相近——避免用「明顯離譜」的違規樣本把分離度灌高。

`authorized` 的判定準則（我逐條判的，判準寫在 `why` 欄）：
  - 授權：`from` 是 ASR 造成的音近錯字／疊字／截斷，`to` 與 `from` 指同一個詞
  - 未授權：`to` 改變了指涉對象（事實修正、語意改寫）、
            增加了原文沒有的內容（插入）、或根本不是術語問題（格式、潤稿）
"""

from __future__ import annotations

#: (from, to, 是否授權, 類別, 說明, 是否為模型實際輸出)
CASES: list[dict] = [
    # ================= 授權：ASR 同音／近音錯字 =================
    {"from": "憍梵钵提", "to": "憍梵波提", "authorized": True, "kind": "音近專名",
     "why": "尊者名號的異體字，同音，指涉不變", "real": True},
    {"from": "意地論", "to": "瑜伽師地論", "authorized": True, "kind": "音近專名",
     "why": "講者口誤吞字，「地論」同音，指涉同一部論", "real": True},
    {"from": "學古學的", "to": "學古文學的", "authorized": True, "kind": "音近漏字",
     "why": "漏一個「文」字，音近，指涉不變", "real": True},
    {"from": "時運", "to": "識蘊", "authorized": True, "kind": "音近專名",
     "why": "shi-yun 完全同音的術語誤寫", "real": False},
    {"from": "波若", "to": "般若", "authorized": True, "kind": "音近專名",
     "why": "bo-re 同音異寫", "real": False},
    {"from": "涅盤", "to": "涅槃", "authorized": True, "kind": "音近專名",
     "why": "pan 同音錯字", "real": False},
    {"from": "十二因原", "to": "十二因緣", "authorized": True, "kind": "音近專名",
     "why": "yuan 同音錯字", "real": False},
    {"from": "陰陽不劃", "to": "陰陽布化", "authorized": True, "kind": "音近經文",
     "why": "bu-hua 音近，經文原句", "real": False},
    {"from": "承其宿夜", "to": "承其宿業", "authorized": True, "kind": "音近經文",
     "why": "ye 同音錯字", "real": False},
    {"from": "精血英也", "to": "精血凝也", "authorized": True, "kind": "音近經文",
     "why": "ying/ning 音近", "real": False},

    # ================= 授權：疊字與截斷 =================
    {"from": "家家當", "to": "家當", "authorized": True, "kind": "疊字",
     "why": "ASR 把「家當」的首字重複了", "real": True},
    {"from": "未", "to": "未來", "authorized": True, "kind": "截斷補全",
     "why": "詞被截斷，後文已補全為「未來」", "real": True},
    {"from": "這這個", "to": "這個", "authorized": True, "kind": "疊字", "why": "疊字", "real": False},
    {"from": "因因為", "to": "因為", "authorized": True, "kind": "疊字", "why": "疊字", "real": False},
    {"from": "菩", "to": "菩薩", "authorized": True, "kind": "截斷補全",
     "why": "詞被截斷", "real": False},

    # ================= 未授權：插入原文沒有的內容 =================
    {"from": "陽神為三魂", "to": "陽神為三魂，動而生也", "authorized": False, "kind": "插入",
     "why": "補上講者根本沒說的後半句經文；§5.3 不變量 10 只驗 from，驗不到這個", "real": True},
    {"from": "一月為胞", "to": "一月為胞，精血凝也", "authorized": False, "kind": "插入",
     "why": "同上，從投影片補全講者未唸的部分", "real": False},
    {"from": "五蘊", "to": "五蘊，色受想行識", "authorized": False, "kind": "插入",
     "why": "附加解釋，不是錯字校正", "real": False},
    {"from": "內觀經", "to": "太上老君內觀經", "authorized": False, "kind": "插入",
     "why": "補全書名。講者用簡稱是正常說法，不是錯字", "real": False},

    # ================= 未授權：事實修正 =================
    {"from": "啟示經", "to": "創世記", "authorized": False, "kind": "事實修正",
     "why": "qi-shi vs chuang-shi 不同音；改的是講者的事實錯誤而非聽寫錯誤", "real": True},
    {"from": "唐朝", "to": "宋朝", "authorized": False, "kind": "事實修正",
     "why": "改年代。講者若講錯，那是講者的話，不該由校正抹掉", "real": False},
    {"from": "六祖惠能", "to": "六祖慧能", "authorized": False, "kind": "事實修正",
     "why": "兩種寫法都通行，屬編輯偏好不是錯字", "real": False},
    {"from": "三百年", "to": "五百年", "authorized": False, "kind": "事實修正",
     "why": "改數字。這是 §5.4 明令要溯源的那類資訊", "real": False},

    # ================= 未授權：語意改寫 =================
    {"from": "投胎轉世", "to": "十個月懷胎", "authorized": False, "kind": "語意改寫",
     "why": "指涉完全不同的東西，非同音非近音", "real": True},
    {"from": "買房子", "to": "受精卵著床", "authorized": False, "kind": "語意改寫",
     "why": "把比喻替換成本體，抹掉講者的修辭", "real": False},
    {"from": "他進來以後", "to": "識蘊進入胎體之後", "authorized": False, "kind": "語意改寫",
     "why": "把代詞展開成詮釋。展開是 content_block 的工作，不是校正的", "real": False},
    {"from": "很醜", "to": "相貌平庸", "authorized": False, "kind": "語意改寫",
     "why": "潤飾用語，改變了語氣", "real": False},

    # ================= 未授權：格式化／潤稿 =================
    {"from": "沒sense", "to": "沒Sense", "authorized": False, "kind": "格式化",
     "why": "大小寫，不是術語錯字", "real": True},
    {"from": "sensor", "to": "感測器", "authorized": False, "kind": "格式化",
     "why": "翻譯外來語，講者原本就講英文", "real": False},
    {"from": "好像很像", "to": "很像", "authorized": False, "kind": "潤稿",
     "why": "刪贅詞。逐字稿的贅詞是講者實際說的話", "real": False},
    {"from": "那個那個就是", "to": "", "authorized": False, "kind": "潤稿",
     "why": "刪填充詞（空 to 會被既有檢查擋掉，列此確認）", "real": False},
]
