# 🥚 Kosto Lab｜台股短中長線選股（科斯托蘭尼模型）

從台股約 1,900 檔中，自動挑出適合 **短線 5 日／中線 20 日／長線 60 日** 的精選母池與選股權重，
每天產生推薦清單與可下載的完整測試報告。

**不需要寫程式、不需要設定任何密碼或 token。** 兩種用法擇一：

| | 方法一：雲端全自動（推薦） | 方法二：自己電腦一鍵啟動 |
|---|---|---|
| 適合 | 想用手機隨時看、電腦不用開著 | 不想用 GitHub |
| 你要做的事 | 上傳檔案 → 部署網頁（約 15 分鐘） | 雙擊一個檔案 → 按一個按鈕 |
| 第一次等待 | 約 5 小時（全自動，不用管） | 約 6~8 小時（電腦要開著，可分次） |
| 之後每天 | 全自動更新 | 打開後按「更新到最新交易日」 |

---

## 方法一：雲端全自動（3 步）

### 第 1 步：把程式放上 GitHub（約 5 分鐘）

1. 登入 [github.com](https://github.com)，右上角 **＋ → New repository**。
2. Repository name 填 `kosto-lab`，選 **Public**，其他不用動，按 **Create repository**。
3. 在新頁面點藍色連結 **uploading an existing file**。
4. 打開解壓縮後的 `kosto-lab` 資料夾，**全選裡面所有東西**，拖進網頁，等檔案列表跑完。
   * Windows：直接全選（Ctrl + A）即可。
   * **Mac：先在 Finder 按 `⌘ + Shift + .`（句號）** 顯示隱藏的 `.github` 資料夾，再全選（⌘ + A）拖進去。
5. 按最下面綠色 **Commit changes**。

✅ 檢查：repo 首頁的檔案列表要看得到 **`.github`** 資料夾。

> 上傳完成的那一刻，GitHub 就會**自動開始**下載資料，不用按任何按鈕。

### 第 2 步：部署手機/電腦都能看的網頁（約 5 分鐘）

1. 打開 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 帳號登入。
2. 右上角 **Create app** → 選 **Deploy a public app from GitHub**。
3. 填三格：
   * Repository：`你的帳號/kosto-lab`
   * Branch：`main`
   * Main file path：`app.py`
4. 按 **Deploy**，等 2~5 分鐘。**不需要填 Secrets。**

### 第 3 步：等待（約 5 小時，什麼都不用做）

* 打開剛部署的網頁，會看到 **「GitHub 正在自動建立資料庫」** 和五個進度條。
* GitHub 一輪最多跑 6 小時，沒跑完會**自己接著跑下一輪**，完成後自動做研究、產生報告。
* 完成後網頁自動變成「今日推薦」畫面。之後**每個交易日晚上 23:10 自動更新**。

把網頁網址加入手機主畫面，就像 App 一樣使用。

---

## 方法二：自己電腦一鍵啟動（3 步）

1. **安裝 Python（只要一次）**：到 [python.org/downloads](https://www.python.org/downloads/) 下載安裝。
   Windows 安裝第一個畫面務必勾選 **「Add python.exe to PATH」**。
2. **雙擊啟動檔**（在 `kosto-lab` 資料夾內）：
   * Windows：`start_windows.bat`（若跳出「Windows 已保護您的電腦」→ 其他資訊 → 仍要執行）
   * Mac：`start_mac.command`（第一次請按右鍵 → 打開 → 打開）
   第一次會自動安裝套件（3~5 分鐘），接著瀏覽器自動打開網頁。
3. 在網頁按 **「▶ 開始／繼續建立資料」**。
   * 第一次約 6~8 小時，**電腦保持開機、不要睡眠**；可以關掉網頁，下載在背景繼續。
   * 中途關機也沒關係，下次再按一次會從斷點接著下載。
   * 完成後自動做研究，網頁出現今日推薦。

之後每天：雙擊啟動檔 → 左側按 **「🔄 更新到最新交易日」**（約 5~15 分鐘）。

---

## 網頁怎麼用

| 分頁 | 用途 |
|---|---|
| 📌 今日推薦 | 市場雞蛋階段、短/中/長線各 10 檔推薦（含理由、停損價）、備選、被規則排除的高分股 |
| ⚡ 快速驗證 | 改母池、模型、科斯托蘭尼權重 λ、持股水位、停損…幾秒看回測結果；可**下載此設定的完整報告** |
| 🧪 母池測試 | 5 種母池規則 × 6 種大小的比較 |
| 📊 因子 | 各因子的預測力（IC）與目前權重 |
| ⚖️ 新舊比較 | 上傳舊版 App 匯出的 `V22_TEST_REPORT.zip` 或 `monthly_decisions.csv`，用同一套規則比較報酬 |
| 📥 報告/系統 | **下載完整測試報告 ZIP**（網頁報告＋Excel＋設定檔） |

想先熟悉操作？任何時候都可以按 **「先用示範資料體驗」**（合成資料，數字不代表台股）。

## 想永久改設定

1. 在「⚡ 快速驗證」試出喜歡的設定 → 按「下載此設定 (config_used.yaml)」。
2. 到 GitHub 的 repo → 點 `config` → `default.yaml` → 右上鉛筆圖示 ✏️ → 全選貼上新內容 → **Commit changes**。
3. GitHub 會自動用新設定重跑研究（約 30 分鐘）。（方法二：直接用記事本改 `config/default.yaml`。）

---

## 常見問題

| 狀況 | 怎麼辦 |
|---|---|
| 網頁一直顯示「尚未收到第一次執行的結果」超過 15 分鐘 | 確認 repo 首頁有 `.github` 資料夾（第 1 步第 4 點）；或到 repo 的 **Actions** 分頁 → 左側「Kosto Lab 全自動流程」→ 右側 **Run workflow** → 綠色 **Run workflow** |
| Actions 分頁出現 “I understand my workflows, go ahead and enable them” | 按下該綠色按鈕即可 |
| 進度頁出現「官方網站拒絕連線」 | 證交所/公開資訊觀測站暫時擋了 GitHub 的伺服器，系統會自動重試；超過一天仍未改善，改用方法二（台灣網路下載），或見下方「進階」加 FinMind |
| 進度頁顯示「已自動續跑 8 輪仍未完成」 | 到 Actions 按一次 Run workflow；仍不行請改用方法二 |
| 網頁顯示「找不到您的 GitHub repo 名稱」 | 在框中輸入 `你的帳號/kosto-lab` → 確定，並把網址加入書籤 |
| Streamlit 部署時安裝失敗 | Create app 時點 **Advanced settings**，Python 版本選 **3.12** 後重新部署 |
| 方法二雙擊後視窗一閃就關 | Python 沒裝好或沒勾 PATH，重新安裝 Python 並勾選 “Add python.exe to PATH” |
| 想重新安裝方法二的套件 | 刪除 `kosto-lab` 資料夾內的 `.venv` 資料夾，再雙擊啟動檔 |

## 進階（可略過）

* **FinMind 備援**：官方網站長期擋 GitHub 時才需要。到 repo → Settings → Secrets and variables → Actions →
  New repository secret，名稱 `FINMIND_TOKEN`，值貼上 FinMind 網站的 API token。
* **私人 repo**：Streamlit 的 Advanced settings → Secrets 加上
  `GITHUB_REPO = "帳號/kosto-lab"` 與 `GITHUB_TOKEN = "github_pat_..."`（Fine-grained token，權限 Contents 唯讀即可）。
  注意私人 repo 每月 Actions 只有 2,000 分鐘，第一個月可能不夠，建議用 Public。
* **舊版比較自動化**：把舊版匯出檔上傳到 repo 的 `compare/` 資料夾，下次研究會用全市場資料比較並寫入完整報告。
* 架構、因子、科斯托蘭尼規則對照與研究方法：見 [`docs/technical.md`](docs/technical.md)。

> 本工具僅供研究參考，不構成投資建議，也不會自動下單。歷史回測不保證未來績效。
