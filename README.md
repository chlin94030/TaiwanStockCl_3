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
5. 按最下面綠色 **Commit changes**。

> 上傳完成後 GitHub 會**自動開始**下載資料，不用按任何按鈕。

#### ⚠️ 關於 `.github` 資料夾（Windows、Mac 都常遇到）

`.github` 裡放的是「自動執行」的設定。它的名字以「.」開頭，**電腦常把它藏起來，GitHub 網頁上傳也常把它略過**。
不用擔心，二選一處理即可：

**做法 A（最簡單，推薦）：先不管它，直接做第 2 步。**
部署好的網頁若發現 GitHub 上缺這個檔，會顯示紅色提示和 3 個步驟：
① 按網頁上的連結（打開 GitHub 新增檔案頁，檔名已自動填好）→
② 按網頁上框框右上角的「複製」→ 回 GitHub 編輯框按 **Ctrl+V** 貼上 →
③ 按 GitHub 右上角綠色 **Commit changes** 兩次。完成。

**做法 B（不靠網頁，自己補）：**
1. 在 GitHub 的 repo 頁面按 **Add file → Create new file**。
2. 檔名欄輸入 `.github/workflows/pipeline.yml`（打到「/」會自動變成資料夾）。
3. 用記事本打開 `kosto-lab` 資料夾裡的 **`pipeline_workflow_COPY.txt`**（這是同一份內容的可見副本），
   Ctrl+A 全選 → Ctrl+C 複製 → 貼到 GitHub 的大編輯框。
4. 按綠色 **Commit changes** 兩次。

（想在電腦上看到隱藏資料夾：Windows 11 檔案總管 →「檢視 → 顯示 → 隱藏的項目」；
Windows 10 →「檢視」→ 勾選「隱藏的項目」；Mac Finder 按 `⌘ + Shift + .`）

✅ 檢查：repo 首頁的檔案列表看得到 **`.github`** 資料夾，就代表成功。

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
| 📌 今日推薦 | 市場雞蛋階段、短/中/長線各 10 檔推薦（含理由、警戒價）、備選、被規則排除的高分股 |
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

## 更新程式版本（例如修正錯誤時）

1. 解壓縮收到的更新包（例如 `kosto_update_v1.1.0`），裡面會有 `kosto`、`config` 兩個資料夾和 `app.py`、`cli.py`。
2. 打開 GitHub 的 repo 首頁 → **Add file → Upload files**。
3. 打開更新包資料夾，**全選裡面所有東西**拖進網頁（同名檔案會自動覆蓋）。
4. 按 **Commit changes**。GitHub 會自動重跑（已下載的資料會保留，只補新資料再做研究，約 20~40 分鐘）；
   Streamlit 網頁也會自動換成新版。

## 常見問題

| 狀況 | 怎麼辦 |
|---|---|
| 想知道 v1.1 改了什麼、為什麼 | 見 `docs/technical.md` 最後的「v1.1 變更紀錄」 |
| Actions 的 finish 出現紅色 ✗、訊息 `Process completed with exit code 1` | v1.0.0 的已知錯誤（GitHub 用 pandas 2.3 時名稱欄位格式衝突），已於 v1.0.1 修正，照上方「更新程式版本」上傳即可；資料不用重抓 |
| Actions 出現黃色 “Node.js 20 is deprecated” 或 “ubuntu-latest will migrate” | 只是提醒、不影響結果；v1.0.1 的流程檔已更新（可選，見更新說明） |
| 網頁出現紅色「GitHub 上缺少自動流程檔」 | 照網頁上的 3 步（連結 → 複製 → 貼上 Commit），見第 1 步的「做法 A」 |
| 網頁一直顯示「尚未收到第一次執行的結果」超過 15 分鐘 | 到 repo 的 **Actions** 分頁 → 左側「Kosto Lab 全自動流程」→ 右側 **Run workflow** → 綠色 **Run workflow** |
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
