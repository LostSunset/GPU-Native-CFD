# ONERA M6 的實驗壓力分布（Schmitt & Charpin, AGARD AR-138）

下載：2026-10-23（使用者同意），來源 NASA Glenn 的 NPARC Alliance 驗證檔案庫
`https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing.html`，檔案 `cp<k><l|u>.ex`（k = 1–7 為展向站，
l／u 為下／上翼面）。內容未改動；SHA-256 見 `SHA256SUMS`。

原始文獻：V. Schmitt and F. Charpin, "Pressure Distributions on the ONERA-M6-Wing at Transonic Mach Numbers,"
Experimental Data Base for Computer Program Assessment, AGARD AR 138, May 1979。

條件（Test 2308）：馬赫 0.8395、攻角 3.06°、側滑 0°、Re = 11.72×10⁶（以平均氣動弦長 0.64607 m 為準）。
幾何：半展長 b = 1.1963 m，展弦比 3.8、尖削比 0.562、前緣後掠 30°、後緣後掠 15.8°、無扭轉、ONERA D 對稱翼型。

| 站 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| η = z/b | 0.20 | 0.44 | 0.65 | 0.80 | 0.90 | 0.95 | 0.99 |

檔案的欄位：x/c、−Cp、以及兩個 ±0.02 的誤差欄（NPARC 頁面註明馬赫 0.84 時壓力的精度為 ±0.02）。
