-- 舊文章補登判讀（2026-09-27 依文章內容整理）。在 SQL Editor 執行一次。
-- 需先執行 forum_judgments_v2.sql。判讀時間 = 文章發布時間，標記 backfilled。
-- 重複執行不會重複新增（已有判讀的文章會略過）。

begin;

-- 建準 2421 周線分析
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select '5b71ab10-cb1c-4d9a-a36d-4a1559791a4f', 'tw', '2421', '1w', array['up']::text[], 156.5,
       array['pattern','volume']::text[], '[{"type": "支撐", "price": 148}, {"type": "阻力", "price": 156.5}, {"type": "阻力", "price": 165.5}, {"type": "阻力", "price": 167.5}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/f3bbec34-3c8e-4ff2-b103-2a509b5bfe31.png',
       '週線下降擴散楔形（底部反轉），前提是週收站穩 148；量能不足，第一壓力 156.5、主要出貨區 165.5/167.5', 0, true, '2026-05-12T08:06:08.382204+00:00', '2026-05-12T08:06:08.382204+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = '5b71ab10-cb1c-4d9a-a36d-4a1559791a4f');

-- BTC 上升楔形 周線分析
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select '4ea76af7-3a14-46f5-aba5-b9353d497afc', 'crypto', 'BTC', '1w', array['down']::text[], null,
       array['pattern','other']::text[], '[{"type": "阻力", "price": 85170}, {"type": "阻力", "price": 86626}, {"type": "阻力", "price": 93011}, {"type": "支撐", "price": 73693}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/a12e5fba-6b28-4e9c-91ec-6b0d9400cdec.png',
       '週線熊市結構中的上升楔形末端，85170–86626 共振壓力區開空，站上 93011 認錯', 0, true, '2026-05-13T15:46:38.681080+00:00', '2026-05-13T15:46:38.681080+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = '4ea76af7-3a14-46f5-aba5-b9353d497afc');

-- 光頡3624 周線分析
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select '97b9afb2-7178-4a2b-a341-93ebde852d29', 'tw', '3624', '1w', array['down']::text[], 64,
       array['pattern','volume']::text[], '[{"type": "VAH", "price": 74.5}, {"type": "阻力", "price": 80.9}, {"type": "支撐", "price": 64}, {"type": "支撐", "price": 62.6}, {"type": "支撐", "price": 47.5}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/97a1f95b-f66c-42ae-8d1a-a0e56535049b.png',
       '上升擴散楔形＋價漲量縮，74.5–80.9 為假突破區，偏空但信心打折；跌破 71–72 看 63–64', 0, true, '2026-05-14T10:36:25.080969+00:00', '2026-05-14T10:36:25.080969+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = '97b9afb2-7178-4a2b-a341-93ebde852d29');

-- 微星 2377 周線分析 —  周線級別轉強訊號
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select 'b17cd8ad-d88b-44fe-af95-441b77d6ec99', 'tw', '2377', '1w', array['up']::text[], 124,
       array['volume','other']::text[], '[{"type": "POC", "price": 100.5}, {"type": "支撐", "price": 111}, {"type": "VAL", "price": 88.9}, {"type": "阻力", "price": 127}, {"type": "阻力", "price": 145.5}, {"type": "阻力", "price": 170}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/eb52f347-3235-49df-bdbc-71d9b61e8c55.png',
       '爆量站上 POC 100.5，一年底部換手完成，中長期偏多；前提是週收站穩 111，短線目標 124–127', 0, true, '2026-05-15T02:18:17.168779+00:00', '2026-05-15T02:18:17.168779+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = 'b17cd8ad-d88b-44fe-af95-441b77d6ec99');

-- Zcash (ZEC) 解析：頭肩頂成型與風險指標的強烈警訊
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select '26c35f8f-3162-4138-a905-e614cb34cedc', 'crypto', 'ZEC', '1d', array['down']::text[], 507.24,
       array['pattern','volume','other']::text[], '[{"type": "阻力", "price": 689.85}, {"type": "阻力", "price": 622}, {"type": "VAH", "price": 507.24}, {"type": "POC", "price": 237.55}, {"type": "VAL", "price": 204.61}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/d618bd74-2a70-4573-b7e6-068dc154e97e.png',
       '頭部為 M 頂的複雜頭肩頂，右肩量縮、長上影、受 0.618 壓制；頸線與 VAH 在 507.24', 0, true, '2026-06-03T02:03:37.857233+00:00', '2026-06-03T02:03:37.857233+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = '26c35f8f-3162-4138-a905-e614cb34cedc');

-- BTC 黃金坑：數據與技術共振的抄底時機
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select 'ea66fab2-e158-41dd-a707-5a9bf0988d8d', 'crypto', 'BTC', '1w', array['up']::text[], null,
       array['wyckoff','volume','other']::text[], '[{"type": "支撐", "price": 58994}, {"type": "支撐", "price": 57798}, {"type": "VAL", "price": 54797}, {"type": "支撐", "price": 53109}, {"type": "支撐", "price": 48907}, {"type": "支撐", "price": 47283}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/ab304d33-89d9-475b-b3cf-a2bfcc943e8e.png',
       '威科夫第三次吸籌，MVRV 與 Realized Price 到低檔；58,994–47,283 為分批建倉區，6 月至 11 月築底', 0, true, '2026-06-08T08:56:52.983531+00:00', '2026-06-08T08:56:52.983531+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = 'ea66fab2-e158-41dd-a707-5a9bf0988d8d');

-- ETH技術分析：多重共振支撐與長線抄底佈局策略
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select 'c61f0f43-85f2-44cc-b86c-f5d58254ab91', 'crypto', 'ETH', '1w', array['down']::text[], 1381.85,
       array['other']::text[], '[{"type": "阻力", "price": 1756.88}, {"type": "支撐", "price": 1400.92}, {"type": "支撐", "price": 1381.85}, {"type": "支撐", "price": 1239.77}, {"type": "VAL", "price": 1060.23}, {"type": "支撐", "price": 1034.5}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/7069669d-c381-4d05-b336-12da4b717c3b.png',
       'W-X-Y 調整的 (Y) 浪，等 (B) 浪反彈結束後 (C) 浪下探；1,381.85–1,034.50 為分批建倉區', 0, true, '2026-06-28T12:26:23.395951+00:00', '2026-06-28T12:26:23.395951+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = 'c61f0f43-85f2-44cc-b86c-f5d58254ab91');

-- 大同 2371：周線分析–底部反轉的潛在契機
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select '82afdc92-4576-47b8-9202-78b6dd150d48', 'tw', '2371', '1w', array['up']::text[], 36.15,
       array['pattern','volume','other']::text[], '[{"type": "支撐", "price": 25.55}, {"type": "支撐", "price": 24}, {"type": "冰線", "price": 19.75}, {"type": "POC", "price": 36.15}, {"type": "阻力", "price": 38.85}, {"type": "VAH", "price": 48.95}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/226d2148-6fd5-4f35-b71f-38ec3c0aef97.png',
       '下降收斂楔形末端＋上升通道底＋0.382＋VAL＋月線 OB 五重共振於 25.55，量縮竭盡；TP1 36.15', 0, true, '2026-07-01T04:01:31.500437+00:00', '2026-07-01T04:01:31.500437+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = '82afdc92-4576-47b8-9202-78b6dd150d48');

-- SOL空單交易邏輯：形態結構、流動性掠奪與時空週期的綜合解析
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select 'd751fefa-2cad-4b36-80bc-3d0a62fcff62', 'crypto', 'SOL', '1d', array['down']::text[], 60.13,
       array['pattern','wyckoff','other']::text[], '[{"type": "阻力", "price": 83.42}, {"type": "阻力", "price": 85.81}, {"type": "支撐", "price": 60.13}, {"type": "支撐", "price": 58}, {"type": "支撐", "price": 45.69}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/acf58a04-386e-4c94-8427-6a48f6c3b74c.png',
       '0.618–0.65 黃金口袋＋看跌 OB＋下降趨勢線，掃空頭流動性後類 UTAD；TP1 60.13，站穩 83.42/85.81 離場', 0, true, '2026-07-03T04:41:15.173929+00:00', '2026-07-03T04:41:15.173929+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = 'd751fefa-2cad-4b36-80bc-3d0a62fcff62');

-- BTC 底部確立：從威科夫量價與道氏結構看本次築底反轉
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select '7287359b-cfa5-4859-a796-dcccd83501f6', 'crypto', 'BTC', '1d', array['up']::text[], 82300,
       array['wyckoff','dow','volume']::text[], '[{"type": "支撐", "price": 74268}, {"type": "阻力", "price": 82300}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/e26a984b-5e65-4982-9713-4d6ce3f8b1cb.png',
       '第三個箱體量縮洗盤後放量 SOS，日線實體突破起跌點 74,268 完成 CHoCH；回踩量縮不破即延續，目標 82,300', 0, true, '2026-08-22T17:55:27.889015+00:00', '2026-08-22T17:55:27.889015+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = '7287359b-cfa5-4859-a796-dcccd83501f6');

-- 廣積 (8050) 技術面分析：杯柄型態與關鍵點位解析
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select '35602b7f-5bb7-4f68-beff-ead6773a3e0e', 'tw', '8050', '1d', array['up']::text[], 77.2,
       array['pattern','volume']::text[], '[{"type": "阻力", "price": 62.3}, {"type": "阻力", "price": 77.2}, {"type": "阻力", "price": 83.8}, {"type": "阻力", "price": 100}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/08a09a0f-dfd6-4258-82ac-91df36a2018a.png',
       '杯柄型態，柄部收斂三角；二測 62.3 量縮、縮量收陽止跌；77.2 先調節，等幅測距看 100', 0, true, '2026-08-24T05:51:52.694900+00:00', '2026-08-24T05:51:52.694900+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = '35602b7f-5bb7-4f68-beff-ead6773a3e0e');

-- NVDA：上升楔形量價背離，等待可能的 SOW
insert into public.forum_judgments
    (post_id, market, symbol, timeframe, directions, target_price, methods, levels, chart_url, reason, sort, backfilled, locked_at, created_at)
select '31e2dbe1-3edd-4b63-a90f-a807a5630ca0', 'us', 'NVDA', '1w', array['down']::text[], 183.01,
       array['pattern','wyckoff','volume']::text[], '[{"type": "阻力", "price": 235.56}, {"type": "POC", "price": 183.01}, {"type": "支撐", "price": 159}, {"type": "支撐", "price": 149.23}, {"type": "冰線", "price": 86.07}]'::jsonb, 'https://cimoqkrzuubulinlxako.supabase.co/storage/v1/object/public/forum-images/posts/b250ea2d-1e36-4531-96f6-3e7413d6f777.png',
       '週線上升楔形價漲量縮，8/27 爆量無距離；等 SOW，量幅目標 159，183–149 分批承接', 0, true, '2026-09-18T07:31:01.005605+00:00', '2026-09-18T07:31:01.005605+00:00'
 where not exists (select 1 from public.forum_judgments where post_id = '31e2dbe1-3edd-4b63-a90f-a807a5630ca0');

-- 沒有明確方向的文章：標記「本篇不含判讀」
update public.forum_posts set no_judgment = true where id = 'a176e94e-4714-4d1b-98a0-3b9a5ed85bfe';  -- 寶德 3349：明寫「不急著進場，等月線收盤確認方向」
update public.forum_posts set no_judgment = true where id = '47b4b35c-2f40-40d3-bf92-0bdc464edfe9';  -- 泰鼎 4927：「還沒過 61.1 都只是觀察」，不做方向判斷
update public.forum_posts set no_judgment = true where id = '3e074816-683f-4b83-a426-ec58cbf5ecd4';  -- 燿華 2367：「等確認，別猜方向」，兩種劇本都未證偽
update public.forum_posts set no_judgment = true where id = '7eae06ba-9d50-40c8-85c7-9c26d7a1ceeb';  -- MSFT：「不宜預判方向」，只列本週收盤四種劇本
update public.forum_posts set no_judgment = true where id = '1e163d95-eeda-483e-931f-5bbdf3b50b24';  -- 大同 2371 月線複盤：延續 7/1 的判讀，這篇是「等待 SOS」的觀察，沒有新方向

commit;

-- 檢查：應該看到 12 則補登判讀，每則都有一筆「驗證中」的結果
select j.symbol, j.timeframe, j.directions, j.target_price, j.locked_at, r.status
  from public.forum_judgments j join public.forum_judgment_results r on r.judgment_id = j.id
 where j.backfilled order by j.locked_at;
