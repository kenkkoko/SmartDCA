import os
import requests
import yfinance as yf
import pandas as pd
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    BroadcastRequest,
    TextMessage
)
import google.generativeai as genai

import xml.etree.ElementTree as ET
from supabase import create_client, Client
from pywebpush import webpush, WebPushException
import json
import uuid
import datetime

from macro import (
    fetch_macro_snapshot,
    format_calendar_section,
    format_macro_section,
    get_macro_context,
    save_macro_snapshot,
)

# --- Configuration ---
# ⚠️ Critical: Read tokens from environment variables for security
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
# Use Backend-specific key if available, otherwise fallback to generic key
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY_BACKEND') or os.environ.get('GEMINI_API_KEY')

# NEW: Keys for Stock/Crypto Data
# NEW: Keys for Stock/Crypto Data
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY')
FUGLE_KEY = os.environ.get('FUGLE_KEY')

# Supabase Configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

# VAPID Keys for Web Push
VAPID_PRIVATE_KEY = '0yUZd0zYl6aJz7NscDAnkKhvVQUaQEacKZHo3vLRI9o' # Hardcoded as requested
VAPID_CLAIMS = {"sub": "mailto:admin@smartdca.com"}

# Thresholds
EXTREME_FEAR_THRESHOLD = 25
FEAR_THRESHOLD = 44

def format_price(price):
    """Formats price: 8 decimals if < 1, else 0 decimals (or 2)"""
    if price is None:
        return "N/A"
    if price < 1:
        return f"{price:.8f}"
    return f"{price:,.0f}"

def fetch_crypto_sentiment():
    """Fetches Crypto Fear & Greed Index from Alternative.me"""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        value = int(data['data'][0]['value'])
        return value
    except Exception as e:
        print(f"Error fetching Crypto sentiment: {e}")
        return None

def fetch_us_stock_sentiment():
    """Fetches US Stock Fear & Greed Index from CNN (or fallback)"""
    # Note: CNN often blocks scraper. If AV API key is present, we could calculate RSI, but sticking to CNN for FNG value.
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'fear_and_greed' in data:
            score = int(round(data['fear_and_greed']['score']))
            return score
        return None
    except Exception as e:
        print(f"Error fetching US Stock sentiment: {e}")
        return None

def fetch_tw_stock_rsi(ticker="0050.TW"):
    """Calculates RSI (14) for a TW stock using yfinance"""
    try:
        # Fetch 3 months of data to ensure enough for RSI calculation
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 15:
            return None
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        if isinstance(current_rsi, pd.Series):
             current_rsi = current_rsi.iloc[0]

        return int(round(current_rsi))
    except Exception as e:
        print(f"Error fetching TW Stock RSI: {e}")
        return None

def fetch_price_stats(ticker):
    """Fetches current price and 1-year high/low"""
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="1y")
        if hist.empty:
            return None
        
        current_price = hist['Close'].iloc[-1]
        year_high = hist['Close'].max()
        year_low = hist['Close'].min()
        
        return {
            "current": current_price,
            "high": year_high,
            "low": year_low
        }
    except Exception as e:
        print(f"Error fetching price stats for {ticker}: {e}")
        return None

def fetch_news_headlines(query="財經"):
    """Fetches news headlines from Google News RSS"""
    try:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        headlines = []
        # Get top 5 items
        for item in root.findall('./channel/item')[:5]:
            title = item.find('title').text
            headlines.append(title)
            
        return headlines
    except Exception as e:
        print(f"Error fetching news for {query}: {e}")
        return []

def get_status_emoji(value):
    if value <= EXTREME_FEAR_THRESHOLD:
        return "🔴" # Extreme Fear
    if value <= FEAR_THRESHOLD:
        return "🟠" # Fear
    return "🔵" # Neutral/Greed

def get_status_text(value, is_rsi=False):
    if value <= EXTREME_FEAR_THRESHOLD:
        return "極度恐懼"
    if value <= FEAR_THRESHOLD:
        return "RSI偏低" if is_rsi else "恐懼"
    return "安全/貪婪"

def generate_ai_advice(market_status_list, macro_context=""):
    """Generates DCA advice using Gemini AI"""
    if not GEMINI_API_KEY:
        return "⚠️ AI 建議無法產生 (未設定 API Key)"

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""
        你是一位極度穩健的 DCA (平均成本法) 投資顧問。你的核心策略是嚴格遵守「在市場情緒極度恐懼時才強力買入」的紀律。

        請根據以下觸發的市場數據，提供一個**簡潔、明確**的操作建議 (80字以內)。

        核心任務：
        1. 分析當前的 FNG/RSI 數值所代表的市場情緒強度。
        2. 根據情緒強度，結合資產名稱和當前價格，**相較於最近一年的價格波動 (參考最高/最低價)**，判斷現在的價格是否具有吸引力？(注意：小幣種價格可能包含多位小數)。
        3. 根據以下行動邏輯，生成一段富有洞察力和鼓勵性的建議。
        4. 參考總經背景修正建議的力道：殖利率快速上升、CPI 連續升溫時語氣應更保守；
           24 小時內有高影響事件 (CPI、非農、FOMC) 時，提醒把單筆投入拆到事件公布後。
           情緒指數仍是主要依據，總經只調整力道與時機，不要推翻行動邏輯。

        行動邏輯：
        - 極度恐懼 (<= 25): 立即建議「強力分批買入」或「執行最大額度投入」。
        - 恐懼 (26 - 44): 建議「小額分批買入」，鼓勵保持紀律。
        - 中立 (45 - 55): 建議「維持觀望，不買也不賣」。
        - 貪婪 (56 - 74) 極度貪婪 (>= 75):: 建議「停止買入，開始小額分批賣出 (止盈)」。

        當前觸發的市場狀態:
        {chr(10).join(market_status_list)}

        總經背景與經濟日曆:
        {macro_context or "（今日無總經資料）"}

        根據以上資訊，你的行動建議是？
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating AI advice: {e}")
        return "⚠️ AI 暫時無法提供建議"

def generate_news_summary(headlines):
    """Generates a summary of news headlines using Gemini AI"""
    if not GEMINI_API_KEY or not headlines:
        return ""

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        headlines_text = "\n".join(headlines)
        prompt = f"""
        請閱讀以下今日財經/加密貨幣新聞標題，並總結成一段 50 字以內的「今日市場重點」。
        
        新聞標題:
        {headlines_text}
        
        請用繁體中文回答，語氣專業客觀。
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating news summary: {e}")
        return ""

def broadcast_push_notifications(general_market_status):
    """Sends personalized Web Push to Premium Users"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Skipping Web Push: Supabase credentials missing.")
        return

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Fetch Premium Users with Subscription
        response = supabase.table('user_profiles') \
            .select('id, watchlist, push_subscription') \
            .eq('is_premium', True) \
            .neq('push_subscription', 'null') \
            .execute()
        
        users = response.data
        if not users:
            print("No premium users with push subscription found.")
            return

        print(f"Found {len(users)} premium users for push.")

        # Cache for market data to avoid re-fetching
        data_cache = {}

        for user in users:
            sub_info = user.get('push_subscription')
            watchlist = user.get('watchlist', [])
            if not sub_info or not watchlist:
                continue

            # Build Personalized Message
            user_alerts = []
            
            for symbol in watchlist:
                # Normalize Symbol for Fetcher
                fetch_symbol = symbol
                is_tw = False
                is_crypto = False
                
                # Enhanced Crypto Detection: Check common coins or pattern
                # If symbol contains 'USD' or matches crypto list
                common_crypto = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'bitcoin', 'ethereum', 'solana', 'dogecoin', 'ripple', 'binancecoin', 'BNB']
                if symbol.upper() in [c.upper() for c in common_crypto] or 'USD' in symbol.upper():
                    if 'USD' not in symbol.upper():
                        # Simple mapping for yfinance if needed, but mainly for classification
                        if symbol.lower() == 'bitcoin': fetch_symbol = 'BTC-USD'
                        elif symbol.lower() == 'ethereum': fetch_symbol = 'ETH-USD'
                        elif symbol.lower() == 'solana': fetch_symbol = 'SOL-USD'
                        else: fetch_symbol = symbol.upper() + "-USD"
                        
                    is_crypto = True
                
                elif symbol.isdigit() or '.TW' in symbol.upper():
                    fetch_symbol = symbol.replace('.TW', '') + ".TW" if not symbol.endswith('.TW') else symbol
                    is_tw = True
                
                # Check Cache
                stats = data_cache.get(fetch_symbol)
                if not stats:
                    if is_tw:
                        # For TW, logic is RSI based mainly
                        rsi = fetch_tw_stock_rsi(fetch_symbol)
                        price = fetch_price_stats(fetch_symbol) # May accept .TW
                        if rsi:
                            stats = {'rsi': rsi, 'price': price}
                            data_cache[fetch_symbol] = stats
                    else:
                        # US/Crypto
                        price = fetch_price_stats(fetch_symbol)
                        stats = {'price': price}
                        data_cache[fetch_symbol] = stats

                # Analyze Logic
                # Crypto
                if is_crypto:
                    fng = fetch_crypto_sentiment()
                    if fng and fng <= FEAR_THRESHOLD:
                         p_curr = stats.get('price', {}).get('current', 0) if stats.get('price') else 0
                         # Simplified: Symbol, FNG, Current Price
                         user_alerts.append(f"🪙 {symbol}: FNG {fng} (Buy) ${format_price(p_curr)}")
                
                # TW Stock
                elif is_tw:
                    # Always try to fetch price/rsi
                    rsi = stats.get('rsi') if stats else None
                    p_curr = stats.get('price', {}).get('current', 0) if stats and stats.get('price') else 0
                    
                    if rsi is not None:
                        status = "(Buy)" if rsi <= FEAR_THRESHOLD else "(Neutral)"
                        # Show all or only buy? User said "missing info", so maybe they want to see it regardless or just know if it's broken.
                        # For now, let's show it if it's a Buy OR if it's just for reporting purpose.
                        # But to fix "missing info", we should probably show it if it's in the watchlist.
                        # However, to avoid spam, maybe only show if Buy OR if requested?
                        # Let's show "Neutral" for now to prove data exists.
                        user_alerts.append(f"🇹🇼 {symbol}: RSI {rsi:.1f} {status} ${format_price(p_curr)}")
                    else:
                        # Data missing case
                        user_alerts.append(f"🇹🇼 {symbol}: RSI N/A (Data Error) ${format_price(p_curr)}")
                
                # US Stock (General FNG applied to specific stock)
                elif not is_tw and not is_crypto:
                    fng = fetch_us_stock_sentiment()
                    if fng and fng <= FEAR_THRESHOLD:
                         p_curr = stats.get('price', {}).get('current', 0) if stats.get('price') else 0
                         # Simplified: Symbol, FNG, Current Price
                         user_alerts.append(f"🇺🇸 {symbol}: FNG {fng} (Buy) ${format_price(p_curr)}")

            # Send Push if alerts exist (or daily report?)
            # Modified: Always send a report to ensure user gets feedback
            title = ""
            payload = ""
            if user_alerts:
                title = "🔥 自選股 DCA 訊號 Triggered!"
                payload = "\n".join(user_alerts)
            else:
                # Fallback report for user verification
                title = "📊 每日市場報告"
                payload = "您的自選股今日無 DCA 買入訊號 (市場情緒穩定)。"

            try:
                # 1. Send Web Push
                full_message = f"{title}\n{payload}"
                webpush(
                    subscription_info=sub_info,
                    data=full_message,
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims=VAPID_CLAIMS
                )
                print(f"Push sent to user {user['id']}")

                # 2. Log to Database (Update user_profiles)
                # Fetch current notifications first (to append)
                current_profile = supabase.table('user_profiles').select('notifications').eq('id', user['id']).execute()
                current_notifications = []
                if current_profile.data and current_profile.data[0].get('notifications'):
                    current_notifications = current_profile.data[0]['notifications']
                
                new_notification = {
                    'id': str(uuid.uuid4()), # Generate ID for frontend key
                    'title': title,
                    'body': payload,
                    'created_at': datetime.datetime.now().isoformat(),
                    'is_read': False
                }
                
                # Prepend and slice to keep last 20
                updated_notifications = [new_notification] + current_notifications
                updated_notifications = updated_notifications[:20]
                
                supabase.table('user_profiles').update({
                    'notifications': updated_notifications
                }).eq('id', user['id']).execute()
                
                print(f"Notification logged to user_profiles for user {user['id']}")

            except WebPushException as ex:
                print(f"Push failed for user {user['id']}: {ex}")
                # Remove invalid subscription?

    except Exception as e:
        print(f"Broadcasting Push Error: {e}")

def main():
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("Error: LINE_CHANNEL_ACCESS_TOKEN not set.")
        # return # Allow running locally without LINE token for debug

    print("Fetching market data...")
    crypto_fng = fetch_crypto_sentiment()
    us_stock_fng = fetch_us_stock_sentiment()
    tw_stock_rsi = fetch_tw_stock_rsi()

    print(f"Crypto: {crypto_fng}")
    print(f"US Stock: {us_stock_fng}")
    print(f"TW Stock (RSI): {tw_stock_rsi}")

    # --- Macro data (Alpha Vantage) + Economic calendar ---
    print("Fetching macro data...")
    macro_snapshot = fetch_macro_snapshot(ALPHA_VANTAGE_KEY)
    print(f"Macro: {macro_snapshot}")
    macro_text = format_macro_section(macro_snapshot)
    calendar_text = format_calendar_section(macro_snapshot)
    macro_context = get_macro_context(macro_snapshot)
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            save_macro_snapshot(create_client(SUPABASE_URL, SUPABASE_KEY), macro_snapshot)
        except Exception as e:
            print(f"Error creating Supabase client for macro_snapshot: {e}")

    # Collect status for ALL markets
    market_status_list = []
    has_buy_signal = False
    
    # Crypto Logic
    if crypto_fng is not None:
        status_icon = get_status_emoji(crypto_fng)
        status_text = get_status_text(crypto_fng)
        msg = f"🪙 加密貨幣: {crypto_fng} ({status_text} {status_icon})"
        
        # Always fetch Price Stats for BTC and ETH
        btc_stats = fetch_price_stats("BTC-USD")
        if btc_stats:
            msg += f"\n   - BTC: ${format_price(btc_stats['current'])} (1Y High: ${format_price(btc_stats['high'])}, Low: ${format_price(btc_stats['low'])})"
            
        eth_stats = fetch_price_stats("ETH-USD")
        if eth_stats:
            msg += f"\n   - ETH: ${format_price(eth_stats['current'])} (1Y High: ${format_price(eth_stats['high'])}, Low: ${format_price(eth_stats['low'])})"
            
        market_status_list.append(msg)
        if crypto_fng <= FEAR_THRESHOLD:
            has_buy_signal = True
    
    # US Stock Logic
    if us_stock_fng is not None:
        status_icon = get_status_emoji(us_stock_fng)
        status_text = get_status_text(us_stock_fng)
        msg = f"🇺🇸 美股: {us_stock_fng} ({status_text} {status_icon})"
        
        # Add SPY Stats
        spy_stats = fetch_price_stats("SPY")
        if spy_stats:
             msg += f"\n   - SPY: ${format_price(spy_stats['current'])} (1Y High: ${format_price(spy_stats['high'])}, Low: ${format_price(spy_stats['low'])})"
        
        # Add Gold Stats (GLD)
        gld_stats = fetch_price_stats("GLD")
        if gld_stats:
             msg += f"\n   - GOLD: ${format_price(gld_stats['current'])} (1Y High: ${format_price(gld_stats['high'])}, Low: ${format_price(gld_stats['low'])})"

        market_status_list.append(msg)
        if us_stock_fng <= FEAR_THRESHOLD:
            has_buy_signal = True
        
    # TW Stock Logic
    if tw_stock_rsi is not None:
        status_icon = get_status_emoji(tw_stock_rsi)
        status_text = get_status_text(tw_stock_rsi, is_rsi=True)
        msg = f"🇹🇼 台股(0050): {tw_stock_rsi} ({status_text} {status_icon})"
        
        # Add 0050 Stats
        tw50_stats = fetch_price_stats("0050.TW")
        if tw50_stats:
             msg += f"\n   - 0050: ${format_price(tw50_stats['current'])} (1Y High: ${format_price(tw50_stats['high'])}, Low: ${format_price(tw50_stats['low'])})"

        market_status_list.append(msg)
        if tw_stock_rsi <= FEAR_THRESHOLD:
            has_buy_signal = True

    # Determine Header
    if has_buy_signal:
        header = "🔥 Smart DCA 訊號觸發 🔥"
    else:
        header = "📊 每日市場觀察報告"

    # Construct Message
    message_text = f"{header}\n\n"
    message_text += "\n\n".join(market_status_list)
    if macro_text:
        message_text += f"\n\n{macro_text}"
    if calendar_text:
        message_text += f"\n\n{calendar_text}"

    # Generate News Summary
    print("Fetching and summarizing news...")
    # Fetch both Crypto and Finance/Stock news
    news_headlines = fetch_news_headlines("加密貨幣") + fetch_news_headlines("財經")
    news_summary = generate_news_summary(news_headlines)
    
    if news_summary:
        message_text += f"\n\n📰 AI 今日新聞重點:\n{news_summary}"

    # Generate AI Advice (Always generate)
    print("Generating AI advice...")
    ai_advice = generate_ai_advice(market_status_list, macro_context)
    message_text += f"\n\n🤖 AI 投資顧問建議:\n{ai_advice}"
    
    if has_buy_signal:
        message_text += "\n\n💡 建議分批進場"
    else:
        message_text += "\n\n💡 市場情緒穩定，請持續觀察"

    # --- PWA Web Push Logic (Premium Users) ---
    print("Broadcasting Web Push notifications...")
    broadcast_push_notifications(market_status_list) # Pass general status, but we will customize inside

    # --- LINE Notification ---
    print("Broadcasting LINE notification...")
    if LINE_CHANNEL_ACCESS_TOKEN:
        try:
            configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
            api_client = ApiClient(configuration)
            messaging_api = MessagingApi(api_client)

            # Broadcast Request
            broadcast_request = BroadcastRequest(
                messages=[TextMessage(text=message_text)]
            )
            
            messaging_api.broadcast(broadcast_request)
            print("Broadcast sent successfully!")

        except Exception as e:
            print(f"Error sending LINE notification: {e}")
    else:
        print("Skipped Broadcast (No Token)") # Friendly verify for local run

if __name__ == "__main__":
    main()
