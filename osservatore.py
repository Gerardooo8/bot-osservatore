import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Legge le chiavi in sicurezza dai GitHub Secrets
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

WATCHLIST = {
    "Oro (XAU/USD)": "GC=F",
    "Argento (XAG/USD)": "SI=F",
    "EUR/USD": "EURUSD=X",
    "Nasdaq 100 (US100)": "NQ=F",
    "S&P 500 (US500)": "ES=F"
}

TIMEFRAME = "15m"

def analizza_dati(ticker):
    df = yf.download(ticker, period="5d", interval=TIMEFRAME, progress=False)
    if df.empty or len(df) < 200:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    ultima = df.iloc[-1]
    precedente = df.iloc[-2]
    decimali = 4 if "=X" in ticker else 2

    return {
        "prezzo": round(float(ultima['Close']), decimali),
        "ema50": round(float(ultima['EMA50']), decimali),
        "ema200": round(float(ultima['EMA200']), decimali),
        "rsi_attuale": round(float(ultima['RSI']), 2),
        "rsi_precedente": round(float(precedente['RSI']), 2),
        "atr": round(float(ultima['ATR']), decimali)
    }

def chiedi_a_gemini(nome_asset, dati):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
    Sei un analista tecnico prudente. Analizza questi dati sul timeframe 15 minuti per {nome_asset}:
    - Prezzo attuale: {dati['prezzo']}
    - EMA 50: {dati['ema50']}
    - EMA 200: {dati['ema200']}
    - RSI attuale: {dati['rsi_attuale']} (candela precedente: {dati['rsi_precedente']})
    - ATR (volatilità): {dati['atr']}

    Regole:
    1. Valuta LONG solo se Prezzo > EMA200 e EMA50 > EMA200, e RSI sta rimbalzando dal basso (< 42).
    2. Valuta SHORT solo se Prezzo < EMA200 e EMA50 < EMA200, e RSI sta scendendo dall'alto (> 58).
    3. Calcola Stop Loss a 1.5 * ATR dal prezzo attuale, e Take Profit a 3.0 * ATR dal prezzo attuale.
    
    Se il setup è pulito e ha senso, rispondi ESCLUSIVAMENTE con un JSON valido così strutturato:
    {{"azione": "LONG o SHORT", "sl": numero, "tp": numero, "distanza_sl": numero, "distanza_tp": numero, "motivo": "breve frase in italiano"}}
    
    Se il setup è debole o ambiguo, rispondi solo: NO_TRADE
    """

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload, timeout=15)
    
    if response.status_code == 200:
        testo = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        if "NO_TRADE" in testo:
            return None
        testo = testo.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(testo)
        except json.JSONDecodeError:
            return None
    return None

def invia_telegram(messaggio):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": messaggio,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload, timeout=10)

def esegui_scansione():
    print(f"--- Scansione Cloud delle {datetime.now().strftime('%H:%M:%S')} UTC ---")
    for nome, ticker in WATCHLIST.items():
        dati = analizza_dati(ticker)
        if not dati:
            print(f"{nome}: Dati non disponibili.")
            continue

        possibile_long = dati['prezzo'] > dati['ema200'] and dati['rsi_attuale'] < 42 and dati['rsi_attuale'] > dati['rsi_precedente']
        possibile_short = dati['prezzo'] < dati['ema200'] and dati['rsi_attuale'] > 58 and dati['rsi_attuale'] < dati['rsi_precedente']

        if possibile_long or possibile_short:
            print(f"Pre-allarme su {nome} (RSI: {dati['rsi_attuale']}). Interrogo Gemini...")
            segnale = chiedi_a_gemini(nome, dati)
            if segnale:
                emoji = "🟢" if segnale['azione'] == "LONG" else "🔴"
                msg = (
                    f"{emoji} *SEGNALE {segnale['azione']} — {nome}*\n\n"
                    f"📊 *Prezzo rif.:* `{dati['prezzo']}`\n"
                    f"🛑 *Stop Loss:* `{segnale['sl']}` _(distanza: {segnale['distanza_sl']})_\n"
                    f"🎯 *Take Profit:* `{segnale['tp']}` _(distanza: {segnale['distanza_tp']})_\n\n"
                    f"💡 *Analisi:* {segnale['motivo']}\n"
                    f"⚠️ _Applica la distanza al prezzo attuale su XTrend Lite._"
                )
                invia_telegram(msg)
                print(f"Segnale inviato su Telegram per {nome}!")
            else:
                print(f"{nome}: Scartato da Gemini (NO_TRADE).")
        else:
            print(f"{nome}: Nessun setup (Prezzo: {dati['prezzo']} | EMA200: {dati['ema200']} | RSI: {dati['rsi_attuale']})")

if __name__ == "__main__":
    esegui_scansione()
