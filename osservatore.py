import os
import time
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# Legge le chiavi in sicurezza dai GitHub Secrets
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

WATCHLIST = {
    # Metalli
    "Oro (XAU/USD)": "GC=F",
    "Argento (XAG/USD)": "SI=F",
    
    # Indici
    "Nasdaq 100 (US100)": "NQ=F",
    "S&P 500 (US500)": "ES=F",
    "Dow Jones (US30)": "YM=F",
    "DAX (GER40)": "^GDAXI",
    "FTSE 100 (UK100)": "^FTSE",
    
    # Forex Major & Cross
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X"
}

TIMEFRAME = "15m"
FILE_STORICO = "storico_giorno.json"
FUSO_ITALIA = ZoneInfo("Europe/Rome")

def carica_storico():
    if os.path.exists(FILE_STORICO):
        with open(FILE_STORICO, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"data": "", "buongiorno_inviato": False, "chiusura_inviata": False, "segnali": []}

def salva_storico(dati):
    with open(FILE_STORICO, "w", encoding="utf-8") as f:
        json.dump(dati, f, indent=2, ensure_ascii=False)

def analizza_dati(ticker):
    try:
        df = yf.download(ticker, period="5d", interval=TIMEFRAME, progress=False)
        if df.empty or len(df) < 200:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
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
        
        if "JPY" in ticker or "=X" not in ticker:
            decimali = 2
        else:
            decimali = 4

        return {
            "prezzo": round(float(ultima['Close']), decimali),
            "prezzo_prec": round(float(precedente['Close']), decimali),
            "ema20": round(float(ultima['EMA20']), decimali),
            "ema50": round(float(ultima['EMA50']), decimali),
            "ema50_prec": round(float(precedente['EMA50']), decimali),
            "ema200": round(float(ultima['EMA200']), decimali),
            "rsi_attuale": round(float(ultima['RSI']), 2),
            "rsi_precedente": round(float(precedente['RSI']), 2),
            "atr": round(float(ultima['ATR']), decimali),
            "decimali": decimali
        }
    except Exception as e:
        print(f"Errore download {ticker}: {e}")
        return None

def chiedi_a_gemini(nome_asset, dati, tipo_setup):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
    Sei un analista tecnico operativo sul timeframe 15 minuti. Valuta questa possibile opportunità su {nome_asset} (Segnale preliminare: {tipo_setup}):
    - Prezzo attuale: {dati['prezzo']} (candela precedente: {dati['prezzo_prec']})
    - EMA 20: {dati['ema20']}
    - EMA 50: {dati['ema50']}
    - EMA 200: {dati['ema200']}
    - RSI attuale: {dati['rsi_attuale']} (candela precedente: {dati['rsi_precedente']})
    - ATR (volatilità): {dati['atr']}

    Istruzioni:
    1. Se i dati mostrano una direzione chiara (rialzista o ribassista) tramite rimbalzo sul trend o rottura della EMA50/EMA20, conferma l'operazione (LONG o SHORT).
    2. Scarta con NO_TRADE solo se il prezzo è completamente intrappolato in laterale senza direzione o contro ogni media mobile.
    3. Calcola:
       - distanza_sl = round(1.5 * ATR, {dati['decimali']})
       - distanza_tp = round(2.5 * ATR, {dati['decimali']})
       - Per LONG: sl = Prezzo - distanza_sl, tp = Prezzo + distanza_tp
       - Per SHORT: sl = Prezzo + distanza_sl, tp = Prezzo - distanza_tp
    
    Rispondi ESCLUSIVAMENTE con un JSON valido così strutturato:
    {{"azione": "LONG o SHORT", "confidenza": "ALTA o MEDIA", "sl": numero, "tp": numero, "distanza_sl": numero, "distanza_tp": numero, "motivo": "spiegazione sintetica e chiara in italiano"}}
    
    Oppure rispondi solo: NO_TRADE
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
        "text": messaggio
    }
    risposta = requests.post(url, json=payload, timeout=10)
    if risposta.status_code != 200:
        print(f"ERRORE TELEGRAM ({risposta.status_code}): {risposta.text}")
    else:
        print("Conferma Telegram: messaggio consegnato!")

def esegui_scansione():
    ora_it = datetime.now(FUSO_ITALIA)
    oggi_str = ora_it.strftime("%Y-%m-%d")
    ora_str = ora_it.strftime("%H:%M")
    
    print(f"--- Scansione delle {ora_str} (Ora Italiana) ---")
    
    storico = carica_storico()
    if storico.get("data") != oggi_str:
        storico = {
            "data": oggi_str,
            "buongiorno_inviato": False,
            "chiusura_inviata": False,
            "segnali": []
        }

    riepilogo_principali = []
    asset_principali = ["Oro (XAU/USD)", "EUR/USD", "GBP/USD", "Nasdaq 100 (US100)", "S&P 500 (US500)"]

    for nome, ticker in WATCHLIST.items():
        dati = analizza_dati(ticker)
        if not dati:
            print(f"{nome}: Dati non disponibili.")
            continue

        if nome in asset_principali:
            riepilogo_principali.append(f"• {nome}: {dati['prezzo']} (RSI: {dati['rsi_attuale']})")

        # 1. Setup Rimbalzo (Pullback) più flessibile (RSI < 50 per Long, > 50 per Short)
        pullback_long = dati['prezzo'] > dati['ema200'] and dati['rsi_attuale'] < 50 and dati['rsi_attuale'] > dati['rsi_precedente'] + 1.5
        pullback_short = dati['prezzo'] < dati['ema200'] and dati['rsi_attuale'] > 50 and dati['rsi_attuale'] < dati['rsi_precedente'] - 1.5

        # 2. Setup Momentum (Rottura decisa della EMA50)
        breakout_long = dati['prezzo_prec'] <= dati['ema50_prec'] and dati['prezzo'] > dati['ema50'] and dati['rsi_attuale'] > 52
        breakout_short = dati['prezzo_prec'] >= dati['ema50_prec'] and dati['prezzo'] < dati['ema50'] and dati['rsi_attuale'] < 48

        tipo_setup = None
        if pullback_long:
            tipo_setup = "Rimbalzo Rialzista (Pullback LONG)"
        elif pullback_short:
            tipo_setup = "Rimbalzo Ribassista (Pullback SHORT)"
        elif breakout_long:
            tipo_setup = "Rottura Rialzista EMA50 (Breakout LONG)"
        elif breakout_short:
            tipo_setup = "Rottura Ribassista EMA50 (Breakout SHORT)"

        if tipo_setup:
            print(f"Pre-allarme su {nome}: {tipo_setup}. Interrogo Gemini...")
            segnale = chiedi_a_gemini(nome, dati, tipo_setup)
            if segnale:
                emoji = "🟢" if segnale['azione'] == "LONG" else "🔴"
                msg = (
                    f"{emoji} SEGNALE {segnale['azione']} — {nome}\n"
                    f"⚡ Confidenza: {segnale.get('confidenza', 'MEDIA')} ({tipo_setup})\n\n"
                    f"📊 Prezzo rif.: {dati['prezzo']}\n"
                    f"🛑 Stop Loss: {segnale['sl']} (distanza: {segnale['distanza_sl']})\n"
                    f"🎯 Take Profit: {segnale['tp']} (distanza: {segnale['distanza_tp']})\n\n"
                    f"💡 Analisi: {segnale['motivo']}\n"
                    f"⚠️ Su MetaTrader 4 applica la distanza al prezzo attuale."
                )
                invia_telegram(msg)
                storico["segnali"].append(f"{ora_str} — {segnale['azione']} {nome} ({segnale.get('confidenza', 'MEDIA')})")
                print(f"Segnale inviato su Telegram per {nome}!")
                time.sleep(2)
            else:
                print(f"{nome}: Scartato da Gemini (NO_TRADE).")
        else:
            print(f"{nome}: Nessun setup (Prezzo: {dati['prezzo']} | RSI: {dati['rsi_attuale']})")
        
        time.sleep(1)

    # 1. MESSAGGIO DI APERTURA (dalle 08:00 in poi)
    if ora_it.hour >= 8 and not storico["buongiorno_inviato"]:
        testo_prezzi = "\n".join(riepilogo_principali)
        msg_apertura = (
            f"☀️ Mercati aperti, sono operativo!\n"
            f"Monitoraggio attivo su 18 mercati (Metalli, Indici e Forex) su timeframe 15m.\n\n"
            f"Panoramica principali:\n{testo_prezzi}"
        )
        invia_telegram(msg_apertura)
        storico["buongiorno_inviato"] = True
        print("Messaggio di apertura inviato!")

    # 2. MESSAGGIO DI CHIUSURA (dalle 22:00 in poi)
    if ora_it.hour >= 22 and not storico["chiusura_inviata"]:
        totale = len(storico["segnali"])
        if totale == 0:
            dettaglio = "Nessun segnale inviato oggi: nessuno dei 18 mercati ha offerto setup direzionali."
        else:
            lista = "\n".join([f"• {s}" for s in storico["segnali"]])
            dettaglio = f"Abbiamo individuato {totale} operazioni:\n{lista}"
            
        msg_chiusura = (
            f"🌙 Giornata conclusa, mercati in chiusura.\n\n"
            f"{dettaglio}\n\n"
            f"A domani mattina alle 08:00!"
        )
        invia_telegram(msg_chiusura)
        storico["chiusura_inviata"] = True
        print("Messaggio di chiusura inviato!")

    salva_storico(storico)

if __name__ == "__main__":
    esegui_scansione()
