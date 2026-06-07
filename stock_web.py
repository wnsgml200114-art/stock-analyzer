import time
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import feedparser
import plotly.graph_objects as go


st.set_page_config(page_title="AI 주식 분석기", layout="wide")
st.title("AI 주식 분석기 웹버전")

KST = ZoneInfo("Asia/Seoul")


def now_kst():
    return datetime.now(KST)


def format_kst(dt=None):
    if dt is None:
        dt = now_kst()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# =========================
# 포맷
# =========================

def format_price(value, market):
    try:
        value = float(value)
        if market == "국내":
            return f"{value:,.0f}원"
        return f"${value:,.2f}"
    except Exception:
        return "정보 없음"


def format_number(value):
    try:
        if value is None or pd.isna(value):
            return "정보 없음"
        return f"{float(value):,.2f}"
    except Exception:
        return "정보 없음"


def format_percent(value):
    try:
        if value is None or pd.isna(value):
            return "정보 없음"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "정보 없음"


def format_korean_money(value):
    try:
        if value is None or pd.isna(value):
            return "정보 없음"

        value = int(value)
        jo = value // 1_0000_0000_0000
        eok = (value % 1_0000_0000_0000) // 1_0000_0000

        if jo > 0 and eok > 0:
            return f"{jo}조 {eok:,}억"
        if jo > 0:
            return f"{jo}조"
        if eok > 0:
            return f"{eok:,}억"
        return f"{value:,}"
    except Exception:
        return "정보 없음"


def format_usd_money(value):
    try:
        if value is None or pd.isna(value):
            return "정보 없음"

        value = float(value)
        if value >= 1_000_000_000_000:
            return f"${value / 1_000_000_000_000:.2f}T"
        if value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        if value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        return f"${value:,.0f}"
    except Exception:
        return "정보 없음"


def format_market_cap(value, market):
    if market == "국내":
        return format_korean_money(value)
    return format_usd_money(value)


# =========================
# 종목 검색 / 데이터
# =========================

@st.cache_data(ttl=3600)
def get_krx_listing():
    stock_list = fdr.StockListing("KRX")
    return stock_list[["Code", "Name", "Market"]]


def find_korean_stock_code(keyword):
    stock_list = get_krx_listing()

    exact = stock_list[stock_list["Name"] == keyword]
    if not exact.empty:
        row = exact.iloc[0]
        return row["Code"], row["Name"], row["Market"]

    contains = stock_list[stock_list["Name"].str.contains(keyword, na=False)]
    if not contains.empty:
        row = contains.iloc[0]
        return row["Code"], row["Name"], row["Market"]

    if keyword.isdigit():
        return keyword.zfill(6), keyword, "KOSPI"

    return None, None, None


@st.cache_data(ttl=30)
def load_stock_data(keyword, market):
    if market == "국내":
        code, name, krx_market = find_korean_stock_code(keyword)

        if code is None:
            return None, None, None, None

        df = fdr.DataReader(code, "2000-01-01")
        yf_symbol = code + ".KQ" if krx_market == "KOSDAQ" else code + ".KS"
        return df.dropna(), code, name, yf_symbol

    code = keyword.upper()
    df = yf.download(code, start="2000-01-01", auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna(), code, code, code


def convert_candle(df, candle):
    if candle == "일봉":
        return df

    rule = {
        "주봉": "W",
        "월봉": "ME",
        "년봉": "YE"
    }[candle]

    return df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


def filter_range(df, candle, period):
    if period == "전체":
        return df

    n = int("".join([x for x in period if x.isdigit()]))

    if "일" in period:
        return df.tail(n)
    if "주" in period:
        return df.tail(n)
    if "개월" in period:
        return df.tail(n)
    if "년" in period:
        if candle == "년봉":
            return df.tail(n)
        if candle == "월봉":
            return df.tail(n * 12)
        if candle == "주봉":
            return df.tail(n * 52)
        return df.tail(n * 245)

    return df


# =========================
# 지표
# =========================

def add_indicators(df):
    df = df.copy()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    df["BB_MID"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + std * 2
    df["BB_LOWER"] = df["BB_MID"] - std * 2

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - prev_close),
        abs(df["Low"] - prev_close)
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    df["OBV"] = 0.0
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("OBV")] = df["OBV"].iloc[i - 1] + df["Volume"].iloc[i]
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("OBV")] = df["OBV"].iloc[i - 1] - df["Volume"].iloc[i]
        else:
            df.iloc[i, df.columns.get_loc("OBV")] = df["OBV"].iloc[i - 1]

    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATE"] = ((df["Volume"] / df["VOL_MA20"]) - 1) * 100

    return df.dropna()


# =========================
# 분석
# =========================

def analyze_volume(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    total_volume = latest["Volume"]
    avg_volume = latest["VOL_MA20"]
    volume_rate = latest["VOL_RATE"]

    if latest["Close"] > prev["Close"]:
        volume_signal = "매수세 우세 추정"
        buy_est = total_volume
        sell_est = 0
    elif latest["Close"] < prev["Close"]:
        volume_signal = "매도세 우세 추정"
        buy_est = 0
        sell_est = total_volume
    else:
        volume_signal = "중립 추정"
        buy_est = 0
        sell_est = 0

    obv_now = latest["OBV"]
    obv_prev = df["OBV"].iloc[-5] if len(df) >= 5 else prev["OBV"]

    if obv_now > obv_prev:
        obv_signal = "OBV 상승: 거래량 흐름 개선"
    elif obv_now < obv_prev:
        obv_signal = "OBV 하락: 거래량 흐름 약화"
    else:
        obv_signal = "OBV 보합"

    return {
        "total_volume": total_volume,
        "avg_volume": avg_volume,
        "volume_rate": volume_rate,
        "buy_est": buy_est,
        "sell_est": sell_est,
        "volume_signal": volume_signal,
        "obv": obv_now,
        "obv_signal": obv_signal
    }


def get_score(df, financial_score=None, news_score=None):
    latest = df.iloc[-1]
    score = 50
    reasons = []

    if latest["MA5"] > latest["MA20"]:
        score += 10
        reasons.append("5선이 20선 위에 있어 단기 흐름이 좋습니다.")
    else:
        score -= 8
        reasons.append("5선이 20선 아래에 있어 단기 흐름이 약합니다.")

    if latest["MA20"] > latest["MA60"]:
        score += 8
        reasons.append("20선이 60선 위에 있어 중기 흐름이 좋습니다.")
    else:
        score -= 8
        reasons.append("20선이 60선 아래에 있어 중기 흐름이 약합니다.")

    if latest["Close"] > latest["MA60"]:
        score += 8
        reasons.append("현재가가 60선 위에 있어 장기 흐름이 양호합니다.")
    else:
        score -= 8
        reasons.append("현재가가 60선 아래에 있어 장기 흐름이 약합니다.")

    if 30 <= latest["RSI"] <= 45:
        score += 12
        reasons.append("RSI가 저점 반등 관심 구간입니다.")
    elif latest["RSI"] < 30:
        score += 5
        reasons.append("RSI가 과매도 구간입니다. 반등 가능성은 있지만 추가 하락도 주의해야 합니다.")
    elif latest["RSI"] >= 70:
        score -= 15
        reasons.append("RSI가 과열 구간입니다.")
    else:
        reasons.append("RSI가 보통 구간입니다.")

    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 10
        reasons.append("MACD가 Signal 위에 있어 상승 흐름 가능성이 있습니다.")
    else:
        score -= 8
        reasons.append("MACD가 Signal 아래에 있어 모멘텀이 약합니다.")

    if latest["Volume"] > latest["VOL_MA20"] * 1.5:
        score += 10
        reasons.append("거래량이 20개 봉 평균보다 크게 증가했습니다.")
    elif latest["Volume"] > latest["VOL_MA20"]:
        score += 5
        reasons.append("거래량이 평균보다 증가했습니다.")
    else:
        reasons.append("거래량 증가가 약합니다.")

    obv_now = latest["OBV"]
    obv_prev = df["OBV"].iloc[-5] if len(df) >= 5 else df["OBV"].iloc[-2]

    if obv_now > obv_prev:
        score += 5
        reasons.append("OBV가 상승해 거래량 흐름이 개선되고 있습니다.")
    else:
        score -= 5
        reasons.append("OBV가 하락 또는 정체되어 거래량 흐름이 약합니다.")

    if financial_score is not None:
        if financial_score >= 70:
            score += 5
            reasons.append("재무 점수가 양호해 종합 점수에 긍정적으로 반영했습니다.")
        elif financial_score <= 40:
            score -= 5
            reasons.append("재무 점수가 낮아 종합 점수에 부정적으로 반영했습니다.")

    if news_score is not None:
        if news_score > 0:
            score += min(news_score, 5)
            reasons.append("뉴스 분위기가 긍정적으로 나타났습니다.")
        elif news_score < 0:
            score += max(news_score, -5)
            reasons.append("뉴스 분위기가 부정적으로 나타났습니다.")

    score = max(0, min(100, score))

    if score >= 80:
        grade = "A / 매수 관심 강함"
    elif score >= 65:
        grade = "B / 분할매수 검토"
    elif score >= 50:
        grade = "C / 관망"
    elif score >= 35:
        grade = "D / 주의"
    else:
        grade = "E / 위험"

    return score, grade, reasons


def get_entry_and_levels(df):
    latest = df.iloc[-1]

    close = latest["Close"]
    ma20 = latest["MA20"]
    ma60 = latest["MA60"]
    atr = latest["ATR"]
    bb_upper = latest["BB_UPPER"]
    bb_lower = latest["BB_LOWER"]

    target1 = close + atr * 1.5
    target2 = close + atr * 2.5
    take_profit = min(bb_upper, close + atr * 2.0)
    stop_loss = max(bb_lower, close - atr * 1.2)

    reasons = []

    if close > ma20 and latest["MA5"] > ma20 and latest["MACD"] > latest["MACD_SIGNAL"]:
        entry = "분할매수 검토 가능"
        reasons.append("현재가가 20선 위에 있고 5선도 20선 위에 있습니다.")
        reasons.append("MACD가 Signal 위에 있어 상승 흐름 가능성이 있습니다.")
    elif close < ma20 and 30 <= latest["RSI"] <= 45:
        entry = "반등 확인 후 진입"
        reasons.append("RSI는 저점 반등 관심 구간입니다.")
        reasons.append("현재가가 20선 아래라 20선 회복 확인이 필요합니다.")
    elif latest["RSI"] < 30:
        entry = "과매도 구간, 신중한 분할 접근"
        reasons.append("RSI가 30 이하로 과매도 구간입니다.")
        reasons.append("반등 가능성은 있지만 추가 하락도 주의해야 합니다.")
    else:
        entry = "관망"
        reasons.append("이동평균선, RSI, MACD 조건이 아직 강하지 않습니다.")

    grounds = [
        f"ATR 기준 변동폭: {atr:,.2f}",
        "1차 목표가는 현재가 + ATR × 1.5 기준입니다.",
        "2차 목표가는 현재가 + ATR × 2.5 기준입니다.",
        "익절가는 볼린저밴드 상단과 ATR 기준 목표가 중 보수적인 값을 사용했습니다.",
        "손절가는 볼린저밴드 하단과 ATR 기준 손절가 중 보수적인 값을 사용했습니다."
    ]

    return entry, ma20, ma60, target1, target2, take_profit, stop_loss, reasons, grounds


def check_alerts(current_price, market, target_price, take_profit_price, stop_loss_price):
    results = []

    if target_price > 0:
        if current_price >= target_price:
            results.append(("🚨 목표가 도달", f"현재가 {format_price(current_price, market)} / 목표가 {format_price(target_price, market)}"))
        else:
            results.append(("🟢 목표가 대기중", f"현재가 {format_price(current_price, market)} / 목표가 {format_price(target_price, market)}"))

    if take_profit_price > 0:
        if current_price >= take_profit_price:
            results.append(("🚨 익절가 도달", f"현재가 {format_price(current_price, market)} / 익절가 {format_price(take_profit_price, market)}"))
        else:
            results.append(("🟢 익절가 대기중", f"현재가 {format_price(current_price, market)} / 익절가 {format_price(take_profit_price, market)}"))

    if stop_loss_price > 0:
        if current_price <= stop_loss_price:
            results.append(("🚨 손절가 도달", f"현재가 {format_price(current_price, market)} / 손절가 {format_price(stop_loss_price, market)}"))
        else:
            results.append(("🟢 손절가 대기중", f"현재가 {format_price(current_price, market)} / 손절가 {format_price(stop_loss_price, market)}"))

    return results


# =========================
# 뉴스
# =========================

@st.cache_data(ttl=600)
def load_news(name):
    query = name.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}+주식+증권&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)
    news_items = []

    for entry in feed.entries[:10]:
        published = "시간 정보 없음"
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), KST)
            published = published_dt.strftime("%Y-%m-%d %H:%M")

        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "published": published
        })

    return news_items


def summarize_news(title):
    if "실적" in title or "영업이익" in title or "매출" in title:
        return ["실적 관련 뉴스입니다.", "매출과 이익 변화가 주가에 영향을 줄 수 있습니다."]
    if "배당" in title:
        return ["배당 관련 뉴스입니다.", "배당 확대는 투자 심리에 긍정적일 수 있습니다."]
    if "목표가" in title or "투자의견" in title:
        return ["증권사 리포트 관련 뉴스입니다.", "목표가 조정은 단기 주가에 영향을 줄 수 있습니다."]
    if "반도체" in title or "AI" in title or "HBM" in title:
        return ["산업 업황 관련 뉴스입니다.", "관련 섹터 전체에 영향을 줄 수 있습니다."]
    if "금리" in title or "환율" in title:
        return ["거시경제 관련 뉴스입니다.", "금리와 환율은 시장 전체에 영향을 줄 수 있습니다."]
    return ["종목 또는 시장 분위기 관련 뉴스입니다.", "제목만으로는 정확한 판단이 어려워 원문 확인이 필요합니다."]


def get_news_sentiment(title):
    positive_words = [
        "호실적", "상승", "수주", "증가", "흑자", "성장", "목표가 상향",
        "강세", "기대", "개선", "돌파", "최대", "확대", "계약", "매수"
    ]

    negative_words = [
        "하락", "부진", "적자", "감소", "소송", "규제", "목표가 하향",
        "약세", "우려", "악화", "급락", "축소", "리콜", "매도", "손실"
    ]

    pos = sum(1 for word in positive_words if word in title)
    neg = sum(1 for word in negative_words if word in title)

    if pos > neg:
        return "긍정", "green", 1
    if neg > pos:
        return "부정", "red", -1
    return "중립", "gray", 0


def get_news_score(news):
    total = 0
    for item in news:
        _, _, score = get_news_sentiment(item["title"])
        total += score
    return total


# =========================
# 재무 / 배당
# =========================

@st.cache_data(ttl=86400)
def load_financial(yf_symbol):
    try:
        ticker = yf.Ticker(yf_symbol)
        return ticker.info
    except Exception:
        return {}


def get_financial_score(info):
    score = 50
    reasons = []

    per = info.get("trailingPE")
    pbr = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    dividend = info.get("dividendYield")

    if per and per > 0:
        if per < 10:
            score += 10
            reasons.append("PER이 낮아 저평가 가능성이 있습니다.")
        elif per > 30:
            score -= 8
            reasons.append("PER이 높아 고평가 부담이 있습니다.")

    if pbr and pbr > 0:
        if pbr < 1:
            score += 8
            reasons.append("PBR이 1 이하로 자산 대비 저평가 가능성이 있습니다.")
        elif pbr > 3:
            score -= 6
            reasons.append("PBR이 높아 자산 대비 부담이 있습니다.")

    if roe is not None:
        if roe >= 0.15:
            score += 10
            reasons.append("ROE가 높아 수익성이 좋습니다.")
        elif roe < 0.05:
            score -= 6
            reasons.append("ROE가 낮아 수익성이 약합니다.")

    if dividend is not None:
        if dividend >= 0.03:
            score += 5
            reasons.append("배당률이 비교적 양호합니다.")

    score = max(0, min(100, score))

    if not reasons:
        reasons.append("무료 데이터에서 확인 가능한 재무 정보가 제한적입니다.")

    return score, reasons


# =========================
# 환율
# =========================

@st.cache_data(ttl=300)
def load_exchange_data(symbol):
    daily = yf.download(symbol, period="1mo", interval="1d", progress=False, auto_adjust=False)
    hourly = yf.download(symbol, period="5d", interval="1h", progress=False, auto_adjust=False)

    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    if isinstance(hourly.columns, pd.MultiIndex):
        hourly.columns = hourly.columns.get_level_values(0)

    return daily.dropna(), hourly.dropna()


# =========================
# 사이드바
# =========================

with st.sidebar:
    st.header("검색 설정")

    market = st.radio("시장", ["국내", "해외"])
    keyword = st.text_input("종목명 / 티커", "삼성전자" if market == "국내" else "AAPL")

    candle = st.selectbox("봉 종류", ["일봉", "주봉", "월봉", "년봉"])

    if candle == "일봉":
        period = st.selectbox("표시 기간", ["30일", "60일", "90일", "180일", "1년", "3년", "5년", "10년", "전체"])
    elif candle == "주봉":
        period = st.selectbox("표시 기간", ["12주", "24주", "52주", "1년", "3년", "5년", "10년", "전체"])
    elif candle == "월봉":
        period = st.selectbox("표시 기간", ["3개월", "6개월", "12개월", "24개월", "3년", "5년", "10년", "전체"])
    else:
        period = st.selectbox("표시 기간", ["1년", "3년", "5년", "10년", "20년", "전체"])

    st.divider()
    st.subheader("가격 알림")
    alert_enabled = st.checkbox("알림 사용", value=False)
    alert_target_price = st.number_input("목표가", min_value=0.0, value=0.0, step=100.0)
    alert_take_profit_price = st.number_input("익절가", min_value=0.0, value=0.0, step=100.0)
    alert_stop_loss_price = st.number_input("손절가", min_value=0.0, value=0.0, step=100.0)


tab1, tab2, tab3, tab4 = st.tabs(["주식 분석", "재무 / 배당", "금일 환율", "용어 설명"])


@st.fragment(run_every="30s")
def render_stock_area(keyword, market, candle, period, alert_enabled, alert_target_price, alert_take_profit_price, alert_stop_loss_price):
    df, code, name, yf_symbol = load_stock_data(keyword, market)

    if df is None or df.empty:
        st.error("종목 데이터를 불러오지 못했습니다.")
        return None, None, None

    candle_df = convert_candle(df, candle)
    analyzed = add_indicators(candle_df)

    if len(analyzed) < 30:
        st.error("분석할 데이터가 부족합니다.")
        return None, None, None

    display_df = filter_range(analyzed, candle, period)

    latest = analyzed.iloc[-1]
    prev = analyzed.iloc[-2]

    change = latest["Close"] - prev["Close"]
    change_rate = change / prev["Close"] * 100

    news = load_news(name)
    news_score = get_news_score(news)

    info = load_financial(yf_symbol)
    financial_score, financial_reasons = get_financial_score(info)

    score, grade, score_reasons = get_score(analyzed, financial_score=financial_score, news_score=news_score)
    entry, entry_ma20, entry_ma60, target1, target2, take_profit, stop_loss, entry_reasons, level_grounds = get_entry_and_levels(analyzed)
    volume_info = analyze_volume(analyzed)

    with tab1:
        col1, col2, col3 = st.columns([1.1, 2.2, 1.3])

        with col1:
            st.subheader("분석 결과")
            st.caption(f"주가 업데이트(KST): {format_kst()} / 주가 데이터 캐시 30초")

            st.metric(
                label=f"{name} 현재가",
                value=format_price(latest["Close"], market),
                delta=f"{change:,.2f} / {change_rate:.2f}%"
            )

            st.write(f"시장: {market}")
            st.write(f"코드/티커: {code}")
            st.write(f"봉 종류: {candle}")
            st.write(f"표시 기간: {period}")

            if alert_enabled:
                st.divider()
                st.write("### 알림 상태")
                alerts = check_alerts(
                    latest["Close"],
                    market,
                    alert_target_price,
                    alert_take_profit_price,
                    alert_stop_loss_price
                )

                if alerts:
                    for title, desc in alerts:
                        if "🚨" in title:
                            st.warning(f"{title}\n\n{desc}\n\n알림 시간(KST): {format_kst()}")
                        else:
                            st.info(f"{title}\n\n{desc}\n\n확인 시간(KST): {format_kst()}")
                else:
                    st.info("알림 사용은 켜져 있지만 목표가/익절가/손절가가 입력되지 않았습니다.")

            st.divider()

            st.write("### AI 종합 점수")
            st.write(f"점수: **{score}점**")
            st.write(f"판단: **{grade}**")
            for r in score_reasons:
                st.write(f"- {r}")

            st.divider()

            st.write("### 거래량 분석")
            st.write(f"총 거래량: {volume_info['total_volume']:,.0f}")
            st.write(f"20개 봉 평균 거래량: {volume_info['avg_volume']:,.0f}")
            st.write(f"평균 대비 증가율: {volume_info['volume_rate']:.2f}%")
            st.write(f"매수 우세 추정: {volume_info['buy_est']:,.0f}")
            st.write(f"매도 우세 추정: {volume_info['sell_est']:,.0f}")
            st.write(f"판단: **{volume_info['volume_signal']}**")
            st.write(f"OBV: {volume_info['obv']:,.0f}")
            st.write(f"- {volume_info['obv_signal']}")
            st.caption("매수/매도 거래량은 실제 체결 데이터가 아니라 가격 방향 기준 추정입니다.")

            st.divider()

            st.write("### 진입 타이밍")
            st.write(f"판단: **{entry}**")
            st.write(f"기준 진입가: {format_price(entry_ma20, market)}")
            st.write(f"보수적 진입가: {format_price(entry_ma60, market)}")
            for r in entry_reasons:
                st.write(f"- {r}")

            st.divider()

            st.write("### 목표가 / 익절가 / 손절가")
            st.write(f"1차 목표가: {format_price(target1, market)}")
            st.write(f"2차 목표가: {format_price(target2, market)}")
            st.write(f"익절가 참고: {format_price(take_profit, market)}")
            st.write(f"손절가 참고: {format_price(stop_loss, market)}")
            for g in level_grounds:
                st.write(f"- {g}")

        with col2:
            st.subheader("차트")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=display_df.index, y=display_df["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=display_df.index, y=display_df["MA5"], name="MA5"))
            fig.add_trace(go.Scatter(x=display_df.index, y=display_df["MA20"], name="MA20"))
            fig.add_trace(go.Scatter(x=display_df.index, y=display_df["MA60"], name="MA60"))
            fig.add_trace(go.Scatter(x=display_df.index, y=display_df["BB_UPPER"], name="BB Upper"))
            fig.add_trace(go.Scatter(x=display_df.index, y=display_df["BB_LOWER"], name="BB Lower"))

            fig.update_layout(height=500, xaxis_title="날짜", yaxis_title="가격", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("거래량")
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(x=display_df.index, y=display_df["Volume"], name="Volume"))
            volume_fig.update_layout(height=250, xaxis_title="날짜", yaxis_title="거래량")
            st.plotly_chart(volume_fig, use_container_width=True)

            st.subheader("RSI")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=display_df.index, y=display_df["RSI"], name="RSI"))
            rsi_fig.add_hline(y=70, line_dash="dash", annotation_text="과열")
            rsi_fig.add_hline(y=30, line_dash="dash", annotation_text="과매도")
            rsi_fig.update_layout(height=250, yaxis=dict(range=[0, 100]), xaxis_title="날짜", yaxis_title="RSI")
            st.plotly_chart(rsi_fig, use_container_width=True)

            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=display_df.index, y=display_df["MACD"], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=display_df.index, y=display_df["MACD_SIGNAL"], name="Signal"))
            macd_fig.add_trace(go.Bar(x=display_df.index, y=display_df["MACD_HIST"], name="Histogram"))
            macd_fig.update_layout(height=250, xaxis_title="날짜", yaxis_title="MACD")
            st.plotly_chart(macd_fig, use_container_width=True)

            st.subheader("OBV")
            obv_fig = go.Figure()
            obv_fig.add_trace(go.Scatter(x=display_df.index, y=display_df["OBV"], name="OBV"))
            obv_fig.update_layout(height=250, xaxis_title="날짜", yaxis_title="OBV")
            st.plotly_chart(obv_fig, use_container_width=True)

        with col3:
            st.subheader("관련 뉴스")
            st.caption("뉴스 캐시 10분 / 뉴스 시간은 KST 기준")

            if news:
                news_pos = 0
                news_neg = 0
                news_neu = 0

                for item in news:
                    sentiment, color, sentiment_score = get_news_sentiment(item["title"])

                    if sentiment == "긍정":
                        news_pos += 1
                    elif sentiment == "부정":
                        news_neg += 1
                    else:
                        news_neu += 1

                    st.markdown(
                        f"<span style='color:{color}; font-weight:bold;'>[{sentiment}]</span> {item['title']}",
                        unsafe_allow_html=True
                    )

                    with st.expander("뉴스 정리 보기"):
                        st.caption(f"뉴스 시간(KST): {item['published']}")
                        for line in summarize_news(item["title"]):
                            st.write(f"- {line}")
                        st.link_button("원문 보기", item["link"])

                st.divider()
                st.write("### 뉴스 분위기")
                st.write(f"긍정: {news_pos}개 / 부정: {news_neg}개 / 중립: {news_neu}개")
                if news_score > 0:
                    st.success("뉴스 분위기: 긍정 우세")
                elif news_score < 0:
                    st.error("뉴스 분위기: 부정 우세")
                else:
                    st.info("뉴스 분위기: 중립")
            else:
                st.write("관련 뉴스를 찾지 못했습니다.")

    return yf_symbol, name, code, financial_score, financial_reasons


yf_symbol, selected_name, selected_code, financial_score, financial_reasons = render_stock_area(
    keyword,
    market,
    candle,
    period,
    alert_enabled,
    alert_target_price,
    alert_take_profit_price,
    alert_stop_loss_price
)


with tab2:
    st.subheader("재무 / 배당")
    st.caption("재무/배당 데이터는 1일 캐시입니다.")

    if yf_symbol:
        info = load_financial(yf_symbol)

        st.write("### 재무 점수")
        st.write(f"점수: **{financial_score}점**")
        for reason in financial_reasons:
            st.write(f"- {reason}")

        f1, f2, f3 = st.columns(3)
        f1.metric("PER", info.get("trailingPE", "정보 없음"))
        f2.metric("PBR", info.get("priceToBook", "정보 없음"))
        f3.metric("ROE", format_percent(info.get("returnOnEquity")))

        f4, f5, f6 = st.columns(3)
        f4.metric("배당률", format_percent(info.get("dividendYield")))
        f5.metric("배당금", info.get("dividendRate", "정보 없음"))
        f6.metric("시가총액", format_market_cap(info.get("marketCap"), market))

        st.caption(f"재무/배당 조회 시간(KST): {format_kst()}")
    else:
        st.info("종목을 먼저 조회하면 재무/배당 정보가 표시됩니다.")


with tab3:
    st.subheader("금일 환율 / 최근 1개월 일별 환율")
    st.caption("환율 데이터는 5분 캐시입니다. Yahoo Finance 무료 데이터 기준이라 은행 고시환율과 차이가 있을 수 있습니다.")

    exchange_options = {
        "미국 달러 / 원": "KRW=X",
        "일본 엔 / 원": "JPYKRW=X",
        "유로 / 원": "EURKRW=X",
        "중국 위안 / 원": "CNYKRW=X"
    }

    selected_exchange = st.selectbox("환율 선택", list(exchange_options.keys()))
    symbol = exchange_options[selected_exchange]

    daily_df, hourly_df = load_exchange_data(symbol)

    if not daily_df.empty:
        latest_source = hourly_df if not hourly_df.empty else daily_df
        latest = latest_source.iloc[-1]
        prev = daily_df.iloc[-2] if len(daily_df) >= 2 else daily_df.iloc[-1]

        latest_close = float(latest["Close"])
        prev_close = float(prev["Close"])

        change = latest_close - prev_close
        change_rate = change / prev_close * 100

        if selected_exchange == "일본 엔 / 원":
            display_value = latest_close * 100
            display_change = change * 100
            unit_label = "100엔 기준"
        else:
            display_value = latest_close
            display_change = change
            unit_label = "1통화 기준"

        st.metric(
            label=f"{selected_exchange} ({unit_label})",
            value=f"{display_value:,.2f}원",
            delta=f"{display_change:,.2f}원 / {change_rate:.2f}%"
        )

        st.caption(f"환율 업데이트(KST): {format_kst()}")

        fig_ex = go.Figure()
        y_values = daily_df["Close"] * 100 if selected_exchange == "일본 엔 / 원" else daily_df["Close"]

        fig_ex.add_trace(go.Scatter(
            x=daily_df.index,
            y=y_values,
            mode="lines+markers",
            name=selected_exchange
        ))

        fig_ex.update_layout(height=450, xaxis_title="날짜", yaxis_title="환율", hovermode="x unified")
        st.plotly_chart(fig_ex, use_container_width=True)

        table_df = daily_df[["Open", "High", "Low", "Close"]].copy()

        if selected_exchange == "일본 엔 / 원":
            table_df = table_df * 100

        table_df = table_df.rename(columns={
            "Open": "시가",
            "High": "고가",
            "Low": "저가",
            "Close": "종가"
        })

        st.dataframe(table_df.tail(30).style.format("{:,.2f}"), use_container_width=True)
    else:
        st.warning("환율 데이터를 불러오지 못했습니다.")


with tab4:
    st.subheader("용어 설명")

    st.write("""
### RSI
과매수/과매도 지표입니다. 70 이상은 과열, 30 이하는 과매도 구간입니다.

### 거래량
거래량이 평균보다 증가하면 시장 관심이 커졌다는 의미로 볼 수 있습니다.

### OBV
가격 상승일의 거래량은 더하고, 하락일의 거래량은 빼서 수급 흐름을 보는 지표입니다.

### MACD
추세 전환을 보는 지표입니다. MACD가 Signal 위면 상승 흐름 가능성이 있습니다.

### AI 점수
이동평균선, RSI, MACD, 거래량, OBV, 재무점수, 뉴스 분위기를 점수화한 참고용 지표입니다.

### 목표가 / 익절가 / 손절가
ATR과 볼린저밴드를 활용한 참고값입니다.

### 알림
목표가, 익절가, 손절가를 입력하면 30초마다 현재가와 비교해 도달 여부를 표시합니다.

### 뉴스 긍정/부정
뉴스 제목의 키워드를 기준으로 긍정, 부정, 중립을 추정합니다.

### 배당률
현재 주가 대비 배당금 비율입니다.

### 환율
무료 데이터 기준이라 실제 은행 고시환율과 약간 차이가 있을 수 있습니다.
""")

st.caption(f"마지막 화면 렌더링(KST): {format_kst()} / 주가 30초, 환율 5분, 뉴스 10분, 재무·배당 1일 캐시")
