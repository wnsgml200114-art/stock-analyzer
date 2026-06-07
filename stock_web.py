import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import feedparser
import plotly.graph_objects as go

st.set_page_config(page_title="AI 주식 분석기", layout="wide")
st.title("AI 주식 분석_junhee")


def find_korean_stock_code(keyword):
    stock_list = fdr.StockListing("KRX")
    stock_list = stock_list[["Code", "Name", "Market"]]

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


def load_stock_data(keyword, market):
    if market == "국내":
        code, name, krx_market = find_korean_stock_code(keyword)
        if code is None:
            st.error("국내 종목을 찾지 못했습니다.")
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

    rule = {"주봉": "W", "월봉": "ME", "년봉": "YE"}[candle]

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

    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - prev_close),
        abs(df["Low"] - prev_close)
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    return df.dropna()


def format_price(value, market):
    if market == "국내":
        return f"{value:,.0f}원"
    return f"${value:,.2f}"


def get_score(df):
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

    if 30 <= latest["RSI"] <= 45:
        score += 12
        reasons.append("RSI가 저점 반등 관심 구간입니다.")
    elif latest["RSI"] < 30:
        score += 5
        reasons.append("RSI가 과매도 구간입니다.")
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

    target1 = close + atr * 1.5
    target2 = close + atr * 2.5
    take_profit = close + atr * 2.0
    stop_loss = close - atr * 1.2

    if close > ma20 and latest["MA5"] > ma20 and latest["MACD"] > latest["MACD_SIGNAL"]:
        entry = "분할매수 검토 가능"
    elif close < ma20 and 30 <= latest["RSI"] <= 45:
        entry = "반등 확인 후 진입"
    elif latest["RSI"] < 30:
        entry = "과매도 구간, 신중한 분할 접근"
    else:
        entry = "관망"

    return entry, ma20, ma60, target1, target2, take_profit, stop_loss


def load_news(name):
    query = name.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}+주식+증권&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)
    return feed.entries[:10]


def load_financial(yf_symbol):
    try:
        ticker = yf.Ticker(yf_symbol)
        return ticker.info
    except Exception:
        return {}


def load_exchange_data(symbol):
    df = yf.download(symbol, period="1mo", interval="1d", progress=False, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()


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

    alert_price = st.number_input("알림가", min_value=0.0, value=0.0)
    alert_direction = st.selectbox("알림 조건", ["이상 도달", "이하 도달"])

    run = st.button("분석하기")


tab1, tab2, tab3, tab4 = st.tabs(["주식 분석", "재무 / 배당", "금일 환율", "용어 설명"])


if run:
    df, code, name, yf_symbol = load_stock_data(keyword, market)

    if df is not None and not df.empty:
        candle_df = convert_candle(df, candle)
        analyzed = add_indicators(candle_df)

        if len(analyzed) < 30:
            st.error("분석할 데이터가 부족합니다.")
            st.stop()

        display_df = filter_range(analyzed, candle, period)

        latest = analyzed.iloc[-1]
        prev = analyzed.iloc[-2]

        change = latest["Close"] - prev["Close"]
        change_rate = change / prev["Close"] * 100

        score, grade, reasons = get_score(analyzed)
        entry, entry_ma20, entry_ma60, target1, target2, take_profit, stop_loss = get_entry_and_levels(analyzed)

        with tab1:
            col1, col2, col3 = st.columns([1.1, 2.2, 1.3])

            with col1:
                st.subheader("분석 결과")

                st.metric(
                    label=f"{name} 현재가",
                    value=format_price(latest["Close"], market),
                    delta=f"{change:,.2f} / {change_rate:.2f}%"
                )

                st.write(f"시장: {market}")
                st.write(f"코드/티커: {code}")
                st.write(f"봉 종류: {candle}")
                st.write(f"표시 기간: {period}")

                st.divider()

                st.write("### 이동평균선")
                st.write(f"MA5: {format_price(latest['MA5'], market)}")
                st.write(f"MA20: {format_price(latest['MA20'], market)}")
                st.write(f"MA60: {format_price(latest['MA60'], market)}")

                st.write("### 보조지표")
                st.write(f"RSI: {latest['RSI']:.2f}")
                st.write(f"MACD: {latest['MACD']:.2f}")
                st.write(f"Signal: {latest['MACD_SIGNAL']:.2f}")
                st.write(f"ATR: {latest['ATR']:.2f}")

                st.divider()

                st.write("### 진입 타이밍")
                st.write(f"판단: **{entry}**")
                st.write(f"기준 진입가: {format_price(entry_ma20, market)}")
                st.write(f"보수적 진입가: {format_price(entry_ma60, market)}")

                st.write("### 목표가 / 익절가 / 손절가")
                st.write(f"1차 목표가: {format_price(target1, market)}")
                st.write(f"2차 목표가: {format_price(target2, market)}")
                st.write(f"익절가 참고: {format_price(take_profit, market)}")
                st.write(f"손절가 참고: {format_price(stop_loss, market)}")

                st.divider()

                st.write("### 종합 점수")
                st.write(f"점수: **{score}점**")
                st.write(f"판단: **{grade}**")

                for r in reasons:
                    st.write(f"- {r}")

                if alert_price > 0:
                    current = latest["Close"]

                    if alert_direction == "이상 도달" and current >= alert_price:
                        st.warning(f"가격 알림 도달: 현재가 {format_price(current, market)}")
                    elif alert_direction == "이하 도달" and current <= alert_price:
                        st.warning(f"가격 알림 도달: 현재가 {format_price(current, market)}")
                    else:
                        st.info(f"알림 대기 중: 현재가 {format_price(current, market)}")

            with col2:
                st.subheader("차트")

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=display_df.index, y=display_df["Close"], name="Close"))
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df["MA5"], name="MA5"))
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df["MA20"], name="MA20"))
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df["MA60"], name="MA60"))
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df["BB_UPPER"], name="BB Upper"))
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df["BB_LOWER"], name="BB Lower"))

                fig.update_layout(
                    height=520,
                    xaxis_title="날짜",
                    yaxis_title="가격",
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.subheader("관련 뉴스")

                news = load_news(name)

                if news:
                    for item in news:
                        with st.expander(item.title):
                            st.write("뉴스 제목 기준 정리")
                            title = item.title

                            if "실적" in title or "영업이익" in title or "매출" in title:
                                st.write("- 실적 관련 뉴스입니다.")
                                st.write("- 매출과 이익 변화가 주가에 영향을 줄 수 있습니다.")
                            elif "배당" in title:
                                st.write("- 배당 관련 뉴스입니다.")
                                st.write("- 배당 확대는 투자 심리에 긍정적일 수 있습니다.")
                            elif "목표가" in title or "투자의견" in title:
                                st.write("- 증권사 리포트 관련 뉴스입니다.")
                                st.write("- 목표가 조정은 단기 주가에 영향을 줄 수 있습니다.")
                            elif "반도체" in title or "AI" in title or "HBM" in title:
                                st.write("- 산업 업황 관련 뉴스입니다.")
                                st.write("- 관련 섹터 전체에 영향을 줄 수 있습니다.")
                            elif "금리" in title or "환율" in title:
                                st.write("- 거시경제 관련 뉴스입니다.")
                                st.write("- 금리와 환율은 시장 전체에 영향을 줄 수 있습니다.")
                            else:
                                st.write("- 종목 또는 시장 분위기 관련 뉴스입니다.")
                                st.write("- 제목만으로는 정확한 판단이 어려워 원문 확인이 필요합니다.")

                            st.link_button("원문 보기", item.link)
                else:
                    st.write("관련 뉴스를 찾지 못했습니다.")

        with tab2:
            st.subheader("재무 / 배당")
            info = load_financial(yf_symbol)

            f1, f2, f3 = st.columns(3)
            f1.metric("PER", info.get("trailingPE", "정보 없음"))
            f2.metric("PBR", info.get("priceToBook", "정보 없음"))
            f3.metric("ROE", f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get("returnOnEquity") else "정보 없음")

            f4, f5, f6 = st.columns(3)
            f4.metric("배당률", f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get("dividendYield") else "정보 없음")
            f5.metric("배당금", info.get("dividendRate", "정보 없음"))
            f6.metric("시가총액", info.get("marketCap", "정보 없음"))

        st.caption("무료 데이터 기반 참고용입니다. 실제 투자 판단은 본인 책임입니다.")


with tab3:
    st.subheader("금일 환율 / 최근 1개월 일별 환율")

    exchange_options = {
        "미국 달러 / 원": "KRW=X",
        "일본 엔 / 원": "JPYKRW=X",
        "유로 / 원": "EURKRW=X"
    }

    selected_exchange = st.selectbox("환율 선택", list(exchange_options.keys()))
    symbol = exchange_options[selected_exchange]

    ex_df = load_exchange_data(symbol)

    if not ex_df.empty:
        latest = ex_df.iloc[-1]
        prev = ex_df.iloc[-2] if len(ex_df) >= 2 else latest

        change = latest["Close"] - prev["Close"]
        change_rate = change / prev["Close"] * 100

        st.metric(
            label=selected_exchange,
            value=f"{latest['Close']:,.2f}원",
            delta=f"{change:,.2f}원 / {change_rate:.2f}%"
        )

        fig_ex = go.Figure()
        fig_ex.add_trace(go.Scatter(
            x=ex_df.index,
            y=ex_df["Close"],
            mode="lines+markers",
            name=selected_exchange
        ))

        fig_ex.update_layout(
            height=450,
            xaxis_title="날짜",
            yaxis_title="환율",
            hovermode="x unified"
        )

        st.plotly_chart(fig_ex, use_container_width=True)

        table_df = ex_df[["Open", "High", "Low", "Close"]].copy()
        table_df = table_df.rename(columns={
            "Open": "시가",
            "High": "고가",
            "Low": "저가",
            "Close": "종가"
        })

        st.dataframe(table_df.tail(30), use_container_width=True)
    else:
        st.warning("환율 데이터를 불러오지 못했습니다.")


with tab4:
    st.subheader("용어 설명")

    st.write("""
### Close
종가입니다. 해당 기간의 마지막 가격입니다.

### MA5 / MA20 / MA60
이동평균선입니다. 5선은 단기, 20선은 중기, 60선은 장기 흐름을 봅니다.

### RSI
과매수/과매도 지표입니다. 70 이상은 과열, 30 이하는 과매도 구간입니다.

### MACD
추세 전환을 보는 지표입니다. MACD가 Signal 위면 상승 흐름 가능성이 있습니다.

### 볼린저밴드
가격의 변동 범위를 보는 지표입니다. 상단 근처는 과열, 하단 근처는 반등 가능성 구간으로 봅니다.

### ATR
평균 변동폭입니다. 목표가, 익절가, 손절가 참고값 계산에 사용했습니다.

### 환율
원/달러 환율이 상승하면 수입 비용 부담이 커질 수 있고, 수출 기업에는 긍정적으로 작용할 수 있습니다.
""")
