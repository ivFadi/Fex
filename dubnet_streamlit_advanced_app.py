
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# بيانات تدريب واقعية مستوحاة من "الدب نت" + ملف الأداء العالي
internal_data = {
    "Ad_Nature": ["تغطية ميدانية", "إعلان منتج", "إعلان موسمي", "إعلان سريع"] * 3,
    "Ad_Duration": ["1 يوم", "3 أيام", "أسبوع"] * 4,
    "Platform": ["Snapchat", "Instagram", "TikTok"] * 4,
    "Offered_Price": [12000, 15000, 18000, 9000, 14000, 20000, 23000, 11000, 17500, 14500, 21000, 16000]
}
df_internal = pd.DataFrame(internal_data)

# بيانات منافسين تقريبية (تمثيلية لأغراض المقارنة)
market_data = {
    "Ad_Nature": ["تغطية ميدانية", "إعلان منتج", "إعلان موسمي", "إعلان سريع"] * 2,
    "Ad_Duration": ["1 يوم", "3 أيام", "أسبوع"] * 2 + ["1 يوم", "3 أيام"],
    "Platform": ["Snapchat", "Instagram", "TikTok"] * 2,
    "Offered_Price": [13500, 17000, 20000, 11000, 16000, 22000, 25000, 13000, 18500, 16500]
}
df_market = pd.DataFrame(market_data)

# ترميز البيانات
encoders = {}
def encode(df):
    for col in ["Ad_Nature", "Ad_Duration", "Platform"]:
        encoders[col] = LabelEncoder()
        df[col + "_Code"] = encoders[col].fit_transform(df[col])
    return df

df_internal = encode(df_internal)
df_market = encode(df_market)

# تدريب النموذجين
model_internal = RandomForestRegressor(n_estimators=100, random_state=42)
model_internal.fit(df_internal[["Ad_Nature_Code", "Ad_Duration_Code", "Platform_Code"]], df_internal["Offered_Price"])

model_market = RandomForestRegressor(n_estimators=100, random_state=42)
model_market.fit(df_market[["Ad_Nature_Code", "Ad_Duration_Code", "Platform_Code"]], df_market["Offered_Price"])

# واجهة Streamlit
st.set_page_config(page_title="توقع سعر الإعلان - الدب نت", layout="centered")
st.title("🔮 توقع سعر الإعلان حسب طبيعة الإعلان، مدته، والمنصة")

# إدخال البيانات
ad_nature = st.selectbox("📝 طبيعة الإعلان", encoders["Ad_Nature"].classes_)
ad_duration = st.selectbox("⏱️ مدة الإعلان", encoders["Ad_Duration"].classes_)
platform = st.selectbox("📱 المنصة", encoders["Platform"].classes_)

show_market_comparison = st.checkbox("📊 مقارنة مع أسعار السوق", value=True)

if st.button("🔎 توقع السعر"):
    encoded = {
        "Ad_Nature_Code": encoders["Ad_Nature"].transform([ad_nature])[0],
        "Ad_Duration_Code": encoders["Ad_Duration"].transform([ad_duration])[0],
        "Platform_Code": encoders["Platform"].transform([platform])[0],
    }
    input_data = [[encoded["Ad_Nature_Code"], encoded["Ad_Duration_Code"], encoded["Platform_Code"]]]

    price_internal = model_internal.predict(input_data)[0]
    total_internal = price_internal + (price_internal * 0.15)
    st.success(f"✅ السعر المتوقع حسب منصة الدب نت (بدون ضريبة): {round(price_internal)} ريال سعودي")
    st.info(f"💰 السعر بعد الضريبة 15٪: {round(total_internal)} ريال سعودي")

    if show_market_comparison:
        price_market = model_market.predict(input_data)[0]
        st.divider()
        st.subheader("📈 مقارنة بالسوق الإعلاني")
        st.write(f"💼 متوسط السعر لدى المنصات المنافسة: {round(price_market)} ريال سعودي")
        st.caption("📝 تم التقدير بناءً على بيانات تمثيلية لمنصات مثل عرب جي تي وسعودي شفت.")
