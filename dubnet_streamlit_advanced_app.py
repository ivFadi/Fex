
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# بيانات تدريب افتراضية واقعية مستوحاة من أداء منصة الدب نت وشركة الأداء العالي
data = {
    "Ad_Nature": [
        "تغطية ميدانية", "إعلان منتج", "إعلان موسمي", "إعلان سريع",
        "إعلان منتج", "تغطية ميدانية", "إعلان سريع", "إعلان موسمي"
    ],
    "Ad_Duration": [
        "1 يوم", "3 أيام", "أسبوع", "1 يوم",
        "3 أيام", "أسبوع", "1 يوم", "أسبوع"
    ],
    "Platform": [
        "Snapchat", "Instagram", "TikTok", "Snapchat",
        "Instagram", "TikTok", "Snapchat", "Instagram"
    ],
    "Offered_Price": [
        12000, 15000, 20000, 9000,
        14000, 23000, 10000, 17000
    ]
}

df = pd.DataFrame(data)

# ترميز البيانات
le_nature = LabelEncoder()
le_duration = LabelEncoder()
le_platform = LabelEncoder()
df["Nature_Code"] = le_nature.fit_transform(df["Ad_Nature"])
df["Duration_Code"] = le_duration.fit_transform(df["Ad_Duration"])
df["Platform_Code"] = le_platform.fit_transform(df["Platform"])

# بناء النموذج
X = df[["Nature_Code", "Duration_Code", "Platform_Code"]]
y = df["Offered_Price"]
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X, y)

# واجهة المستخدم
st.set_page_config(page_title="سعر الإعلان التنبؤي - منصة الدب نت", layout="centered")
st.title("🔮 توقع سعر الإعلان حسب النوع، المدة، والمنصة")

ad_nature = st.selectbox("📝 طبيعة الإعلان", le_nature.classes_)
ad_duration = st.selectbox("⏱️ مدة الإعلان", le_duration.classes_)
platform = st.selectbox("📱 المنصة", le_platform.classes_)

if st.button("توقع السعر"):
    nat_code = le_nature.transform([ad_nature])[0]
    dur_code = le_duration.transform([ad_duration])[0]
    plat_code = le_platform.transform([platform])[0]
    predicted_price = model.predict([[nat_code, dur_code, plat_code]])[0]
    tax = predicted_price * 0.15
    total = predicted_price + tax

    st.success(f"✅ السعر المتوقع بدون ضريبة: {round(predicted_price)} ريال سعودي")
    st.info(f"💸 السعر بعد الضريبة 15%: {round(total)} ريال سعودي")
    st.caption("📈 يستند التنبؤ على بيانات الأداء العالي ومنصة الدب نت ويعكس تطور المنصة ونمو جمهورها.")
