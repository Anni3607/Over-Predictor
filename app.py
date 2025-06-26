
import streamlit as st
import joblib

@st.cache(allow_output_mutation=True)
def load_artifacts():
    model = joblib.load('model.pkl')
    le_batsman = joblib.load('le_batsman.pkl')
    le_bowler = joblib.load('le_bowler.pkl')
    le_venue = joblib.load('le_venue.pkl')
    return model, le_batsman, le_bowler, le_venue

model, le_batsman, le_bowler, le_venue = load_artifacts()

st.title("🏏 Will This Over Be Expensive? (IPL)")
st.write("Enter batsman, bowler, and venue to predict runs in next over:")

batsman = st.text_input("Batsman Name")
bowler = st.text_input("Bowler Name")
venue = st.text_input("Venue Name")

if st.button("Predict"):
    for name, le in [("batsman", le_batsman), ("bowler", le_bowler), ("venue", le_venue)]:
        val = batsman if name == "batsman" else bowler if name == "bowler" else venue
        if val not in le.classes_:
            st.warning(f"⚠️ New {name} detected. Accuracy may drop.")

    try:
        be = le_batsman.transform([batsman])[0] if batsman in le_batsman.classes_ else -1
        bo = le_bowler.transform([bowler])[0]  if bowler in le_bowler.classes_ else -1
        ve = le_venue.transform([venue])[0]    if venue in le_venue.classes_ else -1

        pred = model.predict([[be, bo, ve]])[0]
        emoji = "🔥" if pred >= 10 else "😢"
        bg = "#ffebcc" if pred >= 10 else "#e0e0e0"

        st.markdown(f"""
        <div style="background-color:{bg}; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="font-size:60px;">{emoji}</h1>
            <h3>Estimated runs: {pred:.1f}</h3>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not predict: {e}")
