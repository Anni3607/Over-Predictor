import streamlit as st
import joblib

# Use st.cache_resource for loading machine learning models and encoders
# This will cache the loaded objects and ensure they are only loaded once
# across different user sessions, improving performance.
@st.cache_resource
def load_artifacts():
    """
    Loads the pre-trained machine learning model and label encoders.
    The paths 'model.pkl', 'le_batsman.pkl', etc., assume these files
    are in the same directory as app.py or a relative path from app.py.
    """
    model = joblib.load('model.pkl')
    le_batsman = joblib.load('le_batsman.pkl')
    le_bowler = joblib.load('le_bowler.pkl')
    le_venue = joblib.load('le_venue.pkl')
    return model, le_batsman, le_bowler, le_venue

# Load all the necessary artifacts (model and encoders)
model, le_batsman, le_bowler, le_venue = load_artifacts()

st.title("🏏 Will This Over Be Expensive? (IPL)")
st.write("Select batsman, bowler, and venue to predict runs in next over:")

# --- Prepare options for the selectboxes from the loaded label encoders ---
# Get the unique classes (names/venues) that the label encoders were trained on.
# Converting to list is good practice for st.selectbox options.
known_batsmen = list(le_batsman.classes_)
known_bowlers = list(le_bowler.classes_)
known_venues = list(le_venue.classes_)

# --- Replace st.text_input with st.selectbox ---
# This ensures users can only select values that the model has seen during training,
# eliminating the "New ... detected" warnings and improving prediction accuracy.
batsman = st.selectbox("Batsman Name", options=known_batsmen)
bowler = st.selectbox("Bowler Name", options=known_bowlers)
venue = st.selectbox("Venue Name", options=known_venues)

if st.button("Predict"):
    # Since we are using selectboxes, users can only choose known values.
    # Therefore, the checks for `val not in le.classes_` and the warnings
    # related to "New ... detected" are no longer strictly necessary if all
    # inputs are coming from selectboxes populated with `le.classes_`.
    # However, keeping them as a fallback or for debugging is fine.

    try:
        # Transform the selected values using the respective label encoders.
        # Since they are selected from `le.classes_`, they should always be found.
        be = le_batsman.transform([batsman])[0]
        bo = le_bowler.transform([bowler])[0]
        ve = le_venue.transform([venue])[0]

        # Make the prediction using the loaded model
        pred = model.predict([[be, bo, ve]])[0]

        # Determine emoji and background color based on prediction
        emoji = "🔥" if pred >= 10 else "😢"
        bg = "#4CAF50" if pred >= 10 else "#FFC107" # Changed colors for better contrast/meaning

        # Display the prediction using Streamlit's Markdown for custom styling
        st.markdown(f"""
        <div style="background-color:{bg}; padding:20px; border-radius:10px; text-align:center; color: #FFFFFF;">
            <h1 style="font-size:60px;">{emoji}</h1>
            <h3>Estimated runs: {pred:.1f}</h3>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        # Catch any potential errors during prediction or transformation
        st.error(f"An error occurred during prediction: {e}")

