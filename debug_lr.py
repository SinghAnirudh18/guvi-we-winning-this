import joblib
import os
import sklearn
import numpy as np

print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Joblib version: {joblib.__version__}")

# 1. Load Files
print("\n--- Loading Artifacts ---")
if not os.path.exists("model.pkl"):
    print("❌ ERROR: 'model.pkl' not found.")
    exit()
if not os.path.exists("vectorizer.pkl"):
    print("❌ ERROR: 'vectorizer.pkl' not found.")
    exit()

try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print(f"✅ Model Loaded: {type(model)}")
    print(f"✅ Vectorizer Loaded: {type(vectorizer)}")
except Exception as e:
    print(f"❌ CRITICAL LOAD ERROR: {e}")
    exit()

# 2. Test Input
text = "URGENT: Your SBI account has been blocked. Click here to update pan."
print(f"\n--- Testing Input: '{text}' ---")

try:
    # 3. Preprocess (Manual match of your API logic)
    clean_text = text.lower()
    
    # 4. Vectorize
    vector = vectorizer.transform([clean_text])
    print(f"Vector Shape: {vector.shape}")
    print(f"Non-zero features found: {vector.nnz}")
    
    if vector.nnz == 0:
        print("⚠️ WARNING: The vector is empty! The model sees NO words it knows.")
        print("Did you use the same vectorizer you trained with?")
    else:
        # Print which words were found (if possible)
        feature_names = vectorizer.get_feature_names_out()
        found_words = [feature_names[i] for i in vector.indices]
        print(f"Words recognized: {found_words}")

    # 5. Predict
    # This is where it likely crashes or returns [0, 1]
    raw_prob = model.predict_proba(vector)
    print(f"Raw Probabilities: {raw_prob}")
    
    scam_score = raw_prob[0][1]
    print(f"✅ FINAL SCAM SCORE: {scam_score}")

except Exception as e:
    print(f"\n❌ RUNTIME ERROR: {e}")