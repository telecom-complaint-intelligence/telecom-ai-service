import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training dataset for Telecom Complaints
DATA = [
    # Router / Wifi / Device issues
    {"complaint": "i dont know how to turn on my router", "component": "router", "failure_type": "device_failure", "scope": "individual", "service_impact": "minor", "occurrence_pattern": "one_time"},
    {"complaint": "wifi router light is blinking red and internet is offline", "component": "wifi_router", "failure_type": "device_instability", "scope": "household", "service_impact": "complete_outage", "occurrence_pattern": "continuous"},
    {"complaint": "my ont box has no power light", "component": "ont", "failure_type": "device_failure", "scope": "household", "service_impact": "complete_outage", "occurrence_pattern": "continuous"},
    {"complaint": "modem keeps rebooting every 10 minutes", "component": "modem", "failure_type": "device_instability", "scope": "individual", "service_impact": "degraded", "occurrence_pattern": "recurring"},

    # Fiber cable / Physical damage / Outage issues
    {"complaint": "Our local fiber cable is damaged, leaving many customers without service since this morning.", "component": "fiber_cable", "failure_type": "physical_damage", "scope": "multiple_customers", "service_impact": "complete_outage", "occurrence_pattern": "continuous"},
    {"complaint": "fiber cable got cut by road construction crew whole area is down", "component": "fiber_cable", "failure_type": "cable_cut", "scope": "area_wide", "service_impact": "complete_outage", "occurrence_pattern": "continuous"},
    {"complaint": "network cable severed near exchange building affecting multiple users", "component": "network_cable", "failure_type": "cable_cut", "scope": "multiple_users", "service_impact": "complete_outage", "occurrence_pattern": "continuous"},

    # Tower / Base station / Infrastructure issues
    {"complaint": "cell tower in our town lost power complete blackout for entire region", "component": "network_tower", "failure_type": "infrastructure_failure", "scope": "regional", "service_impact": "complete_outage", "occurrence_pattern": "continuous"},
    {"complaint": "base station failure causing nationwide degraded mobile data service", "component": "base_station", "failure_type": "major_network_failure", "scope": "nationwide", "service_impact": "degraded", "occurrence_pattern": "intermittent"},
    {"complaint": "core network routing problem creating widespread slow speeds across all devices", "component": "core_network", "failure_type": "network_failure", "scope": "widespread", "service_impact": "degraded", "occurrence_pattern": "intermittent"},

    # Speed & intermittent connectivity issues
    {"complaint": "internet speed is extremely slow since yesterday afternoon", "component": "router", "failure_type": "slow_speed", "scope": "household", "service_impact": "degraded", "occurrence_pattern": "recurring"},
    {"complaint": "wifi keeps dropping intermittent connection every hour", "component": "wifi_router", "failure_type": "intermittent_connection", "scope": "individual", "service_impact": "degraded", "occurrence_pattern": "intermittent"},
    {"complaint": "unstable connection while streaming video during peak hours", "component": "router", "failure_type": "unstable_connection", "scope": "individual", "service_impact": "minor", "occurrence_pattern": "recurring"}
]

CATEGORICAL_TARGETS = [
    "component", "failure_type", "scope",
    "service_impact", "occurrence_pattern"
]

def train_and_save_models(output_dir: str = "trained_models"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(DATA)

    # Replicate dataset slightly so multi-class fitting has sufficient examples
    df_expanded = pd.concat([df] * 5, ignore_index=True)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words="english"
    )
    Xtr = vectorizer.fit_transform(df_expanded["complaint"])

    models = {}
    for target in CATEGORICAL_TARGETS:
        ytr = df_expanded[target].astype(str)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=5.0)
        clf.fit(Xtr, ytr)
        models[target] = clf

    vec_path = os.path.join(output_dir, "vectorizer.joblib")
    models_path = os.path.join(output_dir, "field_models.joblib")

    joblib.dump(vectorizer, vec_path)
    joblib.dump(models, models_path)

    print(f"Successfully trained & saved vectorizer to {vec_path}")
    print(f"Successfully trained & saved models to {models_path}")

if __name__ == "__main__":
    train_and_save_models()
