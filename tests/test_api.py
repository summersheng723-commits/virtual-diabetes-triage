import os, subprocess, time, requests

def test_health_and_predict():
    # start container locally for smoke test (in CI-release we do similar)
    img = os.environ.get("TEST_IMAGE", "ghcr.io/org/repo:local")
    proc = subprocess.Popen(["docker","run","-p","8080:8080",img])
    try:
        time.sleep(3)
        r = requests.get("http://localhost:8080/health", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

        payload = {"age": 0.02,"sex": -0.044,"bmi": 0.06,"bp": -0.03,
                   "s1": -0.02,"s2": 0.03,"s3": -0.02,"s4": 0.02,"s5": 0.02,"s6": -0.001}
        p = requests.post("http://localhost:8080/predict", json=payload, timeout=5)
        assert p.status_code == 200
        assert isinstance(p.json().get("prediction"), float)
    finally:
        proc.terminate()
