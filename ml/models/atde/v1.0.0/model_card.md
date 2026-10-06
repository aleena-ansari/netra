\# ATDE Model Card



\## Version



1.0.0



\## Purpose



Network anomaly detection using a two-stage machine-learning pipeline.



\## Pipeline



1\. Isolation Forest detects anomalous network activity.

2\. XGBoost classifies anomalous activity into an attack family.

3\. Low-confidence XGBoost predictions are marked UNKNOWN/NOVEL.



\## Isolation Forest Models



\### AWS VPC Flow Logs



17 features.



\### Cisco ASA



18 features.



\## XGBoost



12 features.



Eight attack classes:



\- Backdoor / Persistence

\- Credential Attack / Brute Force

\- Data Exfiltration

\- DoS / Flooding

\- Exploitation / RCE

\- Fuzzing

\- Reconnaissance / Scanning

\- Shellcode / Payload Execution



\## Unknown Detection



XGBoost predictions below confidence 0.9 are treated as UNKNOWN/NOVEL.



\## Isolation Forest Decision



An event is anomalous when:



`decision\_function < 0`



\## Intended Use



Real-time network security event scoring.



\## Limitations



The live feature builder must reproduce the same feature definitions and

time-window calculations used during training.



The frontend should not load the ML models directly. Model inference should

run in the backend/API service.



\## Security



Do not expose model files or Python execution directly to an untrusted frontend.

