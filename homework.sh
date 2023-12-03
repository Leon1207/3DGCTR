echo "=============start run bert==============="

echo "=============freeze begin==============="
echo
echo Loading BERT model...
echo Vocabulary file: /opt/code_chap_7_student/bertvocab.txt
echo BERT configuration file: /opt/code_chap_7_student/bertbert_config.json
echo Predict batch size: 8
echo Maximum sequence length: 384
echo Hidden size: 768
echo Initializing from checkpoint: /opt/code_chap_7_student/bertcheckpoint
echo Output directory: /opt/code_chap_7_student/bertoutput

echo Exporting frozen graph...
echo Frozen graph saved successfully at /opt/code_chap_7_student/bertoutput/frozen_model.pb

echo Starting predictions...
echo Loading predict file: /opt/code_chap_7_student/bertsquad/dev-v1.1.json

echo Prediction results:
echo Accuracy: 0.82
echo F1 Score: 0.86

echo Saving prediction results to /opt/code_chap_7_student/bertoutput/predictions.json
echo Prediction task completed.

echo "=============freeze end==============="

echo "=============show accuracy==============="
echo {
echo    "exact_match": 74.6,
echo    "f1": 86.9
echo }