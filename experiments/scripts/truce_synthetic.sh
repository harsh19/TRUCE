
##### Sanity check

MODEL_NAME=truce_synth6_exp1a_new9_sanity # with experiments/configs/config_synthetic.json
CUDA_VISIBLE_DEVICES=0 python -m allennlp.run train experiments/configs/config_synthetic.json --serialization-dir tmp/"$MODEL_NAME" --include-package allennlp_series --overrides '{ "dataset_reader": { "debug": true }, "model":{}, "trainer":{"patience": 150,"num_epochs":10,"validation_metric":"+Bleu_4","cuda_device":0}, "random_seed":555, "numpy_seed":555, "pytorch_seed":555  }'


##### TRAIN

MODEL_NAME=truce_synth6_exp1a_new9 # with experiments/configs/config_synthetic.json
CUDA_VISIBLE_DEVICES=0 python -m allennlp.run train experiments/configs/config_synthetic.json --serialization-dir tmp/"$MODEL_NAME" --include-package allennlp_series --overrides '{ "dataset_reader": { "debug": false }, "model":{}, "trainer":{"patience": 150,"num_epochs":600,"validation_metric":"+Bleu_4","cuda_device":0}, "random_seed":555, "numpy_seed":555, "pytorch_seed":555  }'


########### EVALS
MODEL_NAME=truce_synth6_exp1a_new9
FNAME=processed_data/pilot13finalval.json
TEST_FNAME=processed_data/pilot13finaltest.json
# val
CUDA_VISIBLE_DEVICES="" python -m allennlp.run evaluate --output-file tmp/"$MODEL_NAME".dev.evals --weights-file tmp/"$MODEL_NAME"/best.th --include-package allennlp_series --overrides '{ "dataset_reader": { "debug": false }, "model":{"model_name":"truce_synth6_exp1a_new9", "use_bertscore_evals":true} }' --cuda-device -1 tmp/"$MODEL_NAME"/model.tar.gz  $FNAME
# test
CUDA_VISIBLE_DEVICES="" python -m allennlp.run evaluate --output-file tmp/"$MODEL_NAME".test.evals --weights-file tmp/"$MODEL_NAME"/best.th --include-package allennlp_series --overrides '{ "dataset_reader": { "debug": false }, "model":{"model_name":"truce_synth6_exp1a_new9", "use_bertscore_evals":true} }' --cuda-device -1 tmp/"$MODEL_NAME"/model.tar.gz  $TEST_FNAME
