import bleurt
from bleurt import score

scorer = score.BleurtScorer("bleurt-base-128")

for model in "inquisitive_model_ct_setting10_better greedy_model repetition_model_settinginf repetition_model_setting05 repetition_model_setting12 inquisitive_model_ct_setting07 interesting_nidf_model_bfw_setting_08 responsiveness_model_bfw_setting_00 inquisitive_model_ct_setting01 repetition_model_setting35_settinginf interesting_nidf_model_ct_setting9 interesting_nidf_model_ct_setting2 responsiveness_model_bfw_setting_10 responsiveness_model_bfw_setting_minus_10 interesting_nidf_model_ct_setting7 interesting_nidf_model_bfw_setting_06 repetition_model_setting35 interesting_nidf_model_bfw_setting_minus_10 responsiveness_model_bfw_setting_05 inquisitive_model_ct_setting10 inquisitive_model_ct_setting04 interesting_nidf_model_ct_setting0 interesting_nidf_model_ct_setting4 interesting_nidf_model_bfw_setting_minus_04 inquisitive_model_ct_setting00 interesting_nidf_model_bfw_setting_04 responsiveness_model_bfw_setting_13 baseline_model".split(' '):
  print(model)
  with open(model+"_ref.txt") as f:
    refs = f.readlines()
    refs = [i.strip() for i in refs]
  with open(model+"_gen.txt") as f:
    gens = f.readlines()
    gens = [i.strip() for i in gens]

# references = ["This is a test."]
# candidates = ["This is the test."]

  scores = scorer.score(refs, gens)
  # assert type(scores) == list and len(scores) == 1
  print(sum(scores)/len(scores))
