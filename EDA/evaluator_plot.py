import pandas as pd
import matplotlib.pyplot as plt


df_normal = pd.read_csv("./output/evaluation_results/20250401_synthesized_data_eval/v2_plos_synthesized_evaluation_results_normal_summaries.csv")
df_bad = pd.read_csv("./output/evaluation_results/20250401_synthesized_data_eval/v2_plos_synthesized_evaluation_results_bad_summaries.csv")
df_paraphrase = pd.read_csv("./output/evaluation_results/20250401_synthesized_data_eval/v2_plos_synthesized_evaluation_results_paraphrase_summaries.csv")


plt.figure(figsize=(10, 6))
plt.boxplot(
    [df_paraphrase['BioSumm (GEval)_score'],df_normal['BioSumm (GEval)_score'], df_bad['BioSumm (GEval)_score'], ],
    labels=['Paraphrased ref Summaries','Normal Summaries', 'Bad Summaries'],
)
plt.ylabel('GEval Score', fontsize=14)
plt.title('')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.savefig('./figures/EDA/boxplot_geval_scores_evaluator_evaluation.png')
plt.close()