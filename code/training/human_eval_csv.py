import json
import pandas as pd

def generate_comprehensive_csv_v4(base_json_path, ft_json_path, output_csv):
    # Load both JSON files
    with open(base_json_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    with open(ft_json_path, 'r', encoding='utf-8') as f:
        ft_data = json.load(f)

    df_base = pd.DataFrame(base_data['results'])
    df_ft = pd.DataFrame(ft_data['results'])

    # 1. Format List fields for CSV safety
    list_fields = ['cited_cases', 'section_titles', 'section_texts', 'cited_case_ids', 'cited_case_judgments']
    for col in list_fields:
        if col in df_base.columns:
            df_base[col] = df_base[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

    # 2. Rename model-specific columns in the Base dataframe
    df_base = df_base.rename(columns={'per_sample_loss': 'loss_base'})

    # 3. Prepare the Fine-tuned model columns (Output + Loss)
    df_ft_subset = df_ft[['Case Name', 'finetuned_model_output', 'per_sample_loss']].rename(
        columns={
            'finetuned_model_output': 'finetuned_model_output_v10',
            'per_sample_loss': 'loss_ft_v10'
        }
    )

    # 4. Merge on 'Case Name'
    final_df = pd.merge(df_base, df_ft_subset, on='Case Name', how='inner')

    # 5. Save final result
    final_df.to_csv(output_csv, index=False)
    print(f"File saved to {output_csv}")

# Execution
generate_comprehensive_csv_v4(
    '/nfs/iitkanpur/parjanya/nyayamind/logs/base_eval_qwen3_pilot.json',
    '/nfs/iitkanpur/parjanya/nyayamind/logs/Qwen3_DDP_v14_final_finetuned_eval_pilot.json', 
    '/nfs/iitkanpur/parjanya/nyayamind/logs/human_eval_v15_qwen3.csv'
)