import csv
import json

input_file_path = 'md_QA_embedded.csv'
output_json_file_path = 'output_for_fine_tuning.json'

# Read the CSV file
with open(input_file_path, 'r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    data = list(reader)

# Prepare data for JSON output
json_data = []
for row in data:
    question = row['Questions']
    answer = row['Answers']

    if question and answer:  # Exclude rows with empty questions or answers
        message_system = {"role": "system", "content": "你是一位人力资源专家。"}
        message_user = {"role": "user", "content": question}
        message_assistant = {"role": "assistant", "content": answer}

        json_object = {"messages": [message_system, message_user, message_assistant]}
        json_data.append(json_object)

# Write the prepared data to a new JSON file
with open(output_json_file_path, 'w', encoding='utf-8') as json_file:
    for json_object in json_data:
        json.dump(json_object, json_file, ensure_ascii=False)
        json_file.write('\n')

print(f'JSON data has been written to {output_json_file_path}')
