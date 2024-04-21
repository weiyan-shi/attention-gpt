import os
import json

folder_path = 'C:\\Users\\weiyan.shi\\Desktop\\attention-gpt\\DREAMdataset\\User 3'

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if 'task' in data and 'ability' in data['task'] and data['task']['ability'] == 'JA':
                    print(f'Found in file: {filename}')
            except json.JSONDecodeError as e:
                print(f'Invalid JSON in file: {filename}. Error: {e}')

print('Search completed.')
