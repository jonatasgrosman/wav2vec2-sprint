#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import csv


# In[20]:


with open('wav2vec_languages.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # This skips the first row of the CSV file because it's a header
    next(csv_reader)
    for (language_code, language_full_name) in csv_reader:
        print(f"#Launching Training for {language_code}-{language_full_name}")
        cmd = f"ovhai job run --gpu 1 --name '{language_code}-{language_full_name}' --volume output_models@GRA/{language_code}:/workspace/output_models:RW:cache -e model_name_or_path='facebook/wav2vec2-large-xlsr-53' -e dataset_config_name={language_code} -e output_dir='/workspace/output_models/wav2vec2-large-xlsr-{language_code}-{language_full_name}-demo' -e cache_dir='/workspace/data' -e num_train_epochs=10 databuzzword/hf-wav2vec -- sh /workspace/wav2vec/finetune_with_params.sh"
        print(cmd)
        stream = os.popen(cmd)
        output = stream.read()
        output


# In[3]:





# In[ ]:




