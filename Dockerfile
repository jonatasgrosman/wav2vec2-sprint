FROM ovhcom/ai-training-one-for-all

RUN apt-get update && \
    apt install -y bash \
    build-essential \
    libsndfile1-dev \
    git-lfs \
    sox

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs && \
    git lfs install

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch==1.8.1 \
    torchaudio==0.8.1 \
    datasets==1.5.0 \
    jiwer==2.2.0 \
    soundfile==0.10.3.post1 \
    lang-trans==0.6.0 \
    librosa==0.8.0 \
    samplerate==0.1.0 \
    scipy==1.5.4 \
    audiomentations==0.16.0 \
    torch-audiomentations==0.6.0 \
    pyloudnorm==0.1.0 \
    wandb==0.10.23 \
    homoglyphs==2.0.4 \
    gdown

RUN pip3 uninstall -y typing allennlp

RUN pip3 install git+https://github.com/huggingface/transformers.git

RUN mkdir -p /workspace/wav2vec/

COPY finetune.sh finetune_with_params.sh run_common_voice.py dataset_ext.py cer.py wer.py common_voice_eval.py /workspace/wav2vec/

COPY home-server.html run_all.sh /usr/bin/

RUN chown -R 42420:42420 /workspace

RUN chown -R 42420:42420 /usr/bin/run_all.sh

#Default training env variables
ENV model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    dataset_config_name="fr" \
    output_dir="/workspace/output_models/wav2vec2-large-xlsr-french-demo" \
    cache_dir="/workspace/data" \
    num_train_epochs="1" \
    per_device_train_batch_size="32" \
    evaluation_strategy="steps" \
    learning_rate="3e-4" \
    warmup_steps="500" \
    save_steps="10" \
    eval_steps="10" \
    save_total_limit="1" \
    logging_steps="10" \
    feat_proj_dropout="0.0" \
    layerdrop="0.1" \
    max_train_samples=100 \
    max_val_samples=100

WORKDIR /workspace
ENTRYPOINT []
#CMD ["sh", "/usr/bin/run_all.sh"]
CMD ["supervisord", "-n", "-u", "42420", "-c", "/etc/supervisor/supervisor.conf"]
