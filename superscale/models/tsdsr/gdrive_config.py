"""Google Drive configuration for TSD-SR models.

このファイルを編集して、あなたのGoogle DriveファイルIDを設定してください。

手順:
1. TSD-SRのファイルをあなたのGoogle Driveにアップロード
2. 各ファイルを右クリック → 「リンクを取得」
3. 「リンクを知っている全員」に設定
4. URLからファイルIDを抽出:
   https://drive.google.com/file/d/[FILE_ID]/view?usp=sharing
   
5. 下記の YOUR_*_FILE_ID を実際のファイルIDに置き換える
"""

# TSD-SR model file IDs
TSDSR_FILE_IDS = {
    "tsdsr": {
        "transformer.safetensors": "YOUR_TRANSFORMER_FILE_ID",  # 置き換えてください
        "vae.safetensors": "YOUR_VAE_FILE_ID",  # 置き換えてください
        "prompt_embeds.pt": "YOUR_PROMPT_EMBEDS_FILE_ID",  # 置き換えてください
        "pool_embeds.pt": "YOUR_POOL_EMBEDS_FILE_ID",  # 置き換えてください
    },
    "tsdsr-mse": {
        "transformer.safetensors": "YOUR_MSE_TRANSFORMER_FILE_ID",  # 置き換えてください
        "vae.safetensors": "YOUR_MSE_VAE_FILE_ID",  # 置き換えてください
        "prompt_embeds.pt": "YOUR_PROMPT_EMBEDS_FILE_ID",  # 共通の場合は同じID
        "pool_embeds.pt": "YOUR_POOL_EMBEDS_FILE_ID",  # 共通の場合は同じID
    },
    "tsdsr-gan": {
        "transformer.safetensors": "YOUR_GAN_TRANSFORMER_FILE_ID",  # 置き換えてください
        "vae.safetensors": "YOUR_GAN_VAE_FILE_ID",  # 置き換えてください
        "prompt_embeds.pt": "YOUR_PROMPT_EMBEDS_FILE_ID",  # 共通の場合は同じID
        "pool_embeds.pt": "YOUR_POOL_EMBEDS_FILE_ID",  # 共通の場合は同じID
    },
}

# Optional: File checksums for integrity verification
FILE_CHECKSUMS = {
    # "transformer.safetensors": "md5_hash",
    # "vae.safetensors": "md5_hash",
    # "prompt_embeds.pt": "md5_hash",
    # "pool_embeds.pt": "md5_hash",
}

# 例: 実際のファイルIDを設定した場合
"""
TSDSR_FILE_IDS = {
    "tsdsr": {
        "transformer.safetensors": "1AbCdEfGhIjKlMnOpQrStUvWxYz123456",
        "vae.safetensors": "1BcDeFgHiJkLmNoPqRsTuVwXyZ234567",
        "prompt_embeds.pt": "1CdEfGhIjKlMnOpQrStUvWxYz345678",
        "pool_embeds.pt": "1DeFgHiJkLmNoPqRsTuVwXyZ456789",
    },
}
"""