from tokenizers import BertWordPieceTokenizer
import os
import random
import argparse

"""
python tokenizer_utils.py --file_name "/work2/huchida/PSD_DC3/LogGeneratorForAnomalyDetection/datasets_for_models/sample_param/train/dataset_train_10000.txt" --vocab_size 100
"""


class TokenizerTrainer:
    def __init__(self, file_name, vocab_size=20000, add_special_tokens=False, shuffle_special_tokens=False):
        self.file_name = file_name
        self.vocab_size = vocab_size
        self.tokenizer = BertWordPieceTokenizer()
        self.save_path = f'trained_tokenizer/vocab_size_{vocab_size}'
        self.add_special_tokens_flag = add_special_tokens
        self.shuffle_special_tokens_flag = shuffle_special_tokens

    def add_special_tokens(self):
        # 0埋め5桁の数値を特別なトークンとして追加
        if self.add_special_tokens_flag:
            additional_special_tokens = [str(i) for i in range(10001)]
            if self.shuffle_special_tokens_flag:
                random.shuffle(additional_special_tokens)
        else:
            additional_special_tokens = []

        default_special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        combined_special_tokens = default_special_tokens + additional_special_tokens
        # print("combined_special_tokens: ", combined_special_tokens)
        print(len(default_special_tokens), len(combined_special_tokens))

        return combined_special_tokens

    def train_tokenizer(self):
        # トークナイザーの学習
        self.tokenizer.train(
            files=[self.file_name],
            vocab_size=self.vocab_size,
            min_frequency=1,
            special_tokens=self.add_special_tokens()
        )
        print(self.vocab_size)

    def save_tokenizer(self):
        # 学習したトークナイザーを保存
        os.makedirs(self.save_path, exist_ok=True)
        self.tokenizer.save_model(self.save_path)

    def reload_tokenizer(self):
        # トークナイザーを再ロード
        self.tokenizer = BertWordPieceTokenizer(os.path.join(self.save_path, "vocab.txt"))

    def check_special_tokens(self, token):
        # 追加したトークンが存在するかどうかを確認
        print("check_special_tokens: ", self.tokenizer.encode(token).tokens)

    def check_mask_token(self, text):
        encoded_input = self.tokenizer.encode(text).ids
        mask_index = encoded_input.index(self.tokenizer.token_to_id("[MASK]"))
        print(text)
        print(encoded_input)
        print(self.tokenizer.token_to_id("[MASK]"))
        print(mask_index)


if __name__ == "__main__":
    """
    - 特別なトークンを追加し、それをシャッフルする場合
    python tokenizer_utils.py --file_name "my_dataset.txt" --vocab_size 100 --add_special_tokens --shuffle_special_tokens
    - 特別なトークンを追加するが、シャッフルしない場合
    python tokenizer_utils.py --file_name "my_dataset.txt" --vocab_size 100 --add_special_tokens
    - 特別なトークンを追加しない場合
    python tokenizer_utils.py --file_name "my_dataset.txt" --vocab_size 100


    python tokenizer_utils.py --file_name "my_dataset.txt" --vocab_size 30000
    python tokenizer_utils.py --file_name "/work2/huchida/PSD_DC3/LogGeneratorForAnomalyDetection/datasets_for_models/sample_param/train/dataset_train_10000.txt" --vocab_size 100

    # For Sample
    python tokenizer_utils.py --file_name /work2/huchida/PSD_DC3/LogGeneratorForAnomalyDetection/datasets_for_models/sample_param/train/dataset_train_10000.txt  --add_special_tokens --shuffle_special_tokens
    """

    parser = argparse.ArgumentParser(description="Train a tokenizer for text processing.")
    parser.add_argument("--file_name", type=str, default="dataset_train_modified.txt",
                        help="The name of the file to train the tokenizer on.")
    parser.add_argument("--vocab_size", type=int, default=20000,
                        help="The vocabulary size for the tokenizer.")
    parser.add_argument("--add_special_tokens", action='store_true',
                        help="Whether to add additional special tokens.")
    parser.add_argument("--shuffle_special_tokens", action='store_true',
                        help="Whether to shuffle the additional special tokens.")

    args = parser.parse_args()

    # TokenizerTrainerのインスタンスを作成し、トークナイザーを訓練
    trainer = TokenizerTrainer(file_name=args.file_name, vocab_size=args.vocab_size,
                               add_special_tokens=args.add_special_tokens,
                               shuffle_special_tokens=args.shuffle_special_tokens)
    trainer.train_tokenizer()
    trainer.save_tokenizer()
    trainer.reload_tokenizer()
    trainer.check_special_tokens("1")
    trainer.check_special_tokens("<mask>")
    trainer.check_special_tokens("[MASK]")
    trainer.check_mask_token("state = C, value = 903, [MASK]")