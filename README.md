# forge-neo-moebius

[hustvl/Moebius](https://github.com/hustvl/Moebius)（0.2B 軽量インペイントモデル）を **Stable Diffusion WebUI Forge - Neo** のインペイント補助（Masked content）・単体インペイント（Extras）・一括処理（Batch）として利用するための拡張機能です。

## 主な機能

- **img2img -> Inpaint「Masked content」拡張**
  - `Moebius (Places2)`（自然風景・建物・背景）
  - `Moebius (CelebA-HQ)`（ポートレート）
  - `Moebius (FFHQ)`（顔修復・クローズアップ）
  - `Moebius (Pretrained)`（汎用：デフォルト OFF、Settings で有効化可能）
- **カスタム FT モデル対応**
  - `weights/custom/` に `.bin` / `.safetensors` / `.pt` / `.pth` を配置すると、再起動後に `Moebius (Custom: <ファイル名>)` として自動認識されます。
- **Settings -> Moebius Inpaint**
  - 各モデルの有効/無効、Upscaler、Resolution（短辺 / 長辺上限）、Steps、CFG Scale、Boundary compensate（境界補償）、Seed モード、推論後の CPU アンロードを設定可能
- **img2img -> Batch / Extras（Postprocessing）対応**
  - ディレクトリ一括処理および Extras タブでの Moebius 単体インペイントに対応

## インストール

Forge Neo の **Extensions -> Install from URL** にこのリポジトリの URL を入力してインストールし、WebUI を再起動してください。

- 追加の `pip install` は不要です（Forge Neo 標準環境で動作します）。
- モデル重みおよび VAE（`hustvl/PixelHacker`）は初回実行時に Hugging Face から `weights/` へ自動ダウンロードされます。

## ライセンス

- 本拡張機能および同梱の推論モジュール（`moebius_vendor/`）は [Apache License 2.0](LICENSE) に基づきます。
- 詳細は [NOTICE](NOTICE) をご参照ください。
