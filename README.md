# 強化学習アルゴリズム実装リポジトリ (Genesis & Gymnasium)

## 概要 (Overview)

このリポジトリは、強化学習 (Reinforcement Learning, RL) の様々なアルゴリズムを実装し、その動作を理解することを目的としています。将来的には、Genesisプラットフォームを活用した高度な強化学習タスクへの応用も視野に入れています。

このプロジェクトは、AIアシスタント「Jules」との共同作業によって開発されています。

## 現在のステータス (Current Status)

現在実装済みの主なアルゴリズムは以下の通りです。

*   **方策反復法 (Policy Iteration):** Gymnasiumの`FrozenLake-v1`環境向けに実装されています。
    *   実装ファイル: `algorithms/gymnasium_examples/model_based/policy_iteration_frozenlake.py`
    *   関連ドキュメント:
        *   `mdp_explanation.md`: マルコフ決定過程 (MDP) の基本説明。
        *   `frozen_lake_explanation.md`: `FrozenLake-v1` 環境の詳細説明。
        *   `policy_iteration_explanation.md`: 方策反復法のアルゴリズム解説。

## ディレクトリ構造 (Directory Structure)

現在の主要なディレクトリ構造は以下の通りです。

```
/
├── algorithms/              # 強化学習アルゴリズムのコード
│   ├── gymnasium_examples/  # Gymnasium環境を使用したアルゴリズムの例
│   │   └── model_based/     # モデルベースのアルゴリズム
│   │       └── policy_iteration_frozenlake.py  # FrozenLake環境の方策反復法実装
│   └── genesis_algorithms/  # Genesisプラットフォーム用アルゴリズム (今後追加予定)
├── mdp_explanation.md       # マルコフ決定過程(MDP)の説明
├── frozen_lake_explanation.md # FrozenLake-v1環境の説明
├── policy_iteration_explanation.md # 方策反復法のアルゴリズム説明
├── pyproject.toml           # プロジェクト設定と依存関係
├── .python-version          # Pythonバージョン指定 (uvが利用)
├── README.md                # このファイル
```

## セットアップ方法 (Setup Instructions)

1.  **リポジトリのクローン:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Python環境の準備:**
    *   このプロジェクトでは **Python 3.12以上** が必要です (`pyproject.toml` を参照)。
    *   プロジェクトルートに `.python-version` ファイルを配置して、使用する正確なPythonバージョン (例: `3.12.4`) を指定することを推奨します。`uv` はこのファイルを尊重して適切なPythonインタプリタを探します。
    *   `uv` を使用して仮想環境を作成し、有効化します。
        ```bash
        uv venv  # .venv という名前で仮想環境を作成します
        source .venv/bin/activate  # Linux/macOS の場合
        # .venv\Scripts\activate    # Windows の場合
        ```
    *   `uv` がインストールされていない場合は、まず `pip install uv` を実行してください。

3.  **依存ライブラリのインストール:**
    *   仮想環境を有効にした後、`uv` を使用してプロジェクトの依存関係をインストールします。
    ```bash
    uv pip install .
    ```
    *   もし `uv` を使用しない場合は、従来の `pip` でもインストール可能です（Pythonバージョンと仮想環境の管理は手動で行う必要があります）。
    ```bash
    pip install .
    ```

## 実行方法 (How to Run Examples)

### 方策反復法 (FrozenLake-v1)

仮想環境が有効化されていることを確認してから、以下のコマンドで `FrozenLake-v1` 環境における方策反復法の実装を実行できます。

```bash
python algorithms/gymnasium_examples/model_based/policy_iteration_frozenlake.py
```
実行後、コンソールに学習過程、最適価値関数、最適方策、およびテストエピソードの結果が出力されます。

## 今後の展望 (Future Work)

今後は以下の項目に取り組んでいく予定です。

*   価値反復法 (Value Iteration) の実装。
*   Q学習 (Q-Learning)、SARSAなどのモデルフリーアルゴリズムの実装。
*   より複雑なGymnasium環境への適用。
*   Genesisプラットフォームを利用したカスタム環境でのアルゴリズム開発および実験。
*   深層強化学習 (Deep Reinforcement Learning) アルゴリズムの導入。

## コントリビューションについて (Contribution Guidelines)

このプロジェクトへのコントリビューションを歓迎します。新しいアルゴリズムや改善提案がある場合は、以下の簡単な指針に従ってください。

*   **アルゴリズムの配置:**
    *   Gymnasium環境向けのアルゴリズムは `algorithms/gymnasium_examples/` 以下に、適切なサブディレクトリ（例: `model_based`, `model_free`）を作成して配置してください。
    *   Genesisプラットフォーム向けのアルゴリズムは `algorithms/genesis_algorithms/` 以下に配置予定です。
*   **コードのコメント:**
    *   コードには、処理内容やロジックを説明するコメントを適切に付与してください。特に、関数やクラスの目的、主要な処理ブロックについては詳細な説明が望ましいです。
    *   日本語でのコメントを推奨しますが、英語でも構いません。
*   **ドキュメント:**
    *   可能であれば、実装したアルゴリズムの理論的背景や使用方法について簡単な説明ファイルをMarkdown形式で追加してください。
*   **Pull Request:**
    *   変更内容は、明確なコミットメッセージと共にPull Requestとして送信してください。

ご不明な点があれば、Issueでお気軽にご質問ください。
```
