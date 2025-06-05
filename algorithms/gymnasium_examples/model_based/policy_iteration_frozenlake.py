import gymnasium
import numpy as np
import time # For potential rendering delays (未使用だが、デバッグ用に残すことも可能)

# --- 可視化関数 ---
def print_value_function(V, rows, cols, message="Value Function:"):
    """
    価値関数をグリッド形式で表示します。
    Args:
        V (np.array): 表示する価値関数 (1次元配列)。
        rows (int): グリッドの行数。
        cols (int): グリッドの列数。
        message (str): 表示するメッセージ。
    """
    print(message)
    grid_v = V.reshape((rows, cols)) # 価値関数をグリッドの形状に変形
    for row in grid_v:
        # 各セルの値を整形して表示
        print(" ".join([f"{val:8.4f}" for val in row]))
    print("-" * (cols * 9)) # 区切り線

def print_policy(policy, rows, cols, message="Policy:"):
    """
    方策をグリッド形式で表示します。行動を記号で表現します。
    Args:
        policy (np.array): 表示する方策 (1次元配列)。各要素は状態における行動を示します。
        rows (int): グリッドの行数。
        cols (int): グリッドの列数。
        message (str): 表示するメッセージ。
    """
    print(message)
    # 行動を記号にマッピング (0: 左, 1: 下, 2: 右, 3: 上)
    action_symbols = {0: '<', 1: 'v', 2: '>', 3: '^'}
    # 方策配列を記号配列に変換し、グリッドの形状に変形
    grid_policy = np.array([action_symbols[action] for action in policy]).reshape(rows, cols)
    for row in grid_policy:
        print(" ".join(row)) # 各行の記号を表示
    print("-" * (cols * 2)) # 区切り線

# --- 方策評価関数 ---
def policy_evaluation(policy, P, n_states, n_actions, gamma=0.99, theta=1e-8):
    """
    与えられた方策に対して、状態価値関数を反復的に計算して評価します。
    ベルマン期待方程式に基づき、価値関数が収束するまで更新を繰り返します。

    Args:
        policy (np.array): 評価対象の方策。サイズn_statesの1次元配列で、各要素policy[s]が状態sで取る行動を示す。
        P (dict): 環境の遷移モデル。P[state][action] は (prob, next_state, reward, terminated) タプルのリスト。
        n_states (int): 環境の状態数。
        n_actions (int): 環境の行動数。(この関数では決定的方策を仮定するため直接的には使用しない)
        gamma (float): 割引率。将来の報酬をどれだけ重視するか。
        theta (float): 収束判定のための小さな閾値。価値関数の更新幅がこの値を下回ったら収束とみなす。

    Returns:
        np.array: 収束した状態価値関数V。サイズn_statesの1次元配列。
    """
    # V: 状態価値関数。全ての状態で0に初期化。
    V = np.zeros(n_states)
    eval_iter = 0 # 評価のイテレーション回数カウンタ
    while True: # 価値関数が収束するまでループ
        delta = 0 # 1回のイテレーションでの価値関数の最大更新幅
        V_old = np.copy(V) # 更新前の価値関数を保持（比較用）

        # 各状態sについて価値関数を更新
        for s in range(n_states):
            v_s = 0 # 現在の状態sの新しい価値
            action = policy[s] # 現在の方策policyに従って、状態sで取る行動actionを取得

            # P[s][action]には、状態sで行動actionを取った場合の遷移確率、次の状態、報酬、終了フラグのタプルリストが格納されている
            # 例: [(probability1, next_state1, reward1, terminated1), (probability2, next_state2, reward2, terminated2), ...]
            # FrozenLakeのis_slippery=Trueの場合、意図した方向に1/3、直角方向にそれぞれ1/3の確率で遷移する
            for prob, next_state, reward, terminated in P[s][action]:
                # ベルマン期待方程式の計算: Σ p(s',r|s,a) * (r + γ * V(s'))
                # ここでは方策が決定的なので、π(a|s)は1 (選択された行動に対して) または0 (それ以外の行動に対して)
                v_s += prob * (reward + gamma * V_old[next_state])
            V[s] = v_s # 状態sの価値を更新
            delta = max(delta, abs(V[s] - V_old[s])) # 更新幅の最大値を記録

        eval_iter += 1
        # (オプション) イテレーションごとの価値関数表示（デバッグ用、詳細すぎる場合はコメントアウト）
        # if eval_iter % 10 == 0:
        #     print(f"Policy Evaluation Iteration: {eval_iter}")
        #     print_value_function(V, int(np.sqrt(n_states)), int(np.sqrt(n_states)))

        # 収束判定:価値関数の更新幅が閾値thetaより小さくなったらループを終了
        if delta < theta:
            # print(f"Policy Evaluation converged in {eval_iter} iterations.") # 収束メッセージ
            break
    return V # 収束した価値関数を返す

# --- 方策改善関数 (policy_iteration関数内で直接実装) ---
# policy_improvement関数は、メインのpolicy_iterationループ内でロジックが展開されているため、
# ここでは独立した関数としては呼び出されていません。
# もし独立させる場合は以下のようなシグネチャになりますが、現在のコードでは不要です。
# def policy_improvement(V, P, n_states, n_actions, gamma=0.99):
#     """ (この関数は現在のコードでは直接使用されていません)
#     与えられた状態価値関数Vに基づいて方策を改善します。
#     各状態でQ値を最大化する行動を選択することで、新しい欲張り方策を生成します。
#     Args:
#         V (np.array): 状態価値関数。
#         P (dict): 環境の遷移モデル。
#         n_states (int): 状態数。
#         n_actions (int): 行動数。
#         gamma (float): 割引率。
#     Returns:
#         np.array: 改善された新しい方策。
#     """
#     # new_policy = np.zeros(n_states, dtype=int)
#     # for s in range(n_states):
#     #     q_values_s = np.zeros(n_actions)
#     #     for a in range(n_actions):
#     #         q_sa = 0
#     #         for prob, next_state, reward, terminated in P[s][a]:
#     #             q_sa += prob * (reward + gamma * V[next_state])
#     #         q_values_s[a] = q_sa
#     #     new_policy[s] = np.argmax(q_values_s)
#     # return new_policy

# --- 方策反復法メイン関数 ---
def policy_iteration(env, P, n_states, n_actions, rows, cols, gamma=0.99, theta=1e-8):
    """
    方策反復法を実行して、最適方策と最適価値関数を見つけ出します。
    方策評価と方策改善を方策が安定するまで繰り返します。

    Args:
        env: Gym環境インスタンス (P, n_states, n_actionsの取得に使用)。
        P (dict): 環境の遷移モデル。
        n_states (int): 状態数。
        n_actions (int): 行動数。
        rows (int): グリッドの行数（表示用）。
        cols (int): グリッドの列数（表示用）。
        gamma (float): 割引率。
        theta (float): 方策評価の収束判定用閾値。

    Returns:
        tuple: (optimal_policy, optimal_value_function)
               optimal_policy (np.array): 見つかった最適方策。
               optimal_value_function (np.array): 最適方策に対応する最適状態価値関数。
    """
    # 1. 初期化
    # policy: 方策。最初はランダムまたは単純な方策（例：全状態で行動0（左）を選択）で初期化。
    # policy = np.random.randint(0, n_actions, n_states) # ランダムな初期方策
    policy = np.zeros(n_states, dtype=int) # 全ての状態で左に行く初期方策
    V = np.zeros(n_states) # V: 状態価値関数。0で初期化。

    iteration = 0 # 方策反復のイテレーション回数カウンタ
    print_policy(policy, rows, cols, f"初期方策 (イテレーション {iteration}):")

    while True: # 方策が安定するまでループ
        iteration += 1
        print(f"\n--- 方策反復: ループ {iteration} ---")

        # 2. 方策評価 (Policy Evaluation)
        # 現在の方策 policy に基づいて、状態価値関数 V を計算する。
        print("方策評価を開始...")
        V = policy_evaluation(policy, P, n_states, n_actions, gamma, theta)
        print_value_function(V, rows, cols, f"評価後の価値関数 (方策反復ループ {iteration}):")

        # 3. 方策改善 (Policy Improvement)
        # 計算された価値関数 V を使って、より良い方策 new_policy_candidate を見つける。
        print("\n方策改善を開始...")
        new_policy_candidate = np.zeros(n_states, dtype=int) # 改善候補の方策

        # 各状態 s について、Q(s,a) を最大化する行動 a を選択する。
        for s in range(n_states):
            q_values_s = np.zeros(n_actions) # 状態sにおける各行動のQ値を格納
            for a in range(n_actions): # 各行動 a についてQ値を計算
                q_sa = 0 # Q(s,a) の初期値
                # P[s][a] から遷移情報を取得し、Q値を計算
                # Q(s,a) = Σ p(s',r|s,a) * (r + γ * V(s'))
                for prob, next_state, reward, terminated in P[s][a]:
                    q_sa += prob * (reward + gamma * V[next_state])
                q_values_s[a] = q_sa
            new_policy_candidate[s] = np.argmax(q_values_s) # Q値が最大の行動を新しい方策として選択

        print_policy(new_policy_candidate, rows, cols, f"改善候補の方策 (方策反復ループ {iteration}):")

        # 方策の安定性チェック
        # 新しい方策 new_policy_candidate が現在の policy と同じであれば、方策は安定しており、最適解に到達したとみなす。
        if np.array_equal(new_policy_candidate, policy):
            print("\n方策が安定しました！")
            policy = new_policy_candidate # 最終的な最適方策を更新
            break # ループを終了
        else:
            print("\n方策が更新されました。反復を継続します。")
            policy = new_policy_candidate # 方策を更新して次のイテレーションへ

        # 安全停止（通常、方策反復は速く収束するが、念のため）
        if iteration > 50:
            print("最大反復回数に到達しました。終了します。")
            break

    return policy, V # 最適方策と最適価値関数を返す

# --- 学習済み方策のテスト関数 ---
def test_policy(env, policy, n_episodes=10, render_mode=None):
    """
    学習された方策を環境でテストし、その性能を評価します。

    Args:
        env: Gym環境インスタンス。
        policy (np.array): テスト対象の方策。
        n_episodes (int): テストするエピソード数。
        render_mode (str, optional): 'ansi'ならテキストベースで描画、Noneなら描画なし。
    """
    total_rewards = [] # 各エピソードで得られた総報酬を格納するリスト
    print(f"\n--- {n_episodes}エピソードでの方策テスト ---")

    # テスト用に新しい環境インスタンスを作成することもできるが、
    # FrozenLakeの場合は既存のenvをresetして使っても通常問題ない。
    for episode in range(n_episodes):
        state, info = env.reset() # 環境を初期状態にリセット
        terminated = False # エピソード終了フラグ
        truncated = False # エピソード打ち切りフラグ (時間制限など)
        episode_reward = 0 # このエピソードでの累積報酬
        step_count = 0 # ステップ数カウンタ

        print(f"\nエピソード {episode + 1}:")
        if render_mode == 'ansi':
            print(env.render()) # 初期状態を描画 (ANSI形式のテキスト)

        # エピソードが終了 (terminated or truncated) するまでループ
        while not (terminated or truncated):
            action = policy[state] # 現在の状態 state で方策 policy に従って行動 action を選択
            # 選択した行動を実行し、次の状態、報酬、終了・打ち切りフラグ、情報（デバッグ用）を取得
            next_state, reward, terminated, truncated, info = env.step(action)

            if render_mode == 'ansi':
                print(f"ステップ {step_count}: 状態: {state}, 行動: {action}, 報酬: {reward}, 次状態: {next_state}, 終了: {terminated}")
                print(env.render()) # ステップ後の状態を描画
                # time.sleep(0.1) # (オプション) 表示を見やすくするための遅延

            episode_reward += reward # 報酬を累積
            state = next_state # 状態を更新
            step_count += 1
            if step_count > 100: # 無限ループを防ぐための最大ステップ数制限
                print("エピソードが最大ステップ数に達したため打ち切り。")
                truncated = True

        total_rewards.append(episode_reward) # このエピソードの総報酬を記録
        print(f"エピソード {episode + 1} 終了。総報酬: {episode_reward}")

    avg_reward = np.mean(total_rewards) # 全エピソードの平均報酬
    print(f"\n{n_episodes}エピソードの平均報酬: {avg_reward:.2f}")
    # 成功率: ゴールに到達した場合の報酬は1、それ以外は0なので、報酬が0より大きいエピソードの割合
    print(f"成功率: {np.sum(np.array(total_rewards) > 0) / n_episodes :.2f}")


# --- メイン実行ブロック ---
if __name__ == '__main__':
    # 1. 環境設定
    # FrozenLake-v1 環境を初期化
    # map_name: "4x4" または "8x8" を選択可能
    # is_slippery=True: 床が滑りやすい設定（エージェントの行動が確率的になる）
    # render_mode='ansi': テキストベースでの描画モード。ヘッドレス環境での実行に適している。
    # 'human'モードはGUI表示だが、サーバー環境などでは使えないことが多い。
    map_name = "4x4"
    env = gymnasium.make('FrozenLake-v1', map_name=map_name, is_slippery=True, render_mode='ansi')

    # n_states: 状態空間のサイズ (例: 4x4なら16)
    n_states = env.observation_space.n
    # n_actions: 行動空間のサイズ (例: 上下左右の4行動)
    n_actions = env.action_space.n

    # P: 遷移モデル (Transition Model)
    # P[state][action] は (probability, next_state, reward, terminated) のタプルのリストを返す。
    # FrozenLakeでは env.P で直接アクセス可能。他の環境では env.unwrapped.P が必要な場合もある。
    P = env.unwrapped.P

    # グリッドの行数・列数を設定（表示用）
    if map_name == "4x4":
        rows, cols = 4, 4
    elif map_name == "8x8":
        rows, cols = 8, 8
    else: # カスタムマップの場合のフォールバック（正方形を仮定）
        dim = int(np.sqrt(n_states))
        if dim * dim == n_states: # 正方形グリッドか確認
            rows, cols = dim, dim
        else:
            rows, cols = 1, n_states # 正方形でなければ1行で表示

    print(f"環境: FrozenLake-v1 ({map_name}, slippery)")
    print(f"状態数: {n_states}")
    print(f"行動数: {n_actions}")
    # print("遷移モデル P (状態0, 行動0の例):")
    # print(P[0][0]) # 例: [(0.333..., 0, 0.0, False), ...]

    # 2. ハイパーパラメータ設定
    gamma = 0.99   # 割引率 (gamma): 将来の報酬をどれだけ割り引くか (0に近いほど近視眼的、1に近いほど遠視眼的)
    theta = 1e-9   # 収束閾値 (theta): 方策評価の際の価値関数の更新量の許容誤差

    # 3. 方策反復法の実行
    print("\n方策反復アルゴリズムを開始...")
    optimal_policy, optimal_V = policy_iteration(env, P, n_states, n_actions, rows, cols, gamma, theta)

    # 4. 最終結果の表示
    print("\n--- 最適解 ---")
    print_value_function(optimal_V, rows, cols, "最適価値関数 (V*):")
    print_policy(optimal_policy, rows, cols, "最適方策 (π*):")

    # 5. 学習済み方策のテスト実行
    # ANSIモードでテストを行う。
    test_policy(env, optimal_policy, n_episodes=10, render_mode='ansi')

    env.close() # 環境を閉じる
