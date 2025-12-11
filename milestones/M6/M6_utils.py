# --- environment simulator for Q-learning ---
rng = np.random.default_rng(0)

def sample_next_state(s, a):
    return rng.choice(n_states, p=trans_mat[s, a])

def reward_of(s, a):
    return r_sa[s, a]

class PatientEnv:
    def __init__(self, pi, trans_mat, r_sa, horizon=6, rng=None):
        self.pi = pi
        self.P = trans_mat
        self.R = r_sa
        self.horizon = horizon  
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.t = 0
        self.s = self.rng.choice(n_states, p=self.pi)
        return self.s

    def step(self, a):
        r = float(self.R[self.s, a])
        s_next = self.rng.choice(n_states, p=self.P[self.s, a])
        self.s = s_next
        self.t += 1
        done = (self.t >= self.horizon)
        return s_next, r, done, {}

env = PatientEnv(pi, trans_mat, r_sa, horizon=7, rng=rng)



# --- Generic simulator for any policy ---
def evaluate_policy(env, policy_fn, Q=None, episodes=1000, seed=0):
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(episodes):
        s = env.reset()
        G = 0.0
        for t in range(env.horizon):
            a = policy_fn(s, Q=Q, rng=rng)
            s, r, done, _ = env.step(a)
            G += r
            if done: break
        returns.append(G)
    return np.array(returns)