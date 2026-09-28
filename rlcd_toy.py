"""Toy RLCD-style pipeline with scripted answers and a two-choice policy.

This demonstrates contrastive pairs, preference training, and reward-guided
policy improvement. It does not generate language or implement PPO.
"""

import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"

questions = [
    "What is 2 + 2?",
    "What planet do we live on?",
    "What color is a ripe banana?",
]
answers = [
    ("4", "5"),
    ("Earth", "Mars"),
    ("Yellow", "Blue"),
]
n = len(questions)

# A tiny stand-in for a language model: two possible answers per question.
base_logits = torch.zeros(n, 2, device=device)
positive_prompt = "Answer accurately."
negative_prompt = "Answer carelessly."

def generate_answer(question_id, instruction):
    steering = [3.0, -3.0] if instruction == positive_prompt else [-3.0, 3.0]
    steering = torch.tensor(steering, device=device)
    return int((base_logits[question_id] + steering).argmax().item())

# Stage 1: generate contrasting answer pairs.
pairs = [
    (
        i,
        generate_answer(i, positive_prompt),
        generate_answer(i, negative_prompt),
    )
    for i in range(n)
]

chosen_ids = torch.tensor(
    [2 * i + chosen for i, chosen, rejected in pairs], device=device
)
rejected_ids = torch.tensor(
    [2 * i + rejected for i, chosen, rejected in pairs], device=device
)

print("Contrastive pairs:")
for i, chosen, rejected in pairs:
    print(f"  {questions[i]}  preferred={answers[i][chosen]}  other={answers[i][rejected]}")

# Stage 2: train a reward model to score preferred answers higher.
reward_model = torch.nn.Embedding(2 * n, 1, device=device)
torch.nn.init.zeros_(reward_model.weight)
reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.1)

for step in range(100):
    reward_optimizer.zero_grad()
    chosen_scores = reward_model(chosen_ids).squeeze(-1)
    rejected_scores = reward_model(rejected_ids).squeeze(-1)
    reward_loss = F.softplus(-(chosen_scores - rejected_scores)).mean()
    reward_loss.backward()
    reward_optimizer.step()

# Stage 3: improve a policy using the learned rewards.
policy_logits = torch.nn.Parameter(base_logits.clone())
policy_optimizer = torch.optim.Adam([policy_logits], lr=0.1)
all_answer_ids = torch.arange(2 * n, device=device).reshape(n, 2)

with torch.no_grad():
    rewards = reward_model(all_answer_ids).squeeze(-1).sigmoid()

for step in range(100):
    policy_optimizer.zero_grad()
    log_probs = F.log_softmax(policy_logits, dim=1)
    probs = log_probs.exp()
    expected_reward = (probs * rewards).sum(dim=1).mean()
    kl = (probs * (log_probs + math.log(2))).sum(dim=1).mean()
    policy_loss = -expected_reward + 0.1 * kl
    policy_loss.backward()
    policy_optimizer.step()

with torch.no_grad():
    preference_accuracy = (
        reward_model(chosen_ids) > reward_model(rejected_ids)
    ).float().mean().item()
    before = F.softmax(base_logits, dim=1)[:, 0].mean().item()
    after = F.softmax(policy_logits, dim=1)[:, 0].mean().item()

print("Reward-model training-pair accuracy:", preference_accuracy)
print("Preferred-answer probability before:", before)
print("Preferred-answer probability after:", after)
print("Device:", device)
